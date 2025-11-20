from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import os
from model import load_model
from fgsm import FGSMAttack

app = FastAPI()

# 1. Enable CORS
# This allows your Frontend (which will run on a different port) to talk to this Backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Model
# We force CPU usage to avoid compatibility issues during deployment
device = torch.device("cpu")
model = load_model().to(device)

# Robust logic to find the weights file
weights_path = ""
if os.path.exists("backend/mnist_cnn_weights.pth"):
    weights_path = "backend/mnist_cnn_weights.pth"
elif os.path.exists("mnist_cnn_weights.pth"):
    weights_path = "mnist_cnn_weights.pth"
else:
    print("WARNING: Weights file not found! Predictions will be random.")

if weights_path:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Model loaded successfully from {weights_path}")

model.eval()

# 3. Define Image Preprocessing
# This MUST match the transforms used in train.py exactly
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def tensor_to_base64(tensor):
    """
    Helper to convert a PyTorch tensor (the attacked image) 
    back to a Base64 string so the Frontend can display it.
    """
    # Denormalize to get back to valid pixel range [0, 1]
    # (Multiply by std deviation, add mean)
    tensor = tensor * 0.3081 + 0.1307
    
    # Clamp to ensure no invalid colors
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to standard image format
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    
    # Save to a memory buffer (like a fake file in RAM)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    # Convert that buffer to a string
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/attack")
async def attack_endpoint(
    file: UploadFile = File(...), 
    epsilon: float = Form(...)
):
    # A. Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # B. Preprocess the image (Resize -> Grayscale -> Tensor)
    input_tensor = transform(image).unsqueeze(0).to(device) 
    
    # C. Get Clean Prediction (Original)
    input_tensor.requires_grad = False
    clean_output = model(input_tensor)
    clean_pred = clean_output.max(1, keepdim=True)[1].item()
    
    # D. Run FGSM Attack
    # We clone the tensor because the attack modifies gradients
    attack_input = input_tensor.clone().detach()
    target = torch.tensor([clean_pred]).to(device)
    
    attacker = FGSMAttack(model)
    adversarial_tensor = attacker.generate_adversarial_example(
        attack_input, target, epsilon
    )
    
    # E. Get Adversarial Prediction (Hacked)
    adv_output = model(adversarial_tensor)
    adv_pred = adv_output.max(1, keepdim=True)[1].item()
    
    # F. Prepare image for display
    adv_image_b64 = tensor_to_base64(adversarial_tensor.squeeze(0))
    
    return {
        "clean_prediction": int(clean_pred),
        "adversarial_prediction": int(adv_pred),
        "adversarial_image": adv_image_b64,
        "epsilon": epsilon,
        "success": clean_pred != adv_pred
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)