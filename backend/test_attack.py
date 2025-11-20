import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import load_model
from fgsm import FGSMAttack
import os

# Configuration
EPSILONS = [0, 0.05, 0.1, 0.2, 0.3]

# Robust way to find weights file regardless of where you run the script from
if os.path.exists("backend/mnist_cnn_weights.pth"):
    WEIGHTS_PATH = "backend/mnist_cnn_weights.pth"
elif os.path.exists("mnist_cnn_weights.pth"):
    WEIGHTS_PATH = "mnist_cnn_weights.pth"
else:
    raise FileNotFoundError("Could not find mnist_cnn_weights.pth. Make sure you trained the model!")

def test_attack(model, device, test_loader, epsilon):
    correct = 0
    attacker = FGSMAttack(model)
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        if epsilon == 0:
            perturbed_data = data
        else:
            perturbed_data = attacker.generate_adversarial_example(data, target, epsilon)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc:.4f}")
    return final_acc

def main():
    device = torch.device("cpu")

    model = load_model().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # CRITICAL FIX: Added Normalize to match training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load a subset of 1000 test images
    subset_indices = range(1000)
    subset_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(subset_indices)
    )

    print(f"Model loaded from {WEIGHTS_PATH}. Starting attack evaluation...")
    
    for eps in EPSILONS:
        test_attack(model, device, subset_loader, eps)

if __name__ == "__main__":
    main()