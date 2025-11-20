import requests
from PIL import Image
import io
import numpy as np

# 1. Create a dummy image (a black square with a white line)
# This simulates a handwritten digit "1"
img_data = np.zeros((28, 28), dtype=np.uint8)
img_data[5:23, 12:15] = 255 # Draw a vertical line
image = Image.fromarray(img_data)

# Save it to a memory buffer to send it
buf = io.BytesIO()
image.save(buf, format="PNG")
buf.seek(0)

# 2. Define the server URL
url = "http://127.0.0.1:8000/attack"

# 3. Prepare the data
payload = {'epsilon': '0.3'} # High attack strength
files = {'file': ('test_image.png', buf, 'image/png')}

print(f"Sending request to {url}...")

try:
    # 4. Send the POST request
    response = requests.post(url, data=payload, files=files)
    
    # 5. Print the result
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS! Server responded.")
        print(f"Clean Prediction: {data['clean_prediction']}")
        print(f"Adversarial Prediction: {data['adversarial_prediction']}")
        print(f"Attack Success: {data['success']}")
    else:
        print(f"❌ FAILED. Status Code: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("❌ ERROR: Could not connect. Is the server running in the other terminal?")