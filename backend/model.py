import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) model designed for MNIST (1 input channel, 10 classes).
    This model will be the target for the FGSM attack.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1: Takes 1 grayscale channel input
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        # Fully Connected Layer 1 (The input size 320 comes from the size of the feature maps after Conv and Pooling)
        self.fc1 = nn.Linear(320, 50) 
        
        # Fully Connected Layer 2: Maps features to 10 output classes (digits 0-9)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 1. Conv1 -> Max Pooling -> ReLU activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # 2. Conv2 -> Dropout -> Max Pooling -> ReLU activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # 3. Flatten the tensor for the fully connected layers
        x = x.view(-1, 320)
        
        # 4. FC1 -> ReLU activation
        x = F.relu(self.fc1(x))
        
        # 5. Final layer: Log Softmax for class probabilities (required for NLLLoss)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def load_model():
    """
    Instantiates the model architecture.
    """
    # Later, we will add logic here to load the *trained weights* for this model.
    return SimpleCNN()

if __name__ == '__main__':
    # Test initialization
    model = load_model()
    print("Model architecture defined in backend/model.py.")