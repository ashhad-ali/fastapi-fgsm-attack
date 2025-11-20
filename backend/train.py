import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleCNN
import torch.nn.functional as F

# Configuration
EPOCHS = 3
BATCH_SIZE = 64
SAVE_PATH = "backend/mnist_cnn_weights.pth"

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Detect device
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data transformation and loading
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # Initialize model and optimizer
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        print(f"Epoch {epoch} complete.")
        
    print("Training finished. Testing final model...")
    initial_accuracy = test(model, device, test_loader)

    # Save the model weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model weights saved to {SAVE_PATH}")
    
    # Return the accuracy for observation
    return initial_accuracy

if __name__ == '__main__':
    main()