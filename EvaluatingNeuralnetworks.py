import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
     def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
# Load the test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
# Load the trained model
model = Net()
model.load_state_dict(torch.load('model.pth'))
# Set the model to evaluation mode
model.eval()
# Evaluate the model on the test dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # Flatten the images into a vector
        images = images.view(images.shape[0], -1)
        # Forward pass to get the predicted labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
       # Count the number of correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
# Compute the accuracy of the model on the test dataset
accuracy = 100 * correct / total
print('Accuracy on the test dataset: %.2f %%' % accuracy)
