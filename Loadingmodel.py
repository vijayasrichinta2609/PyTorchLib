import torch
import torch.nn as nn
# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
     def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = Net()
# Load the saved model's state_dict
model.load_state_dict(torch.load('model.pth'))
model.eval()
