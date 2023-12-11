# PyTorchLib

  COMP651DataAnalyticsTechniquesSemesterProject
Tutorial on PyTorch library
                                                                                                                                                                       -By Vijayasri Chinta
Introduction on PyTorch:
PyTorch is a powerful deep learning framework that has become increasingly popular among researchers and developers. Its dynamic computational graph and automatic differentiation capabilities make it easy to build and train neural networks. PyTorch also has strong GPU acceleration capabilities and provides several tools for working with large datasets.
What Is PyTorch?
PyTorch is the largest machine learning library that allow developers to perform tensor computations with an acceleration of GPU, creates dynamic computational graphs, and calculate gradients automatically. Other than this, PyTorch offers rich APIs for solving application issues related to neural networks.This machine learning library is based on Torch, which is an open source machine library implemented in C with a wrapper in Lua.This machine library in Python was introduced in 2017, and since its inception, the library is gaining popularity and attracting an increasing number of machine learning developers.
Why use of PyTorch?
Here are some reasons why PyTorch is widely used:
1.Dynamic Computational Graphs: PyTorch uses dynamic computational graphs which enable users to define and modify the computational graph on-the-fly. This makes it easy to experiment with different model architectures and implement complex models.
2.Automatic Differentiation: PyTorch provides automatic differentiation, which is a powerful tool for computing gradients. With automatic differentiation, developers don't need to manually compute the gradients of their models, which can be time-consuming and error-prone.
3.Pythonic API: PyTorch has a simple and intuitive Pythonic API, which makes it easy to use and understand. The PyTorch API is built on top of Python, which means developers can use all the standard Python libraries and tools they're already familiar with.
4.Strong GPU Acceleration: PyTorch provides strong GPU acceleration for deep learning tasks. It has support for NVIDIA GPUs and can take advantage of their parallel processing capabilities to speed up computations.
5.Active Community: PyTorch has a large and active community of developers and researchers who contribute to the framework and create open-source libraries and tools. This makes it easy to find support, ask questions, and collaborate with others.
Overall, PyTorch is a powerful and flexible deep learning framework that is well-suited for research and production use cases. Its dynamic computational graph, automatic differentiation, Pythonic API, GPU acceleration, and active community make it a popular choice among developers and researchers alike.
Installing PyTorch:
Here's a brief explanation on how to install the latest version of PyTorch:
•	Check if you have Python 3.6 or later installed by running the command python3 --version in your terminal.
•	Next, you need to install PyTorch via pip. You can use the following command to install the latest version of PyTorch:
                pip install torch torchvision torchaudi
This command will install the latest version of PyTorch, along with the necessary dependencies for working with image and audio data.
•	Finally, you can verify that PyTorch has been installed correctly by launching a Python shell and importing the library:
Code:
import torch
# Test if PyTorch is working properly by creating a random tensor
x = torch.rand(5, 3)
print(x)
If PyTorch has been installed successfully, you should see a randomly generated tensor printed to the console. You're now ready to start using PyTorch in your projects.
Features Of PyTorch:
Hybrid Front-End:A new hybrid front-end provides ease-of-use and flexibility in eager mode, while seamlessly transitioning to graph mode for speed, optimization, and functionality in C++ runtime environments.
Distributed Training:Optimize performance in both research and production by taking advantage of native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from Python and C++.
Python First:PyTorch is not a Python binding into a monolithic C++ framework. It’s built to be deeply integrated into Python so it can be used with popular libraries and packages such as Python and Numpy.
Libraries And Tools:An active community of researchers and developers have built a rich ecosystem of tools and libraries for extending PyTorch and supporting development in areas from computer vision to reinforcement learning.
In this ,we will cover the following topics:
1.Tensors in PyTorch
2.Autograd in PyTorch
3.Neural Networks in PyTorch
4.Training a Neural Network in PyTorch
5.Evaluating a Neural Network in PyTorch
6.GPU Acceleration in PyTorch
7.Custom Datasets and Data Loaders in PyTorch
8.Transfer Learning in PyTorch
9.Saving and Loading Models in PyTorch
10.Conclusion
1. Tensors in PyTorch:
Tensors in PyTorch are multi-dimensional arrays that can hold numerical data, such as integers or floating-point numbers. They are similar to NumPy arrays, but they can be used on GPUs to accelerate the computations.Here's an example code to create and manipulate tensors in PyTorch:
Code:
import torch
# Create a 2x3 tensor with random values
tensor = torch.randn(2, 3)
# Print the tensor and its shape
print(tensor)
print(tensor.shape)
# Add 2 to all elements of the tensor
tensor += 2
# Multiply the tensor by 3
tensor *= 3
# Convert the tensor to a NumPy array
numpy_array = tensor.numpy()
# Create a tensor from a NumPy array
tensor_from_numpy = torch.from_numpy(numpy_array)
# Compute the dot product of two tensors
tensor2 = torch.randn(3, 4)
dot_product = torch.mm(tensor, tensor2)
# Print the dot product
print(dot_product)

In this code, we first import the torch library. Then, we create a 2x3 tensor with random values using the torch.randn function. We print the tensor and its shape using the print function.Next, we add 2 to all elements of the tensor using the += operator and multiply the tensor by 3 using the *= operator.We then convert the tensor to a NumPy array using the numpy method and create a tensor from the NumPy array using the from_numpy method.Finally, we compute the dot product of two tensors using the mm function and print the result using the print function.

2. Autograd in PyTorch:
PyTorch's autograd package provides automatic differentiation for all operations on Tensors. This means that we can compute gradients of functions with respect to their input Tensors without explicitly computing the gradients.
Code:
import torch
# create a tensor and set requires_grad to True to track computations on it
x = torch.ones(3, 3, requires_grad=True)
# perform some operations on the tensor
y = x + 2
z = y * y * 3
out = z.mean()
# compute gradients
out.backward()
# print gradients
print(x.grad)

3. Neural Networks in PyTorch:
Neural networks are a fundamental concept in deep learning, and PyTorch provides a powerful and flexible framework for building and training neural networks. A neural network is composed of several layers, each consisting of a set of neurons or units that perform computations on the input data. The output of one layer serves as the input to the next layer, and so on, until the final output is produced.
Here is a simple example code for building a neural network using PyTorch:
Code:
import torch
import torch.nn as nn
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
 In this example, we define a neural network with three fully connected layers. The Net class is a subclass of the nn.Module class, which is the base class for all neural network modules in PyTorch. The __init__ method initializes the layers of the network, and the forward method defines the forward pass of the network.The criterion variable specifies the loss function, which is used to compute the difference between the predicted output and the actual output. The optimizer variable specifies the optimization algorithm, which is used to update the parameters of the network during training.
4.Training a neutral network in PyTorch:
Training a neural network in PyTorch involves several steps, including defining the network architecture, specifying the loss function and optimizer, and iterating over the training dataset to update the network's parameters.Here is an example code for training a simple feedforward neural network to classify images of handwritten digits from the MNIST dataset:
Code:
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
# Set up the training data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)
# Initialize the network and optimizer
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
# Specify the loss function
criterion = nn.CrossEntropyLoss()
# Train the network
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
In this example, we first define a simple feedforward neural network with two fully connected layers using the nn.Linear module. We then set up the training data using the DataLoader class from torch.utils.data, which loads the MNIST dataset and applies some transformations to the images.We then initialize the network and the optimizer, which in this case is stochastic gradient descent (SGD) with a learning rate of 0.01 and momentum of 0.5. We also specify the loss function as cross-entropy loss, which is commonly used for classification problems.Finally, we loop over the training data for 10 epochs and update the network's parameters using backpropagation and the SGD optimizer. We print the loss and epoch number every 100 batches for monitoring the training progress.
5. Evaluating a neutral network in PyTorch: 
Evaluating a neural network in PyTorch involves using the trained model to predict the output of new data and comparing those predictions to the true labels of the data. This is done to measure the performance of the model on unseen data.Here is an example code that demonstrates how to evaluate a neural network in PyTorch:
Code:
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
In this example, we define a neural network with two fully connected layers and a softmax activation function. We then load a test dataset of MNIST images and their corresponding labels, and load a trained model that was previously saved to a file. We set the model to evaluation mode using model.eval(), which disables dropout and batch normalization layers. We then loop over the test dataset in batches, flatten the images into vectors, and perform a forward pass through the model to get the predicted labels. We count the number of correct predictions and compute the accuracy of the model on the test dataset. Finally, we print the accuracy of the model as a percentage.
6. GPU Acceleration in PyTorch: 
PyTorch provides efficient GPU acceleration for deep learning computations. This can significantly speed up training and evaluation of neural networks.To use a GPU with PyTorch, we can simply move our model and data to the GPU device using the .to() method. Here is an example code that demonstrates how to use a GPU for training a neural network:
Code:
import torch
# Define the model architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Load the data
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
# Define the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={running_loss/len(train_data)}")
# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total}")
In this example, we first define a convolutional neural network architecture called Net. We then load some data and define the model, loss function, and optimizer.To use the GPU, we first check if it is available using torch.cuda.is_available(), and then move the model and data to the GPU using .to(device). We can then train the model as usual. During evaluation, we also move the data to the GPU using .to(device).Overall, using a GPU with PyTorch can lead to significant speedups in training and evaluation of deep learning models.
7. Custom Datasets and Data Loaders in PyTorch:
Custom datasets and data loaders are an essential part of training deep learning models on large datasets in PyTorch. A dataset is a collection of data, and a data loader is an object that loads data from the dataset into memory in batches during training.In PyTorch, custom datasets and data loaders can be created by subclassing the torch.utils.data.Dataset and torch.utils.data.DataLoader classes, respectively. The Dataset class provides an interface for accessing data, and the DataLoader class provides a convenient way to load data in batches.Here's an example code to create a custom dataset and data loader in PyTorch:
Code:
import torch
from torch.utils.data import Dataset, DataLoader
# define custom dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x = torch.tensor(self.data[index][0], dtype=torch.float32)
        y = torch.tensor(self.data[index][1], dtype=torch.float32)
        return x, y
# create a toy dataset
data = [(1, 2), (3, 4), (5, 6)]
# create custom dataset object
my_dataset = MyDataset(data)
# create data loader
batch_size = 2
my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
# iterate over batches in data loader
for batch in my_dataloader:
    x, y = batch
    print(x, y)
In this example, we define a custom dataset MyDataset that takes in a list of tuples as input data. The __len__ method returns the length of the dataset, and the __getitem__ method returns a tuple of x and y values for a given index.We then create a toy dataset and create a MyDataset object with it. We also create a data loader with a batch size of 2 and shuffle the data randomFinally, we iterate over the data loader, which returns batches of data in the form of tuples of x and y values. In this example, each batch contains two tuples, since we set the batch size to 2.
8. Transfer Learning in PyTorch:
Transfer learning is a technique where pre-trained models are used as a starting point for training a new model on a different task or dataset. It can save time and computational resources by reusing the pre-trained model's weights, architecture, and learned features. In PyTorch, transfer learning can be achieved by loading a pre-trained model and replacing the final fully connected layer with a new one that matches the number of classes in the new dataset.Here is an example code for transfer learning using PyTorch:
Code:
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# Load the pre-trained model
model = models.resnet18(pretrained=True)
# Freeze all the layers except the last one
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True
# Replace the last fully connected layer with a new one
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Load the data
train_dir = '/path/to/train'
val_dir = '/path/to/val'
train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
# Train the model
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
# Evaluate the model on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%')

9. Saving and Loading models in PyTorch:
Saving and loading models is an important aspect of using neural networks in PyTorch. After training a model, it is useful to be able to save its parameters so that it can be used later or shared with others. PyTorch provides simple methods for saving and loading models.To save a model in PyTorch, we can use the torch.save() function. This function takes two arguments: the first argument is the object to be saved (usually a model state_dict), and the second argument is the file path where the object should be saved.Here's an example code for saving a model:
Code:
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
# Save the model's state_dict
torch.save(model.state_dict(), 'model.pth')

To load a saved model, we can use the torch.load() function. This function takes one argument, which is the file path where the saved object is located. The load() function returns the saved object as a dictionary, so we need to extract the relevant information to reconstruct the model.Here's an example code for loading a saved model:
Code:
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
In this code, we define the same neural network as before and create an instance of it. We then load the saved state_dict using the load_state_dict() method and set the model to evaluation mode using the eval() method.

# Conclusion:
PyTorch is a powerful and flexible deep learning framework that has gained popularity among researchers and practitioners alike. Its dynamic computational graph and ease of use in Python have made it a favorite among many in the deep learning community. With its extensive documentation, active development community, and a growing number of pre-trained models, PyTorch makes it easy to develop and deploy deep learning models.
Overall, PyTorch is a great choice for those looking to experiment with deep learning, conduct research, or build scalable production applications. Its hybrid front-end, distributed training capabilities, Python-first approach, and rich ecosystem of libraries and tools make it a versatile and powerful framework. While there may be some limitations and challenges, PyTorch has proven to be a valuable asset in advancing the field of deep learning.

