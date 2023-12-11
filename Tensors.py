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
