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
