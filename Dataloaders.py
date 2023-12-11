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
