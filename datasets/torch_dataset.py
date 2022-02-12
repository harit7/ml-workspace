
from torch.utils.data import  Dataset
import numpy as np
from PIL import Image

class CustomTensorDataset(Dataset):
    def __init__(self, X,Y, transform=None):
        #assert all(X.size(0) == tensor.size(0) for tensor in tensors)
        self.data = X
        self.targets = Y
        self.transform = transform

    def __getitem__(self, index):
        
        x = self.data[index]
        
        if self.transform:
            x = Image.fromarray(x.numpy().astype(np.uint8))
            x = self.transform(x)
        
        y = self.targets[index]
        
        return x, y

    def __len__(self):
        return self.data.size(0)

    def get_subset(self,idcs):
        idcs = np.array(idcs) 
        return CustomTensorDataset(X=self.data[idcs],Y=self.targets[idcs],transform=self.transform)

    def build_dataset(self):
        pass 
    

    
    
