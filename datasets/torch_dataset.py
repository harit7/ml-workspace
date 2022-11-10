
from torch.utils.data import  Dataset
import numpy as np
from PIL import Image

class CustomTensorDataset(Dataset):
    def __init__(self, X=None,Y=None,num_classes=2, d=None,transform=None):
        #assert all(X.size(0) == tensor.size(0) for tensor in tensors)
        self.X = X
        self.Y = Y
        self.d = d
        self.num_classes  = num_classes
        self.transform = transform
        
    def __getitem__(self, index):
        
        x = self.X[index]
        
        if self.transform:
            x = Image.fromarray(x.numpy().astype(np.uint8))
            x = self.transform(x)
        
        y = self.Y[index]
        
        return x, y

    def __len__(self):

        return self.X.size(0)
    
    def len(self):
        return len(self.X)
    
    def get_subset(self,idcs,Y_=None):
        '''
           Y must have same size as idcs and Y[i] = label of idcs[i].
           If this Y is given that means the labels of the subset are to 
           be populated from it, else populate the labels from original labels 
           if available.
        '''
        idcs = np.array(idcs) 
        X_ = None
        Y_ = None 
        if(self.X is not None):
            X_ = self.X[idcs] 
        if(Y_ is None and self.Y is not None):
            Y_ = self.Y[idcs]
        
        return CustomTensorDataset(X=X_,Y=Y_,transform=self.transform)
    
    def get_random_fraction(self,frac=1.0):
        n = len(self.X)
        idcs = np.random.choice(n,size = int(frac*n),replace=False) 
        return self.get_subset(idcs)

    def build_dataset(self):
        pass 
    

    
    
