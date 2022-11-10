from random import random
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
from .torch_dataset import CustomTensorDataset
from .custom_dataset import CustomDataset

def tensorize(np_dataset,transform=None):
    X_tensor = None 
    Y_tensor = None 
    if(np_dataset.X is not None):
        X_tensor = torch.Tensor(np_dataset.X)
    if(np_dataset.Y is not None):
        Y_tensor = torch.Tensor(np_dataset.Y).long()
    tensor_dataset = CustomTensorDataset(X_tensor,Y_tensor, transform = transform)
    
    return tensor_dataset

def randomly_split_dataset(ds,fraction=0.5,random_state=42):
    n = ds.len()
    idcs = list(range(n))
    np.random.seed(random_state)
    idcs1 = np.random.choice(idcs,int(n*fraction),replace=False)
    idcs1_set = set(list(idcs1))
    idcs2 = np.array( list(set(idcs).difference(idcs1_set) ))
    return ds.get_subset(idcs1), ds.get_subset(idcs2)

def take_subset_of_train_dataset(dataset,idcs):
    subset_train_ds = dataset.train_dataset.get_subset(idcs)
    return CustomDataset(subset_train_ds,dataset.test_dataset)


def getDataLoaderForSubset(tensorDataset,subsetIndices,batchSize,true_labels=True):
    
    X_np = tensorDataset.data.numpy()[subsetIndices]
    if(true_labels):
        Y_np = tensorDataset.targets.numpy()[subsetIndices]
    else:
        Y_np = np.zeros(len(subsetIndices))

    subsetDataset = CustomTensorDataset(torch.Tensor(X_np),torch.Tensor(Y_np).long(),transform=None)
    subsetLoader  = DataLoader(dataset=subsetDataset,batch_size= batchSize, shuffle=False)
    return subsetLoader

def get_data_loader_from_numpy_arrays(X_np,Y_np,batch_size,transform,shuffle):
    subsetDataset = CustomTensorDataset(torch.Tensor(X_np),torch.Tensor(Y_np).long(),transform=transform)
    subsetLoader  = DataLoader(dataset=subsetDataset,batch_size= batch_size, shuffle=shuffle)
    return subsetLoader  
    
