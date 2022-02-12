from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
from .torch_dataset import CustomTensorDataset
from .custom_dataset import CustomDataset

def tensorize(np_dataset,transform=None):
    np_dataset.train_dataset = CustomTensorDataset(torch.Tensor(np_dataset.train_dataset.X), 
                                                    torch.Tensor(np_dataset.train_dataset.Y).long(), transform = transform)

    np_dataset.test_dataset  = CustomTensorDataset(torch.Tensor(np_dataset.test_dataset.X), 
                                                    torch.Tensor(np_dataset.Y_test).long(), transform=transform)


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
    
