from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

from datasets.dataset_utils import *

class MNISTData:
    def __init__(self,conf):
        self.conf = conf
        
        
    def build_dataset(self):
        
        data_conf = self.conf['data_conf']
        data_dir  = data_conf['data_path']
        
        self.transform = transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])
        self.trainData = MNIST(data_dir, download=True,
                                transform=self.transform)
        
        self.testData  = MNIST(data_dir, train=False, download=True,
                               transform=self.transform)

        self.train_labels = self.trainData.targets.numpy()
        self.train_instances= self.trainData.data.numpy()
        
        self.X_train = self.trainData.data.numpy()
        self.Y_train = self.trainData.targets.numpy()

        self.X_test = self.testData.data.numpy()
        self.Y_test = self.testdata.targets.numpy()
        
        if('sub_sample_fraction' in data_conf):
            np.random.seed(data_conf['sub_sample_fraction_seed'])
            n = len(self.trainData)
            idcs = np.array(range(n))
            th = int(n*data_conf['sub_sample_fraction'])
            np.random.shuffle(idcs)
            idcs = idcs[:th]
            
            self.X_train = self.X_train[idcs]
            self.Y_train = self.Y_train[idcs]
            
            self.trainData = CustomTensorDataset(torch.Tensor(self.X_train),
                                           torch.Tensor(self.Y_train).long(),
                                           transform=self.transform)