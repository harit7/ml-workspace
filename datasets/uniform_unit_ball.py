import sys
from .numpy_dataset import DatasetNumpy
#sys.path.append('../')

import numpy as np
from datasets.dataset_utils import *
from sklearn.model_selection import train_test_split
from utils.sampling_utils import *

class UniformUnitBallDataset:
    def __init__(self,data_conf=None):
        self.data_conf = data_conf
     
    def build_dataset(self):
        
        self.transform = None

        d = self.data_conf['dimension']
        
        X = random_ball(self.data_conf['n_samples'],d)

        w = np.ones(d)
    
        w = w/np.linalg.norm(w)
        
        Y = [1 if np.dot(w,x)>= 0 else 0 for x in X]
        
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=self.data_conf['test_size']
                                                            ,random_state=self.data_conf['random_state'] )
        

        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        self.X_test = np.array(X_test )
        self.Y_test = np.array(Y_test)

        self.train_dataset = DatasetNumpy(self.X_train,self.Y_train) 
        self.test_dataset  = DatasetNumpy(self.X_test,self.Y_test) 

    




