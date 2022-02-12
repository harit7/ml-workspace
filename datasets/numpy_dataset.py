
import numpy as np 

class DatasetNumpy:

    def __init__(self,X=None,Y=None):
        self.X = X 
        self.Y = Y 
    
    def build_dataset(self):
        pass 
    
    def get_subset(self,idcs):
        idcs = np.array(idcs) 
        #print(type(idcs))
        #print(self.X.shape,self.Y.shape)
        return DatasetNumpy(X=self.X[idcs.astype(int)],Y=self.Y[idcs.astype(int)])
    
    '''
    def tensorize(self):

    '''