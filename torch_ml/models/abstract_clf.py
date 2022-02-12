from abc import ABC, abstractmethod

class AbstractClassifier(ABC):

    @abstractmethod
    def fit(self,dataset):
        pass
    
    @abstractmethod
    def predict(self,dataset):
        pass
    
    @abstractmethod
    def get_weights(self):
        pass
     
