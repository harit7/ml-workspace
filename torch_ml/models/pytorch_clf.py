
from .abstract_clf import AbstractClassifier
from .logistic_regression import  PyTorchLogisticRegression

from ..training.model_training import *
from ..inference.clf_inference import *
import torch

class PyTorchClassifier(AbstractClassifier):

    def __init__(self,model_conf,logger):
        
        self.model_conf = model_conf 
        self.logger = logger 

        if(model_conf['model_name']=='binary_logistic_regression'):
            assert model_conf['num_classes']==2 
            self.model= PyTorchLogisticRegression(model_conf,logger)
    
    def fit(self,dataset,training_conf):
        model_training = ModelTraining(self.logger)
        out = model_training.train(self.model,dataset,training_conf)
        return out 
    
    def predict(self, dataset, inference_conf):
        clf_inference = ClassfierInference(self.logger)
        return clf_inference.predict(self.model,dataset,inference_conf)

    def get_weights(self):
        return torch.Tensor([x for x in self.model.parameters()])