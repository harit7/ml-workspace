
from ..abstract_clf import AbstractClassifier
from .logistic_regression import  PyTorchLogisticRegression

from .model_training import *
from .clf_inference import *
from .clf_get_embedding import *
import torch
from .lenet import *
from .one_layer_softmax import OneLayerSoftmax
from .cifar_small_net import CifarSmallNet
from .resnet import ResNet18
from .cifar_medium_net import CifarMediumNet


class PyTorchClassifier(AbstractClassifier):

    def __init__(self,model_conf,logger):
        
        self.model_conf = model_conf 
        self.logger = logger 

        if(model_conf['model_name']=='binary_logistic_regression'):
            assert model_conf['num_classes']==2 
            self.model= PyTorchLogisticRegression(model_conf,logger)
        if(model_conf['model_name']=='lenet'):
            self.model = LeNet5(model_conf['num_classes'])
        if(model_conf['model_name']=='one_layer_softmax'):
            self.model = OneLayerSoftmax(model_conf['num_classes'])
        if(model_conf['model_name']=='cifar_small_net'):
            self.model = CifarSmallNet(model_conf['num_classes'])
        if(model_conf['model_name']=='resnet18'):
            self.model = ResNet18(model_conf['num_classes'])
        if(model_conf['model_name']=='cifar_med_net'):
            self.model = CifarMediumNet(model_conf['num_classes'])
            
    
    def fit(self,dataset,training_conf,val_set=None):
        model_training = ModelTraining(self.logger)
        out = model_training.train(self.model,dataset,training_conf,val_set=val_set)
        return out 
    
    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        return clf_inference.predict(self.model,dataset,inference_conf)

    def get_weights(self):
        w = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().cpu()
        return w

    def get_grad_embedding(self,dataset, inference_conf):
        clf_embedding = ClassifierEmbedding(self.logger)
        return clf_embedding.get_gard_embedding(self.model,dataset,inference_conf)

    def get_embedding(self,dataset, inference_conf):
        clf_embedding = ClassifierEmbedding(self.logger)
        return clf_embedding.get_embedding(self.model,dataset,inference_conf)