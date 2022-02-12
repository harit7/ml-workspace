
import sys
#sys.path.append('../')
sys.path.append('../../')
#sys.path.append('../../../')

from torch_ml.models.pytorch_clf import *
from sklearn_ml.models.sklearn_clf import *

class Classifier:
    def __init__(self,model_conf,logger=None):
        # conf should have, lib = sklearn or torch
        # and model_conf, train_conf, inference_conf
        self.logger = logger 
        if(model_conf['lib']=='pytorch'):
            self.model = PyTorchClassifier(model_conf,logger=logger)
        elif(model_conf['lib']=='sklearn'):
            self.model = SkLearnClassifier(model_conf,logger=logger)
        else:
            logger.info('invalid lib')
    
    def fit(self,train_dataset, train_conf):
        return self.model.fit(train_dataset,train_conf)
    
    def predict(self,test_dataset,inference_conf=None):
        return self.model.predict(test_dataset,inference_conf)
    




        
