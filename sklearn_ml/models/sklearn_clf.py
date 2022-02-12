from sklearn.linear_model import LogisticRegression
from .abstract_clf import AbstractClassifier
import numpy as np
import sys 
sys.path.append('../../')
from multipledispatch import dispatch

class SkLearnClassifier(AbstractClassifier):

    def __init__(self,model_conf,logger):
        self.input_dim = model_conf['input_dimension']
        self.num_classes = model_conf['num_classes']
        self.model_conf = model_conf
        self.logger = logger

    def create_model(self):
        model_conf = self.model_conf
        training_conf = self.training_conf

        if(model_conf['model_name']=='binary_logistic_regression'):
            assert self.num_classes == 2
            assert self.input_dim   >= 1
            self.model = LogisticRegression(fit_intercept=model_conf['fit_intercept'],
                                            max_iter =  training_conf['max_epochs'],
                                            tol = training_conf['loss_tolerance'],
                                            solver= training_conf['optimizer_name'],
                                            penalty=training_conf['regularization']
                                            )
        
    def set_default_conf(self,training_conf) :
        #training_conf.setdefault('weight_decay',0)
        #training_conf.setdefault('momentum',0)
        training_conf.setdefault('regularization','none')
        training_conf.setdefault('optimizer_name','lbfgs')
        #training_conf.setdefault('learning_rate',1e-2)
        training_conf.setdefault('loss_tol',1e-6)
        training_conf.setdefault('max_epochs',100)
        #training_conf.setdefault('shuffle',False)
        #training_conf.setdefault('batch_size',32)
        #training_conf.setdefault('device','cpu')

    
    def fit(self,dataset,training_conf):
        train_dataset = dataset.train_dataset
        self.set_default_conf(training_conf)
        self.training_conf = training_conf
        self.create_model()
        X = train_dataset.X 
        Y = train_dataset.Y 

        n,d = X.shape
        assert d == self.input_dim
        assert Y.min() < self.num_classes and Y.min() > -1
        self.model.fit(X,Y)

    def predict(self,dataset,inference_conf=None):
        X = dataset.test_dataset.X
        Y_hat = self.model.predict(X)
        P_hat = self.model.predict_proba(X)
        return Y_hat,P_hat 

    def get_weights(self):
        if(self.model_conf['fit_intercept']):
            return  self.model.coef_, self.model.intercept_
        else:
            return self.model.coef_, 0.0