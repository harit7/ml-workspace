import pickle
import torch
import numpy as np 
import models.clf_factory 
from .conf_defaults import *
from sklearn.metrics import accuracy_score

class PassiveLearning:
    
    def __init__(self,conf, ds_train, ds_val=None,ds_test=None,logger=None):
                
        self.ds_train = ds_train
        self.ds_val    = ds_val 
        self.ds_test   = ds_test 

        self.conf = conf 

        ## set default params
        set_defaults(conf)  

        self.num_classes =self.ds_train.num_classes
        self.lst_classes = np.arange(0,self.num_classes,1)

        self.logger = logger


    def run(self):
        logger =self.logger
        conf = self.conf 
        train_conf = conf.training_conf
        model_conf = conf.model_conf
        inference_conf = conf.inference_conf

        out = {}
        
        train_ds = self.ds_train

        n_train = len(train_ds)

        logger.info('Train data size: {}'.format(n_train))

        if(train_conf.ckpt_load_path is not None):
            logger.info('Loading model from path: {}'.format(train_conf.ckpt_load_path))
            self.load_state(train_conf.ckpt_load_path)

        else:
            # create a new model for training.         
            self.cur_clf = models.clf_factory.get_classifier(model_conf,self.logger)
        
        logger.info('--------------- Begin Model Training ------------')
        
        logger.info('Training conf :{}'.format(train_conf))
        logger.info('Model conf : {}'.format(model_conf))

        self.cur_clf = self.train_model(train_ds,model_conf, train_conf,val_set = self.ds_val)
        logger.info('--------------- End Model Training ------------')


        if(train_conf.ckpt_save_path is not None):
            logger.info('Saved model checkpoint to path')
            self.save_state(train_conf.ckpt_save_path)
        
        # get validation error..
        
        train_err = self.get_test_error(self.cur_clf,train_ds,inference_conf)
        val_err = self.get_test_error(self.cur_clf,self.ds_val,inference_conf)
        test_err = self.get_test_error(self.cur_clf,self.ds_test,inference_conf)
        out['train_error'] = train_err
        out['val_error']  = val_err 
        out['test_error'] = test_err 
        self.cur_test_err = test_err 
        self.cur_val_err = val_err 
        self.cur_train_err = train_err

        logger.info(f'Training Error : {train_err}  Validation Error : {val_err}  Test Error : {test_err}')
        
        
        if(train_conf.embedding_save_path is not None): # save the embeddings
            
            embeddings = {}
            
            train_embeddings = self.cur_clf.get_embedding(train_ds,inference_conf)['embedding']
            val_embeddings = self.cur_clf.get_embedding(self.ds_val,inference_conf)['embedding']
            test_embeddings = self.cur_clf.get_embedding(self.ds_test,inference_conf)['embedding']

            self.logger.info('Get Train Embedding, shape: {}'.format(train_embeddings.shape))
            self.logger.info('Get Val Embedding, shape: {}'.format(val_embeddings.shape))
            self.logger.info('Get Test Embedding, shape: {}'.format(test_embeddings.shape))

            embeddings["train_embed"] =  [train_embeddings,train_ds.Y]
            embeddings["val_embed"] =  [val_embeddings,self.ds_val.Y]
            embeddings["test_embed"] =  [test_embeddings,self.ds_test.Y]

            out["embed"] = embeddings 
            with open(train_conf.embedding_save_path, 'wb') as handle:
                pickle.dump(out, handle) 
            
        
        return out 
    
    def train_model(self,train_dataset,model_conf,training_conf,val_set=None):
        
        # in some cases we might use the previous model and retrain.
        self.cur_clf.fit(train_dataset, training_conf,val_set)
        return self.cur_clf

    
    def get_test_error(self,clf,test_ds,inference_conf):
        inf_out = clf.predict(test_ds, inference_conf) 
        test_err = 1-accuracy_score(inf_out['labels'],test_ds.Y)
        return test_err 


    def load_state(self,path):
        checkpoint = torch.load(path)
        if(self.conf.model_conf.lib=='pytorch'):
            self.cur_clf = models.clf_factory.get_classifier(self.conf.model_conf,self.logger)
            self.cur_clf.model.load_state_dict(checkpoint['model_state_dict'])
        