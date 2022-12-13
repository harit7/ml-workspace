import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader
from .clf_inference import *
from sklearn.metrics import accuracy_score

class ModelTraining:
    
    def __init__(self,logger):
        self.logger = logger
    
    def init_optimizer(self,model,train_params):
        
        self.optimizer = None 

        opt_name = train_params.optimizer_name

        if(opt_name=='adam'):        
            optimizer = optim.Adam(model.parameters(), lr=train_params.learning_rate,
                                    weight_decay= train_params.weight_decay
                                    )
        if(opt_name == 'lbfgs'):
            optimizer = torch.optim.LBFGS(model.parameters())
        
        else:
            optimizer = optim.SGD(model.parameters(), lr = train_params.learning_rate, 
                                momentum= train_params.momentum, 
                                weight_decay= train_params.weight_decay)

        self.optimizer  = optimizer 

    def set_defaults(self, train_params):
        train_params.setdefault('weight_decay',1e-4)
        train_params.setdefault('momentum',0.9)
        train_params.setdefault('optimizer_name','sgd')
        train_params.setdefault('learning_rate',1e-2)
        train_params.setdefault('loss_tol',1e-6)
        train_params.setdefault('max_epochs',200)
        train_params.setdefault('max_max_epochs',10000)
        train_params.setdefault('shuffle',False)
        train_params.setdefault('batch_size',32)
        train_params.setdefault('device','cpu')
        train_params.setdefault('stopping_criterion','max_epochs')
        train_params.setdefault('log_val_err',False)
        train_params.setdefault('train_err_tol',0.001) #default less than 0.1%

        # set this to 0 to disable loss prints with batches.
        train_params.setdefault('log_batch_loss_freq',20) 


    def train_one_epoch(self,model,train_data_loader,train_params,epoch_num=0):
        
        logger = self.logger 

        device = train_params.device
        
        epoch_loss = 0
        num_pts = len(train_data_loader.dataset)
        
        model.train()
        
        model = model.to(device)
        
      
        log_batch_loss_freq = train_params.log_batch_loss_freq
        y_hat = []
        y_true = []
        
        for batch_idx, (data, target) in enumerate(train_data_loader):
            
            data, target = data.to(device), target.to(device)
            
            self.optimizer.zero_grad()   # set gradient to 0
            out     = model.forward(data)
            probs   = out['probs']
            logits  = out['logits']
            
            #print(probs[0])
            #print(logits[0])
            #print(sum(probs[0]))

            #loss    = model.criterion(probs, target) 
            # it expects unnormalized scores. internally converts to probs.
            #print(target[0])
            
            loss    = model.criterion(logits, target) 
            
            loss.backward()    # compute gradient
            #loss2 = model.criterion(probs,target)
            #loss2.backward()
            confidence, y_hat_ = torch.max(probs, 1)
            y_hat.extend(y_hat_.cpu().numpy())
            y_true.extend(target.cpu().numpy())

            if log_batch_loss_freq >0 and batch_idx%20 == 0:
                #self.logger.debug(probs[0].max().item())
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch_num, batch_idx * len(data), num_pts,
                                100. * batch_idx / num_pts, loss.item()))

            self.optimizer.step()             
            epoch_loss += loss.item()
            #print('loss',loss.item())

        training_err = 1-accuracy_score(y_hat,y_true)
        epoch_loss= epoch_loss/num_pts
        
        #print('num pts',num_pts)

        if(train_params.normalize_weights):
            model.normalize_weights()

        return epoch_loss, training_err


    def train(self,model,dataset, train_params = {} , val_set=None):
        '''
            max_epochs, loss_threshold=1e-6
        '''
        logger = self.logger 

        self.set_defaults(train_params)

        train_data_loader = DataLoader( dataset=dataset,
                                        batch_size= train_params.batch_size, 
                                        shuffle=train_params.shuffle,
                                        pin_memory=True,
                                        num_workers=2)

        train_loss = 0
        epoch_loss = 1e10
        epoch = 0
        self.optimizer = None 
        stop_crit = train_params['stopping_criterion']
        
        # just sets some flags
        model.train() 
        
        logger.debug('Training conf : {}'.format(train_params))

        logger.debug('Using stopping criterion {}'.format(stop_crit))

        inf_conf = {'device':train_params.device,'batch_size':128,'shuffle':False}
        
        if(self.optimizer is None):
            self.init_optimizer(model,train_params) 

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        while(epoch< train_params.max_epochs):
            logger.debug('------------------------------------------------------')
            logger.debug('Training Epoch {} Begins '.format(epoch))

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.debug('Epoch:{} Using learning rate : {}'.format(epoch,lr))

            epoch_loss, training_err = self.train_one_epoch(model,train_data_loader,train_params,epoch_num=epoch)
            
            scheduler.step()

            logger.info('Epoch: {} Training Error : {}'.format(epoch,training_err))
            
            stop = False 

            if(training_err <= train_params.train_err_tol):
                stop = True 

            if(stop_crit=='max_epochs' and epoch> train_params.max_epochs):
                stop = True 
                
            if(train_params.log_val_err and epoch%5==0):
                val_err  = self.get_validation_error(model,val_set,inf_conf)
                logger.debug('Epoch:{} Validation Error:{}'.format(epoch,val_err))
            
            if(train_params.stopping_criterion=='val_err_threshold'):
                val_err  = self.get_validation_error(model,val_set,inf_conf)
                if(val_err <= train_params.val_err_threshold):
                    stop= True 
                logger.debug('Epoch:{} Validation Error:{}'.format(epoch,val_err))

            if(stop_crit == 'loss_tol' and epoch_loss <= train_params.loss_tol):
                stop = True 
            
            if(stop):    
                logger.debug('Training Stopping criterion met. ')
                logger.debug('')
                break 

            train_loss += epoch_loss
            epoch += 1
            logger.debug(' Epoch loss : {}'.format(epoch_loss))
            logger.debug('Training Epoch {} Ends '.format(epoch))
            logger.debug('------------------------------------------------------')

        avg_train_loss = (1/epoch)*train_loss
        logger.debug('Average training loss : {}'.format(avg_train_loss))

        return epoch_loss

    def get_validation_error(self,model,val_set,inf_conf):
        
        inf_out = ClassfierInference(logger=self.logger).predict(model,val_set,inf_conf)
        val_error = 1 - accuracy_score(val_set.Y,inf_out['labels'])
        return val_error 