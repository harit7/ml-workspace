import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader
 
class ModelTraining:
    
    def __init__(self,logger):
        self.logger = logger
    
    def init_optimizer(self,model,train_params):
        
        self.optimizer = None 

        opt_name = train_params['optimizer_name']

        if(opt_name=='adam'):        
            optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'],
                                    weight_decay= train_params['weight_decay']
                                    )
        if(opt_name == 'lbfgs'):
            optimizer = torch.optim.LBFGS(model.parameters())
        
        else:
            optimizer = optim.SGD(model.parameters(), lr = train_params['learning_rate'], 
                                momentum= train_params['momentum'], 
                                weight_decay= train_params['weight_decay'])

        self.optimizer  = optimizer 


    def train_one_epoch(self,model,train_data_loader,train_params,epoch_num=0):
        
        device = train_params['device']

        if(self.optimizer is None):
            self.init_optimizer(model,train_params) 

        epoch_loss = 0
        num_pts = len(train_data_loader.dataset)
        
        model = model.to(device)

        for batch_idx, (data, target) in enumerate(train_data_loader):
            
            data, target = data.to(device), target.to(device)
            
            self.optimizer.zero_grad()   # set gradient to 0
            output  = model.forward(data)
            loss    = model.criterion(output, target) 
            loss.backward()    # compute gradient
            
            #if batchIdx%100 == 0:
            self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch_num, batch_idx * len(data), num_pts,
                                100. * batch_idx / num_pts, loss.item()))
            self.optimizer.step() 
            epoch_loss += loss.item()
        
        epoch_loss= epoch_loss/num_pts
            
        return epoch_loss

    def train(self,model,dataset, train_params = {} ):
        '''
            max_epochs, loss_threshold=1e-6
        '''
        
        train_params.setdefault('weight_decay',0)
        train_params.setdefault('momentum',0)
        train_params.setdefault('optimizer_name','sgd')
        train_params.setdefault('learning_rate',1e-2)
        train_params.setdefault('loss_tol',1e-6)
        train_params.setdefault('max_epochs',100)
        train_params.setdefault('shuffle',False)
        train_params.setdefault('batch_size',32)
        train_params.setdefault('device','cpu')

        train_data_loader = DataLoader(dataset=dataset.train_dataset,batch_size= train_params['batch_size'], shuffle=train_params['shuffle'])

        train_loss = 0
        epoch_loss = 1e10
        num_epochs = 0
        self.optimizer = None 
        
        model.train() # just sets some flags

        while(epoch_loss > train_params['loss_tol'] and num_epochs< train_params['max_epochs']):
            epoch_loss = self.train_one_epoch(model,train_data_loader,train_params,epoch_num=num_epochs)
            train_loss += epoch_loss
            num_epochs += 1
        avg_train_loss = (1/num_epochs)*train_loss
        return epoch_loss



