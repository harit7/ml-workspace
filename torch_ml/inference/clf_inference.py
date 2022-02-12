from asyncio.log import logger
import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader


class ClassfierInference:
    def __init__(self,logger):
        self.logger = logger
    
    def predict(self,model,dataset,inference_conf={}):
        
        inference_conf.setdefault('batch_size',64)
        inference_conf.setdefault('shuffle',False)
        inference_conf.setdefault('device','cpu')
        
        device = inference_conf['device']
        data_loader = DataLoader(dataset=dataset.test_dataset,batch_size= inference_conf['batch_size'], shuffle=inference_conf['shuffle'])

        model = model.to(device)
        
        with torch.no_grad():
            model.eval() 
            lst_confs = []
            lst_preds = []
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)

                output  = model.forward(data)
                confidences, y_hat = torch.max(output, 1)
                lst_confs.extend(confidences.cpu().numpy())
                lst_preds.extend(y_hat.cpu().numpy())
            return torch.Tensor(lst_preds).long(),torch.Tensor(lst_confs)
