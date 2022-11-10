import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class OneLayerSoftmax(nn.Module):

    def __init__(self,n_class=10):
        super(OneLayerSoftmax, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=28*28, out_features=n_class) 
        #self.act1 = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits  = self.fc1(x)
        #logits  = self.act1(logits)
        probs = F.softmax(logits)
        #print(logits.shape,probs.shape)
        out = {}
        out['probs'] = probs 
        out['abs_logits'] =  torch.abs(logits)
        return out
    
    def embedding(self,x):
        logits  = self.fc1(x)
        probs = F.softmax(logits)
        out = {}
        out['embedding'] = logits.detach().numpy()
        probs.mean().backward()
        out['grad_embedding'] = probs
        return out

    def criterion(self,input,targets):
        loss = nn.CrossEntropyLoss()
        return loss(input,targets)