import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELossFunc(nn.Module):
    def __init__(self):
        super(MSELossFunc,self).__init__()
        # self.input = input
        # self.target = target
        
    def forward(self,input,target):
        loss = torch.sum((torch.mean((input.float()-target.float())**2)))
        return loss
    
