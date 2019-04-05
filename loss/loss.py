import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class MSELossFunc(nn.Module):
    def __init__(self):
        super(MSELossFunc,self).__init__()
        # self.input = input
        # self.target = target
        
    def forward(self,input,target):
        loss = torch.sum((torch.mean((input.float()-target.float())**2)))
        return loss

class MSELoss_Landmark(nn.Module):
    def __init__(self):
        super(MSELoss_Landmark,self).__init__()
    def forward(self,input,target):
        weight = np.ones((input.size()[0],input.size()[1],input.size()[2],input.size()[3]))
        weight[:,0:40,:,:] = weight[:,0:40,:,:]*5.0
        weight = torch.from_numpy(weight.astype(np.float32)).cuda()
        loss = torch.sum((torch.mean(((input.float()-target.float()).mul(weight))**2)))
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self,weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss()
    '''
    def forward(self ,output,target,weight=None,size_average=True):
        n,c,h,w = output.size()
        nt,ht,wt = target.squeeze().size()
        m = nn.LogSoftmax(dim=1)
        # example
        N,C = 5,4
        data = torch.randn(N,16,10,10)
        conv = nn.Conv2d(16,C,(3,3))
        output1 = conv(data)
        out1 = m(output1)
        target1 = torch.empty(N,8,8,dtype=torch.long).random_(0,C)
        # network
        out = m(output)
        pdb.set_trace()
        output = output.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
        target = target.view(-1)
        
        loss = F.cross_entropy(output,target,weight=weight,size_average = size_average,\
               ignore_index=250)
        return loss
    '''

    def forward(self,outputs,targets):
        return self.loss(F.log_softmax(outputs,1),torch.squeeze(targets))
