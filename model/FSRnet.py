import torch

import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

class _Residual_Block(nn.Module):
    def __init__(self,out_channels,in_channels=64):
        super(_Residual_Block,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels,affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels,affine=True)
        
    def forward(self, x):
        identity_data = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = torch.add(out,identity_data)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.InstanceNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)
    
    
class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 128
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.InstanceNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.InstanceNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out
    

def hg2(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=2, num_blocks=4,
                         num_classes=kwargs['num_classes'])
    return model

class Course_SR_Network(nn.Module):
    def __init__(self):
        super(Course_SR_Network,self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        
        self.relu = nn.PReLU(64)
        
        self.residual = self.make_layer(_Residual_Block,3,out_channel=64)
        self.dropout = nn.Dropout2d(p=0.5,inplace=True)
        self.conv_mid = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn_mid = nn.InstanceNorm2d(64,affine=True)
        
    def make_layer(self,block,num_of_layer,out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(out_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn_mid(self.conv_input(x)))
        out = self.dropout(out)
        out = self.residual(out)
        out = self.residual(out)
        # out = self.residual(out)
        # out = self.residual(out)
        #out = self.residual(out)
        out_coarse = self.conv_mid(out)
        
        return out,out_coarse
    
class Fine_SR_Encoder(Course_SR_Network):
    def __init__(self):
        super(Fine_SR_Encoder,self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu = nn.PReLU(64)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)
        
        self.residual = self.make_layer(_Residual_Block,3,out_channel=64)
        
        self.conv_end = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
    
    def make_layer(self,block,num_of_layer,out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(out_channels=out_channel))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.relu(self.bn_mid(self.conv_input(x)))
        out = self.dropout(out)
        # 12 residual blocks
        out = self.residual(out)
        out = self.dropout(out)
        out = self.residual(out)
        out = self.residual(out)
        out = self.residual(out)
        # out = self.residual(out)
        # out = self.residual(out)
        # out = self.residual(out)
        # out = self.residual(out)
        # out = self.residual(out)
        # out = self.residual(out)
        #out = self.residual(out)
        # out = self.residual(out)

        out = self.relu(self.bn_mid(self.conv_end(out)))
        
        return out
    
class Prior_Estimation_Network(nn.Module):
    def __init__(self):
        super(Prior_Estimation_Network,self).__init__()
        self.conv = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=7,stride=2,padding=3,bias=True)
        self.bn = nn.InstanceNorm2d(128,affine=True)
        self.relu = nn.PReLU(128)
        self.residual = self.make_layer(_Residual_Block,3,out_channel=128,in_channel=128)
        self.residual_next = self.make_layer(_Residual_Block,3,out_channel=128,in_channel=128)
        self.hg = Hourglass(planes=64,depth=4,block=Bottleneck,num_blocks=2)
        self.dropout = nn.Dropout2d(p=0.5,inplace=True)
        self.fc = nn.Conv2d(in_channels=128, out_channels=11, kernel_size=1, bias=True)
        self.fc_landmark = nn.Conv2d(in_channels=128,out_channels=97,kernel_size=1,bias=True)
        # self.fc_landmark1 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,bias=False)
        # self.landmark_fc = nn.Linear(in_features=112*112*11,out_features=194*2)


    def make_layer(self, block, num_of_layer,in_channel, out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(out_channel,in_channels=in_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.dropout(out)
        out = self.residual(out)
        out = self.dropout(out)
        out = self.residual_next(out)
        # out = self.residual_next(out)
        # out = self.hg(out)
        
        out = self.hg(out)      # planes = 128
        parsing_out = self.fc(out)
        # landmark_out = self.fc_landmark1(out)
        landmark_out = self.fc_landmark(out)
        # landmark_out = landmark_out.view(landmark_out.size(0), -1)
        # landmark_out = self.landmark_fc(landmark_out)
        # landmark_out = self.fc(landmark_out)
        return out,landmark_out,parsing_out
        
class Fine_SR_Decoder(nn.Module):
    def __init__(self):
        super(Fine_SR_Decoder,self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=192,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.relu = nn.PReLU(64)
        self.bn_mid = nn.InstanceNorm2d(64,affine=True)
        
        self.deconv = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,bias=True,padding=1,output_padding=1)
        self.residual = self.make_layer(_Residual_Block, 3,out_channel=64)
        self.dropout = nn.Dropout2d(p=0.5,inplace=True)
        self.conv_out = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1,bias=True)
        self.instance_norm = nn.InstanceNorm2d(3,affine=True)
        
    def make_layer(self,block,num_of_layer,out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(out_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn_mid(self.conv_input(x)))
        # out = self.dropout(out)
        out = self.relu(self.bn_mid(self.deconv(out)))
        # out = self.dropout(out)
        out = self.residual(out)
        out = self.residual(out)
        # out = self.residual(out)
        
        out = self.conv_out(out)
        # out = self.instance_norm(out)
        return out
        
class OverallNetwork(nn.Module):
    def __init__(self):
        super(OverallNetwork,self).__init__()
        self._coarse_sr_network = Course_SR_Network()
        self._prior_estimation_network = Prior_Estimation_Network()
        self._fine_sr_encoder = Fine_SR_Encoder()
        self._fine_sr_decoder = Fine_SR_Decoder()
        self.softmax = nn.Softmax()
        # self.deconv = nn.ConvTranspose2d(in_channels=16,out_channels=11,kernel_size=3,stride=2,bias=False,padding=1,output_padding=1)
    def forward(self,x):
        out,coarse_out = self._coarse_sr_network(x)
        out_sr = self._fine_sr_encoder(out)
        out_pe,landmark_out,parsing_out = self._prior_estimation_network(out)
        #landmark_out = self.softmax(landmark_out)
        #parsing_out = self.deconv(parsing_out)
        # out = torch.cat((out_sr,landmark_out),1)
        # out = torch.cat((out,parsing_out),1)
        out = torch.cat((out_pe,out_sr),1)
        out = self._fine_sr_decoder(out)
        # pdb.set_trace()
        return coarse_out,out,landmark_out,parsing_out
    
if __name__=='__main__':
    model = OverallNetwork().cuda()
    input_data = Variable(torch.rand(3, 3, 224, 224)).cuda()

    print(model(input_data)[0].size())
    pdb.set_trace()
    
        
