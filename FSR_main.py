import os
import time
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets.FaceLandmarksDataset import *

from model.FSRnet import *
import argparse

from utils import Bar,Logger,AverageMeter,normalizedME,mkdir_p,savefig,visualize
from helen_loader import HelenLoader

from loss.loss import MSELossFunc

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-1,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.1,  # "lr_policy: step"
        step_size=1000000,  # "lr_policy: step"
        interval_validate=1000,
    ),
}

def main():
    parser = argparse.ArgumentParser("FSR Network on pytorch")
    
    # Datasets
    parser.add_argument('--dataset_dir',type=str,default="")
    parser.add_argument('-j','--workers',default=1,type=int,metavar='N',help='number of data loading workers (default = 4)')
    
    # Optimization option
    parser.add_argument('--epochs',default=600,type=int,metavar='N',help='number of total epochs to run')
    parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manual epoch number (useful to restart)')
    parser.add_argument('--train_batch',type=int,default=4,help='train batch size',metavar='N')
    parser.add_argument('--test_batch',type=int,default=4,help='test batch size',metavar='N')
    parser.add_argument('--lr','--learning_rate',default=0.01,type=float,help='initial learning rate',metavar='LR')
    parser.add_argument('--drop','--dropout',default=0.0,type=float,metavar='Dropout',help='Dropout ratio')
    parser.add_argument('--schedule',type=int,default=[60,120],nargs='+',help='Decrease learning rate at these epochs')
    parser.add_argument('--gamma',type=float,default=0.1,help='LR is multiplied by gamma on shedule')
    parser.add_argument('--momentum',default=0.9,type=float,metavar='M',help='momentum')
    parser.add_argument('--weight_decay','--wd',default=5e-4,type=float,metavar='W',help='weight decay')
    
    # Checkpoint
    parser.add_argument('-c','--checkpoint',default='./checkpoint/',type=str,metavar='PATH',help='Path to save checkpoint')
    parser.add_argument('--resume',default='',type=str,metavar='PATH',help='path to ltest checkpoint (default=None)')
    
    # Miscs
    parser.add_argument('--manualSeed',type=int,help='manual seed')
    parser.add_argument('-e','--evaluate',dest='evaluate',action='store_true',help='eval model on validation set')
    
    # Device option
    parser.add_argument('--gpu_id',type=str,default='0')
    args = parser.parse_args()
    # state = {k:v for k,v in args._get_kwards()}
    # parser.add_argument('--train_img_list_file',type=str,default='')
    # parser.add_argument('--test_img_list_file',type=str,default='')
    # parser.add_argument('--gpu',type=int,default=0)
    # args = parser.parse_args()
    print(args)
    
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
        
    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1,10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        
    global best_loss
    best_loss = 9999999
    
    # Data
    print('====> Preparing dataset <====')
    trainset = HelenLoader(is_transform=True,split='train')
    trainloader = data.DataLoader(trainset,batch_size=args.train_batch,shuffle=True,num_workers=args.workers)
    
    testset = HelenLoader(is_transform=True,split='test')
    testloader = data.DataLoader(testset,batch_size=args.test_batch,shuffle=True,num_workers=args.workers)
    
    # model
    model = OverallNetwork()
    cudnn.benchmark = True
    print('Total params:  %.2fM' %(sum(p.numel() for p in model.parameters())/100000.0) )
    
    start_epoch = 0
    start_iteration = 0
    
    title = 'FSRNetwork'
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        
    if use_cuda:
        model = model.cuda()
        
    # Optimizer
    optim = torch.optim.Adam(model.parameters(),lr =args.lr,weight_decay=args.weight_decay)
    
    # Criterion
    criterion = MSELossFunc().cuda()
    

    # Train and Val
    for epoch in range(start_epoch,args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, args.lr))
        
        train_loss = train(trainloader,model,criterion,optim,epoch,use_cuda)
        test_loss = test(testloader,model,criterion,optim,epoch,use_cuda)
        
        is_best = best_loss>test_loss
        best_loss = min(test_loss,best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optim.state_dict(),
        }, is_best, checkpoint=args.checkpoint, filename=title + '_' + str(epoch) + '.pth.tar')
        

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    NormMS = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    count = 0

    for batch_idx, [batch_lr_img, batch_sr_img, batch_lbl, batch_landmark] in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
    
        if use_cuda:
            batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = batch_lr_img.cuda(), batch_sr_img.cuda(), batch_lbl.cuda(), batch_landmark.cuda()
        batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = Variable(batch_lr_img), Variable(batch_sr_img), \
                                                                Variable(batch_lbl), Variable(batch_landmark)
    
        # compute output
        coarse_out, out_sr, out_landmark, out_lbl = model(batch_lr_img)
        
        loss = criterion(out_sr, batch_sr_img) + criterion(coarse_out, batch_sr_img) + \
               criterion(out_landmark,batch_landmark) + criterion(out_lbl, batch_lbl)
    
        losses.update(loss.data, batch_lr_img.size(0))
    
        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
        )
        print(bar.suffix)
        
        count += 1
        if count%200 ==0:
            # count = 0
            rand_id = random.randint(0, 4)
            random_img, random_landmark, random_parsing = out_sr[0], out_landmark[0], out_lbl[0]
            random_img, random_landmark, random_parsing = random_img.detach().cpu().numpy(), random_landmark.detach().cpu().numpy(), random_parsing.detach().cpu().numpy()
            visualize.save_image(random_img, random_landmark, random_parsing, epoch,if_train=True,count=int(count/200))
        
        bar.next()

    bar.finish()
    return losses.avg


def test(test_loader, model, criterion, optimizer, epoch, use_cuda):
    global best_loss
    
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    bar = Bar('Processing', max=len(test_loader))
    count = 0

    for batch_idx, [batch_lr_img, batch_sr_img, batch_lbl, batch_landmark] in enumerate(test_loader):
        with torch.no_grad():
    
            # measure data loading time
            data_time.update(time.time() - end)
            
            if use_cuda:
                batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = batch_lr_img.cuda(), batch_sr_img.cuda(), batch_lbl.cuda(), batch_landmark.cuda()
            batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = Variable(batch_lr_img), Variable(batch_sr_img), Variable(batch_lbl), Variable(batch_landmark)
            
            # compute output
            coarse_out, out_sr, out_landmark, out_lbl = model(batch_lr_img)
            loss = criterion(out_sr, batch_sr_img) + criterion(coarse_out, batch_sr_img) + criterion(out_landmark,batch_landmark) + criterion(out_lbl, batch_lbl)
            
            # rand_id = random.randint(0,4)
            count += 1
            if count%10 ==0:
                rand_id = random.randint(0, 4)
                # count = 0
                random_img, random_landmark, random_parsing = out_sr[0], out_landmark[0], out_lbl[0]
                random_img, random_landmark, random_parsing = random_img.detach().cpu().numpy(), random_landmark.detach().cpu().numpy(), random_parsing.detach().cpu().numpy()
                visualize.save_image(random_img, random_landmark, random_parsing, epoch,if_train=False,count=int(count/200))
            
            losses.update(loss.data, batch_lr_img.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(test_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
        )
        print(bar.suffix)
        bar.next()
    
    bar.finish()
    return losses.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    
if __name__=='__main__':
    main()
        
    
    
    
    