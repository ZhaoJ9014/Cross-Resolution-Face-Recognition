import os
import math
import time
import random
import shutil
import pdb
from torch.nn.init import xavier_uniform as xavier
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
from helen_loader import *
from loss.loss import MSELossFunc,CrossEntropyLoss2d,MSELoss_Landmark

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

def weights_init(m):
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        xavier(m.weight.data)
        xavier(m.bias.data)
    '''
    for each in m.modules():
        if isinstance(each,nn.Conv2d):
            nn.init.xavier_uniform_(each.weight.data)
            if each.bias is not None:
                each.bias.data.zero_()
        elif isinstance(each,nn.BatchNorm2d):
            each.weight.data.fill_(1)
            each.bias.data.zero_()
        # elif isinstance(each,nn.InstanceNorm2d):
        #     each.weight.data.fill_(1)
        #     each.bias.data.zero_()
        elif isinstance(each,nn.Linear):
            nn.init.xavier_uniform_(each.weight.data)
            each.bias.data.zero_()


def main():
    parser = argparse.ArgumentParser("FSR Network on pytorch")
   
    # Datasets
    parser.add_argument('--dataset_dir',type=str,default="")
    parser.add_argument('-j','--workers',default=4,type=int,metavar='N',help='number of data loading workers (default = 4)')
    
    # Optimization option
    parser.add_argument('--epochs',default=600,type=int,metavar='N',help='number of total epochs to run')
    parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manual epoch number (useful to restart)')
    parser.add_argument('--train_batch',type=int,default=6,help='train batch size',metavar='N')
    parser.add_argument('--test_batch',type=int,default=6,help='test batch size',metavar='N')
    parser.add_argument('--lr','--learning_rate',default=0.001,type=float,help='initial learning rate',metavar='LR')
    parser.add_argument('--lr_decay_rate',default=0.99,type=float)
    parser.add_argument('--drop','--dropout',default=0.0,type=float,metavar='Dropout',help='Dropout ratio')
    parser.add_argument('--schedule',type=int,default=[100,300],nargs='+',help='Decrease learning rate at these epochs')
    parser.add_argument('--gamma',type=float,default=0.1,help='LR is multiplied by gamma on shedule')
    parser.add_argument('--momentum',default=0.9,type=float,metavar='M',help='momentum')
    parser.add_argument('--weight_decay','--wd',default=1e-5,type=float,metavar='W',help='weight decay')
    
    # Checkpoint
    parser.add_argument('-c','--checkpoint',default='./checkpoint/',type=str,metavar='PATH',help='Path to save checkpoint')
    # parser.add_argument('--resume',default='',type=str,metavar='PATH',help='path to ltest checkpoint (default=None)')
    parser.add_argument('--resume',default = './checkpoint/FSRNetwork_102.pth.tar',type=str)

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
    os.environ['CUDA_ENABLE_DEVICES'] = '0'
    torch.cuda.set_device(0)
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
    trainset = HelenLoader(is_transform=True,split='train_no_rotate')
    trainloader = data.DataLoader(trainset,batch_size=args.train_batch,shuffle=True,num_workers=args.workers)
    
    testset = HelenLoader(is_transform=True,split='test_no_rotate')
    testloader = data.DataLoader(testset,batch_size=args.test_batch,shuffle=True,num_workers=args.workers)
    
    # model
    model = OverallNetwork()
    #model = Course_SR_Network()
    # model = Prior_Estimation_Network()
    model.apply(weights_init)
    cudnn.benchmark = True
    print('Total params:  %.2fM' %(sum(p.numel() for p in model.parameters())/100000.0) )
    # pdb.set_trace()
    
    start_epoch = 0
    start_iteration = 0
    
    title = 'FSRNetwork'
    #title = 'CoarseNetwork'
    # title='PriorEstimationNetwork'
    if args.resume:
        checkpoint = torch.load(args.resume)
        
        
        pretrained_dict = {k:v for k,v in checkpoint['state_dict'].items() if '_fine_sr_decoder' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(checkpoint['state_dict'])
        
        start_epoch = checkpoint['epoch']
        # start_iteration = checkpoint['iteration']
        
    if use_cuda:
        model = model.cuda()
        
    decoder_params = list(map(id,model._fine_sr_decoder.parameters()))
    encoder_params = list(map(id,model._fine_sr_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in decoder_params+encoder_params, model.parameters())
    
    # Criterion
    criterion_mse = MSELossFunc().cuda()
    criterion_cross_entropy = CrossEntropyLoss2d().cuda()
    criterion_landmark = MSELoss_Landmark().cuda()

    # lr = args.lr
    # Train and Val
    for epoch in range(start_epoch,args.epochs):
        # Learning Rate Schedule
        '''
        lr_schedule = args.schedule*2.0/(epoch*1.0)
        if epoch < lr_schedule[0]:
            lr = args.lr
        elif epoch < lr_schedule[1]:
            lr = args.lr * 0.1
        else:
            lr = args.lr*0.01
        '''
        # lr = args.lr*/(epoch*1.5+1.0)
        # lr = args.lr 
        # if epoch%20==0:
        lr = args.lr*args.lr_decay_rate**epoch
        # Optimizer
        optim = torch.optim.RMSprop(model.parameters(),lr =lr,weight_decay=args.weight_decay,alpha=0.99)
        # optim = torch.optim.RMSprop(
        #     [{'params':base_params, 'lr':lr},
        #      {'params':model._fine_sr_encoder.parameters(),'lr':lr,'weight_decay':args.weight_decay*2.0},
        #      {'params':model._fine_sr_decoder.parameters(), 'lr':lr , 'weight_decay':args.weight_decay*2.0}],lr=lr,alpha=0.99,weight_decay=args.weight_decay)
        # optim = torch.optim.RMSprop(
        #     [{'params':model._fine_sr_decoder.parameters(), 'lr':lr , 'weight_decay':args.weight_decay*10.0}],lr=lr,alpha=0.99,weight_decay=args.weight_decay)

        print('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, args.epochs, lr))
        
        train_loss = train(trainloader,model,criterion_mse,criterion_cross_entropy,criterion_landmark,optim,epoch,use_cuda,train_batch=args.train_batch,lr=lr)
        test_loss = test(testloader,model,criterion_mse,criterion_cross_entropy,criterion_landmark,optim,epoch,use_cuda,test_batch=args.test_batch)
        
        is_best = best_loss>test_loss
        best_loss = min(test_loss,best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optim.state_dict(),
        }, is_best, checkpoint=args.checkpoint, filename=title + '_' + str(epoch) + '.pth.tar')
        

def train(train_loader, model, criterion_mse, criterion_cross_entropy,criterion_landmark,optimizer, epoch, use_cuda,train_batch,lr):
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
        # pdb.set_trace() 
        if use_cuda:
            batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = batch_lr_img.cuda(), batch_sr_img.cuda(), batch_lbl.cuda(), batch_landmark.cuda()
        batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = Variable(batch_lr_img), Variable(batch_sr_img), \
                                                                Variable(batch_lbl), Variable(batch_landmark)
        
        #--------------------------------------------------------------------------
        #train overall network   
        # compute output
        coarse_out, out_sr, out_landmark, out_lbl = model(batch_lr_img)
        # pdb.set_trace()
        loss = (5.*criterion_mse(out_sr, batch_sr_img) + 5.*criterion_mse(coarse_out, batch_sr_img) + \
               criterion_landmark(out_landmark,batch_landmark) + criterion_cross_entropy(out_lbl, batch_lbl))/(2.0*train_batch)
        # pdb.set_trace() 
        
        ## ---------------------------------------------------------------------------
        ##train coarse sr network 
        #out,coarse_out = model(batch_lr_img)
        #loss = (criterion_mse(coarse_out,batch_sr_img))/(2.0*train_batch)
        
        #-------------------------------------------------------------------------
        #train prior estimation network
        # out,landmark_out,parsing_out = model(batch_sr_img)
        # loss = (criterion_landmark(landmark_out,batch_landmark)+criterion_cross_entropy(parsing_out,batch_lbl))/(2.0*train_batch)
        losses.update(loss.data, batch_lr_img.size(0))
    
        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        # plot progress
        bar.suffix = '(Epoch: {epoch} | Learning Rate: {lr:.8f} | {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.6f}'.format(
			epoch = epoch,
			lr=lr,
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
        )
        print(bar.suffix)
        
        count += 1
        if count% 200 ==0:
            ## count = 0
            #rand_id = random.randint(0, 4)
            random_img, random_landmark, random_parsing,random_coarse,sr_img,lr_img = out_sr[0], out_landmark[0], out_lbl[0],coarse_out[0],batch_sr_img[0],batch_lr_img[0]
            ## pdb.set_trace()
            random_img, random_landmark, random_parsing,random_coarse= random_img.detach().cpu().numpy(), random_landmark.detach().cpu().numpy(), random_parsing.max(dim=0)[1].detach().cpu().numpy(), random_coarse.detach().cpu().numpy()
            sr_img = sr_img.detach().cpu().numpy()
            lr_img = lr_img.detach().cpu().numpy()
            ## pdb.set_trace()
            visualize.save_image(random_coarse,random_img, random_landmark, random_parsing,lr_img,sr_img, epoch,if_train=True,count=int(count/200))
            ##-----------------------------------------------------------------------
            #visualize coarse network
            #random_coarse = coarse_out[0]
            # random_landmark = landmark_out[0]
            # random_parsing = parsing_out[0]
            # #random_coarse = random_coarse.detach().cpu().numpy()
            # random_landmark = random_landmark.detach().cpu().numpy()
            # random_parsing = random_parsing.max(dim=0)[1].detach().cpu().numpy()
            # visualize.save_image(landmark=random_landmark,parsing=random_parsing,epoch=epoch,if_train=True,count=int(count/100))
        bar.next()

    bar.finish()
    return losses.avg


def test(test_loader, model, criterion_mse,criterion_cross_entropy, criterion_landmark ,optimizer, epoch, use_cuda,test_batch):
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
            loss = (7.0*criterion_mse(out_sr, batch_sr_img) + 7.0*criterion_mse(coarse_out, batch_sr_img) + criterion_landmark(out_landmark,batch_landmark) + criterion_cross_entropy(out_lbl, batch_lbl))/(2.0*test_batch)
             
            # ---------------------------------------------------------------------------
            #train coarse sr network 
            #out,coarse_out = model(batch_lr_img)
            #loss = criterion_mse(coarse_out,batch_sr_img)/(2.0*test_batch) 
            # rand_id = random.randint(0,4)
            #-------------------------------------------------------------------------
            #train prior estimation network
            # out,landmark_out,parsing_out = model(batch_sr_img)
            # loss = (criterion_landmark(landmark_out,batch_landmark)+criterion_cross_entropy(parsing_out,batch_lbl))/(2.0*test_batch)
            losses.update(loss.data, batch_lr_img.size(0))

            count += 1
            if count%90 ==0:
                #rand_id = random.randint(0, 4)
                ## count = 0
                random_img, random_landmark, random_parsing, random_coarse, sr_img, lr_img = out_sr[0], out_landmark[0], \
                                                                                             out_lbl[0], coarse_out[0], \
                                                                                             batch_sr_img[0], \
                                                                                             batch_lr_img[0]
                random_img, random_landmark, random_parsing ,random_coarse= random_img.detach().cpu().numpy(), random_landmark.detach().cpu().numpy(), random_parsing.max(dim=0)[1].detach().cpu().numpy() , random_coarse.detach().cpu().numpy()
                sr_img = sr_img.detach().cpu().numpy()
                lr_img = lr_img.detach().cpu().numpy()

                visualize.save_image(random_coarse, random_img, random_landmark, random_parsing, lr_img, sr_img, epoch,
                                     if_train=False, count=int(count / 90))

                ##-----------------------------------------------------------------------
                ##visualize coarse network
                #random_coarse = coarse_out[0]
                #random_coarse = random_coarse.detach().cpu().numpy()
                #visualize.save_image(coarse_image=random_coarse,epoch=epoch,if_train=True,count=int(count/90))        

                # random_landmark = landmark_out[0]
                # random_parsing = parsing_out[0]
                #random_coarse = random_coarse[0].detach().cpu().numpy()
                # random_landmark = random_landmark.detach().cpu().numpy()
                # random_parsing = random_parsing.max(dim=0)[1].detach().cpu().numpy()
                # visualize.save_image(landmark=random_landmark,parsing=random_parsing,epoch=epoch,if_train=False,count=int(count/5))

            losses.update(loss.data, batch_lr_img.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.6f}'.format(
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
        
    
    
    
    
