import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

import torchvision

import numpy as np
import os
import time
import shutil

from model.model_irse import IR_50
from model.resnet import ResNet_34
import argparse
import pdb

from utils import Bar,Logger,AverageMeter,normalizedME,mkdir_p,savefig,visualize
from utils.utils import calculate_roc
from loss.loss import MSELossFunc,CrossEntropyLoss2d,MSELoss_Landmark
from distill_loader import DistillTestLoader,DistillTrainLoader

def load_teacher_model(teacher_net,model_name):
    print('*'*60)
    print('\n'*2)
    print('Loading Teacher Model')
    checkpoint = torch.load('./ms1m-ir50/' + model_name)
    model_dict = teacher_net.state_dict()
    pretrained_dict = {k:v for k,v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    
    teacher_net.load_state_dict(model_dict)
    print('\n'*2)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        
def train(train_loader,teacher_model,student_model,assistant_model,student_optimizer,assistant_optimizer,criterion,epoch,lr,use_cuda):
    
    student_model.train()
    assistant_model.train()
    teacher_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    student_losses = AverageMeter()
    assistant_losses = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    
    for batch_idx,batch_img in enumerate(train_loader):
        if use_cuda:
            batch_img = batch_img.cuda()
        batch_img = Variable(batch_img)
        t_outputs, t_out1,t_out2, t_out3, t_out4 = teacher_model(batch_img)
        s_outputs, s_out1, s_out2, s_out3, s_out4 = student_model(batch_img)
        a_outputs, a_out1, a_out2, a_out3, a_out4 = assistant_model(batch_img)
        
        student_loss = criterion(s_outputs,t_outputs.detach())
        student_optimizer.zero_grad()
        student_loss.backward(retain_graph=True)
        student_optimizer.step()
        
        assistant_loss = criterion(t_out1-s_out1,a_out1) + criterion(t_out2-s_out2,a_out2) \
            + criterion(t_out3-s_out3,a_out3) + criterion(t_out4-s_out4,a_out4) \
                         + criterion(t_outputs-s_outputs,a_outputs)
        
        assistant_optimizer.zero_grad()
        assistant_loss.backward()
        student_optimizer.step()
        
        assistant_losses.update(assistant_loss,batch_img.size(0))
        student_losses.update(student_loss,batch_img.size(0))
        
        batch_time.update(time.time()-end)
        end = time.time()

        bar.suffix = '(Epoch: {epoch} | Learning Rate: {lr:.8f} | {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Student Loss: {student_loss:.6f} | Assistant Loss: {assistant_loss:.6f}'.format(
            epoch=epoch,
            lr=lr,
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            student_loss=student_losses.avg,
            assistant_loss=assistant_losses.avg,
            
        )
        print(bar.suffix)
        bar.next()
    bar.finish()
    return student_losses.avg,assistant_losses.avg
        
def test(test_loader,teacher_model,student_model,assistant_model):
    global best_acc
    teacher_model.eval()
    student_model.eval()
    assistant_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    teacher_accs = AverageMeter()
    student_accs = AverageMeter()
    bar = Bar('Processing', max=len(test_loader))

    end = time.time()
    
    for idx,[img1,img2,label] in enumerate(test_loader):
        with torch.no_grad():
            img1 = img1.cuda()
            img2 = img2.cuda()
            t_embeddings1,t_out1,t_out2, t_out3, t_out4 = teacher_model(img1)
            t_embeddings2,t_out1,t_out2, t_out3, t_out4 = teacher_model(img2)
            t_embeddings1 = t_embeddings1.cpu().numpy()
            t_embeddings2 = t_embeddings2.cpu().numpy()

            thresholds = np.arange(0, 12000, 3)

            t_tpr, t_fpr, t_accuracy, t_best_thresholds = calculate_roc(thresholds, t_embeddings1, t_embeddings2,
                                                                np.asarray(label), nrof_folds=10,
                                                                pca=0)

            s_output, s_out1, s_out2, s_out3, s_out4 = student_model(img1)
            a_output, a_out1, a_out2, a_out3, a_out4 = assistant_model(img1)
            s_embeddings1 = s_output.cpu().numpy()+a_output.cpu().numpy()

            s_output, s_out1, s_out2, s_out3, s_out4 = student_model(img2)
            a_output, a_out1, a_out2, a_out3, a_out4 = assistant_model(img2)
            s_embeddings2 = s_output.cpu().numpy() + a_output.cpu().numpy()
            s_tpr, s_fpr, s_accuracy, s_best_thresholds = calculate_roc(thresholds, s_embeddings1, s_embeddings2,
                                                                        np.asarray(label), nrof_folds=10,
                                                                        pca=0)
            student_accs.update(s_accuracy,img1.size(0))
            teacher_accs.update(t_accuracy,img1.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Teacher Acc: {teacher_accs:.6f} | Student Acc: {student_accs:.6f}'.format(
                batch=idx + 1,
                size=len(test_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                teacher_accs=teacher_accs.avg,
                student_accs=student_accs.avg,
            )
            print(bar.suffix)
            bar.next()

    bar.finish()
    return teacher_accs.avg,student_accs.avg
    
    
def main():
    parser = argparse.ArgumentParser(description="Train Student Model")
    parser.add_argument('--input_size',default=[112, 112],help='image size')
    parser.add_argument('--workers',default=4,type=int)
    #********************  Optimization option ********************
    parser.add_argument('--epochs',default=200,type=int,help="num of total train epochs")
    parser.add_argument('--start_epoch',default=0,type=int,help="manual epoch number (use for restart)")
    parser.add_argument('--train_batch',type=int,default=32,help='train batch size')
    parser.add_argument('--test_batch',type=int,default=150,help='test batch size')
    parser.add_argument("--lr",default=0.0001,help="learning rate")
    parser.add_argument("--lr_decay_rate",default=0.99,type=float,help="decay weight of learning rate")
    parser.add_argument("--weight_decay",default=1e-5,help="weight decay")
    
    #********************  Checkpoint ********************
    parser.add_argument('--checkpoint',default="./distill_checkpoint",type=str,help="checkpoint path")
    parser.add_argument('--resume',default="",type=str,help="path to load checkpoint")
    parser.add_argument('--teacher_model_name',default='backbone_ir50_ms1m_epoch120.pth',help='teacher model name')
    
    #********************  Device Option  ********************
    parser.add_argument("--gpu_id",type=str,default="0")
    
    args = parser.parse_args()
    print(args)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_ENABLE_DEVICES'] = '0'
    torch.cuda.set_device(0)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
        device = 'cuda'

    global best_acc
    best_acc = 0.0
    
    print('===> Preparing Dataset <===')
    trainset = DistillTrainLoader()
    train_loader = data.DataLoader(trainset,batch_size=args.train_batch,shuffle=True,num_workers=args.workers)
    
    testset = DistillTestLoader()
    test_loader = data.DataLoader(testset,batch_size=args.test_batch,shuffle=True,num_workers=args.workers)
    print('===> Building Model <===')
    teacher_net = IR_50(args.input_size)
    student_net = ResNet_34()
    assistant_net = ResNet_34()
    
    if use_cuda:
        teacher_net = teacher_net.cuda()
        student_net = student_net.cuda()
        assistant_net = assistant_net.cuda()
        
    criterion_MSE = nn.MSELoss()
    
    load_teacher_model(teacher_net,args.teacher_model_name)
    
    for epoch in range(args.start_epoch,args.epochs):
        lr = args.lr * args.lr_decay_rate ** epoch
        # if epoch<50:
        #     lr = args.lr
        # elif epoch < 60:
        #     lr = args.lr*0.1
        # else:
        #     lr = args.lr*0.03
        student_optimizer = optim.RMSprop(student_net.parameters(), lr=lr, weight_decay=args.weight_decay,
                                          alpha=0.99)
        assistant_optimizer = optim.RMSprop(assistant_net.parameters(), lr=lr, alpha=0.99,
                                            weight_decay=args.weight_decay)

        print('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, args.epochs, lr))
        
        student_loss,assistant_loss = train(train_loader=train_loader,teacher_model=teacher_net,
                                            student_model=student_net,assistant_model=assistant_net,
                                            student_optimizer=student_optimizer,assistant_optimizer=assistant_optimizer,
                                            criterion=criterion_MSE,epoch=epoch,lr=lr,use_cuda=True)
        teacher_acc,student_acc = test(test_loader=test_loader,teacher_model=teacher_net,
                                       student_model=student_net,assistant_model=assistant_net)
        is_best = best_acc<student_acc
        best_acc = max(student_acc,best_acc)
        
        if is_best:
            save_checkpoint({
                'epoch':epoch+1,
                'state_dict':student_net.state_dict(),
                'best_acc':best_acc,
                'optimizer':student_optimizer.state_dict(),
            }, is_best,checkpoint=args.checkpoint,filename='student_'+'.pth.tar')
    
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': assistant_net.state_dict(),
                'best_acc': best_acc,
                'optimizer': assistant_optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint, filename='assistant_' +'.pth.tar')
    
if __name__=="__main__":
    main()