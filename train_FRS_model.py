import datetime

import time

from utils import utils


class FRS_Trainer(object):
    def __init(self,cmd,cuda,model,criterion,optimizer,train_loader,
               val_loader,log_file,max_iter,interval_validate=None,lr_scheduler=None,
               checkpoint_dir=None,print_freq=1,model_dict=None):
        self.cmd = cmd
        self.cuda = cuda
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.lr_scheduler = lr_scheduler
        self.model_dict = model_dict
        
        self.timestamp_start = datetime.datetime.now()
        
        if cmd == 'train':
            self.interval_validate = len(self.train_loader) if interval_validate is None else interval_validate

        self.epoch = 0
        self.iteration = 0

        self.max_iter = max_iter
        self.best_top1 = 0
        self.best_top5 = 0
        self.print_freq = print_freq

        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file
        
    def print_log(self, log_str):
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')
            
    def validate(self):
        batch_time = utils.AverageMeter()
        loss = utils.AverageMeter()
        
        training = self.model.training
        self.model.eval()
        
        end = time.time()
        
        for batch_idx,
        