import numpy as np
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import os
import pdb
import random
import collections

from PIL import Image,ImageEnhance


class DistillTrainLoader(data.Dataset):
    '''
    Trainloader for distillation
    '''
    def __init__(self,
                 helen_root = '/media/hyo/文档/Dataset/face_dataset_v3/',
                 split = 'train_no_rotate',
                 is_transform = True,
                 train_mode = True
                 ):
        self.helen_root = helen_root
        self.train_mode = train_mode
        self.split = split
        self.is_transform = is_transform
        self.files = collections.defaultdict(list)
        
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ]
        )
        
        for split in ['train_no_rotate','test_no_rotate']:
            path = os.path.join(self.helen_root,split+".txt")
            file_list = tuple(open(path,'r'))
            self.files[split] = file_list
            
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        if self.train_mode:
            img_name = self.files[self.split][index]
    
            random_angle = random.uniform(-10, 10)
            random_contrast = random.uniform(0.95, 1.05)
            random_brightness = random.uniform(0.9, 1.1)
            random_sharpness = random.uniform(0.95, 1.05)
            
            img = Image.open(os.path.join(self.helen_root,'Helen_aligned_224_renew',img_name.strip('\n'))).resize((112,112))
            img = img.rotate(random_angle)
            
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(random_contrast)
    
            brightness = ImageEnhance.Contrast(img)
            img = brightness.enhance(random_brightness)
    
            sharpness = ImageEnhance.Contrast(img)
            img = sharpness.enhance(random_sharpness)
            
            if self.is_transform:
                img = self.tf(img)
                
            return img

class DistillTestLoader(data.Dataset):
    '''
    Testloader for distillation
    '''

    def __init__(self,
                 img_root='',
                 pair_txt_root='/media/hyo/文档/Dataset/face_dataset_v3/test_img_pair.txt',
                 helen_root= '/media/hyo/文档/Dataset/face_dataset_v3/',
                 ):
        self.img_root = img_root
        self.pair_txt_root = pair_txt_root
        self.helen_root = helen_root
        self.files = collections.defaultdict(list)
    
        self.file_list = tuple(open(self.pair_txt_root))
    
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_pair_list = self.file_list[index].strip('\n').split(' , ')
        img_name1 = img_pair_list[0]
        img_name2 = img_pair_list[1]
        label = int(img_pair_list[2])
        img1 = Image.open(os.path.join(self.helen_root, 'Helen_aligned_224_renew', img_name1))
        img1 = img1.resize((112, 112))
        img2 = Image.open(os.path.join(self.helen_root, 'Helen_aligned_224_renew', img_name2)).resize(
            (112, 112))
        img1 = self.tf(img1)
        img2 = self.tf(img2)
        return [img1, img2, label]
        

        