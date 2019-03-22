import pandas as pd
import numpy as np
import collections
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import glob
import os
import scipy.io as io
import scipy.misc as m
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import trange
from os.path import join as pjoin

import pdb

class HelenLoader(data.Dataset):
    """
    Data loader for helen dataset.
    """
    def __init__(
            self,
            parsing_root='/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/labels_aligned/',
            target_path='/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/Segmentation/',
            helen_root='/media/hyo/文档/Dataset/',
            landmark_root='/home/hyo/xixixi.csv',
            is_transform=False,
            img_size=224,
            split = 'train',
            img_norm=True,
            
            test_mode=False):
        
        self.helen_root = helen_root
        self.target_path = target_path
        self.parsing_root = parsing_root
        self.test_mode = test_mode
        self.landmark_root = landmark_root
        self.n_classes = 11
        self.is_transform = is_transform
        self.split = split
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.df = pd.read_csv(self.landmark_root)

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        # self.files 是一个dict，格式为{‘train’：[img1_name , img2_name , ...] , 'val':[... , ... , ...]}
        # 无jpg或png等后缀
        if not self.test_mode:
            for split in ['train','test']:
                path = pjoin(self.helen_root,split+'.txt')
                file_list = tuple(open(path,'r'))
                self.files[split] = file_list
            
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        # 需要返回的是lr和hr图片对/landmark坐标/parsing map
        img_name = self.files[self.split][index]        # train image name or test image name
        # lr image
        lr_img = Image.open(pjoin(self.helen_root,'Helen_aligned_28',self.split,img_name.strip('\n')+'.jpg'))
        # SR image
        sr_img = Image.open(pjoin(self.helen_root,'Helen_aligned',self.split,img_name.strip('\n')+'.jpg'))
        # parsing map

        parsing = Image.open(pjoin(self.target_path,img_name.strip('\n')+'.png'))
        # landmark
        id = pjoin(self.helen_root,'Helen_aligned',self.split,img_name.strip('\n')+'.jpg')
        landmark = self.df[self.df['img_name']==id].values[0][2:]
        if self.is_transform:
            lr_img,sr_img,parsing,landmark = self.transform(lr_img,sr_img,parsing,landmark)
        return [lr_img,sr_img,parsing,landmark]
        
        
    def transform(self, lr_img,sr_img, lbl,landmark):
        # if self.img_size == ("same", "same"):
        #     pass
        # else:
        #     img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        #     lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        lr_img = self.tf(lr_img)
        sr_img = self.tf(sr_img)
        lbl = torch.from_numpy(np.expand_dims(np.array(lbl),axis=0)).long()
        landmark = torch.from_numpy(landmark.astype(np.float32)).long()
        lbl[lbl == 255] = 0
        return lr_img,sr_img, lbl,landmark
    
    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (11, 3)
        """
        return np.asarray(
            [
                [255, 255, 255],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 128],
                [0, 64, 0],
                [0, 192, 0],
                [0, 64, 128],
            ]
        )
    
    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask1 = mask/(mask.max()-mask.min())*11.0
        # pdb.set_trace()
        if mask1.min()<0:
            mask1 += mask1.min()
        mask = mask1.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == ii, axis=-1))[:2]] = label
        label_mask = label_mask.astype(int)
        return label_mask
    
    
    def decode_segmap(self, label_mask, plot=False,save=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()

        label_mask1 = label_mask / (label_mask.max() - label_mask.min()) * 11.0
        # pdb.set_trace()
        if label_mask1.min() < 0:
            label_mask1 = label_mask1 - label_mask1.min()
        label_mask1 = label_mask1.astype(int)
        
        
        r = label_mask1.copy()
        g = label_mask1.copy()
        b = label_mask1.copy()
        for ll in range(0, self.n_classes):
            r[label_mask1 == ll] = label_colours[ll, 0]
            g[label_mask1 == ll] = label_colours[ll, 1]
            b[label_mask1 == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask1.shape[0], label_mask1.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
        
    def setup_annotation(self):
        # '/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/labels_aligned/'
        parsing_path = self.parsing_root
        target_path = self.target_path
        subfolder = os.listdir(parsing_path)
        for sub in subfolder:
            img_name = sub+'.png'
            result = np.zeros([224,224],dtype=np.int16)
            for lbl in os.listdir(pjoin(parsing_path,sub)):
                label = int(lbl[-6:-4])     # 0~ 10
                img = Image.open(pjoin(parsing_path,sub,lbl))
                img_array = np.array(img)
                result[img_array>150] = label
            # result = result - 1
            lbl_new = m.toimage(result,high=result.max(),low=result.min())
            temp = np.array(lbl_new)
            print(img_name)
            m.imsave(pjoin('/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original','Segmentation',img_name),lbl_new)
        
        # for img in os.listdir(pjoin('/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original','Segmentation')):
        #     img_path = pjoin('/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original','Segmentation',img)
        #     lbl = self.encode_segmap(m.imread(img_path))
        #     lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
        #     print('xixixixixi',img)
        #     m.imsave(pjoin(target_path,img), lbl)
            
if __name__=='__main__':
    h = HelenLoader()
    # h.setup_annotation()
    labelmask = np.array(Image.open('/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/Segmentation/2268738156_1.png'))
    h.decode_segmap(labelmask,plot=True)
    
    
    