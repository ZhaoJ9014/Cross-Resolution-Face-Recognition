import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .misc import *
from helen_loader import HelenLoader as helenloader
from PIL import Image
from helen_loader import HelenLoader

import os
import scipy.misc as m
import pdb

# __all__ = ['make_image', 'show_batch', 'show_mask', 'show_mask_single']

def save_image(coarse_image=None,image=None,landmark=None,parsing=None,epoch=0,if_train=True,count=0):
    root = './results/'
    if if_train:
        flag = 'train_epoch_'
    else:
        flag = 'test_epoch_'
    img_name = flag + str(epoch) +'_'+str(count)+ '_HR_Image.jpg'
    parsing_name = flag + str(epoch) +'_'+str(count)+ '_Parsing_Map.jpg'
    landmark_name = flag + str(epoch) + '_' +str(count) + '_Landmark.jpg'
    coarse_name = flag+str(epoch) + '_' + str(count) + '_Coarse.jpg'
    # image.convert('RGB').save(os.path.join(root,))
    # landmark = np.array(landmark).reshape(-1,2)
    if image is None:
        pass
    else:
        image = m.toimage(image)
        image.save(os.path.join(root,img_name))
    if coarse_image is None:
        pass
    else:
        coarse = m.toimage(coarse_image)
        coarse.save(os.path.join(root,coarse_name))
    #fig1 = plt.figure()
    # pdb.set_trace()
    #plt.imshow(image)
    #plt.scatter(landmark[:,0],landmark[:,1],s=5,c='r')
    #plt.savefig(os.path.join(root,landmark_name))
    # plt.imshow(image)
    # pdb.set_trace()
    # landmark = m.toimage(landmark[0])
    # landmark.save(os.path.join(root,landmark_name))
    if landmark is None:
        pass
    else:
        h = HelenLoader()
        # landmark =  
        hm = np.sum(landmark,axis = 0) 
        plt.imshow(hm)
        plt.savefig(os.path.join(root,landmark_name))
    if parsing is None:
        pass
    else:
        h = helenloader()
        parsing_map = h.decode_segmap(parsing)
        # pdb.set_trace()
        plt.imsave(os.path.join(root,parsing_name),parsing_map)
    
    
# functions to show an image
def make_image(img, mean=(0,0,0), std=(1,1,1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.imshow(images)
    plt.show()


def show_mask_single(images, mask, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(2, 1, 1)
    plt.imshow(images)
    plt.axis('off')

    # for b in range(mask.size(0)):
    #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
    mask_size = mask.size(2)
    # print('Max %f Min %f' % (mask.max(), mask.min()))
    mask = (upsampling(mask, scale_factor=im_size/mask_size))
    # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
    # for c in range(3):
    #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

    # print(mask.size())
    mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
    # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.axis('off')

def show_mask(images, masklist, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(1+len(masklist), 1, 1)
    plt.imshow(images)
    plt.axis('off')

    for i in range(len(masklist)):
        mask = masklist[i].data.cpu()
        # for b in range(mask.size(0)):
        #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
        mask_size = mask.size(2)
        # print('Max %f Min %f' % (mask.max(), mask.min()))
        mask = (upsampling(mask, scale_factor=im_size/mask_size))
        # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
        # for c in range(3):
        #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

        # print(mask.size())
        mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
        # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
        plt.subplot(1+len(masklist), 1, i+2)
        plt.imshow(mask)
        plt.axis('off')



# x = torch.zeros(1, 3, 3)
# out = colorize(x)
# out_im = make_image(out)
# plt.imshow(out_im)
# plt.show()
