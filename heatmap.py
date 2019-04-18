import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import pandas as pd
import os
import cv2

import pdb

def gaussian_k(x0,y0,sigma,width,height):
    """
    Make a square gaussian kernel centered at (x0,y0) with sigma
    :param x0:
    :param y0:
    :param sigma:
    :param width:
    :param height:
    :return: Square gaussian kernel
    """
    x = np.arange(0,width,1,float)
    y = np.arange(0,height,1,float)[:,np.newaxis]
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))


def generate_hm(height, width, landmarks, s=1.2):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
    """
    # pdb.set_trace()
    Nlandmarks = len(landmarks)
    # hm = np.zeros((Nlandmarks,height, width), dtype=np.float32)
    hm = np.zeros((height, width), dtype=np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1, -1]):
            # hm[i,:, :] = gaussian_k(landmarks[i][0],
            #                          landmarks[i][1],
            #                          s, height, width)
            hm += gaussian_k(landmarks[i][0],landmarks[i][1],s,height,width)
        else:
            hm[:, :, i] = np.zeros((height, width))
    return hm

if __name__=='__main__':
    df = pd.read_csv('/home/hyo/xixixi_v2.csv')
    idd = '10405146_1.jpg'
    landmark = df[df['img_name'] == idd].values[0][2:].reshape(-1,2)/2.0
    landmark1 = []
    n = np.shape(landmark)[0]
    # for i in range(n):
    #     if i%2==0:
    #         landmark1.append(landmark[i,:])
    hm = generate_hm(112,112,landmark,s=1.0)
    # plt.imshow(np.sum(hm,axis=0))
    plt.imshow(hm)
    plt.show()
    

