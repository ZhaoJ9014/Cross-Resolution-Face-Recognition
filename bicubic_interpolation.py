import cv2
import numpy as np
import math
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from skimage import data,filters
from skimage.morphology import disk
from PIL import ImageFilter

def S(x):
    x = np.abs(x)
    if 0<=x<1:
        return 1-2*x*x+x**3
    elif 1<=x<2:
        return 4-8*x+5*x**2-x**3
    else:
        return 0


def function(img, m, n):
    height, width, channels = img.shape
    emptyImage = np.zeros((m, n, channels), np.uint8)
    # emptyImage = torch.zeros((m,n,channels))
    sh = m / height
    sw = n / width
    for i in range(m):
        for j in range(n):
            x = i / sh
            y = j / sw
            p = (i + 0.0) / sh - x
            q = (j + 0.0) / sw - y
            # x = int(x) - 2
            # y = int(y) - 2
            x = math.floor(x)-2
            y = math.floor(y)-2
            A = np.array([
                [S(1 + p), S(p), S(1 - p), S(2 - p)]
            ])
            if x >= m - 3:
                m - 1
            if y >= n - 3:
                n - 1
            if x >= 1 and x <= (m - 3) and y >= 1 and y <= (n - 3):
                B = np.array([
                    [img[x - 1, y - 1], img[x - 1, y],
                     img[x - 1, y + 1],
                     img[x - 1, y + 1]],
                    [img[x, y - 1], img[x, y],
                     img[x, y + 1], img[x, y + 2]],
                    [img[x + 1, y - 1], img[x + 1, y],
                     img[x + 1, y + 1], img[x + 1, y + 2]],
                    [img[x + 2, y - 1], img[x + 2, y],
                     img[x + 2, y + 1], img[x + 2, y + 1]],
                
                ])
                C = np.array([
                    [S(1 + q)],
                    [S(q)],
                    [S(1 - q)],
                    [S(2 - q)]
                ])
                blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]
                green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
                red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]
                
                # ajust the value to be in [0,255]
                # def adjust(value):
                #     if value > 255:
                #         value = 255
                #     elif value < 0:
                #         value = 0
                #     return value
                #
                # blue = adjust(blue)
                # green = adjust(green)
                # red = adjust(red)
                
                emptyImage[i, j] = np.array([blue, green, red], dtype=np.uint8)
    
    return emptyImage




def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2.*(x**2)+(x**3)
    elif x<2:
        return 4-8.*x+5*(x**2)-(x**3)
    else:
        return 0.

def BiCubic_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg

def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    return retimg

def NN_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=round((i+1)*(scrH/dstH))
            scry=round((j+1)*(scrW/dstW))
            retimg[i,j]=img[scrx-1,scry-1]
    return retimg
# img = cv2.imread("/media/hyo/文档/Dataset/Helen_aligned_28/train_1/13601661_1.jpg")
# zoom = function(img, 224, 224)
# cv2.imshow("cubic", zoom)
# cv2.imshow("image", img)
# cv2.waitKey(0)


# img = "/media/hyo/文档/Dataset/Helen_aligned_28/train_1/117634057_1.jpg"
# image=np.array(Image.open(img))
# # image = Image.open(img)
# image3=BiCubic_interpolation(image,image.shape[0]*8,image.shape[1]*8)
# # image3 = BiCubic_interpolation(image,image3.shape[0]*4,image3.shape[1]*4)
# # image3 = interpolate.interp2d(image3,)
# # image3 = filters.median(np.array(image3),disk(5))
# # image3 = image.resize((224,224),Image.ANTIALIAS)
# # image3 = np.array(image3)
# image3=Image.fromarray(image3.astype('uint8')).convert('RGB')
# # image3 = image3.filter((ImageFilter.GaussianBlur(radius=3)))
# # plt.savefig(image3,)
# image3.save('/home/hyo/temp1.jpg')
# plt.imshow(image3,interpolation = 'bicubic')
# plt.show()
# cv2.waitKey(0)


def rotate_img_name(img_name,angle,if_flip=False):
    temp = img_name.split('.')
    angle = str(angle)
    if not if_flip:
        new_img_name = temp[0]+'_'+angle+'.jpg'
    else:
        new_img_name = temp[0]+'_'+angle+'_r.jpg'
    return new_img_name
    
    
root = '/media/hyo/文档/Dataset/Helen_aligned_28_renew'
dir_list = os.listdir(root)
for dir in dir_list:
    image_dir = os.path.join(root,dir)
    image_list = os.listdir(image_dir)
    for image in image_list:
        img_loc = os.path.join(image_dir,image)

        img = Image.open(img_loc)
        # zoom = BiCubic_interpolation(img, 224, 224)
        zoom = img.resize((224,224),Image.BICUBIC)
        # zoom1 = Image.fromarray(zoom.astype('uint8')).convert('RGB')
        # zoom1 = zoom1.filter((ImageFilter.GaussianBlur(radius=2.7)))
        zoom.save(img_loc)
        print(img_loc)
#
#         img = cv2.imread(img_loc)
#         zoom = function(img,224,224)
#         cv2.imshow("",zoom)
#         cv2.waitKey(0)
#         print(img_loc)