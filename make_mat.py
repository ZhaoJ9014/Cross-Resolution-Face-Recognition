import scipy.io as io
import os
import pandas as pd
import numpy as np
import pdb


df = pd.read_csv('/home/hyo/xixixi.csv')
data = df.values[:,1:]

# /media/hyo/文档/Dataset/Helen_aligned/test/image.jpg
# /media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/labels_aligned/image_name/image_name_lblxx.png

root = '/media/hyo/文档/Dataset/'
parsing_root = '/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/labels_aligned/'
dir_list = ['Helen_aligned','Helen_aligned_28']
img_pair = []

# 构建HR image和LR image图片路径对
subfolder = os.listdir(os.path.join(root,'Helen_aligned_28'))      # test/train
for sub in subfolder:
    img_list = os.listdir(os.path.join(root, 'Helen_aligned_28', sub))
    for img in img_list:
        img_pair.append((os.path.join(root,'Helen_aligned',sub,img) , os.path.join(root,'Helen_aligned_28',sub,img)))
        
# 构建landmark的array
landmarks = data[:,1:]

for landmark in landmarks:
    landmark.reshape(-1,2)

# 总图片名称列表：
img_list = os.listdir(parsing_root)

for pair in img_pair:
    img_name = pair[0].split('/')[-1]
    img_root = pair[0]
    
    # 存储landmark
    
    



