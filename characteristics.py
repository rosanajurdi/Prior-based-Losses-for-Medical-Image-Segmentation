#!/usr/bin/env python3.6
from skimage.measure import label, regionprops

print('hi')

import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
import csv

sys.path.append('/home/eljurros/spare-workplace/surface-loss-master')
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, probs2one_hot, class2one_hot, numpy_haussdorf
root = '/media/eljurros/Transcend/Decathlone/Task02_Heart/nifty/FOLD_4/npy/val'
#print(net)
fieldnames = ['SLICE_ID', 'dice','haus',  'c_error']
folder_path = Path(root, 'characteristics')
n_classes = 2 
folder_path.mkdir(parents=True, exist_ok=True)
fold_clean_H1 = open(os.path.join(folder_path, 'characteristics.csv'), "w")
fold_clean_H1.write(f"file, size, cc, \n")


for _,_,files in os.walk(os.path.join(root, 'gt_npy')): 

    print('walking into', os.path.join(root, 'gt_npy'))
    for file in files: 
        #print(file)
        image = np.load(os.path.join(root,'in_npy', file))
        gt = np.load(os.path.join(root,'gt_npy', file))
           
        gt_label = len(np.unique(label(gt))) 
        gt_label = len(np.unique(label(class2one_hot(torch.tensor(gt), n_classes)[0][1])))
        
        print(f"{file}, {np.float(gt_label)}, {np.float(gt_label)},{np.float(gt_label)} \n")

        fold_clean_H1.write(f"{file}, {np.float(gt_label)}, {np.float(gt_label)},{np.float(gt_label)} \n")
        fold_clean_H1.flush()
            
        
  
        

        
        
        
        

