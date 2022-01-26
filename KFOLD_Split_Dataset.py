'''
Created on Feb 13, 2020

@author: eljurros
'''
'''
Created on Mar 20, 2019

@author: eljurros
'''
from DataSEt_Classes import WoodScapeDataSet
from Label_Estimate_Helper_Functions import extract_bbox, rect_mask, get__mask, Get_Upper_Lower_boundaries
from torchvision import transforms
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import imageio
import h5py
import sys
import os
import shutil
import random
sys.path.append('../medicaltorch-0.2')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet_Functions')

typ = 'ROOT'
root_path = '/media/eljurros/Crucial X6/Synthetic_Data'
fold = 'FOLD_1'
if os.path.exists(os.path.join(root_path, fold)) is False:
    os.mkdir(os.path.join(root_path,  fold))
    os.mkdir(os.path.join(root_path,   fold, 'train'))
    os.mkdir(os.path.join(root_path,   fold, 'val'))
inner_arr = []
outer_arr = []
ds = WoodScapeDataSet(root_dir=root_path, typ=typ)
train_path = [os.path.join(root_path,fold,'train', 'rgb_images'), os.path.join(root_path, fold,'train', 'gtLabels')]
val_path = [os.path.join(root_path, fold,'val', 'rgb_images'), os.path.join(root_path, fold,'val', 'gtLabels')]
if os.path.exists(train_path[0]) is False:
    os.mkdir(train_path[0])
    os.mkdir(train_path[1])

nb_val = 400
if os.path.exists(val_path[0]) is False:
    os.mkdir(val_path[0])
    os.mkdir(val_path[1])


for i, patient_path in enumerate(ds.filename_pairs):
    patient_name = os.path.basename(patient_path[0])
    input_filename, gt_filename = patient_path[0], \
                                  patient_path[1]
    
    

    i = random.randint(1,101)
        
    if i <50 and nb_val >0:
        nb_val = nb_val - 1
        
        shutil.copy(patient_path[0], val_path[0])
        shutil.copy(patient_path[1], val_path[1])
        with open(os.path.join(os.path.join(root_path,fold,'val.txt')), 'a') as the_file:
            the_file.write(patient_name)
            the_file.write('\n')

    else:
        shutil.copy(patient_path[0], train_path[0])
        shutil.copy(patient_path[1], train_path[1])
        with open(os.path.join(os.path.join(root_path,fold,'train.txt')), 'a') as the_file:
            the_file.write(patient_name)
            the_file.write('\n')

    
        







        
