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
root = '/media/eljurros/Transcend/Decathlone/benchmark/isles/FOLD_1'
net_path = '/media/eljurros/Transcend/Decathlone/benchmark/isles/FOLD_1/results/eroded_distance/best2.pkl'
net = torch.load(net_path, map_location=torch.device('cpu'))
#print(net)
fieldnames = ['SLICE_ID', 'dice','haus',  'c_error']
n_classes = 4
#assert os.path.exists(os.path.join(net_path.split(os.path.basename(net_path))[0], 'predictions'))== False
print('started this stupid work')
exp_path = net_path.split('/best2.pkl')[0]
name =os.path.basename(exp_path)
folder_path = Path(exp_path, 'CSV_RESULTS')

folder_path.mkdir(parents=True, exist_ok=True)
file_path = os.path.join(exp_path, name)
fold_clean_H1 = open(os.path.join(folder_path, '{}_clean.csv'.format(name)), "w")
fold_all_H1 = open(os.path.join(folder_path, '{}_all.csv'.format(name)), "w")
 

fold_all_H1.write(f"file, dice, haussdorf,connecterror \n")
fold_clean_H1.write(f"file, dice, haussdorf,connecterror \n")

 
path=os.path.join(net_path.split(os.path.basename(net_path))[0])

pred_path = Path(path,'predictions')
gt_path = Path(path,'gt')

pred_path.mkdir(parents=True, exist_ok=True)
gt_path.mkdir(parents=True, exist_ok=True)

for _,_,files in os.walk(os.path.join(root, 'img')): 

    print('walking into', os.path.join(root, 'img'))
    for file in files: 
        print(file)
        image = np.array(Image.open(os.path.join(root,'img', file)))
        gt = np.array(Image.open(os.path.join(root,'gt', file)))        
        if len(np.unique(gt)) >0:
            #print('infering {} of shape {} and classes {}, max {} and min {} '.format( file, image.shape, np.unique(gt), image.max(), image.min()))
            image = image.reshape(-1, 5, 256, 256)/255.00
            image = torch.tensor(image, dtype=torch.float)
            image = Variable(image, requires_grad=True)
            pred = net(image)
            pred = F.softmax(pred, dim=1).to('cpu')
            predicted_output = probs2one_hot(pred.detach())
            np.save(os.path.join(path, 'predictions', '{}'.format(file)), pred.to('cpu').detach().numpy())
            dice = dice_coef(predicted_output.to('cpu'), class2one_hot(torch.tensor(gt).to('cpu'), n_classes))
            hauss = haussdorf(predicted_output,class2one_hot(torch.tensor(gt).to('cpu'), n_classes))
            plt.imsave(os.path.join(path, 'predictions', '{}.png'.format(file.split('.npy')[0])), np.argmax(predicted_output,1)[0]) 
            plt.imsave(os.path.join(path, 'gt', '{}.png'.format(file.split('.npy')[0])), gt)            
           
            pred_label = len(np.unique(label(np.array(pred.argmax(axis = 1).detach().numpy()))))
            gt_label = len(np.unique(label(gt)))
            pred_label = len(np.unique(label(predicted_output[0][1])))
            gt_label = len(np.unique(label(class2one_hot(torch.tensor(gt), n_classes)[0][2])))
            error = np.abs(pred_label - gt_label)

            
            print(f"{file}, {np.float(dice[0][2])}, {np.float(hauss[0][2])},{np.float(error)} \n")

            fold_all_H1.write(f"{file}, {np.float(dice[0][2])}, {np.float(hauss[0][2])},{np.float(error)} \n")

            if len(np.unique(gt)) == 2:
                fold_clean_H1.write(f"{file}, {np.float(dice[0][2])}, {np.float(hauss[0][2])},{np.float(error)} \n")
                fold_clean_H1.flush()
            #folders.write("hi")
            fold_all_H1.flush()
            
        
  
        

        
        
        
        

