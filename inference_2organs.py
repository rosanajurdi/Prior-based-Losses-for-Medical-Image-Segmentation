#!/usr/bin/env python3.6
from skimage.measure import label, regionprops



import torch.nn.functional as F

import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
import csv

sys.path.append('/home/eljurros/spare-workplace/surface-loss-master')
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, probs2one_hot, class2one_hot
root = '/media/eljurros/Transcend/Decathlone/Task04_Hippocampus/FOLD_3/npy/val'
net_path = '/media/eljurros/Transcend/Decathlone/Hippu_Cont/FOLD_3/results/clDice/best2.pkl'
path=os.path.join(net_path.split(os.path.basename(net_path))[0])
#assert os.path.exists(os.path.join(net_path.split(os.path.basename(net_path))[0], 'predictions'))== False
#os.mkdir(os.path.join(net_path.split(os.path.basename(net_path))[0], 'predictions'))
net = torch.load(net_path, map_location=torch.device('cpu'))
fieldnames = ['SLICE_ID', 'dice','haus',  'c_error']
n_classes = 3

exp_path = net_path.split('/best2.pkl')[0]
name =os.path.basename(exp_path)
file_path = os.path.join(exp_path, name)
fold_clean_H1 = open(os.path.join(exp_path, '{}H1_clean.csv'.format(name)), "w")
fold_all_H1 = open(os.path.join(exp_path, '{}H1_all.csv'.format(name)), "w")
 

fold_all_H1.write(f"file, dice, haussdorf,connecterror \n")
fold_clean_H1.write(f"file, dice, haussdorf,connecterror \n")

fold_clean_H2 = open(os.path.join(exp_path, '{}H2_clean.csv'.format(name)), "w")
fold_all_H2 = open(os.path.join(exp_path, '{}H2_all.csv'.format(name)), "w")
 

fold_all_H2.write(f"file, dice, haussdorf,connecterror \n")
fold_clean_H2.write(f"file, dice, haussdorf,connecterror \n")


for _,_,files in os.walk(os.path.join(root, 'in_npy')): 
    for file in files: 
        image = np.load(os.path.join(root,'in_npy', file))
        gt = np.load(os.path.join(root,'gt_npy', file))
        
        #print('infering {} of shape {} and classes {}, max {} and min {} '.format( file, image.shape, np.unique(gt), image.max(), image.min()))
        image = image.reshape(-1, 1, 256, 256)
        image = torch.tensor(image, dtype=torch.float)
        image = Variable(image, requires_grad=True)
        pred = net(image)
        pred = F.softmax(pred, dim=1)
        predicted_output = probs2one_hot(pred.detach())
        #np.save(os.path.join(path, 'predictions', '{}'.format(file)), pred.detach().numpy())
        dice = dice_coef(predicted_output, class2one_hot(torch.tensor(gt), n_classes))
        hauss = haussdorf(predicted_output, class2one_hot(torch.tensor(gt), n_classes))
        
        #pred_label = len(np.unique(label(np.array(pred.argmax(axis = 1).detach().numpy()))))
        #gt_label = len(np.unique(label(gt)))
        pred_label = len(np.unique(label(predicted_output[0][1])))
        gt_label = len(np.unique(label(class2one_hot(torch.tensor(gt), n_classes)[0][1])))
        error = np.abs(pred_label - gt_label)
        pred_label2 = len(np.unique(label(predicted_output[0][2])))
        gt_label2 = len(np.unique(label(class2one_hot(torch.tensor(gt), n_classes)[0][2])))
        error2 = np.abs(pred_label2 - gt_label2)
        
        print(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")
        print(f"{file}, {np.float(dice[0][2])}, {np.float(hauss[0][2])},{np.float(error2)} \n")

        fold_all_H1.write(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")
        fold_all_H2.write(f"{file}, {np.float(dice[0][2])}, {np.float(hauss[0][2])},{np.float(error2)} \n")

        if len(np.unique(gt)) == 2:
            fold_clean_H1.write(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")
            fold_clean_H1.flush()
            fold_clean_H2.write(f"{file}, {np.float(dice[0][2])}, {np.float(hauss[0][2])},{np.float(error2)} \n")
            fold_clean_H2.flush()

        fold_all_H1.flush()
        fold_all_H2.flush()
        
        
        
        
        

        
        
        
        


