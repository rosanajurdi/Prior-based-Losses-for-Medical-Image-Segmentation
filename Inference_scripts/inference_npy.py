#!/usr/bin/env python3.6
'''
Script to compute metrics : Dice accuracy, hausdorf distance and the error on the number of connected components.
'''
import sys

from skimage.measure import label, regionprops
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
import csv

from Label_Estimate_Helper_Functions import Get_contour_characteristics

from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, probs2one_hot, class2one_hot, numpy_haussdorf
#root = '/home/2017011/reljur01/surface-loss-master/data/Prostate/FOLD_2/npy/val'
#net_path = '/home/2017011/reljur01/surface-loss-master/data/Prostate/FOLD_2/results/clDice_with_coord/best2.pkl'
root='/media/eljurros/Transcend/CoordConv/ACDC/ACDC/FOLD_1/npy/val'
net_path = '/media/eljurros/Transcend/Decathlone/ACDC/FOLD_1/size/best2.pkl'

net = torch.load(net_path, map_location=torch.device('cpu'))
n_classes = 4
n = 3

fieldnames = ['SLICE_ID', 'dice','haus',  'c_error']

#assert os.path.exists(os.path.join(net_path.split(os.path.basename(net_path))[0], 'predictions'))== False

exp_path = net_path.split('/best2.pkl')[0]. #Include the name of the checkpoint you want to use 
name =os.path.basename(exp_path)
folder_path = Path(exp_path, 'CSV_RESULTS')

#create the directory you want your result text files to be stored in 
folder_path.mkdir(parents=True, exist_ok=True)
file_path = os.path.join(exp_path, name)
fold_clean_H1 = open(os.path.join(folder_path, '{}_{}_clean.csv'.format(name,n)), "w")
fold_all_H1 = open(os.path.join(folder_path, '{}_{}_all.csv'.format(name,n)), "w")
 

fold_all_H1.write(f"file, dice, haussdorf,connecterror \n")
fold_clean_H1.write(f"file, dice, haussdorf,connecterror \n")

 
path=os.path.join(net_path.split(os.path.basename(net_path))[0])

pred_path = Path(path,'predictions')
gt_path = Path(path,'gt')

pred_path.mkdir(parents=True, exist_ok=True)
gt_path.mkdir(parents=True, exist_ok=True)

for _,_,files in os.walk(os.path.join(root, 'in_npy')): 

    print('walking into', os.path.join(root, 'in_npy'))
    for file in files: 
        print(file)
        image = np.load(os.path.join(root,'in_npy', file))
        gt = np.load(os.path.join(root,'gt_npy', file))        
        if len(np.unique(gt)) >0:
            #print('infering {} of shape {} and classes {}, max {} and min {} '.format( file, image.shape, np.unique(gt), image.max(), image.min()))
            image = image.reshape(-1, 1, 256, 256)/255.00
            image = torch.tensor(image, dtype=torch.float)
            image = Variable(image, requires_grad=True)
            pred = net(image)
            pred = F.softmax(pred, dim=1).to('cpu')
            predicted_output = probs2one_hot(pred.detach())
            #print(predicted_output.to('cpu')[:,:2:].shape,class2one_hot(torch.tensor(gt).to('cpu'), n_classes).shape )
            #np.save(os.path.join(path, 'predictions', '{}'.format(file)), pred.to('cpu').detach().numpy())
            #dice = dice_coef(predicted_output.to('cpu'), class2one_hot(torch.tensor(gt).to('cpu'), n_classes))[:,n,]
            dice = dice_coef(predicted_output.to('cpu'), class2one_hot(torch.tensor(gt).to('cpu'), n_classes))[:,n,]

            hauss = haussdorf(predicted_output, class2one_hot(torch.tensor(gt), n_classes))[:,n,]
            '''
            fig, ax = plt.subplots()
            ax.imshow(np.argmax(predicted_output.detach().numpy(), axis=1)[0], cmap=plt.cm.gray)
            #r, contours= Get_contour_characteristics(np.argmax(predicted_output.detach().numpy(), axis=1)[0])
            g, contours = Get_contour_characteristics(np.array(gt).round())
            total_summ = 0
            
            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1].astype(int), contour[:, 0].astype(int),color='red', linewidth=-1)

                ax.axis('image')
                ax.set_xticks([])
                ax.set_yticks([])
                #plt.show()
            
            plt.savefig(os.path.join(path, 'predictions', '{}.png'.format(file.split('.npy')[0])))
            #plt.imsave(os.path.join(path, 'predictions', '{}.png'.format(file.split('.npy')[0])), np.argmax(predicted_output,1)[0]) 
            #plt.imsave(os.path.join(path, 'gt', '{}.png'.format(file.split('.npy')[0])), gt)            
            '''
            gt_label = len(np.unique(label((gt==n)*1.00)))
            pred_label = len(np.unique(label(predicted_output[:,n:][0][0])))
            gt_label = len(np.unique(label(class2one_hot(torch.tensor(gt), n_classes)[0][n])))
            error = np.abs(pred_label - gt_label)

            '''
            print(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")

            fold_all_H1.write(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")

            if len(np.unique(gt)) == 2:
                fold_clean_H1.write(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")
                fold_clean_H1.flush()
            #folders.write("hi")
            fold_all_H1.flush()
            '''
            print(f"{file}, {np.float(dice[0])}, {np.float(hauss[0])},{np.float(error)} \n")

            fold_all_H1.write(f"{file}, {np.float(dice[0])}, {np.float(hauss[0])},{np.float(error)} \n")

            if len(np.unique(gt)) != 1:
                fold_clean_H1.write(f"{file}, {np.float(dice[0])}, {np.float(hauss[0])},{np.float(error)} \n")
                fold_clean_H1.flush()
            #folders.write("hi")
            fold_all_H1.flush()
        
  
        

        
        
        
        


