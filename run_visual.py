#!/usr/bin/env python3.6
from skimage.measure import label, regionprops

print('hi')

import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm

import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
import csv
import sys
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet_Functions')

sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Multi_Organ_Seg')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Common_Scripts')
from Label_Estimate_Helper_Functions import Get_contour_characteristics
sys.path.append('/home/eljurros/spare-workplace/surface-loss-master')
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, probs2one_hot, class2one_hot, numpy_haussdorf
#root = '/home/2017011/reljur01/surface-loss-master/data/Prostate/FOLD_2/npy/val'
#net_path = '/home/2017011/reljur01/surface-loss-master/data/Prostate/FOLD_2/results/clDice_with_coord/best2.pkl'
root='/media/eljurros/Transcend/Decathlone/Task04_Hippocampus/FOLD_2/npy/val'
net_path = '/media/eljurros/Transcend/Decathlone/Hippu_Cont/FOLD_2/results/HDDT/best2.pkl'

net = torch.load(net_path, map_location=torch.device('cpu'))
#print(net)
fieldnames = ['SLICE_ID', 'dice','haus',  'c_error']
n_classes = 4
n = 1
#assert os.path.exists(os.path.join(net_path.split(os.path.basename(net_path))[0], 'predictions'))== False
print('started this stupid work')
exp_path = net_path.split('/best2.pkl')[0]
name =os.path.basename(exp_path)
folder_path = Path(exp_path, 'CSV_RESULTS')

folder_path.mkdir(parents=True, exist_ok=True)
file_path = os.path.join(exp_path, name)


 
path=os.path.join(net_path.split(os.path.basename(net_path))[0])

pred_path = Path(path,'predictions')
gt_path = Path(path,'gt')

pred_path.mkdir(parents=True, exist_ok=True)
gt_path.mkdir(parents=True, exist_ok=True)
counter = 100
for _,_,files in os.walk(os.path.join(root, 'in_npy')): 
    while counter > 0:
        print('walking into', os.path.join(root, 'in_npy'))
        for file in files: 
            print(file)
            image = np.load(os.path.join(root,'in_npy', file))
            gt = np.load(os.path.join(root,'gt_npy', file))        
            if 1 in np.unique(gt):
                #print('infering {} of shape {} and classes {}, max {} and min {} '.format( file, image.shape, np.unique(gt), image.max(), image.min()))
                image = image.reshape(-1, 1, 256, 256)
                image = torch.tensor(image, dtype=torch.float)
                image = Variable(image, requires_grad=True)
                pred = net(image)
                pred = F.softmax(pred, dim=1).to('cpu')
                #predicted_output = probs2one_hot(pred.detach())
                predicted_output = np.argmax(pred.detach().numpy(), axis = 1)[0]
                fig, ax = plt.subplots()
                ax.imshow(np.transpose(image.squeeze().detach().numpy()), cmap = cm.gray)
                color_list = ['#009ACD','#008080', 'purple']
                gt_color_list = ['#FF1493', '#00FF00']
                for id in range(1,3):
                    new_gt = (gt == id)*1.00 
                    new_predicted = (predicted_output == id)*1.00
                    #g, contours= Get_contour_characteristics(predicted_output[0][id].detach().numpy())
                    #g, contours = Get_contour_characteristics(np.array(new_gt).round())
                    g, gt_contours= Get_contour_characteristics(np.transpose(new_gt))
                    
                    # '#009ACD' -----RVC
                    #'#00FF00' ----- spleen
                    for n, contour in enumerate(gt_contours):
                        ax.fill(contour[:, 1].astype(int), contour[:, 0].astype(int),color=gt_color_list[id-1], linewidth=5)
        
                        ax.axis('image')
                        ax.set_xticks([])
                        ax.set_yticks([])
                    total_summ = 0
                    g, contours= Get_contour_characteristics(np.transpose(new_predicted))
                    for n, contour in enumerate(contours):
                        ax.plot(contour[:, 1].astype(int), contour[:, 0].astype(int),color=color_list[id -1], linewidth=5)
        
                        ax.axis('image')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        #plt.show()
                    
                plt.savefig(os.path.join(path, 'predictions', '{}.png'.format(file.split('.npy')[0])))
                    #plt.imsave(os.path.join(path, 'predictions', '{}.png'.format(file.split('.npy')[0])), np.argmax(predicted_output,1)[0]) 
                    #plt.imsave(os.path.join(path, 'gt', '{}.png'.format(file.split('.npy')[0])), gt)  
                counter = counter - 1
                print(counter)          
                
 

            


  
        

        
        
        
        


