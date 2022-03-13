  
This repository contains the code for the paper: **Effect of Prior-based Losses on Segmentation Performance: A Benchmark** and **A Surprisingly Effective Perimeter-based Loss for Medical Image Segmentation.**  
  
  
The benchmark of establishes performance of four recent prior-based losses for across 8 different medical datasets of various tasks and modalities. The main objective is to provide intuition onto which losses to choose given a particular task or dataset, based on dataset characteristics and properties. The proposed benchmark is conducted via a unified segmentation network and learning environment chich can be found (https://github.com/LIVIAETS/boundary-loss).   
   
The perimeter loss paper (https://openreview.net/forum?id=NDEmtyb4cXu) can be found in losses.py script.   
   
# Installation and Dependencies  
  
For installation and dependencies, please check this [repository](https://github.com/LIVIAETS/boundary-loss). The code for the nechmark in this repository is an extension of https://github.com/LIVIAETS/boundary-loss with additinal scripts, functions and Modalities.  
In addition to the requirements in [repository](https://github.com/LIVIAETS/boundary-loss), you are also required to install :  
- pillow (for visualization)  
- tqdm  

# Difference from the orginal framework in [repository](https://github.com/LIVIAETS/boundary-loss):  
- addition of the Decathlon class(Dataset section).   
- addition of the different losses (in losses.py script)   
- addition of an inference script to compute prior metrics (inference_npy.py connected component error)   
  
  
# Datasets   
The datasets explored are from a variety of medical image segmentation challenges including the [Decathlon](http://medicaldecathlon.com), the  [ISLES](http://www.isles-challenge.org) and the [WMH](https://wmh.isi.uu.nl) challenges. The data format from the Decathlon is in niffty format.;   
  
## Preparing the dataset for the code  
### Splitting into train and validation . 

Download the required data (nifty format) from the [Decathlone challenge](http://medicaldecathlon.com)  
   - the dataset will be in nifty format in 3 folders : (imagesTr, labelsTr, imagesTs)  
   - In this benchmark, we include results on validation datasets i.e we split the (imagesTr, labelsTr) into train and validation and benchmark results on the validation set   
   - Validation was conducted via 3 monte carlo simulations.  
   - Place the (imagesTr, labelsTr) in a folder under the name ../nifty/ROOT  
2. Run the [KFOLD_split_dataset.py](https://github.com/rosanajurdi/DataSET_module) with the proper dataset class (See documentation).The script will create the required fold_K: (train, val) and their corresponding text files (Make sure to specify )   
### From nifty to numpy: 

Conversion from nifty to numpy can be done via [slice_decathlone.py(with retain=0)](https://github.com/rosanajurdi/Prior-based-Losses-for-Medical-Image-Segmentation/blob/master/slice_decathlon.py) to transform the data from nifty to .npy format:  
Variables to initialize: \\  
- source_dir: the path to your split data (obtained by running the KFOLD_split_dataset.py script )  
- dest_dir: the path where you want to store the npy converted data. (usually ../npy)  
#### Note: - for decathlon datasets of two organs, please check [slice_decathlon_2organs.py](https://github.com/rosanajurdi/Prior-based-Losses-for-Medical-Image-Segmentation/blob/master/slice_decathlon_2organs.py)  
- for the ISLES and WMH datasets, please refer to the original [repository](https://github.com/LIVIAETS/boundary-loss).  
- make sure to specify the number of samples in the validation set. In the paper,  splits were conducted according to 80 % training 20 % testing.   
- make sure to specify the proper dataset class corresponding to the name of dataset.  
- There is a slicing script for each dataset : Decathlon(single), Decthlon(multi), isles, wmh, Prostate, and acdc.  
                         Example: `ds = Decathlon(root_dir=root_path, typ=typ)` # Training the network across the different losses and datasets Training can be done either for single organ or multi organ datasets and across either the dice loss or Dice loss + prior.   
To train single organ segmentation with only the dice loss:  
  
## Variables to initialize:  
  
parser = argparse.ArgumentParser(description='Hyperparams')  
- dataset: path to the numpy folder containing your dataset.  
- csv : metrics file nama you want to store the loss evolution in.   
- workdir: where to store the results.   
- batch_size : set yp to 8 in our experimetns  
- n_epoch: SET TO 200 in our experiments.  
- metric_axis: for single organ segmentation set to [1], multi-organ set to [1,2, ...(number of organs to be segmented)]  
- losses": List of (loss_name, loss_params, bounds_name, bounds_params, fn, weight):  
- scheduler: if the network is to be trained with dice loss, set to Dummy else, set to StealWeight.  
- scheduler_params: if network trained in conjuntion of losses set to 'to_steal': 0.01, else,   
  
### Example Scripts:  
 **single organ segmentation with two losses:**  
```  
losses:  [('GeneralizedDice', {'idc': [0, 1]}, None, None, None, 1), ('contour_loss',{'idc': [0, 1]}, None, None, None, 0.01)]") 
folders: [('in_npy', torch.tensor, False), ('gt_npy', gt_transform, True)]+[('gt_npy', gt_transform, True), \ ('gt_npy', gt_transform, True)]
scheduler: "StealWeight" 
scheduler_params: "{'to_steal': 0.01}"
n_class: 2
 ```  
**Multi-organ with one loss:**
``` 
n_class:3  
metric_axis: [1,2]
losses: [('GeneralizedDice', {'idc': [1]}, None, None, None, 1)]
folders: [('in_npy', torch.tensor, False), ('gt_npy', gt_transform, True)]+[('gt_npy', gt_transform, True)]  
scheduler: DummyScheduler
scheduler_params: {}
``` 
 
 
  
  
## Inference:   
### the output of the training script: 

After you train your networks, you will have:
- metrics.py script which contain the loss evolution as well as the generic accuracy metrics such as the Dice accuracy (as specified by the script)   
- best.pkl  which is the model you have trained;   
- Best epoch folder that contains the image predictions corresponding to the best model saved.

Aside from these results you can also run [inference_npy.py](https://github.com/rosanajurdi/Prior-based-Losses-for-Medical-Image-Segmentation/blob/master/inference_npy.py)
this script computes the metrics (dice accuracy, haussdorf distance, connected component error) on each sample in the given validation set.

**description of variables:**

``` 
root: path to validation set under consideration. ex: '/media/eljurros/Transcend/CoordConv/ACDC/ACDC/FOLD_1/npy/val'
net_path = : path to checkpoint.  ex: '/media/eljurros/Transcend/Decathlone/ACDC/FOLD_1/size/best2.pkl'
net = torch.load(net_path, map_location=torch.device('cpu'))
n_classes : number of classes with  background
n = 3 : number of classes without background 
``` 
  
## Citation  
  
If you use this benchmark in your experiments or for your loss, please consider citating the following paper:  
  
