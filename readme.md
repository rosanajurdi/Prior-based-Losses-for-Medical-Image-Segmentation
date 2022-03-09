This repository contains the code for the paper: **Effect of Prior-based Losses on Segmentation Performance: A Benchmark** and **A Surprisingly Effective Perimeter-based Loss for Medical Image Segmentation.**


The benchmark of establishes performance of four recent prior-based losses for across 8 different medical datasets of various tasks and modalities. The main objective is to provide intuition onto which losses to choose given a particular task or dataset, based on dataset characteristics and properties. The code is an extension of https://github.com/LIVIAETS/boundary-loss with additinal scripts, functions and Modalities. The proposed benchmark is conducted via a unified segmentation network and learning environment chich can be found (https://github.com/LIVIAETS/boundary-loss). 
 
The perimeter loss paper (https://openreview.net/forum?id=NDEmtyb4cXu) can be found in losses.py script. 
 

# Datasets 

The datasets explored are from a variety of medical image segmentation challenges including the [Decathlon](http://medicaldecathlon.com), the  [ISLES](http://www.isles-challenge.org) and the [WMH](https://wmh.isi.uu.nl) challenges. The data format from the Decathlon is in niffty format.; 

## Step 1: Preparing the dataset for the code
### For the Decathlon Datasets 
1. Download the required data (nifty format) from the [Decathlone challenge](http://medicaldecathlon.com)
2. Run slice_decathlone.py(with retain=0) to transform the data from nifty to .npy format (The output will be the transformed data nifty -> .npy)
3. Run the [KFOLD_split_dataset.py](https://github.com/rosanajurdi/DataSET_module) with the proper dataset class (See documentation).The script will create the required fold_K: (train, val) and their corresponding text files (Make sure to specify the number of samples in the validation set) 

You can just plugin the name of the loss in the losses argument in the main script and run the program. 


**TO DO** \\
- Code should be cleaned and divided into project compartements . \\
- Functions should be documented
- Documentation should be provided !!! 
