This repository contains the code for the paper: **Effect of Prior-based Losses on Segmentation Performance: A Benchmark** and **A Surprisingly Effective Perimeter-based Loss for Medical Image Segmentation.**


The benchmark of establishes performance of four recent prior-based losses for across 8 different medical datasets of various tasks and modalities. The main objective is to provide intuition onto which losses to choose given a particular task or dataset, based on dataset characteristics and properties. The proposed benchmark is conducted via a unified segmentation network and learning environment chich can be found (https://github.com/LIVIAETS/boundary-loss). 
 
The perimeter loss paper (https://openreview.net/forum?id=NDEmtyb4cXu) can be found in losses.py script. 
 
# Installation and Dependencies

For installation and dependencies, please check this [repository](https://github.com/LIVIAETS/boundary-loss). The code for the nechmark in this repository is an extension of https://github.com/LIVIAETS/boundary-loss with additinal scripts, functions and Modalities.

# Difference from the orginal framework in [repository](https://github.com/LIVIAETS/boundary-loss):
- addition of the Decathlon class(Dataset section). 
- addition of the different losses (in losses.py script) 
- addition of an inference script to compute prior metrics (inference_npy.py connected component error) 


# Datasets 

The datasets explored are from a variety of medical image segmentation challenges including the [Decathlon](http://medicaldecathlon.com), the  [ISLES](http://www.isles-challenge.org) and the [WMH](https://wmh.isi.uu.nl) challenges. The data format from the Decathlon is in niffty format.; 

## Step 1: Preparing the dataset for the code

1. Download the required data (nifty format) from the [Decathlone challenge](http://medicaldecathlon.com)
2. Run slice_decathlone.py(with retain=0) to transform the data from nifty to .npy format (The output will be the transformed data nifty -> .npy)
3. Run the [KFOLD_split_dataset.py](https://github.com/rosanajurdi/DataSET_module) with the proper dataset class (See documentation).The script will create the required fold_K: (train, val) and their corresponding text files (Make sure to specify ) 

Note: for the ISLES and WMH datasets, please refer to the original [repository](https://github.com/LIVIAETS/boundary-loss).

**Notes**:
- make sure to specify the number of samples in the validation set. In the paper,  splits were conducted according to 80 % training 20 % testing. 
- make sure to specify the propper dataset class corresponding to the name of dataset.

                                        Example: `ds = Decathlon(root_dir=root_path, typ=typ)` 



You can just plugin the name of the loss in the losses argument in the main script and run the program. 

Citation

If you use this benchmark in your experiments or for your loss, please consider citating the following paper:




**TO DO** \\
- Code should be cleaned and divided into project compartements . \\
- Functions should be documented
- Documentation should be provided !!! 
