Repository for papers:

Benchmark: Effect of Prior-based Losses on Segmentation Performance: A Benchmark

Midl: A Surprisingly Effective Perimeter-based Loss for Medical Image Segmentation (https://openreview.net/forum?id=NDEmtyb4cXu)

The code is an extension of https://github.com/LIVIAETS/boundary-loss with additinal scripts, functions and Modalities.

Step 1: Preparing the dataset for the code
1. Download the required data (nifty format) from the Decathlone challenge
2. Run slice_decathlone.py(with retain=0) to transform the data from nifty to .npy format 
3. Go to https://github.com/rosanajurdi/DataSET_module and run KFOLD_split_dataset.py with the proper dataset class (See documentation).The script will create the required fold_K: (train, val) and their corresponding text files. 


In the .py script losses you can find all the losses used in the benchmark as well as the Contour-based loss. 

You can just plugin the name of the loss in the losses argument in the main script and run the program. 


**TO DO** \\
- Code should be cleaned and divided into project compartements . \\
- Functions should be documented
- Documentation should be provided !!! 
