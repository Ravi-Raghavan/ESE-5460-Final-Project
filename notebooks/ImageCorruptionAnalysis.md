# File Overview
This file serves to identify the set of images in our training, validation, and test splits that are corrupted and cannot be used as inputs to our model. 

The script goes through the designated image folder, checks each file for corruption, and logs any corrupted images into a text file. This text file can then be used in future training runs to filter out corrupted samples, ensuring that our model only sees valid image data.