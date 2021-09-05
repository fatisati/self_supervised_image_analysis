import random
from dermoscopic_dataset import *

# Split our img paths into a training and a validation set



# Instantiate data Sequences for each split
batch_size, img_size = 32, 128
data_path = '../../data/ISIC/dermoscopic/'
image_path = 'resized256/'
mask_path = 'ISIC2018_Task1-2_Training_Input.zip'


import os
print(os.listdir(data_path + mask_path))
# train_gen = DermoscopicImage(
#     batch_size, img_size,
# )
# val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths