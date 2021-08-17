from my_dataset import *
from barlow import barlow_model

model_path = '../../models/twins/'
barlow_pretrain_path = 'pretrain_imgsize{0}bs{1}e{2}proj_dim{3}'
barlow_fine_tune_path = 'fine_tune_e{0}bs{1}'

barlow_pretrain_epochs = [1]#[25, 50]
barlow_batch_sizes = [2]#[16, 32, 64]
barlow_crop_to = [32]#[128, 256]
barlow_project_dim = [2048]#[2048, 1024]

barlow_fine_tune_epochs = [2]#[25, 50, 100]
barlow_fine_tune_bs = [3]#[32, 64, 128]

if __name__ == '__main__':
    ds = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', balanced=True, data_size=5)
