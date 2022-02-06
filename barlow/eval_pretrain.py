import os

from barlow import barlow_pretrain
from ham_dataset import MyDataset
from barlow.data_utils import prepare_data_loader
from barlow.augmentation_utils import *
from utils.model_utils import load_model
import pandas as pd


def calc_val_loss(barlow_model, val_data):
    ds_one = val_data.map(lambda a, b : a)
    ds_two = val_data.map(lambda a, b : b)
    z_a, z_b = barlow_model.predict(ds_one), barlow_model.predict(ds_two)
    loss = barlow_pretrain.compute_loss(z_a, z_b, 5e-3)
    return loss


if __name__ == '__main__':
    bs, crop_to = 64, 128
    aug_func = get_tf_augment(crop_to)
    ds = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/')
    x_train, x_test = ds.get_x_train_test_ds()
    val_ssl_ds = prepare_data_loader(x_test, bs, aug_func)

    model_path = '../../models/barlow/pretrain/'
    model_path = model_path + f'batchnorm_ct{crop_to}_bs{bs}_aug_tf'

    res = []
    for epoch in os.listdir(model_path):
        print(f'loading model for epoch {epoch}...')
        model = load_model(model_path + epoch)
        print('done')

        print('calculating val loss...')
        val_loss = calc_val_loss(model, val_ssl_ds)
        print(f'done. loss: {val_loss}')

        res.append({'epoch': epoch, 'val_loss': val_loss})
    pd.DataFrame(res).to_excel(model_path + 'val_loss.xlsx')
