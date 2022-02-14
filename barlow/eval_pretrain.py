import os

from barlow import barlow_pretrain
from ham_dataset import MyDataset
from barlow.data_utils import prepare_data_loader
from barlow.augmentation_utils import *
from utils.model_utils import load_model
import pandas as pd


def calc_val_loss(barlow_model, val_data):
    ds_one = val_data.map(lambda a, b: a)
    ds_two = val_data.map(lambda a, b: b)
    z_a, z_b = barlow_model.predict(ds_one), barlow_model.predict(ds_two)
    loss, on_diog, off_diog = barlow_pretrain.compute_loss_detail(z_a, z_b, 5e-3)
    return loss, on_diog, off_diog


def model_val_loss(model_path, ds, bs, aug_func):
    x_train, x_test = ds.get_x_train_test_ds()
    val_ssl_ds = prepare_data_loader(x_test, bs, aug_func)
    res = []

    for epoch in os.listdir(model_path):
        try:
            model = load_model(model_path + f'/{epoch}')
        except:
            continue
        print('calculating val loss...')
        val_loss, on_diog, off_diog = calc_val_loss(model, val_ssl_ds)
        print(f'done. loss: {val_loss}, on_diog: {on_diog}, off_diog: {off_diog}')

        res.append(
            {'epoch': epoch, 'val_loss': val_loss.numpy(), 'on_diog': on_diog.numpy(), 'off_diog': off_diog.numpy()})
        pd.DataFrame(res).to_excel(model_path + '/val_loss.xlsx')


if __name__ == '__main__':
    bs, crop_to = 64, 128
    aug_func = get_tf_augment(crop_to)
    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/')

    model_path = '../models/twins/pretrain/'
    model_path = model_path + f'adam_ct{crop_to}_bs{bs}_aug_tf'
    model_val_loss(model_path, ds, bs, aug_func)
