import os
import traceback

# from my_dataset import *
from barlow.barlow_run import *
from model_eval import *

from utils.model_utils import *

model_path = '../models/twins/'
# barlow_pretrain_path = 'pretrain_imgsize{0}bs{1}e{2}proj_dim{3}'
# barlow_fine_tune_path = 'fine_tune_e{0}bs{1}'


barlow_pretrain_checkpoints = [2]  # ,5, 10, 15, 20, 25, 30, 40, 50, 75, 100]  # [25, 50]

barlow_batch_sizes = [16]  # [16, 32, 64]
barlow_crop_to = [16]  # [128, 256]
barlow_project_dim = [2048]  # [2048, 1024]


def pretrain_inception(ds):
    for bs in barlow_batch_sizes:
        pretrain_params = PretrainParams(-1, bs, -1, barlow_pretrain_checkpoints, model_path, backbone='inception')
        run_pretrain(ds, pretrain_params, debug=True)


def pretrain(ds):
    for bs in barlow_batch_sizes:
        for crop_to in barlow_crop_to:
            for project_dim in barlow_project_dim:
                pretrain_params = PretrainParams(crop_to, bs, project_dim, barlow_pretrain_checkpoints, model_path)
                barlow_encoder = run_pretrain(ds, pretrain_params)


def fine_tune(ds):
    checkpoints = [1, 3, 5]  # [25, 50, 75, 100]
    batch_sizes = [32, 64, 128, 256]
    pretrain_params = PretrainParams(256, 16, 2048, [], model_path)

    for batch_size in batch_sizes:
        params = FineTuneParams(checkpoints, batch_size, pretrain_params, 10, model_path, 'test')
        # params.loss = ds.weighted_loss #'categorical_crossentropy'
        run_fine_tune(ds, params)


if __name__ == '__main__':
    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=100)
    pretrain_inception(ds)
