import os
import traceback

# from my_dataset import *
from barlow.barlow_run import *
from model_eval import *

from utils.model_utils import *

model_path = '../models/twins/'
# barlow_pretrain_path = 'pretrain_imgsize{0}bs{1}e{2}proj_dim{3}'
# barlow_fine_tune_path = 'fine_tune_e{0}bs{1}'


barlow_pretrain_checkpoints = [5, 36]  # ,5, 10, 15, 20, 25, 30, 40, 50, 75, 100]  # [25, 50]

barlow_batch_sizes = [128]  # [16, 32, 64]
barlow_crop_to = [64]  # [128, 256]
barlow_project_dim = [2048]  # [2048, 1024]


def pretrain_inception(ds):
    for bs in barlow_batch_sizes:
        pretrain_params = PretrainParams(-1, bs, -1, barlow_pretrain_checkpoints, model_path, backbone='inception')
        run_pretrain(ds, pretrain_params, debug=True)


def pretrain(ds):
    for bs in barlow_batch_sizes:
        for crop_to in barlow_crop_to:

            for project_dim in barlow_project_dim:
                pretrain_params = PretrainParams(crop_to, bs, project_dim, barlow_pretrain_checkpoints, model_path,
                                                 augment_func='tf')
                run_pretrain(ds, pretrain_params)


pretrain_crop_batch_epoch = [[128, 'no-pretrain', 'no-pretrain']]


def fine_tune(ds):
    checkpoints = [10, 25, 50, 75, 100, 150, 200]
    batch_sizes = [128]
    for cr, bs, pretrain_epoch in pretrain_crop_batch_epoch:
        pretrain_params = PretrainParams(cr, bs, 2048, -1, -1)
        # print(pretrain_params.get_summary())
        # pretrain_params.save_path = model_path + 'pretrain-old/'
        backbone = resnet20.get_network(cr, hidden_dim=2048, use_pred=False,
                                        return_before_head=False)
        for batch_size in batch_sizes:
            params = FineTuneParams(checkpoints, batch_size, pretrain_params, -1, model_path, f'10percent-no-pretrain')
            # params.loss = ds.weighted_loss
            print(params.get_model_path())
            print('before run')
            run_fine_tune(ds, params, backbone)


if __name__ == '__main__':
    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=100)
    pretrain(ds)
