import numpy as np
import pandas as pd

from utils.model_utils import *
from barlow.barlow_run import *

if __name__ == '__main__':
    img_sizes = [[32, 512], [64, 128], [128, 64]]
    model_path = '../models/twins/'
    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=1000)
    x_train, x_test = ds.get_x_train_test_ds()

    res = []
    for crop_to, batch_size in img_sizes:
        pretrain_params = PretrainParams(crop_to, batch_size, 2048, [], model_path, 'adam')

        pretrain_params.save_path = model_path + 'pretrain-old/'

        encoder = load_model(pretrain_params.get_model_path() + f'_e{25}')

        backbone = get_backbone(encoder)

        input_x = prepare_x(x_train, batch_size, crop_to)
        backbone_out = backbone.predict(input_x)
        res.append({'max': backbone_out.max(), 'min': backbone_out.min(),
                    'input-size': crop_to, 'batch-size': batch_size, 'train-epochs': 25,
                    'mean': np.mean(backbone_out), 'avg': np.average(backbone_out),
                    'std': np.std(backbone_out)})
    pd.DataFrame(res).to_excel(model_path + 'embedding_statistics.xlsx')
