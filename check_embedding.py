import numpy as np
import pandas as pd

from utils.model_utils import *
from barlow.barlow_run import *

if __name__ == '__main__':
    crop_batch_epoch = [[256, 16, 10]]
    model_path = '../models/twins/'
    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=1000)
    x_train, x_test = ds.get_x_train_test_ds()

    res = []
    for crop_to, batch_size, epoch in crop_batch_epoch:
        pretrain_params = PretrainParams(crop_to, batch_size, 2048, [], model_path, 'adam')

        # pretrain_params.save_path = model_path + 'pretrain-old/'
        encoder = load_model(pretrain_params.get_model_path() + f'/e{epoch}')

        backbone = get_resnet_encoder(encoder)

        input_x = prepare_x(x_train, batch_size, crop_to)
        backbone_out = backbone.predict(input_x)
        res.append({'max': backbone_out.max(), 'min': backbone_out.min(),
                    'input-size': crop_to, 'batch-size': batch_size, 'train-epochs': 10,
                    'mean': np.mean(backbone_out), 'avg': np.average(backbone_out),
                    'std': np.std(backbone_out)})
    try:
        df = pd.read_excel('embedding_statistics.xlsx')
    except:
        df = pd.DataFrame([])
    df = df.append(res)
    df.to_excel(model_path + 'embedding_statistics.xlsx')
