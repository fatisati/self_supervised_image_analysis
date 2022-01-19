from barlow.barlow_pretrain import *
from utils.model_utils import *
from razi_dataset import *

if __name__ == '__main__':
    checkpoints = []
    save_path = '../../models/twins/pretrain/razi/'
    bs = 64
    crop_to = 128
    name = f'aug_multi_instance_bs{bs}_cr{crop_to}'

    ds = RaziDataset('../../data/razi/', crop_to)
    train_samples, test_samples = ds.get_pretrain_samples()
    ssl_ds = ds.prepare_ssl_ds(train_samples, bs)

    backbone = get_backbone('resnet', crop_to, 2048)
    optimizer = 'adam'
    model = compile_barlow(backbone, optimizer)


    def compile_function(): return model.compile(optimizer=optimizer)


    train_model(model, ssl_ds, checkpoints, save_path, name, load_latest_model=True,
                compile_function=compile_function)
