from razi_dataset import RaziDataset
from utils.data_utils import get_train_test_idx
from barlow import resnet20
from barlow.barlow_finetune import *


def prepare_data(train_ds, test_ds, bs):
    train_ds = (
        train_ds.shuffle(1024)
            .batch(bs)
            .prefetch(AUTO)
    )
    test_ds = test_ds.batch(bs).prefetch(AUTO)
    return train_ds, test_ds


if __name__ == '__main__':
    img_size = 128
    checkpoints = [2, 3]
    save_path = '../../models/twins/'
    model_name = 'razi-no-pretrain-bs128'
    bs = 128

    ds = RaziDataset('../../data/razi/', img_size)
    train_ratio = 0.01
    train_ds, test_ds = ds.get_supervised_ds(train_ratio)
    train_ds, test_ds = prepare_data(train_ds, test_ds, bs)
    outshape = len(ds.all_labels)
    print(f'train-ratio: {train_ratio}, train-size: {len(train_ds)}')

    print('loading backbone...')
    backbone = resnet20.get_network(img_size, hidden_dim=2048, use_pred=False,
                                    return_before_head=False)
    print('done')

    print('preparing model...')
    linear_model = get_linear_model(backbone, img_size, outshape)
    linear_model.compile(
        loss='binary_crossentropy',
        metrics=get_metrics(),
        optimizer=tf.keras.optimizers.Adam()
    )
    print('done. start training...')
    train_model(linear_model, train_ds, checkpoints, save_path, model_name, test_ds)
