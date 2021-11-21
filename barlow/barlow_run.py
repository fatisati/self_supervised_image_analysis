import os

import matplotlib.pyplot as plt
import numpy as np

from barlow.barlow_pretrain import *
from barlow.barlow_finetune import *
from utils.model_utils import *

from utils.model_utils import *
import barlow.inception_v3 as inception_v3
import segmentation.unet_model as unet_model

class PretrainParams:
    def __init__(self, crop_to, batch_size, project_dim, checkpoints, save_path, name='adam',
                 optimizer=tf.keras.optimizers.Adam(), backbone='resnet'):
        self.crop_to = crop_to
        self.batch_size = batch_size
        self.project_dim = project_dim
        self.checkpoints = checkpoints
        self.save_path = save_path + 'pretrain/'
        self.name = name

        self.backbone = backbone

        if backbone == 'inception':
            self.crop_to = 299
            self.normalized = True
        else:
            self.normalized = False
        self.optimizer = optimizer

    def get_summary(self):
        summary = f'{self.name}_ct{self.crop_to}_bs{self.batch_size}'
        if self.backbone == 'resnet':
            return summary
        return summary + f'_{self.backbone}'

    def get_old_summary(self):
        return f'{self.name}_pretrain_projdim{self.project_dim}_bs{self.batch_size}_ct{self.crop_to}'

    def get_model_path(self):
        if 'old' in self.save_path:
            return self.save_path + self.get_old_summary()
        return self.save_path + self.get_summary()

    def get_report(self):
        return 'pretrain params: epochs {0}, bs {1}, image size {2}, project dim{3}' \
            .format(self.checkpoints[-1], self.batch_size, self.crop_to, self.project_dim)


class FineTuneParams:
    def __init__(self, checkpoints, batch_size, pretrain_params: PretrainParams, pretrain_epoch, save_path, name,
                 loss=None):
        self.checkpoints = checkpoints
        self.batch_size = batch_size
        self.crop_to = pretrain_params.crop_to
        self.pretrain_params = pretrain_params
        self.pretrain_epoch = pretrain_epoch
        self.save_path = save_path + 'finetune/'
        if loss is None:
            self.loss = "binary_crossentropy"
            self.loss_name = 'normal'
        else:
            self.loss = loss
            self.loss_name = 'weighted'
        self.name = name

    def get_summary(self):

        return f'{self.name}_ct{self.crop_to} _bs{self.batch_size}_loss_{self.loss_name}'

    def get_old_summary(self):

        return f'finetune_bs{self.batch_size}_ct{self.crop_to}_loss_{self.loss_name}'

    def get_model_path(self):
        return self.save_path + self.get_summary()


def run_pretrain(ds, params: PretrainParams, debug=False):
    if params.backbone == 'resnet':
        backbone = resnet20.get_network(params.crop_to, hidden_dim=params.project_dim, use_pred=False,
                                        return_before_head=False)
    elif params.backbone == 'inception':
        backbone = inception_v3.get_network()

    elif params.backbone == 'unet':
        backbone = unet_model.get_unet_backbone((params.crop_to, params.crop_to))

    x_train, x_test = ds.get_x_train_test_ds()
    ssl_ds = prepare_data_loader(x_train, params.crop_to, params.batch_size, params.normalized)

    # lr_decayed_fn = get_lr(x_train, params.batch_size, params.checkpoints[-1])
    optimizer = params.optimizer  # .SGD(learning_rate=lr_decayed_fn, momentum=0.9)
    model = compile_barlow(backbone, optimizer)

    train_model(model, ssl_ds, params.checkpoints, params.save_path, params.get_summary(), load_latest_model=False, debug=debug)

    return model.encoder


def run_fine_tune(ds, params: FineTuneParams, barlow_enc=None):
    print('running-finetune')
    outshape = ds.train_labels.shape[-1]
    train_ds, test_ds = ds.get_supervised_ds()
    train_ds, test_ds = prepare_supervised_data_loader(train_ds, test_ds, params.batch_size, params.crop_to)

    if barlow_enc is None:
        print('loading pretrained-encoder')
        pretrain_path = params.pretrain_params.get_model_path()
        if 'old' in pretrain_path:
            pretrain_path += f'_e{params.pretrain_epoch}'
        else:
            pretrain_path += f'/e{params.pretrain_epoch}'

        barlow_enc = tf.keras.models.load_model(pretrain_path)
    else:
        print('not loading backbone. using function inputs.')

    # cosine_lr = 0.01  # get_cosine_lr(params.checkpoints[-1], len(train_ds), params.batch_size)
    linear_model = get_linear_model(barlow_enc, params.crop_to, outshape)
    # Compile model and start training.

    linear_model.compile(
        loss=params.loss,
        metrics=get_metrics(),
        optimizer=tf.keras.optimizers.Adam()
    )

    train_model(linear_model, train_ds, params.checkpoints, params.save_path, params.get_summary(),
                test_ds)

    # test_acc = linear_model.evaluate(test_ds)
    # print(f'test acc {test_acc}')
    # _, test_acc = linear_model.evaluate(test_ds)
    # print("Test accuracy: {:.2f}%".format(test_acc * 100))

    # linear_model.save(params.get_model_path())
    # plt.plot(history.history['loss'])
    # plt.savefig('{0}figures/{1}.png'.format(params.pretrain_params.save_path, params.get_summary()))
    return linear_model
