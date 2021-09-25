import os

import matplotlib.pyplot as plt
import numpy as np

from barlow.barlow_pretrain import *
from barlow.barlow_finetune import *
from utils.model_utils import *

from utils.model_utils import *


class PretrainParams:
    def __init__(self, crop_to, batch_size, project_dim, checkpoints, save_path, name):
        self.crop_to = crop_to
        self.batch_size = batch_size
        self.project_dim = project_dim
        self.checkpoints = checkpoints
        self.save_path = save_path
        self.name = name

    def get_summary(self):
        return f'{self.name}_pretrain_projdim{self.project_dim}_bs{self.batch_size}_ct{self.crop_to}'

    def get_model_path(self):
        return self.save_path + self.get_summary()

    def get_report(self):
        return 'pretrain params: epochs {0}, bs {1}, image size {2}, project dim{3}' \
            .format(self.checkpoints[-1], self.batch_size, self.crop_to, self.project_dim)


class FineTuneParams:
    def __init__(self, checkpoints, batch_size, pretrain_params: PretrainParams, pretrain_epoch, loss=None):
        self.checkpoints = checkpoints
        self.batch_size = batch_size
        self.crop_to = pretrain_params.crop_to
        self.pretrain_params = pretrain_params
        self.pretrain_epoch = pretrain_epoch
        if loss is None:
            self.loss = "binary_crossentropy"
            self.loss_name = 'normal'
        else:
            self.loss = loss
            self.loss_name = 'weighted'

    def get_summary(self):

        return f'finetune_bs{self.batch_size}_ct{self.crop_to}_loss_{self.loss_name}'

    def get_model_path(self):
        return self.pretrain_params.save_path + self.get_summary()


def run_pretrain(ds, params: PretrainParams):
    x_train, x_test = ds.get_x_train_test_ds()

    ssl_ds = prepare_data_loader(x_train, params.crop_to, params.batch_size)

    lr_decayed_fn = get_lr(x_train, params.batch_size, params.checkpoints[-1])

    resnet_enc = resnet20.get_network(params.crop_to, hidden_dim=params.project_dim, use_pred=False,
                                      return_before_head=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

    model = get_model(resnet_enc, optimizer)

    # history = model.fit(ssl_ds, epochs=params.epochs)
    # plt.plot(history.history["loss"])
    # plt.savefig('{0}figures'.format(params.save_path, params.get_summary()))
    # model.encoder.save(params.get_model_path())

    train_model(model, ssl_ds, params.checkpoints, params.save_path, params.get_summary())

    return model.encoder


def run_fine_tune(ds, params: FineTuneParams, barlow_enc=None):
    outshape = ds.train_labels.shape[-1]
    train_ds, test_ds = ds.get_supervised_ds()
    train_ds, test_ds = prepare_supervised_data_loader(train_ds, test_ds, params.batch_size, params.crop_to)

    if barlow_enc is None:
        barlow_enc = tf.keras.models.load_model(params.pretrain_params.get_model_path() + f'_e{params.pretrain_epoch}')

    cosine_lr = 0.01  # get_cosine_lr(params.checkpoints[-1], len(train_ds), params.batch_size)
    linear_model = get_linear_model(barlow_enc, params.crop_to, outshape)
    # Compile model and start training.

    linear_model.compile(
        loss=params.loss,
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.Adam()
    )

    print(linear_model.predict(train_ds))
    print('-----------test res-------------')
    print(linear_model.predict(test_ds))

    # history = linear_model.fit(
    #     train_ds, validation_data=test_ds, epochs=params.epochs
    # )

    train_model(linear_model, train_ds, params.checkpoints, params.pretrain_params.save_path, params.get_summary(),
                test_ds)

    _, test_acc = linear_model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))

    # linear_model.save(params.get_model_path())
    # plt.plot(history.history['loss'])
    # plt.savefig('{0}figures/{1}.png'.format(params.pretrain_params.save_path, params.get_summary()))
    return linear_model
