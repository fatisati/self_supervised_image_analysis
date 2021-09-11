import matplotlib.pyplot as plt
import numpy as np

from barlow.barlow_pretrain import *
from barlow.barlow_finetune import *
from utils.model_utils import *

from utils.model_utils import *


class PretrainParams:
    def __init__(self, crop_to, batch_size, project_dim, checkpoints, save_path):
        self.crop_to = crop_to
        self.batch_size = batch_size
        self.project_dim = project_dim
        self.checkpoints = checkpoints
        self.save_path = save_path

    def get_summary(self):
<<<<<<< HEAD
        return 'pretrain_{0}e_{1}projdim_{2}bs_{3}ct'.format(self.epochs, self.project_dim, self.batch_size,
                                                             self.crop_to)
=======
        return f'pretrain_projdim{self.project_dim}_bs{self.batch_size}_ct{self.crop_to}'
>>>>>>> pretrain_autosave

    def get_model_path(self):
        return self.save_path + self.get_summary()

    def get_report(self):
        return 'pretrain params: epochs {0}, bs {1}, image size {2}, project dim{3}' \
            .format(self.checkpoints[-1], self.batch_size, self.crop_to, self.project_dim)



class FineTuneParams:
    def __init__(self, epochs, batch_size, crop_to, pretrain_params: PretrainParams):
        self.epochs = epochs
        self.batch_size = batch_size
        self.crop_to = crop_to
        self.pretrain_params = pretrain_params

    def get_summary(self):
        return f'finetune_{self.batch_size}bs_{self.crop_to}ct'

    def get_model_path(self):
        return self.pretrain_params.save_path + self.get_summary()

    def get_report(self):
        return 'fine-tune params: bs {0}, epochs {1}, crop size {2}'.format(self.batch_size, self.epochs, self.crop_to)


def save_checkpoint(model, epoch_loss_list, path, name):
    model.encoder.save(path + name)
    epoch_loss = []
    for hist in epoch_loss_list:
        epoch_loss += list(hist.history['loss'])

    plt.plot(epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig(f'{path}/figures/{name}.png')


def train_barlow(model, data, checkpoints, path, name):
    history = []

    checkpoints = np.array(checkpoints)
    epoch_change = np.diff(checkpoints)

    epoch_change = np.insert(epoch_change, 0, checkpoints[0])
    current_epoch = 0

    for change in epoch_change:
        hist = model.fit(data, epochs=change)
        history.append(hist)

        current_epoch += change
        save_checkpoint(model, history, path, name + f'_e{current_epoch}')

    return model, history


def run_pretrain(ds, params: PretrainParams):
    x_train, x_test = ds.get_x_train_test_ds()

    ssl_ds = prepare_data_loader(x_train, params.crop_to, params.batch_size)
<<<<<<< HEAD
    val_ds = prepare_data_loader(x_test, params.crop_to, params.batch_size)
    lr_decayed_fn = get_lr(x_train, params.batch_size, params.epochs)
=======
    lr_decayed_fn = get_lr(x_train, params.batch_size, params.checkpoints[-1])
>>>>>>> pretrain_autosave

    resnet_enc = resnet20.get_network(params.crop_to, hidden_dim=params.project_dim, use_pred=False,
                                      return_before_head=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

    model = get_model(resnet_enc, optimizer)
<<<<<<< HEAD
    save_callback = get_checkpoint_callback(params.save_path + 'checkpoints')
    history = model.fit(ssl_ds, validation_data=val_ds, epochs=params.epochs, callbacks=[save_callback])
    plt.plot(history.history["loss"])
    plt.savefig('{0}figures/{1}.png'.format(params.save_path, params.get_summary()))
    model.encoder.save(params.get_model_path())
=======

    # history = model.fit(ssl_ds, epochs=params.epochs)
    # plt.plot(history.history["loss"])
    # plt.savefig('{0}figures'.format(params.save_path, params.get_summary()))
    # model.encoder.save(params.get_model_path())

    train_barlow(model, ssl_ds, params.checkpoints, params.save_path, params.get_summary())

>>>>>>> pretrain_autosave
    return model.encoder


def run_fine_tune(ds, params: FineTuneParams, barlow_enc=None, loss=None):
    outshape = ds.train_labels.shape[-1]
    train_ds, test_ds = ds.get_supervised_ds()
    train_ds, test_ds = prepare_supervised_data_loader(train_ds, test_ds, params.batch_size, params.crop_to)
    if barlow_enc is None:
        barlow_enc = tf.keras.models.load_model(params.pretrain_params.get_model_path())
    if loss is None:
        loss = "binary_crossentropy"

    cosine_lr = get_cosine_lr(params.epochs, len(train_ds), params.batch_size)
    linear_model = get_linear_model(barlow_enc, params.crop_to, outshape)
    # Compile model and start training.

    linear_model.compile(
        loss=loss,
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.SGD(cosine_lr, momentum=0.9),
    )
    history = linear_model.fit(
        train_ds, validation_data=test_ds, epochs=params.epochs
    )
    _, test_acc = linear_model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))

    linear_model.save(params.get_model_path())
    plt.plot(history.history['loss'])
    plt.savefig('{0}figures/{1}.png'.format(params.pretrain_params.save_path, params.get_summary()))
    return linear_model
