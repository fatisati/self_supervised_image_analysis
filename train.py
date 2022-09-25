import barlow.resnet20 as resnet20
import models
from utils.model_utils import *
from datasets.supervised_ds import SupervisedDs
from barlow.augmentation_utils import *


def train_linear_resnet(train_ds, test_ds, in_shape, out_shape, loss,
                        name, model_path = '../models/'):
    resnet = resnet20.ResNet(True, False, None)
    backbone = resnet.get_network(in_shape, hidden_dim=2048)
    model = models.get_classifier(backbone, in_shape, out_shape, False, False)

    # Compile model and start training.
    model.compile(
        loss=loss,
        metrics=get_metrics(),
        optimizer=tf.keras.optimizers.Adam()
    )

    train_model(model, train_ds, [10, 25, 50, 100],
                model_path, name,
                test_ds)


def train_linear_isic(isic_img_folder):
    bs = 64
    ct = 128
    loss = 'binary_crossentropy'
    aug_func = lambda img: custom_augment(img, ct)
    train = SupervisedDs(isic_img_folder + 'train', 'image_name', 'target').get_ds(aug_func, bs)
    test = SupervisedDs(isic_img_folder + 'test', 'image_name', 'target').get_ds(aug_func, bs)
    train_linear_resnet(train, test, ct, 2, loss, 'linear-isic')
