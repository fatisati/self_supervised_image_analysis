from swav.fine_tune.fine_tune_params import *
import tensorflow_datasets as tfds


class Data:


    @staticmethod
    def prepare_ssl_data(train_ds):
        # Get multiple data loaders
        trainloaders = multicrop_dataset.get_multires_dataset(train_ds,
                                                              size_crops=SIZE_CROPS,
                                                              num_crops=NUM_CROPS,
                                                              min_scale=MIN_SCALE,
                                                              max_scale=MAX_SCALE,
                                                              options=options)

        # Zipping
        trainloaders_zipped = tf.data.Dataset.zip(trainloaders)

        # Final trainloader
        trainloaders_zipped = (
            trainloaders_zipped
                .batch(BS)
                .prefetch(AUTO)
        )
        return trainloaders_zipped

    @staticmethod
    def download_flower_ds(supervised=False, split=["train[:85%]", "train[85%:]"]):
        # Gather Flowers dataset
        ds = tfds.load(
            "tf_flowers",
            split=split,
            as_supervised=supervised
        )
        return ds

    @staticmethod
    def visualize_ds(train_ds):
        # Visualization
        plt.figure(figsize=(10, 10))
        for i, image in enumerate(train_ds.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image['image'])
            plt.axis("off")

def get_lr_fn(decay_steps=1000):
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.1, decay_steps=decay_steps)
    return lr_decayed_fn


def get_sgd_optimizer(lr_decayed_fn):
    opt = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn)
    return opt