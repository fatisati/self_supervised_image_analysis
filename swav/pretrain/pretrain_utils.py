from swav.pretrain.pretrain_params import *


def get_flower_ds():
    # Gather Flowers dataset
    train_ds, validation_ds = tfds.load(
        "tf_flowers",
        split=["train[:85%]", "train[85%:]"]
    )
    return train_ds, validation_ds


def prepare_data(train_ds, size_crops, batch_size):
    # Get multiple data loaders
    trainloaders = multicrop_dataset.get_multires_dataset(train_ds,
                                                          size_crops=size_crops,
                                                          num_crops=NUM_CROPS,
                                                          min_scale=MIN_SCALE,
                                                          max_scale=MAX_SCALE,
                                                          options=options)

    # Prepare the final data loader

    AUTO = tf.data.experimental.AUTOTUNE

    # Zipping
    trainloaders_zipped = tf.data.Dataset.zip(trainloaders)

    # Final trainloader
    trainloaders_zipped = (
        trainloaders_zipped
            .batch(batch_size)
            .prefetch(AUTO)
    )

    return trainloaders_zipped
