from swav.fine_tune.fine_tune_params import *


def prepare_fine_tune_augment_data(train_ds):
    trainloader = (
        train_ds
            .shuffle(1024)
            .map(tie_together, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    trainloader = trainloader.with_options(options)
    return trainloader


def prepare_fine_tune_data(ds):
    res_ds = (
        ds
            .map(scale_resize_image, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    return res_ds


@tf.function
def scale_resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))  # Resizing to highest resolution used while training swav
    return (image, label)


def load_pretrained_weights():
    pass
    # feature_backbone_urlpath = "https://storage.googleapis.com/swav-tf/feature_backbone_10_epochs.h5"
    # prototype_urlpath = "https://storage.googleapis.com/swav-tf/projection_prototype_10_epochs.h5"
    #
    # feature_backbone_weights = get_file('swav_feature_weights', feature_backbone_urlpath)
    # prototype_weights = get_file('swav_prototype_projection_weights', prototype_urlpath)
    # return feature_backbone_weights, prototype_weights


def get_early_stopper():
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=2,
                                                     restore_best_weights=True)
    return early_stopper


@tf.function
def scale_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return (image, label)


@tf.function
def random_apply(func, x, p):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)


@tf.function
def random_resize_crop(image, label):
    # Conditional resizing
    image = tf.image.resize(image, (260, 260))
    # Get the crop size for given min and max scale
    size = tf.random.uniform(shape=(1,), minval=MIN_SCALE * 260,
                             maxval=MAX_SCALE * 260, dtype=tf.float32)
    size = tf.cast(size, tf.int32)[0]
    # Get the crop from the image
    crop = tf.image.random_crop(image, (size, size, 3))
    crop_resize = tf.image.resize(crop, (CROP_SIZE, CROP_SIZE))

    return crop_resize, label


@tf.function
def tie_together(image, label):
    # Scale the pixel values
    image, label = scale_image(image, label)
    # random horizontal flip
    image = random_apply(tf.image.random_flip_left_right, image, p=0.5)
    # Random resized crops
    image, label = random_resize_crop(image, label)

    return image, label
