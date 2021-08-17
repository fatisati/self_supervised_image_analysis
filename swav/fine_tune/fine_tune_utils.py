from swav.fine_tune.fine_tune_params import *


def prepare_fine_tune_augment_data(train_ds, batch_size, crop_size):
    trainloader = (
        train_ds
            .shuffle(1024)
            .map(lambda x, y: tie_together(x,y, crop_size), num_parallel_calls=AUTO)

            .batch(batch_size)
            .prefetch(AUTO)
    )

    trainloader = trainloader.with_options(options)
    return trainloader


def prepare_fine_tune_data(ds, batch_size, crop_to):
    res_ds = (
        ds
            .map(lambda x, y: scale_resize_image(x,y, crop_to), num_parallel_calls=AUTO)
            .batch(batch_size)
            .prefetch(AUTO)
    )

    return res_ds


@tf.function
def scale_resize_image(image, label, crop_to):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (crop_to, crop_to))  # Resizing to highest resolution used while training swav
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
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, verbose=2,
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
def random_resize_crop(image, label, crop_size):

    initial_size = int(4*crop_size/3)
    # Conditional resizing
    image = tf.image.resize(image, (initial_size, initial_size))
    # Get the crop size for given min and max scale
    size = tf.random.uniform(shape=(1,), minval=MIN_SCALE * initial_size,
                             maxval=MAX_SCALE * initial_size, dtype=tf.float32)
    size = tf.cast(size, tf.int32)[0]
    # Get the crop from the image
    crop = tf.image.random_crop(image, (size, size, 3))
    crop_resize = tf.image.resize(crop, (crop_size, crop_size))

    return crop_resize, label


@tf.function
def tie_together(image, label, crop_size):
    # Scale the pixel values
    image, label = scale_image(image, label)
    # random horizontal flip
    image = random_apply(tf.image.random_flip_left_right, image, p=0.5)
    # Random resized crops
    image, label = random_resize_crop(image, label, crop_size)

    return image, label
