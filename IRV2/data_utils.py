import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_aug_datagen():
    # Creating Image Data Generator to augment images
    aug_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'

    )
    return aug_datagen


def get_preprocessing_datagen():
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
    return datagen
