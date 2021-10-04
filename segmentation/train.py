import os
import random
from dermoscopic_dataset import *
from unet_model import *

# Split our img paths into a training and a validation set

# Instantiate data Sequences for each split
batch_size, img_size = 32, (128, 128)
data_path = '../../data/ISIC/dermoscopic/'
image_path = 'resized255/'
mask_path = 'ISIC2018_Task2_Training_GroundTruth_v3/'

if __name__ == '__main__':


    x_size = len(os.listdir(data_path + image_path))
    x_train, x_test = get_img_path(data_path, image_path, int(0.8 * x_size))

    train_gen = DermoscopicImage(batch_size, img_size, x_train, data_path + mask_path)
    val_gen = DermoscopicImage(batch_size, img_size, x_test, mask_path)

    model = get_model(img_size, len(train_gen.target_class_names))
    # print(model.summary())
    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("dermoscopic.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 15
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
