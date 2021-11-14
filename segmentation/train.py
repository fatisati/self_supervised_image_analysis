from dermoscopic_dataset import *
from unet_model import *
from utils.model_utils import *

batch_size, img_size = 5, (32, 32)
data_path = '../../data/ISIC/dermoscopic/'
image_path = 'resized255/'
mask_path = 'ISIC2018_Task2_Training_GroundTruth_v3/'

if __name__ == '__main__':


    x_size = len(os.listdir(data_path + image_path))
    x_train, x_test = get_img_path(data_path, image_path, int(0.8 * x_size))

    train_gen = DermoscopicImage(batch_size, img_size, x_train[:10], data_path + mask_path)
    val_gen = DermoscopicImage(batch_size, img_size, x_test[:10], data_path + mask_path)

    model = get_model(img_size, len(train_gen.target_class_names))
    # print(model.summary())
    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="binary_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("dermoscopic.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 5
    train_model(model, train_gen, [5,10,20,30,40,50], model_path, model_name,
                test_ds=val_gen)
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
