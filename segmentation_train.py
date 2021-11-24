from dermoscopic_dataset import *
from segmentation.unet_model import *
from segmentation.evaluation import SegmentationVisualization
from utils.model_utils import train_model

batch_size, img_size = 10, (16, 16)
epochs = 5
data_path = '../data/ISIC/dermoscopic/'
image_path = 'resized255/'
mask_path = 'ISIC2018_Task2_Training_GroundTruth_v3/'
model_path = f'../models/segmentation/'
model_name = f'unet_cr{img_size[0]}_bs{batch_size}'
checkpoints = [2,3]#[5,10,15,20,25,30,35,40,50]

if __name__ == '__main__':

    visualization = SegmentationVisualization(img_size[0])
    x_size = len(os.listdir(data_path + image_path))
    x_train, x_test = get_img_path(data_path, image_path, int(0.8 * x_size))
    x_train = x_train[:50]
    train_gen = DermoscopicImage(batch_size, img_size, x_train, data_path + mask_path)
    val_gen = DermoscopicImage(batch_size, img_size, x_test, data_path + mask_path)

    model = get_model(img_size, len(train_gen.target_class_names))
    # print(model.summary())

    # Configure the model for training.
    model.compile(optimizer="rmsprop", loss="binary_crossentropy")

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    # ]

    # Train the model, doing validation at the end of each epoch.
    # model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    train_model(model, train_gen, checkpoints, model_path, model_name,
                val_gen, checkpoint_function = visualization.visualize_samples)
