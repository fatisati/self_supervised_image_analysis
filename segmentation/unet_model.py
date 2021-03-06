from tensorflow.keras import layers
import keras
from datasets.dermoscopic_dataset import *


def get_unet_backbone(img_size):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    model = keras.Model(inputs, x)
    return model


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    backbone = get_unet_backbone(img_size)
    x = backbone(inputs)
    # Add a per-pixel classification layer
    # default: softamx - sigmoid for multi label
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    img_size = (255, 255)
    num_classes = 5
    model = get_model(img_size, num_classes)
    input_img_folder = '../../data/ISIC/dermoscopic/resized255/'
    target_img_folder = '../../data/ISIC/dermoscopic/ISIC2018_Task2_Training_GroundTruth_v3/'
    train_input_img_paths, val_input_img_paths = split_train_test(input_img_folder)
    print('model loaded.')

    ds = DermoscopicImage(10, img_size, val_input_img_paths[:20],
                          target_img_folder)
    print(f'dataset generated. val size {len(val_input_img_paths)}')

    predicted_masks = model.predict(ds)
    print(predicted_masks.shape)
    print(predicted_masks)
