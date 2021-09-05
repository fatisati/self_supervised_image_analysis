import matplotlib.pyplot as plt
from ham_dataset import *


def get_linear_model(barlow_encoder, crop_to, y_shape):
    # Extract the backbone ResNet20.
    backbone = tf.keras.Model(
        barlow_encoder.input, barlow_encoder.layers[-8].output
    )

    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = tf.keras.layers.Input((crop_to, crop_to, 3))
    x = backbone(inputs, training=False)
    outputs = tf.keras.layers.Dense(y_shape, activation="softmax")(x)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_model")
    return linear_model


def get_cosine_lr(epochs, train_size, batch_size):
    # Cosine decay for linear evaluation.
    STEPS_PER_EPOCH = train_size // batch_size

    cosine_decayed_lr = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.3, decay_steps=epochs * STEPS_PER_EPOCH
    )
    return cosine_decayed_lr


if __name__ == '__main__':
    epochs = 10
    batch_size = 32
    crop_to = 128
    ds = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=30)
    pretrain_path = ''
    barlow_enc = None
    loss = None

    outshape = ds.train_labels.shape[-1]
    train_ds, test_ds = ds.get_supervised_ds()
    if barlow_enc is None:
        barlow_enc = tf.keras.models.load_model(pretrain_path)

    cosine_lr = get_cosine_lr(epochs, len(train_ds), batch_size)
    linear_model = get_linear_model(barlow_enc, crop_to, outshape)
    # Compile model and start training.
    if loss is None:
        loss = "binary_crossentropy"
    linear_model.compile(
        loss=loss,
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.SGD(cosine_lr, momentum=0.9),
    )
    history = linear_model.fit(
        train_ds, validation_data=test_ds, epochs=epochs
    )
    _, test_acc = linear_model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))

    # linear_model.save(save_path)
    plt.plot(history.history['loss'])
    plt.show()
    # plt.savefig(params.model_path + 'figures/' + params.get_summary() + '.png')
