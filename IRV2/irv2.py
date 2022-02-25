import tensorflow as tf
from IRV2.soft_attention import SoftAttention
from tensorflow.keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, \
    AveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.model_utils import get_metrics

# class_weights = {
#     0: 1.0,  # akiec
#     1: 1.0,  # bcc
#     2: 1.0,  # bkl
#     3: 1.0,  # df
#     4: 5.0,  # mel
#     5: 1.0,  # nv
#     6: 1.0,  # vasc
# }
def get_model(outshape):
    irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax",

    )

    # Excluding the last 28 layers of the model.
    conv = irv2.layers[-28].output

    attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]),
                                          name='soft_attention')(conv)
    attention_layer = (MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer))
    conv = (MaxPooling2D(pool_size=(2, 2), padding="same")(conv))

    conv = concatenate([conv, attention_layer])
    conv = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)

    output = Flatten()(conv)
    output = Dense(outshape, activation='softmax')(output)
    model = Model(inputs=irv2.input, outputs=output)

    print(model.summary())

    opt1 = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=0.1)
    model.compile(optimizer=opt1,
                  loss='categorical_crossentropy',
                  metrics=get_metrics())
    return model


def train_model(model, train_batches, test_batches, train_size, test_size, class_weights, batch_size, save_path):
    checkpoint = ModelCheckpoint(filepath=save_path + 'saved_model.hdf5', monitor='val_accuracy', save_best_only=True,
                                 save_weights_only=True)

    Earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=30, min_delta=0.001)

    history = model.fit(train_batches,
                        steps_per_epoch=(train_size / 10),
                        epochs=150,
                        verbose=2,
                        validation_data=test_batches, validation_steps=test_size / batch_size,
                        callbacks=[checkpoint, Earlystop], class_weight=class_weights)
    return history
if __name__ == '__main__':
    irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax",

    )

    # Excluding the last 28 layers of the model.
    conv = irv2.layers[-28].output
    conv.summary()
