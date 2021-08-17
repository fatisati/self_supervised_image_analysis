from swav.fine_tune.fine_tune_utils import *
from swav.utils.swav_utils import *
from swav.config_ import *

class FineTune:

    def __init__(self, feature_backbone):
        self.feature_backbone = feature_backbone


    def get_warmup_classifier(self, outshape, activation="softmax"):
        # input placeholder
        inputs = Input(shape=(224, 224, 3))

        self.feature_backbone.trainable = False

        x = self.feature_backbone(inputs, training=False)
        outputs = Dense(outshape, activation=activation)(x)
        linear_model = Model(inputs, outputs)

        return linear_model

    def warm_up(self, training_ds, outshape, epochs):

        tf.keras.backend.clear_session()

        model = self.get_warmup_classifier(outshape, activation)
        early_stopper = get_early_stopper()
        model.compile(loss=fine_tune_loss, metrics=["acc"],
                      optimizer=optimizer)
        # train
        history = model.fit(training_ds,
                            # validation_data=(testing_ds),
                            epochs=epochs,
                            callbacks=[early_stopper])
        model.save(warmup_name)
        return model, history

    def get_fine_tune_classifier(self):
        # input placeholder
        inputs = Input(shape=(224, 224, 3))
        # get swav baseline model architecture

        self.feature_backbone.trainable = True

        # load warmup model
        warmup_model = tf.keras.models.load_model(warmup_name)
        # get trained output layer
        last_layer = warmup_model.get_layer('dense')

        x = self.feature_backbone(inputs, training=False)
        outputs = last_layer(x)
        linear_model = Model(inputs, outputs)

        return linear_model

    def fine_tune_model(self, training_ds, epochs):
        # get model and compile
        tf.keras.backend.clear_session()
        full_trainable_model = self.get_fine_tune_classifier()
        full_trainable_model.compile(loss=fine_tune_loss, metrics=["acc"],
                                     optimizer=optimizer)
        # train
        history = full_trainable_model.fit(training_ds,
                                           # validation_data=(testing_ds),
                                           epochs=epochs,
                                           callbacks=[get_early_stopper()])

        return full_trainable_model, history


if __name__ == '__main__':
    train_ds, extra_train_ds, validation_ds = download_flower_ds(supervised=True,
                                                                 split=["train[:10%]", "train[10%:85%]",
                                                                        "train[85%:]"])
    training_ds, testing_ds = prepare_fine_tune_data(train_ds, validation_ds)
    feature_backbone_weights, prototype_weights = load_pretrained_weights()
    ft = FineTune(feature_backbone_weights)
    ft.warm_up(training_ds, testing_ds)
    ft.fine_tune_model(training_ds, testing_ds)

    augmented_training_ds = prepare_fine_tune_augment_data(train_ds)
    ft.warm_up(augmented_training_ds, testing_ds)
    ft.fine_tune_model(augmented_training_ds, testing_ds)
