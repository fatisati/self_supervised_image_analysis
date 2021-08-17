import matplotlib.pyplot as plt

from barlow.augmentation import *
from barlow.barlow_loss import *
from self_supervised_model import *
import barlow.setup as setup


class FineTuneParams:
    def __init__(self, bs, epochs, pretrain_crop_to, model_path):
        self.batch_size = bs
        self.epochs = epochs
        self.pretrain_crop_to = pretrain_crop_to
        self.model_path = model_path

    def get_report(self):
        return 'fine-tune params: bs {0}, epochs {1}'.format(self.batch_size, self.epochs)

    def get_summary(self):
        return 'fine_tune_b{0}_e{1}_cropto{2}'.format(self.batch_size, self.epochs, self.pretrain_crop_to)


class PretrainParams:
    def __init__(self, bs, epochs, project_dim, crop_to, model_path):
        self.batch_size = bs
        self.epochs = epochs
        self.project_dim = project_dim
        self.crop_to = crop_to
        self.model_path = model_path

    def get_report(self):
        return 'pretrain params: epochs {0}, bs {1}, image size {2}, project dim{3}' \
            .format(self.epochs, self.batch_size, self.crop_to, self.project_dim)

    def get_summary(self):
        return 'pretrain_e{0}_b{1}_cropto{2}_projdim{3}'.format(self.epochs, self.batch_size, self.crop_to,
                                                                self.project_dim)


def get_cfar_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(f"Total training examples: {len(x_train)}")
    print(f"Total test examples: {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)


def cast_to_tf_dataset(ds):
    return tf.data.Dataset.from_tensor_slices(ds)


def prepare_data_loader(x_train_ds, params):
    # ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        x_train_ds.shuffle(1024, seed=SEED)
            .map(lambda x: custom_augment(x, params.crop_to), num_parallel_calls=AUTO)
            .batch(params.batch_size)
            .prefetch(AUTO)
    )

    # ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_two = (
        x_train_ds.shuffle(1024, seed=SEED)
            .map(lambda x: custom_augment(x, params.crop_to), num_parallel_calls=AUTO)
            .batch(params.batch_size)
            .prefetch(AUTO)
    )

    # We then zip both of these datasets.
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
    return ssl_ds, ssl_ds_one, ssl_ds_two


class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_a, z_b = self.encoder(ds_one, training=True), self.encoder(ds_two, training=True)
            loss = compute_loss(z_a, z_b, self.lambd)

        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def pretrain(encoder, opt, ssl_ds, params):
    print('pretraining...')
    barlow_twins = BarlowTwins(encoder)
    barlow_twins.compile(optimizer=opt)
    history = barlow_twins.fit(ssl_ds, epochs=params.epochs)
    # Visualize the training progress of the model.
    plt.plot(history.history["loss"])
    plt.grid()
    plt.savefig(params.model_path +'figures/'+ params.get_summary() + '.png')
    # plt.title("Barlow Twin Loss")
    # plt.show()

    return barlow_twins


def prepare_fine_tune_data(train_ds, test_ds, params: FineTuneParams):
    # Then we shuffle, batch, and prefetch this dataset for performance. We
    # also apply random resized crops as an augmentation but only to the
    # training set.
    train_ds = (
        train_ds.shuffle(1024)
            .map(lambda x, y: (flip_random_crop(x, params.pretrain_crop_to), y), num_parallel_calls=AUTO)
            .batch(params.batch_size)
            .prefetch(AUTO)
    )
    test_ds = test_ds.map(lambda x, y: (resize(x, params.pretrain_crop_to), y), num_parallel_calls=AUTO) \
        .batch(params.batch_size).prefetch(AUTO)
    return train_ds, test_ds


def get_linear_model(barlow_encoder, outshape, params: FineTuneParams, weight_path=''):
    backbone = tf.keras.Model(
        barlow_encoder.input, barlow_encoder.layers[-8].output
    )

    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = tf.keras.layers.Input((params.pretrain_crop_to, params.pretrain_crop_to, 3))
    x = backbone(inputs, training=False)
    outputs = tf.keras.layers.Dense(outshape, activation="softmax")(x)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_model")
    if len(weight_path) > 0:
        linear_model.load_weights(weight_path)
    return linear_model


def fine_tune(train_ds, test_ds, loss, lr, outshape,
              barlow_encoder, params: FineTuneParams, weight_path=''):
    print('fine-tuning...')
    train_ds, test_ds = prepare_fine_tune_data(train_ds, test_ds, params)
    linear_model = get_linear_model(barlow_encoder, outshape, params, weight_path)

    # Cosine decay for linear evaluation.

    # Compile model and start training.
    linear_model.compile(
        loss=loss,
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9),
    )
    history = linear_model.fit(
        train_ds, validation_data=test_ds, epochs=params.epochs
    )
    _, test_acc = linear_model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
    return linear_model, history


def run_pretrain_barlow(ds, pretrain_path, params: PretrainParams):
    print('barlow pretrain with bs:{0}, crop_to:{1}, proj_dim: {2}'.format(params.batch_size,
                                                                           params.crop_to,
                                                                           params.project_dim))
    x_train, x_test = ds.get_x_train_test_ds()
    ssl_ds, ssl_ds_one, ssl_ds_two = prepare_data_loader(x_train, params)

    STEPS_PER_EPOCH = len(x_train) // params.batch_size
    WARMUP_STEPS = int(params.epochs * STEPS_PER_EPOCH)

    lr_decayed_fn = lr_scheduler.WarmUpCosine(
        learning_rate_base=1e-3,
        total_steps=params.epochs * STEPS_PER_EPOCH,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS
    )

    resnet_enc = resnet20.get_network(input_shape=(params.crop_to, params.crop_to, 3),
                                      hidden_dim=params.project_dim, use_pred=False,
                                      return_before_head=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

    barlow = pretrain(resnet_enc, optimizer, ssl_ds, params)
    barlow.encoder.save(pretrain_path)
    print('---end---')
    return barlow.encoder


def run_fine_tune(ds, pretrain_path, save_path, params: FineTuneParams):
    outshape = ds.train_labels.shape[-1]
    train_ds, test_ds = ds.get_supervised_ds()
    barlow_enc = tf.keras.models.load_model(pretrain_path)

    STEPS_PER_EPOCH = len(train_ds) // params.batch_size
    cosine_lr = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.3, decay_steps=params.epochs * STEPS_PER_EPOCH
    )
    linear_model, history = fine_tune(train_ds, test_ds,
                                      'binary_crossentropy', cosine_lr,
                                      outshape, barlow_enc,
                                      params)

    linear_model.save(save_path)
    plt.plot(history.history['loss'])
    plt.savefig(params.model_path +'figures/'+ params.get_summary() + '.png')
    return linear_model, history
