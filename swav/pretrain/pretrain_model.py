from swav.pretrain.pretrain_params import *


def sinkhorn(sample_prototype_batch, n_iters=3):
    Q = tf.transpose(tf.exp(sample_prototype_batch / 0.05))
    Q /= tf.keras.backend.sum(Q)
    K, B = Q.shape

    u = tf.zeros_like(K, dtype=tf.float32)
    r = tf.ones_like(K, dtype=tf.float32) / K
    c = tf.ones_like(B, dtype=tf.float32) / B

    for _ in range(n_iters):
        u = tf.keras.backend.sum(Q, axis=1)
        Q *= tf.expand_dims((r / u), axis=1)
        Q *= tf.expand_dims(c / tf.keras.backend.sum(Q, axis=0), 0)

    final_quantity = Q / tf.keras.backend.sum(Q, axis=0, keepdims=True)
    final_quantity = tf.transpose(final_quantity)

    return final_quantity


# @tf.function
# Reference: https://github.com/facebookresearch/swav/blob/master/main_swav.py
def train_step(input_views, feature_backbone, projection_prototype,
               optimizer, crops_for_assign, temperature):
    # ============ retrieve input data ... ============
    im1, im2, im3, im4, im5 = input_views
    inputs = [im1, im2, im3, im4, im5]
    batch_size = inputs[0].shape[0]

    # ============ create crop entries with same shape ... ============
    crop_sizes = [inp.shape[1] for inp in inputs]  # list of crop size of views
    unique_consecutive_count = [len([elem for elem in g]) for _, g in
                                groupby(crop_sizes)]  # equivalent to torch.unique_consecutive
    idx_crops = tf.cumsum(unique_consecutive_count)

    # ============ multi-res forward passes ... ============
    start_idx = 0
    with tf.GradientTape() as tape:
        for end_idx in idx_crops:
            concat_input = tf.stop_gradient(tf.concat(inputs[start_idx:end_idx], axis=0))
            _embedding = feature_backbone(concat_input)  # get embedding of same dim views together
            if start_idx == 0:
                embeddings = _embedding  # for first iter
            else:
                embeddings = tf.concat((embeddings, _embedding), axis=0)  # concat all the embeddings from all the views
            start_idx = end_idx

        projection, prototype = projection_prototype(embeddings)  # get normalized projection and prototype
        projection = tf.stop_gradient(projection)

        # ============ swav loss ... ============
        # https://github.com/facebookresearch/swav/issues/19
        loss = 0
        for i, crop_id in enumerate(crops_for_assign):  # crops_for_assign = [0,1]
            with tape.stop_recording():
                out = prototype[batch_size * crop_id: batch_size * (crop_id + 1)]

                # get assignments
                q = sinkhorn(out)  # sinkhorn is used for cluster assignment

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(NUM_CROPS)),
                               crop_id):  # (for rest of the portions compute p and take cross entropy with q)
                p = tf.nn.softmax(prototype[batch_size * v: batch_size * (v + 1)] / temperature)
                subloss -= tf.math.reduce_mean(tf.math.reduce_sum(q * tf.math.log(p), axis=1))
            loss += subloss / tf.cast((tf.reduce_sum(NUM_CROPS) - 1), tf.float32)

        loss /= len(crops_for_assign)

    # ============ backprop ... ============
    variables = feature_backbone.trainable_variables + projection_prototype.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


def train_swav(feature_backbone,
               projection_prototype,
               dataloader,
               optimizer,
               crops_for_assign,
               temperature,
               epochs=50):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(epochs)):
        w = projection_prototype.get_layer('prototype').get_weights()
        w = tf.transpose(w)
        w = tf.math.l2_normalize(w, axis=1)
        projection_prototype.get_layer('prototype').set_weights(tf.transpose(w))

        for i, inputs in enumerate(dataloader):
            loss = train_step(inputs, feature_backbone, projection_prototype,
                              optimizer, crops_for_assign, temperature)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))

        if epoch % 5 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, [feature_backbone, projection_prototype]


def fit_swav(trainloaders_zipped, epochs):
    # ============ initialize the networks and the optimizer ... ============
    feature_backbone = architecture.get_resnet_backbone()
    projection_prototype = architecture.get_projection_prototype(10)

    # ============ train for 40 epochs ... ============
    epoch_wise_loss, models = train_swav(feature_backbone,
                                         projection_prototype,
                                         trainloaders_zipped,
                                         opt,
                                         crops_for_assign=[0, 1],
                                         temperature=0.1,
                                         epochs=epochs
                                         )
    return epoch_wise_loss, models


def load_models(feature_weights, prototype_weights):
    feature_backbone = architecture.get_resnet_backbone()
    projection_prototype = architecture.get_projection_prototype(10)

    feature_backbone.load_weights(feature_weights)
    projection_prototype.load_weights(prototype_weights)

    return feature_backbone, projection_prototype


def continue_train(trainloaders_zipped, models, epochs):
    feature_backbone, projection_prototype = models
    epoch_wise_loss, models = train_swav(feature_backbone,
                                         projection_prototype,
                                         trainloaders_zipped,
                                         opt,
                                         crops_for_assign=[0, 1],
                                         temperature=0.1,
                                         epochs=epochs
                                         )
    return epoch_wise_loss, models


def visualize_training(epoch_wise_loss):
    plt.plot(epoch_wise_loss)
    plt.show()


def save_models(models, feature_path,
                projection_path):
    # Serialize the models
    feature_backbone, projection_prototype = models
    feature_backbone.save_weights(feature_path)
    projection_prototype.save_weights(projection_path)
