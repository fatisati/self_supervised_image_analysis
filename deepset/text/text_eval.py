from text_params_ import *
from text_data import *


def evaluate_model(get_model, weights, layer_arr):
    metrics = {'acc': [], 'mae': [], 'mse': []}

    lengths = range(min_test_length, max_test_length, step_test_length)
    for l in lengths:
        print('Evaluating at length: ', l)
        K.clear_session()

        # generate test data
        Y, sum_Y = gen_text_test_data(num_test_examples, l)

        # model
        model = get_model(l)

        # load weights
        for i, idx in enumerate(layer_arr):
            model.get_layer(index=idx).set_weights(weights[i])

        # prediction
        preds = model.predict(Y, batch_size=128, verbose=1)
        metrics['acc'].append(1.0 * np.sum(np.squeeze(np.round(preds)) == sum_Y) / len(sum_Y))
        metrics['mae'].append(np.sum(np.abs(np.squeeze(preds) - sum_Y)) / len(sum_Y))
        metrics['mse'].append(np.dot(np.squeeze(preds) - sum_Y, np.squeeze(preds) - sum_Y) / len(sum_Y))

    return metrics
