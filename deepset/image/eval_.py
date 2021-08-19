from data import *
from model import *


def evaluate_models(weights, get_model):
    metrics = {'acc': [], 'mae': [], 'mse': []}
    lengths = range(min_test_length, max_test_length, step_test_length)
    for l in tqdm(lengths):
        print('Evaluating at length: ', l)
        # generate test data
        img, Y, sum_Y = gen_test_data(num_test_examples, l)

        # model
        K.clear_session()
        model = get_model(img, l)

        # load weights
        for i, idx in enumerate([2, 3, 4, 6]):
            model.get_layer(index=idx).set_weights(weights[i])

        # prediction
        preds = model.predict(Y, batch_size=128, verbose=1)
        metrics['acc'].append(1.0 * np.sum(np.squeeze(np.round(preds)) == sum_Y) / len(sum_Y))
        metrics['mae'].append(1.0 * np.sum(np.abs(np.squeeze(preds) - sum_Y)) / len(sum_Y))
        metrics['mse'].append(np.dot(np.squeeze(preds) - sum_Y, np.squeeze(preds) - sum_Y) / len(sum_Y))

    return metrics
