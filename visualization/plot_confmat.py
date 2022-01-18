import numpy as np
from visualization.confusion_matrix.cf_matrix import make_confusion_matrix


def generate_multi_label_confmat(y, y_pred):
    size = len(y[0])
    confmat = np.zeros((size, size))
    for i in range(len(y)):
        true_index = np.argmax(y[i])
        for j in range(size):
            confmat[true_index][j] += y_pred[i][j]
    return confmat


def plot_confusion_matrix(model, x, y, labels):
    print('predicting...')
    y_pred = model.predict(x)
    print('done')

    cf_matrix = generate_multi_label_confmat(y, y_pred)
    make_confusion_matrix(cf_matrix, cbar=False, categories=labels)
