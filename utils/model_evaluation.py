from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
from utils.confusion_matrix.cf_matrix import make_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from ham_dataset import *
from barlow.barlow_run import *

from barlow.data_utils import prepare_input_data


def generate_multi_label_confmat(y, y_pred):
    size = len(y[0])
    confmat = np.zeros((size, size))
    for i in range(len(y)):
        true_index = np.argmax(y[i])
        for j in range(size):
            confmat[true_index][j] += y_pred[i][j]
    return confmat


def plot_confusion_matrix(model, x, y):
    y_pred = model.predict(x)
    cf_matrix = generate_multi_label_confmat(y, y_pred)
    print(cf_matrix)
    make_confusion_matrix(cf_matrix, cbar=False)
    plt.show()


if __name__ == '__main__':

    ds = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/')
    x_train, x_test = ds.get_x_train_test_ds()
    y_train, y_test = ds.train_labels, ds.test_labels

    custom_objects = {"weighted_loss": ds.weighted_loss}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model('../../models/twins/finetune/ct64_bs128_loss_weighted/e100')
    print('model loaded')

    x_train = prepare_input_data(x_train, 64, 128)
    x_test = prepare_input_data(x_test, 64, 128)

    plot_confusion_matrix(model, x_train, y_train)
    plot_confusion_matrix(model, x_test, y_test)
