from sklearn.metrics import accuracy_score
import sklearn.metrics as skm

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from my_dataset import MyDataset


def eval_model(model, ds: MyDataset, report_func, bs):
    x_train, x_test = ds.get_x_train_test_ds()
    report_func('train results:\n')
    eval_(model, x_train.batch(bs), ds.train_labels, report_func)

    report_func('test results:\n')
    eval_(model, x_test.batch(bs), ds.test_labels, report_func)


def eval_(model, x, y, report_func):
    y_pred = model.predict(x)
    eval_result(y_pred, y, report_func)


def eval_result(y_pred, y_true, report_func):
    y_pred = y_pred.round()
    report_func('acc: {0}'.format(accuracy_score(y_true, y_pred)))
    report_func(str(skm.classification_report(y_true, y_pred)))


def plot_conf_mat(classifier, X_test, y_test, class_names):
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
