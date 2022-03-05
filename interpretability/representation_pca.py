from datasets.ham_dataset import HAMDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from visualization.vis_utils import VisUtils
from barlow.data_utils import prepare_input_data


def vis_using_pca(rep, y):
    pca = PCA()
    print('fit pca...')
    trans_rep = pca.fit_transform(rep)
    print('done')
    plt.scatter(trans_rep[:, 0], trans_rep[:, 1], c=y)


def vis_model_out(model, ds: HAMDataset, save_path):
    vis = VisUtils(save_path)

    bs, ct = 64, 128

    y_train, y_test = ds.train_labels, ds.test_labels
    y_train = [np.argmax(row) for row in y_train]
    y_test = [np.argmax(row) for row in y_test]

    print('preparing input data...')
    x_train, x_test = ds.get_x_train_test_ds()
    x_train = prepare_input_data(x_train, ct, bs)
    x_test = prepare_input_data(x_test, ct, bs)
    print('done')

    print('predicting representation')
    train_rep = model.predict(x_train)
    test_rep = model.predict(x_test)
    print('done')

    vis_using_pca(train_rep, y_train)
    plt.title('train')
    vis.save_and_show('train_pca')

    vis_using_pca(test_rep, y_test)
    plt.title('test')
    vis.save_and_show('test_pca')
