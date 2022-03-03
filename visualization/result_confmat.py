from barlow.barlow_run import *

from barlow.data_utils import prepare_input_data
from ham_plots_backup_ import BarlowResVis
from vis_utils.model_utils import load_model

from plot_confmat import *


if __name__ == '__main__':
    print('loading data...')
    ds = HAMDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                    image_col='image', image_folder='resized256/')

    x_train, x_test = ds.get_x_train_test_ds()
    y_train, y_test = ds.train_labels, ds.test_labels
    x_train = prepare_input_data(x_train, 128, 64)
    x_test = prepare_input_data(x_test, 128, 64)

    vis = BarlowResVis()
    labels = ds.label_names
    best_model = load_model(vis.model_path1 + 'e150')

    # plot_confusion_matrix(best_model, x_test, y_test, labels)
    # plt.title('best model confusion matrix for test data')
    # vis.save_and_show('best_confmat_test')

    plot_confusion_matrix(best_model, x_train, y_train, labels)
    plt.title('best model confusion matrix for train data')
    vis.save_and_show('best_confmat_train')

    # no_pretrain = load_model(vis.no_pretrain_path + 'e150')
    # plot_confusion_matrix(no_pretrain, x_test, y_test)
    # plt.title('no-pretrain confusion matrix')
    # vis.save_and_show('no_pretrain_confmat')
