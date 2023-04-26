import matplotlib.pyplot as plt
import pandas as pd
import os


# link for 0 f1 if precision and recall both 0:
# https://towardsdatascience.com/a-look-at-precision-recall-and-f1-score-36b5fd0dd3ec#:~:text=In%20each%20case%20where%20TP,predict%20any%20correct%20positive%20result.
def f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def calc_log_f1_score(log):
    precisions = log['val_precision']
    recalls = log['val_recall']
    f1 = [f1_score(precision, recall) for precision, recall in zip(precisions, recalls)]
    return f1


def make_folder_if_not(path, folder):
    if folder not in os.listdir(path):
        os.mkdir(path + '/' + folder)
    else:
        print('folder existed.')


def multiple_plots(x_arr, y_arr, titles, rows, cols):
    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            axs[row, col].plot(x_arr[idx], y_arr[idx])
            axs[row, col].set_title(titles[idx])
            idx += 1
            if idx == len(x_arr):
                return


class VisUtils:
    def __init__(self, save_path):
        self.save_path = save_path

    def save_with_format(self, name, frm):
        plt.savefig(self.save_path + f'{name}.{frm}')

    def save_and_show(self, name):
        self.save_with_format(name, 'svg')
        self.save_with_format(name, 'png')
        plt.show()


class CompareModels:
    def __init__(self, res_folder, model_folder, experiment):
        self.experiment = experiment
        self.model_folder = model_folder
        self.res_folder = res_folder
        vis_utils = VisUtils(res_folder)
        self.save_and_show = vis_utils.save_and_show
        self.all_metrics = ['val_accuracy', 'val_loss', 'val_precision', 'val_recall',
                            'val_auc', 'val_prc', 'f1-score']

    def read_log(self, model_name):
        log = pd.read_csv(self.model_folder + model_name + '/log.csv')
        log = log[:100]
        return log

    def plot_metric(self, metric, model_names, labels, x_metric=None, plot_func=plt.plot):
        save_name = metric
        for name, label in zip(model_names, labels):
            log = self.read_log(name)

            if metric == 'f1-score':
                log['f1-score'] = calc_log_f1_score(log)

            if x_metric:
                plot_func(log[x_metric], log[metric], label=label)
                plt.xlabel(x_metric)
                save_name += '_' + x_metric
            else:
                plot_func(log[metric], label=label)
                plt.xlabel('epochs')
        plt.ylabel(metric)
        plt.title(self.experiment)
        plt.legend()
        make_folder_if_not(self.res_folder, self.experiment)
        self.save_and_show(f'{self.experiment}/{save_name}')

    def compare_all_metrics(self, models, labels):
        self.plot_metric('val_precision', models,
                         labels, x_metric='val_recall',
                         plot_func=plt.scatter)
        for metric in self.all_metrics:
            self.plot_metric(metric, models, labels)


if __name__ == '__main__':
    path = '../../results/tables/razi-models-eval.xlsx'
    log = pd.read_excel(path)
    # log['f1-score'] = calc_log_f1_score(log)
    #
    # log.to_excel(path)
    for col in log.columns:
        try:
            log[col] = [round(float(num), 2) for num in log[col]]
            print(col)
        except:
            continue
    log.to_excel(path[:-4] + '_round.xlsx')
