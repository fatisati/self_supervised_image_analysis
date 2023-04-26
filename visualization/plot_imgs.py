import matplotlib.pyplot as plt
import pandas as pd
import random


def plot_img_row(fig, imgs, rows, cols, group, cnt):

    has_label = False
    for img in imgs:

        fig.add_subplot(rows, cols, cnt)

        plt.imshow(img)
        if not has_label:
            plt.ylabel(group[0].upper() + group[1:])
            has_label = True
        plt.xticks([])
        plt.yticks([])

        cnt += 1
    return cnt


def save_image(save_path):
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path)


def razi_samples(i):
    data_folder = '../../data/'
    res_path = '../../results/data-plots/razi/samples/'
    data = pd.read_csv(data_folder + 'razi/razi-stratify-split.csv')
    groups = ['hair', 'nail', 'tumor', 'inflimatory']

    fig = plt.figure(figsize=(8, 8))
    rows, cols = 4, 4
    cnt = 1
    for group in groups:
        all = data[data['group'] == group]
        group_names = list(all['img_name'])
        samples = random.sample(group_names, 4)
        imgs = [plt.imread(data_folder + '/razi/imgs/' + name) for name in samples]
        cnt = plot_img_row(fig, imgs, rows, cols, group, cnt)
    save_image(res_path + f'sample{i}.png')


def ham_samples(i):
    data_folder = '../../data/'
    save_path = '../../results/data-plots/ham-samples/'
    data = pd.read_csv(data_folder + 'ISIC/ham10000/HAM10000_metadata.csv')
    label_set = set(list(data['dx']))

    fig = plt.figure(figsize=(4, 9))
    rows, cols = len(label_set), 3
    cnt = 1
    for label in label_set:

        all_names = list(data[data['dx'] == label]['image_id'])
        samples = random.sample(all_names, cols)
        imgs = [plt.imread(data_folder + f'ISIC/ham10000/resized256/{name}.jpg') for name in samples]
        cnt = plot_img_row(fig, imgs, rows, cols, label, cnt)
    save_image(save_path + f'sample{i}.png')


if __name__ == '__main__':

    for i in range(10):
        ham_samples(i)
