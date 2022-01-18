import random

import matplotlib.pyplot as plt
import pandas as pd
import plot_confmat


def generate_random_y(size):
    y = [0] * size
    r = random.randint(0, size-1)
    y[r] = 1
    return y


def generate_random_population(size):
    population = [generate_random_y(7) for i in range(size)]
    print(population)
    return population


if __name__ == '__main__':
    sample_size = 50


    y = generate_random_population(sample_size)
    y_pred = generate_random_population(sample_size)
    confmat = plot_confmat.generate_multi_label_confmat(y, y_pred)

    # data = pd.read_csv('../../data')
    data = pd.read_csv('../../data/ISIC/ham10000/disease_labels.csv')
    labels = list(data.columns[2:-1])

    plot_confmat.make_confusion_matrix(confmat, categories=labels)
    plt.show()
