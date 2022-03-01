import matplotlib.pyplot as plt
import pandas as pd

path = f'../models/twins/pretrain/'

colors = ['blue', 'green', 'red']
for ct, color in zip([128, 64, 32], colors):
    batchnorm = pd.read_csv(path + f'batchnorm_ct{ct}_bs64_aug_tf/log.csv')
    no_batchnorm = pd.read_csv(path + f'no-batchnorm_ct{ct}_bs64_aug_tf/log.csv')

    plt.plot(batchnorm['loss'], label = f'batchnorm-ct{ct}', color=color)
    plt.plot(no_batchnorm['loss'],'r--', label = f'no-batchnorm-ct{ct}', color=color)

plt.title('comparing validation loss between models with and without batch-normalization')
plt.legend()
plt.show()
