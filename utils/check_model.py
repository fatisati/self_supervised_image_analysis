import barlow.inception_v3 as inception_v3
from datasets.ham_dataset import HAMDataset

from barlow.data_utils import prepare_data_loader

def check_inception():
    normalizied_inception = inception_v3.get_network()
    inception = inception_v3.load_inception()
    print('models loaded')
    ds = HAMDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                    image_col='image', image_folder='resized256/', data_size=100)
    x_train, x_test = ds.get_x_train_test_ds()
    ssl_ds = prepare_data_loader(x_train, 299, 128, False)
    data = next(iter(ssl_ds))
    ds_one, ds_two = data
    normalized_out = normalizied_inception.predict(ds_one)
    out = inception.predict(ds_one)

    print('prediction done')
    print(f'normalized min:{normalized_out.min()}, max: {normalized_out.min()}')
    print(f'inception out min: {out.min()}, max: {out.max()}')

