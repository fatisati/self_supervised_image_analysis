import barlow.inception_v3 as inception_v3
from ham_dataset import MyDataset

if __name__ == '__main__':
    normalizied_inception = inception_v3.get_network()
    inception = inception_v3.load_inception()
    print('models loaded')
    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=100)

    normalized_out = normalizied_inception.predict(ds)
    out = inception.predict(ds)
    print('prediction done')
    print(f'normalized min:{min(normalized_out)}, max: {max(normalized_out)}')
    print(f'inception out min: {min(out)}, max: {max(out)}')
