from razi_dataset import RaziDataset

if __name__ == '__main__':
    ds = RaziDataset('../data/razi/', 32)
    data = ds.irv2_augmented_supervised_ds(0.8, 'hair')
    print(list(next(data)))
