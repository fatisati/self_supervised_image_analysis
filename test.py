import pandas as pd


if __name__ == '__main__':
    data_folder = '../data/'
    isic_folder = data_folder + 'ISIC/2020/'
    df = pd.read_csv(isic_folder + 'stratify-split.csv')
    df['image_name'] = [img + '.jpg' for img in df['image_name']]
    df.to_csv(isic_folder + 'stratify-split.csv')
