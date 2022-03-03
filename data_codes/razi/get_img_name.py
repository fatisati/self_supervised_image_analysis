import pandas as pd


def get_name(img_url):
    idx = img_url.find('/')
    return img_url[idx + 1:]


if __name__ == '__main__':
    data_folder = '../../../data/'
    razi_folder = data_folder + 'razi/'
    razi_df = pd.read_excel(razi_folder + 'supervised_samples.xlsx', index_col=0)
    razi_df['img_name'] = [get_name(url) for url in razi_df['img_url']]
    razi_df.to_excel(razi_folder + 'supervised_samples.xlsx')
