import os
import shutil
import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

data_path = 'D:/Fatemeh/razi data/Backup/Images/Images'


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def add_image(dic, img_id, path, img_name):
    if img_id not in dic:
        dic[img_id] = {'path': path, 'names': [], 'ids': []}
    dic[img_id]['names'].append(img_name)


def count_images():
    all_images = {}
    # sample_path = os.path.join(data_path, 'sample')
    last_id = '-'
    img_id = '-'

    img_cnt = 0
    all_folders = os.listdir(data_path)
    for folder in all_folders:
        if folder[-4:] == 'xlsx':
            continue
        if folder == 'sample':
            continue
        folder_path = os.path.join(data_path, folder)
        images = os.listdir(folder_path)
        img_cnt += len(images)

        for img_name in sorted(images):

            current_id = img_name.split('_')[1]
            diff = abs(ord(current_id[-1]) - ord(last_id[-1]))

            if not (current_id[:-1] == last_id[:-1] and diff < 2):
                img_id = current_id

            add_image(all_images, img_id, folder_path, img_name)
            last_id = current_id
    print('number of images: ', img_cnt)
    print('estimated number of uniqe patients: ', len(all_images))

    # for img_id in sorted(list(all_images.keys()))[:10]:
    #     os.mkdir(sample_path + '/' + img_id)
    #     for file in all_images[img_id]['names']:
    #         src = all_images[img_id]['path'] + '/' + file
    #         shutil.copy(src, sample_path + '/' + img_id + '/' + file)


def add_dic_set(dic, k, v):
    if k not in dic:
        dic[k] = set()
    dic[k].add(v)


def check_similarity(disease, optional_set):
    for item in optional_set:
        if similar(disease.lower(), item.lower()) > 0.6:
            return True
    return False


def auto_group_labels():
    df_orig = pd.read_excel('../../data/razi/all_disease.xlsx').fillna(-1)
    disease_ids = {}
    for index, row in df_orig[df_orig['disease ID'] != -1].iterrows():
        add_dic_set(disease_ids, row['disease ID'], row['disease'])

    for index, row in df_orig[df_orig['disease ID'] == -1].iterrows():
        for id_ in disease_ids:
            if check_similarity(row['disease'], disease_ids[id_]):
                print(row['disease'],',', id_)

def change_font(size):
    plt.rcParams.update({'font.size': size})

def analyze_labels():
    all_dis = pd.read_excel('../../')

def class_count_report():
    df = pd.read_excel('./results/razi_processed_disease.xlsx').dropna()
    df = df[df['count'] > 100]
    df = df[df['disease id']!='UNK']
    # df = pd.to_numeric(df)
    all_dis = list(df['disease id'])
    cnts = list(pd.to_numeric(df['count']))

    int_ = 1
    idx = -1
    for i in range(len(all_dis)):
        if type(all_dis[i]) == type(int_):
            idx = i
            print(all_dis[i], cnts[i])
    all_dis.pop(idx)
    cnts.pop(idx)
    change_font(8)
    plt.bar(x=all_dis, height=cnts)
    # plt.xticks([])
    plt.xlabel('disease')
    plt.ylabel('count')
    plt.title('classes with more than 100 sample counts in razi data')

    plt.show()
