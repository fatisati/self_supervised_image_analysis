import pandas as pd


def try_lower(str_):
    try:
        return str_.lower()
    except:
        return str_


def lower_all_cols(path, name):
    df = pd.read_excel(path + name + '.xlsx')
    for col in df.columns:
        df[col] = [try_lower(item) for item in df[col]]
    df.to_excel(path + f'{name}_lower.xlsx')


def analyse_label_group(group_labels, all_samples):
    group_dis = all_samples[all_samples['label'].isin(group_labels)]

    res = []
    for label in group_labels:
        label_rows = group_dis[group_dis['label'] == label]
        sample_cnt = len(label_rows)
        img_cnt = label_rows['valid_cnt'].sum()

        res.append({'label': label, 'sample_cnt': sample_cnt, 'img_cnt': img_cnt,
                    'disease_set': set(label_rows['disease'])})
    return pd.DataFrame(res)


not_found = set()


def find_disease_label(disease, all_disease):
    label = all_disease[all_disease['disease'] == disease]['label']
    if len(label) == 0:
        print(disease)
        print(label)
        not_found.add(disease)
        print('--------------------')
        return -1
    return label.iloc[0]


def label_all_samples(data_folder):
    all_dis = pd.read_excel(data_folder + 'all_disease.xlsx')
    all_samples = pd.read_excel(data_folder + 'all_samples.xlsx')
    all_samples['label'] = [find_disease_label(disease, all_dis) for disease in all_samples['disease']]
    all_samples.to_excel(data_folder + 'all_samples_labeled.xlsx')


def analyse_other(label, all_samples):
    label_samples = all_samples[all_samples['label'] == label]
    disease_set = set(label_samples['disease'])
    res = []
    for dis in disease_set:
        disease_arr = label_samples[label_samples['disease'] == dis]
        img_cnt = disease_arr['valid_cnt'].sum()
        res.append({'disease': dis, 'sample_cnt': len(disease_arr), 'img_cnt': img_cnt})
    return pd.DataFrame(res)


if __name__ == '__main__':
    data_folder = '../../../data/razi/'
    label_all_samples(data_folder)

    label_set = pd.read_excel(data_folder + 'label_set.xlsx')
    all_samples = pd.read_excel(data_folder + 'all_samples_labeled.xlsx')

    for group in label_set.columns:
        res_df = analyse_label_group(list(label_set[group]), all_samples)
        res_df.to_excel(data_folder + f'reports/{group}.xlsx')

    # unk = analyse_other('unk', all_samples)
    # other = analyse_other('other', all_samples)
    # unk.to_excel(data_folder + f'reports/unk.xlsx')
    # other.to_excel(data_folder + f'reports/other.xlsx')
