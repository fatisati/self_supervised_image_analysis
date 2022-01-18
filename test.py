import pandas as pd

if __name__ == '__main__':

    df = pd.read_excel('../data/razi/all_disease.xlsx', index_col=0).fillna('')
    is_automate_id = []
    disease_ids = []
    for index, row in df.iterrows():
        if row['disease ID'] == '':
            disease_ids.append(row['disease'])
            is_automate_id.append(True)
        else:
            disease_ids.append(row['disease ID'])
            is_automate_id.append(False)
    df['disease ID'] = disease_ids
    df['is automate id'] = is_automate_id
    df.to_excel('../data/razi/disease_ids.xlsx')
