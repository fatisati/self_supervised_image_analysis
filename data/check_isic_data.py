import pandas as pd
import zipfile
from io import BytesIO
from PIL import Image
import imghdr
import pandas as pd

data_path = '../../data/ISIC/'


# derm_img_ids = pd.read_csv(data_path + 'labels.csv')['image']
def read_zip(src_path, dst_path):
    imgzip = open(src_path, 'rb')
    z = zipfile.ZipFile(imgzip)

    cnt = 0
    avg_height = 0
    avg_width = 0

    return list(z.namelist())


task3_img_ids = pd.read_csv(data_path +
                            'ISIC2018_Task3_Training_GroundTruth.csv')['image']
import os

derm_img_ids = list(os.listdir(data_path + 'ISIC_task1_resized_input'))

for i in range(len(derm_img_ids)):
    derm_img_ids[i]  = derm_img_ids[i][:-4]
same_ids = set(derm_img_ids).intersection(set(task3_img_ids))

print(derm_img_ids[:10], task3_img_ids[:10])
print(len(same_ids), len(set(derm_img_ids)))
