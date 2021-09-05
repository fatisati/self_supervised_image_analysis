import os
import zipfile

data_path = '../../data/ISIC/dermoscopic/'
image_path = 'resized256/'
mask_path = 'ISIC2018_Task2_Validation_GroundTruth.zip'

zip_filepath = [data_path+mask_path]  # or glob.glob('...zip')
target_dir = data_path + 'sample_mask/'


zipList = zip_filepath


for file in zipList:
    with zipfile.ZipFile(file) as zip_file:
        print(zip_file)
        for member in zip_file.namelist():
            print(member)
            if 'streaks' in member:
                print(member)
                zip_file.extract(member,target_dir)
