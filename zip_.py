import os
import zipfile


def copy_img_from_zip():
    src_path = '../data/ISIC/dermoscopic/ISIC2018_Task1-2_Training_Input.zip'
    dest_path = '../data/ISIC/dermoscopic/'

    zip_file = zipfile.ZipFile(src_path)
    for member in zip_file.namelist():
        if member.endswith('.jpg'):
            print(member)
            zip_file.extract(member, dest_path)
            os.remove(dest_path + member)
            break


def load_img_from_zip(zip_path, zip_subfolder, dest_folder,
                      img_name, load_func):
    zip_file = zipfile.ZipFile(zip_path)
    zip_file.extract(zip_subfolder + img_name, dest_folder + img_name)
    img = load_func(dest_folder + img_name)
    os.remove(dest_folder + img_name)
    return img
