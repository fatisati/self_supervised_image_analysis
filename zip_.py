import os
import zipfile


def copy_img_from_zip(zip_file, dest_path, names):
    for member in names:
        zip_file.extract(member, dest_path)


def load_img_from_zip(zip_path, zip_subfolder, dest_folder,
                      img_name, load_func):
    zip_file = zipfile.ZipFile(zip_path)
    zip_file.extract(zip_subfolder + img_name, dest_folder + img_name)
    img = load_func(dest_folder + img_name)
    os.remove(dest_folder + img_name)
    return img
