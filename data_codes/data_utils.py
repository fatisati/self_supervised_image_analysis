import zipfile
from io import BytesIO
from PIL import Image


def get_name(zip_file_name):
    return zip_file_name.split('/')[-1]


def is_image(filename):
    if filename[-3:] == 'jpg':
        return True
    return False


def read_zip_resize(src_path, dst_path):
    imgzip = open(src_path, 'rb')
    z = zipfile.ZipFile(imgzip)

    all_files = list(z.namelist())
    print(len(all_files), ' files are in zip folder')

    cnt = 0
    for file in z.namelist():
        if is_image(file):

            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)

            data = z.read(file)

            dataEnc = BytesIO(data)
            img = Image.open(dataEnc)
            # height, width = img.size
            img = img.resize((256, 256))
            img.save(dst_path + get_name(file))

if __name__ == '__main__':
    read_zip_resize('../../data/ISIC/ham10000/ISIC2018_Task3_Training_Input.zip',
                    '../../data/ISIC/ham10000/resized256/')
