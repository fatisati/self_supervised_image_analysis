from PIL import Image
import os


class ChangeData:
    def __init__(self, image_folder, dst_path, image_size=256):
        self.image_folder = image_folder
        self.image_size = image_size
        self.dst_path = dst_path

    @staticmethod
    def read_image(image_path):
        img = Image.open(image_path)
        return img

    def resize_image(self, img):
        return img.resize((self.image_size, self.image_size))

    def save_image(self, img, name):
        img.save(self.dst_path + name)

    def resize_all_images(self):
        for name in os.listdir(self.image_folder):
            img = self.read_image(self.image_folder + name)
            resized = self.resize_image(img)
            self.save_image(resized, name)


if __name__ == '__main__':
    data_folder = '../../data/ISIC/ham10000/'
    data_tool = ChangeData(data_folder + 'resized256/',
                           data_folder + 'resized128/', 128)
    data_tool.resize_all_images()
