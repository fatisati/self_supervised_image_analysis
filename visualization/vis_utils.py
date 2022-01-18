import matplotlib.pyplot as plt


class VisUtils:
    def __init__(self, save_path):
        self.save_path = save_path

    def save_with_format(self, name, frm):
        plt.savefig(self.save_path + f'{frm}/{name}.{frm}')

    def save_and_show(self, name):
        self.save_with_format(name, 'svg')
        self.save_with_format(name, 'png')
        plt.show()
