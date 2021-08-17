from my_dataset import *
from swav.swav_model import *

model_path = '../../models/swav/'

pretrain_epochs = [25, 50]
pretrain_size_crops = [[32, 64], [128, 256]]
pretrain_batch_sizes = [32, 64, 16]

fine_tune_warmup_epochs = [25, 50, 100]
fine_tune_epochs = [25, 50, 100]
fine_tune_bs = [16, 32, 128, 256]
fine_tune_img_size = [128, 256]

if __name__ == '__main__':
    ds = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', balanced=True, data_size=20)

    for epoch in pretrain_epochs:
        for size_crops in pretrain_size_crops:
            for bs in pretrain_batch_sizes:
                pretrain_params = PretrainParams(epoch, size_crops, bs, model_path)
                run_pretrain(ds, model_path, pretrain_params)

                for warmup_epochs in fine_tune_warmup_epochs:
                    for fine_tune_epoch in fine_tune_epochs:
                        for ft_bs in fine_tune_bs:
                            for img_size in fine_tune_img_size:
                                fine_tune_params = FineTuneParams(warmup_epochs, fine_tune_epochs, model_path, ft_bs,
                                                                  img_size)

                                run_fine_tune(ds, pretrain_params, fine_tune_params)
