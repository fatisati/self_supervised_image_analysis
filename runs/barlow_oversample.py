from my_dataset import *
from barlow import barlow_model
from model_eval import *

model_path = '../../models/twins/'
barlow_pretrain_path = 'pretrain_imgsize{0}bs{1}e{2}proj_dim{3}'
barlow_fine_tune_path = 'fine_tune_e{0}bs{1}'

barlow_pretrain_epochs = [2]  # [25, 50]
barlow_batch_sizes = [2]  # [16, 32, 64]
barlow_crop_to = [32]  # [128, 256]
barlow_project_dim = [2048]  # [2048, 1024]

barlow_fine_tune_epochs = [2]  # [25, 50, 100]
barlow_fine_tune_bs = [3]  # [32, 64, 128]

if __name__ == '__main__':

    ds = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', balanced=True, data_size=15)

    res = open(model_path + 'results.txt', 'w')
    for epoch in barlow_pretrain_epochs:
        for bs in barlow_batch_sizes:
            for crop_to in barlow_crop_to:
                for project_dim in barlow_project_dim:

                    pretrain_params = barlow_model.PretrainParams(bs, epoch, project_dim, crop_to, model_path)
                    pretrain_path = model_path + barlow_pretrain_path.format(crop_to, bs, epoch, project_dim)
                    #
                    # try:
                    #     barlow_model.run_pretrain_barlow(ds, pretrain_path, pretrain_params)
                    #
                    # except Exception as e:
                    #     print('err pretrain: epoch: {0}, batch size: {1}, image size: {2}, projection dim: {3}'.format(
                    #         epoch, bs,
                    #         crop_to,
                    #         project_dim))
                    #     print('error: ', e)
                    #     print('------------')

                    for fine_tune_epoch in barlow_fine_tune_epochs:
                        for fine_tune_bs in barlow_fine_tune_bs:
                            try:
                                fine_tune_params = barlow_model.FineTuneParams(fine_tune_bs, fine_tune_epoch, crop_to, model_path)
                                fine_tune_path = model_path + barlow_fine_tune_path.format(fine_tune_epoch, fine_tune_bs)
                                model, history = barlow_model.run_fine_tune(ds, pretrain_path, fine_tune_path, fine_tune_params)

                                res.write(pretrain_params.get_report() + '\n')
                                res.write(fine_tune_params.get_report()+'\n')
                                eval_model(model, ds, res.write, fine_tune_bs)
                                res.write('-------------------------------')


                            except Exception as e:
                                print(
                                    'fine tune err: epoch: {0}, batch size: {1}'.format(fine_tune_epoch, fine_tune_bs))
                                print('error: ', e)
                                print('------------')

    res.close()
