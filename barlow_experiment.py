import traceback

# from my_dataset import *
from barlow.barlow_run import *
from model_eval import *

model_path = '../models/twins/'
barlow_pretrain_path = 'pretrain_imgsize{0}bs{1}e{2}proj_dim{3}'
barlow_fine_tune_path = 'fine_tune_e{0}bs{1}'

barlow_pretrain_epochs = [5]  # [25, 50]
barlow_batch_sizes = [16]  # [16, 32, 64]
barlow_crop_to = [128]  # [128, 256]
barlow_project_dim = [2048]  # [2048, 1024]

barlow_fine_tune_epochs = [5]  # [25, 50, 100]
barlow_fine_tune_bs = [5]  # [32, 64, 128]

if __name__ == '__main__':

    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=30)
    res = open(model_path + 'results.txt', 'w')

    for epoch in barlow_pretrain_epochs:
        for bs in barlow_batch_sizes:
            for crop_to in barlow_crop_to:
                for project_dim in barlow_project_dim:

                    pretrain_params = PretrainParams(crop_to, bs, project_dim, epoch, model_path)

                    try:
                        barlow_encoder = run_pretrain(ds, pretrain_params)

                    except Exception as e:
                        print('err pretrain: epoch: {0}, batch size: {1}, image size: {2}, projection dim: {3}'.format(
                            epoch, bs,
                            crop_to,
                            project_dim))
                        print('error: ', e)
                        print('------------')

                    for fine_tune_epoch in barlow_fine_tune_epochs:
                        for fine_tune_bs in barlow_fine_tune_bs:
                            try:
                                fine_tune_params = FineTuneParams(fine_tune_epoch, fine_tune_bs, crop_to,
                                                                  pretrain_params)
                                model = run_fine_tune(ds, fine_tune_params, barlow_encoder, ds.weighted_loss)
                                res.write(pretrain_params.get_report() + '\n')
                                res.write(fine_tune_params.get_report() + '\n')
                                eval_model(model, ds, res.write, fine_tune_bs)
                                res.write('-------------------------------')

                            except Exception as e:
                                print(
                                    'fine tune err: epoch: {0}, batch size: {1}'.format(fine_tune_epoch, fine_tune_bs))
                                print('error: ', e)
                                print(traceback.print_exc())
                                print('------------')

    res.close()
