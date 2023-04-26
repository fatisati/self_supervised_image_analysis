Analysis of dermatological images using self-supervised methods

to train models there are two steps:
1. pretraining: this step inputs unlabeled images. use ssl_ds model in datasets folder to generate dataset for thsi phase of training. to pretrain model use barlow/barlow_pretrain methods.
2. fine-tuning: in this step algorithm needs labeled data and pretrained model. use dataset/supervised_ds to generate dataset for this phase and train model using barlow_finetune file in barlow folder. 

Evaluation: use codes in visualization folder to generatesome plot for model evaluations.
