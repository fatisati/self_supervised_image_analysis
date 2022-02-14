from barlow.resnet20 import ResNet

resnet = ResNet(use_batchnorm=False, use_dropout=False)
b_resnet = ResNet(use_batchnorm=True, use_dropout=False)


resnet.get_network(32, use_pred=False, return_before_head=False).summary()
b_resnet.get_network(32, use_pred=False, return_before_head=False).summary()
