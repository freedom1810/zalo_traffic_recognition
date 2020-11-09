import torch
from torch import nn
import torch.nn.functional as F

#sử dụng pretrain của các model còn lại
import pretrainedmodels

# sử dụng pretrain của efficientnet
from efficientnet_pytorch import EfficientNet

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        # chỉ khai báo cấu trúc model
        self.base_model = EfficientNet.from_name('efficientnet-b4')
        # sử dụng các weight của pretrain khi train vơi imagenet
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b4')

        #sử dụng pretrainedmodels
        # model_name = densenet121
        # self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)

        hdim = 1792

        self.lin = nn.Linear(hdim, 1292)

    def load_model(self, path = ''):
        self.base_model.load_state_dict(path)
    
    def freeze_model(self):
        for param in self.base_model.parameters():
                param.requires_grad = False

    def unfreeze_model(self):
        for param in self.base_model.parameters():
                param.requires_grad = True

    def forward(self, x):
        h = self.base_model.extract_features(x)

        h = F.adaptive_avg_pool2d(h, 1)
        h = h.view(h.size(0), -1)

        h = self.lin(h)

        return h
