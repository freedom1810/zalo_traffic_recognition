from .cutmix_mixup import *

import torch
from torch import nn
import torch.nn.functional as F



class BengaliClassifier1295(nn.Module):
    def __init__(self, 
                    predictor, 
                    n_grapheme=168, 
                    n_vowel=11, 
                    n_consonant=7, 
                    cutmix_ratio=0, 
                    cutmix_bien=0):

        super(BengaliClassifier1295, self).__init__()

        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant

        self.predictor = predictor

        self.metrics_keys = ['loss', 'acc']

    def forward(self, x, y=None):

        preds = self.predictor(x)

        loss =  F.cross_entropy(preds, y)
        metrics = {
            'loss': loss.item(),
            'acc': accuracy(preds, y),
        }

        return loss, metrics, preds

    def forward_eval(self, x, y=None):

        preds = self.predictor(x)

        loss =  F.cross_entropy(preds, y)
        metrics = {
            'loss': loss.item(),
            'acc': accuracy(preds, y),
        }

        return loss, metrics, preds


    def calc(self, data_loader):
        device: torch.device = next(self.parameters()).device
        self.eval()

        output_list = []
        with torch.no_grad():
            for batch in data_loader:
                # TODO: support general preprocessing.
                # If `data` is not `Data` instance, `to` method is not supported!
                batch = batch.to(device)
                # pred = self.predictor(batch)
                # output_list.append(pred)
                pred = self.predictor(batch)
                output_list.append(pred)

            output = torch.cat(output_list, dim=0)
        return output


    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim = 0) for p in preds]