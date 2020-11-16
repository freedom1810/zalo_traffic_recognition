from .cutmix_mixup import *

import torch
from torch import nn
import torch.nn.functional as F


class ZaloClassifier(nn.Module):
    def __init__(self, 
                    predictor, 
                    n_grapheme=168, 
                    n_vowel=11, 
                    n_consonant=7, 
                    cutmix_ratio=0, 
                    cutmix_bien=0):
                    
        super(ZaloClassifier, self).__init__()

        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        # self.predictor = nn.DataParallel(predictor)
        self.predictor = predictor

        self.cutmix_ratio = cutmix_ratio
        self.cutmix_bien = cutmix_bien

    def forward(self, x, y=None):
        
        label1 = y[:, 0]
        label2 = y[:, 1]
        label3 = y[:, 2]

        #----------------------------
        # mixup + cutmix

        '''if np.random.uniform() < ratio_mixup:
            images, targets = mixup(x, label1, label2, label3, 0.4)
            preds = self.predictor(images)

            # output1, output2, output3 = preds
            pred_g, pred_v, pred_c = preds = preds
            loss_grapheme, loss_vowel, loss_consonant = mixup_criterion(pred_g, pred_v, pred_c, targets)
            loss = loss_grapheme + loss_vowel + loss_consonant

            lam = targets[6]
            label1 = targets[0] * lam + targets[1] * (1 - lam)
            label2 = targets[2] * lam + targets[3] * (1 - lam)
            label3 = targets[4] * lam + targets[5] * (1 - lam)

            metrics = {
                'loss': loss.item(),
                'loss_grapheme': loss_grapheme.item(),
                'loss_vowel': loss_vowel.item(),
                'loss_consonant': loss_consonant.item(),
                'acc_grapheme': accuracy(pred_g, label1),
                'acc_vowel': accuracy(pred_v, label2),
                'acc_consonant': accuracy(pred_c, label3),
            }
            '''
        if np.random.uniform() < self.cutmix_ratio:
            images, targets = cutmix(x, label1, label2, label3, self.cutmix_bien)
            preds = self.predictor(images)

            # output1, output2, output3 = preds
            pred_g, pred_v, pred_c = preds
            loss_grapheme, loss_vowel, loss_consonant = cutmix_criterion(pred_g, pred_v, pred_c, targets)
            loss = loss_grapheme + loss_vowel + loss_consonant

            lam = targets[6]
            label1 = targets[0] * lam + targets[1] * (1 - lam)
            label2 = targets[2] * lam + targets[3] * (1 - lam)
            label3 = targets[4] * lam + targets[5] * (1 - lam)

            metrics = {
                'loss': loss.item(),
                'loss_grapheme': loss_grapheme.item(),
                'loss_vowel': loss_vowel.item(),
                'loss_consonant': loss_consonant.item(),
                # 'acc_grapheme': accuracy(pred_g, label1),
                # 'acc_vowel': accuracy(pred_v, label2),
                # 'acc_consonant': accuracy(pred_c, label3),
            }
            
        #----------------------------
        else:
        
            preds = self.predictor(x)
            pred_g, pred_v, pred_c = preds

            loss_grapheme = F.cross_entropy(pred_g, y[:, 0])
            loss_vowel = F.cross_entropy(pred_v, y[:, 1])
            loss_consonant = F.cross_entropy(pred_c, y[:, 2])

            loss = loss_grapheme + loss_vowel + loss_consonant
            metrics = {
                'loss': loss.item(),
                'loss_grapheme': loss_grapheme.item(),
                'loss_vowel': loss_vowel.item(),
                'loss_consonant': loss_consonant.item(),
                # 'acc_grapheme': accuracy(pred_g, y[:, 0]),
                # 'acc_vowel': accuracy(pred_v, y[:, 1]),
                # 'acc_consonant': accuracy(pred_c, y[:, 2]),
            }

        return loss, metrics, torch.cat((pred_g, pred_v, pred_c), dim = 1)


    def forward_eval(self, x, y=None):
        
        preds = self.predictor(x)
        
        pred_g, pred_v, pred_c = preds

        loss_grapheme = F.cross_entropy(pred_g, y[:, 0])
        loss_vowel = F.cross_entropy(pred_v, y[:, 1])
        loss_consonant = F.cross_entropy(pred_c, y[:, 2])

        loss = loss_grapheme + loss_vowel + loss_consonant
        metrics = {
            'loss': loss.item(),
            'loss_grapheme': loss_grapheme.item(),
            'loss_vowel': loss_vowel.item(),
            'loss_consonant': loss_consonant.item(),
            # 'acc_grapheme': accuracy(pred_g, y[:, 0]),
            # 'acc_vowel': accuracy(pred_v, y[:, 1]),
            # 'acc_consonant': accuracy(pred_c, y[:, 2]),
        }

        return loss, metrics, torch.cat((pred_g, pred_v, pred_c), dim = 1)

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
                pred_g, pred_v, pred_c = self.predictor(batch)
                output_list.append(torch.cat((pred_g, pred_v, pred_c), dim = 1))
        output = torch.cat(output_list, dim=0)
        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds


    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim=1) for p in preds]

    def predict(self, data_loader):
        preds = self.calc(data_loader)
        pred_labels = [torch.argmax(p, dim=1) for p in preds]
        return pred_labels


class ZaloClassifier1295(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7, cutmix_ratio=0, cutmix_bien=0):
        super(ZaloClassifier1295, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor

        self.cutmix_ratio = cutmix_ratio
        self.cutmix_bien = cutmix_bien

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

def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc