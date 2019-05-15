import torch
from torch import nn
import torch.nn.functional as F


class MixSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(**kwargs)
        self.ignore_label = ignore_label

    def forward(self, preds, target):
        return dict(loss=F.cross_entropy(preds, target, ignore_index=self.ignore_label))


# TODO: add aux support
class OHEMSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.6, min_kept=256,
                 down_ratio=1, reduction='mean', use_weight=False):
        super(OHEMSoftmaxCrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def base_forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept < num_valid and num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

    def forward(self, preds, target):
        for i, pred in enumerate(preds):
            if i == 0:
                loss = self.base_forward(pred, target)
            else:
                loss = loss + self.base_forward(pred, target)
        return dict(loss=loss)
