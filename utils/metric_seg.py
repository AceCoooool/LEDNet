"""Evaluation Metrics for Semantic Segmentation"""
import torch
from utils.metric import EvalMetric

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union']


class SegmentationMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__('pixAcc & mIoU')
        self.nclass = nclass
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """

        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)
            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            for (label, pred) in zip(labels, preds):
                evaluate_worker(self, label, pred)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

    def get_value(self):
        return {'total_inter': self.total_inter, 'total_union': self.total_union,
                'total_correct': self.total_correct, 'total_label': self.total_label}

    def combine_value(self, values):
        if self.total_inter.is_cuda:
            device = torch.device('cuda')
            self.total_inter += values['total_inter'].to(device)
            self.total_union += values['total_union'].to(device)
        else:
            self.total_inter += values['total_inter']
            self.total_union += values['total_union']
        self.total_correct += values['total_correct']
        self.total_label += values['total_label']

    # def combine_metric(self, metric):
    #     if self.total_inter.is_cuda:
    #         metric.total_inter = metric.total_inter.to(self.total_inter.device)
    #         self.total_inter += metric.total_inter
    #         metric.total_union = metric.total_union.to(self.total_union.device)
    #         self.total_union += metric.total_union
    #     else:
    #         self.total_inter += metric.total_inter
    #         self.total_union += metric.total_union
    #     self.total_correct += metric.total_correct
    #     self.total_label += metric.total_label


def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    predict = torch.argmax(output.long(), 1) + 1

    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict, bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, \
        "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


if __name__ == '__main__':
    a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]]).cuda()
    b = torch.LongTensor([[1, 3], [3, 4]]).cuda()
    metric = SegmentationMetric(4)
    metric.update(a, b)
    print(metric.get())
