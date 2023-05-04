import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets
import numpy as np


class TCMeasurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold
        self.unsupervised_tc_values = []
        self.supervised_tc_values = []
        self.supervised_tc_urban_values = []

        self.eps = 10e-05

    def add_sample(self, y: torch.Tensor, y_hat: torch.Tensor):
        B, T, _, H, W = y.size()

        y = y.bool()
        y_hat = y_hat > self.threshold

        for b in range(B):
            unsup_tc = self.unsupervised_tc(y_hat[b])
            self.unsupervised_tc_values.append(unsup_tc.cpu().item())
            sup_tc = self.supervised_tc(y[b], y_hat[b])
            self.supervised_tc_values.append(sup_tc.cpu().item())
            sup_tc_urban = self.supervised_tc_urban(y[b], y_hat[b])
            self.supervised_tc_urban_values.append(sup_tc_urban.cpu().item())

    def _iou(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.tensor:
        tp = torch.sum(y & y_hat).float()
        fp = torch.sum(y_hat & ~y).float()
        fn = torch.sum(~y_hat & y).float()
        return tp / (tp + fp + fn + self.eps)

    # https://ieeexplore.ieee.org/document/9150870
    def unsupervised_tc(self, y_hat: torch.Tensor) -> torch.tensor:
        # y_hat (T, C, H, W)
        T = y_hat.size(0)
        sum_tc = 0
        for t in range(1, T):
            sum_tc += self._iou(y_hat[t - 1], y_hat[t])
        return (1 / (T - 1)) * sum_tc

    def supervised_tc(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.tensor:
        # y and y_hat (T, C, H, W)
        T = y.size()[0]
        consistency = torch.empty((T - 1))
        for t in range(1, T):
            diff_y_hat = y_hat[t - 1] != y_hat[t]
            diff_y = y[t - 1] != y[t]
            # inconsistencies: predictions of two consecutive timestamps disagree and ground truth is consistent
            inconsistencies = diff_y_hat & torch.logical_not(diff_y)
            # ratio of inconsistencies to consistent pixels in the ground truth
            consistency[t - 1] = 1 - torch.sum(inconsistencies) / torch.sum(torch.logical_not(diff_y))
        return torch.mean(consistency)

    def supervised_tc_urban(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.tensor:
        # y and y_hat (T, C, H, W)
        T = y.size(0)
        consistency_urban = torch.empty((T - 1))
        for t in range(1, T):
            diff_urban_y_hat = (y_hat[t - 1] == 1) != (y_hat[t] == 1)
            diff_urban_y = (y[t - 1] == 1) != (y[t] == 1)
            inconsistencies_urban = diff_urban_y_hat & torch.logical_not(diff_urban_y)
            consistency_urban[t - 1] = 1 - torch.sum(inconsistencies_urban) / torch.sum(torch.logical_not(diff_urban_y))
        return torch.mean(consistency_urban)

    def is_empty(self):
        assert(len(self.unsupervised_tc_values) == len(self.supervised_tc_values))
        return True if len(self.supervised_tc_values) == 0 else False

    def reset(self):
        self.unsupervised_tc_values = []
        self.supervised_tc_values = []
        self.supervised_tc_urban_values = []


class Measurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self._precision = None
        self._recall = None

        self.eps = 10e-05

    def add_sample(self, y: torch.Tensor, y_hat: torch.Tensor):
        y = y.bool()
        y_hat = y_hat > self.threshold

        self.TP += torch.sum(y & y_hat).float()
        self.TN += torch.sum(~y & ~y_hat).float()
        self.FP += torch.sum(y_hat & ~y).float()
        self.FN += torch.sum(~y_hat & y).float()

    def precision(self):
        if self._precision is None:
            self._precision = self.TP / (self.TP + self.FP + self.eps)
        return self._precision

    def recall(self):
        if self._recall is None:
            self._recall = self.TP / (self.TP + self.FN + self.eps)
        return self._recall

    def compute_basic_metrics(self):
        false_pos_rate = self.FP / (self.FP + self.TN + self.eps)
        false_neg_rate = self.FN / (self.FN + self.TP + self.eps)
        return false_pos_rate, false_neg_rate

    def f1(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall() + self.eps)

    def iou(self):
        return self.TP / (self.TP + self.FP + self.FN + self.eps)

    def oa(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN + self.eps)

    def is_empty(self):
        return True if (self.TP + self.TN + self.FP + self.FN) == 0 else False


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    measurer = Measurer()
    measurer_tc = TCMeasurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            logits = net(x)
            logits[:, 1:] if cfg.MODEL.TYPE == 'change' else logits

        y = item['y'].to(device)
        y_hat = torch.sigmoid(logits)

        measurer.add_sample(y, y_hat.detach())
        measurer_tc.add_sample(y, y_hat.detach())

    f1 = measurer.f1()
    sup_tc = np.mean(measurer_tc.supervised_tc_values)
    sup_tc_urban = np.mean(measurer_tc.supervised_tc_urban_values)
    unsup_tc = np.mean(measurer_tc.unsupervised_tc_values)

    wandb.log({
        f'{run_type} f1': f1,
        f'{run_type} unsup_tc': unsup_tc,
        f'{run_type} sup_tc': sup_tc,
        f'{run_type} sup_tc_urban': sup_tc_urban,
        'step': step, 'epoch': epoch,
    })

    return f1

