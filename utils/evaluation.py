import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets
import numpy as np

EPS = 10e-05


class Measurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban mapping
        self.TP_sem = self.TN_sem = self.FP_sem = self.FN_sem = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_cch = self.TN_cch = self.FP_cch = self.FN_cch = 0
        self.TP_flch = self.TN_flch = self.FP_flch = self.FN_flch = 0

        # temporal consistency
        self.unsup_tc_values = []
        self.sup_tc_values = []
        self.sup_tc_urban_values = []

    def add_sample(self, y: torch.Tensor, y_hat: torch.Tensor):
        B, T, _, H, W = y.size()

        y = y.bool()
        y_hat = y_hat > self.threshold

        # urban mapping
        self.TP_sem += torch.sum(y & y_hat).float()
        self.TN_sem += torch.sum(~y & ~y_hat).float()
        self.FP_sem += torch.sum(y_hat & ~y).float()
        self.FN_sem += torch.sum(~y_hat & y).float()

        # urban change
        # continuous change
        for t in range(1, T):
            y_ch = ~torch.eq(y[:, t], y[:, t - 1])
            y_hat_ch = ~torch.eq(y_hat[:, t], y_hat[:, t - 1])
            self.TP_cch += torch.sum(y_ch & y_hat_ch).float()
            self.TN_cch += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_cch += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_cch += torch.sum(~y_hat_ch & y_ch).float()
        # first last change
        y_ch = ~torch.eq(y[:, -1], y[:, 0])
        y_hat_ch = ~torch.eq(y_hat[:, -1], y_hat[:, 0])
        self.TP_flch += torch.sum(y_ch & y_hat_ch).float()
        self.TN_flch += torch.sum(~y_ch & ~y_hat_ch).float()
        self.FP_flch += torch.sum(y_hat_ch & ~y_ch).float()
        self.FN_flch += torch.sum(~y_hat_ch & y_ch).float()

        # temporal consistency
        unsup_tc = unsupervised_tc(y_hat)
        self.unsup_tc_values.append(unsup_tc)
        sup_tc = supervised_tc(y, y_hat)
        self.sup_tc_values.append(sup_tc)
        sup_tc_urban = supervised_tc_urban(y, y_hat)
        self.sup_tc_urban_values.append(sup_tc_urban)

    def is_empty(self):
        assert(len(self.unsup_tc_values) == len(self.sup_tc_values))
        return True if len(self.sup_tc_values) == 0 else False

    def reset(self):
        # urban mapping
        self.TP_sem = self.TN_sem = self.FP_sem = self.FN_sem = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_cch = self.TN_cch = self.FP_cch = self.FN_cch = 0
        self.TP_flch = self.TN_flch = self.FP_flch = self.FN_flch = 0

        # temporal consistency
        self.unsup_tc_values = []
        self.sup_tc_values = []
        self.sup_tc_urban_values = []


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    m = Measurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            logits = net(x)
            logits[:, 1:] if cfg.MODEL.TYPE == 'change' else logits

        y = item['y'].to(device)
        y_hat = torch.sigmoid(logits)

        m.add_sample(y, y_hat.detach())

    f1_sem = f1_score(m.TP_sem, m.FP_sem, m.FN_sem)

    wandb.log({
        f'{run_type} f1': f1_sem,
        f'{run_type} f1 cch': f1_score(m.TP_cch, m.FP_cch, m.FN_cch),
        f'{run_type} f1 flch': f1_score(m.TP_flch, m.FP_flch, m.FN_flch),
        f'{run_type} unsup_tc': np.mean(m.unsup_tc_values),
        f'{run_type} sup_tc': np.mean(m.sup_tc_values),
        f'{run_type} sup_tc_urban': np.mean(m.sup_tc_urban_values),
        'step': step, 'epoch': epoch,
    })

    return f1_sem


# just a bunch of metrics
def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp + EPS)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn + EPS)


def rates(tp: int, fp: int, fn: int, tn: int) -> tuple:
    false_pos_rate = fp / (fp + tn + EPS)
    false_neg_rate = fn / (fn + tp + EPS)
    return false_pos_rate, false_neg_rate


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * p * r) / (p + r + EPS)


def iou(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fp + fn + EPS)


def oa(tp: int, fp: int, fn: int, tn: int) -> float:
    return (tp + tn) / (tp + tn + fp + fn + EPS)


def iou_tensors(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    tp = torch.sum(y & y_hat).float()
    fp = torch.sum(y_hat & ~y).float()
    fn = torch.sum(~y_hat & y).float()
    return tp / (tp + fp + fn + EPS)


# https://ieeexplore.ieee.org/document/9150870
def unsupervised_tc(y_hat: torch.Tensor) -> float:
    # y_hat (B, T, C, H, W)
    T = y_hat.size(1)
    sum_tc = 0
    for t in range(1, T):
        sum_tc += iou_tensors(y_hat[:, t - 1], y_hat[:, t]).cpu().item()
    return (1 / (T - 1)) * sum_tc


def supervised_tc(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    # y and y_hat (B, T, C, H, W)
    T = y.size(1)
    consistency = torch.empty((T - 1))
    for t in range(1, T):
        diff_y_hat = y_hat[:, t - 1] != y_hat[:, t]
        diff_y = y[:, t - 1] != y[:, t]
        # inconsistencies: predictions of two consecutive timestamps disagree and ground truth is consistent
        inconsistencies = diff_y_hat & torch.logical_not(diff_y)
        # ratio of inconsistencies to consistent pixels in the ground truth
        consistency[t - 1] = 1 - torch.sum(inconsistencies) / torch.sum(torch.logical_not(diff_y))
    return torch.mean(consistency).cpu().item()


# TODO: double check this one
def supervised_tc_urban(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    # y and y_hat (B, T, C, H, W)
    T = y.size(1)
    consistency_urban = torch.empty((T - 1))
    for t in range(1, T):
        diff_urban_y_hat = (y_hat[:, t - 1] == 1) != (y_hat[:, t] == 1)
        diff_urban_y = (y[:, t - 1] == 1) != (y[:, t] == 1)
        inconsistencies_urban = diff_urban_y_hat & torch.logical_not(diff_urban_y)
        consistency_urban[t - 1] = 1 - torch.sum(inconsistencies_urban) / torch.sum(torch.logical_not(diff_urban_y))
    return torch.mean(consistency_urban).cpu().item()


if __name__ == '__main__':
    torch.manual_seed(0)
    size = (2, 5, 1, 16, 16)
    y = torch.randint(high=2, size=size)
    y_hat = torch.rand(size)
    m = Measurer()
    m.add_sample(y, y_hat)

    print(np.mean(m.sup_tc_values))
    print(np.mean(m.sup_tc_urban_values))
    print(np.mean(m.unsup_tc_values))

