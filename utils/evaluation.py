import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets


class TCMeasurer(object):
    def __init__(self, aoi_id: str, threshold: float = 0.5):

        self.aoi_id = aoi_id
        self.threshold = threshold
        self.y_hats, self.ys = [], []

        self.eps = 10e-05

    def add_sample(self, y: torch.Tensor, y_hat: torch.Tensor):
        y = y.bool()
        y_hat = y_hat > self.threshold

        self.y_hats.append(y_hat.squeeze())
        self.ys.append(y.squeeze())

    def iou(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.tensor:
        tp = torch.sum(y & y_hat).float()
        fp = torch.sum(y_hat & ~y).float()
        fn = torch.sum(~y_hat & y).float()
        return tp / (tp + fp + fn + self.eps)

    def tc(self) -> torch.tensor:
        # https://ieeexplore.ieee.org/document/9150870
        T = len(self.y_hats)
        sum_tc = 0
        for t in range(1, T):
            sum_tc += self.iou(self.ys[t], self.y_hats[t])
        return (1 / (T - 1)) * sum_tc

    def is_empty(self):
        return True if len(self.y_hats) == 0 else False


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

    ds = datasets.EvalSingleDateDataset(cfg, run_type)

    net.to(device)
    net.eval()

    measurer = Measurer()

    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            logits = net(x)
            y_hat = torch.sigmoid(logits)

        y = item['y'].to(device)
        measurer.add_sample(y, y_hat.detach())

    f1 = measurer.f1()

    wandb.log({
        f'{run_type} f1': f1,
        'step': step, 'epoch': epoch,
    })

    return f1


def model_evaluation_timeseries(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalTimeseriesDataset(cfg, run_type)

    net.to(device)
    net.eval()

    measurer = Measurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        # => TimeStep, BatchSize, ...
        x = item['x'].to(device).transpose(0, 1)
        T = x.shape[0]

        lstm_states = None
        logits = []
        with torch.no_grad():
            if net.module.is_lstm_net():
                for t in range(T):
                    if t != 0:
                        lstm_states = lstm_states_prev
                    logits_t, lstm_states_prev = net(x[t].unsqueeze(0), lstm_states)
                    logits.append(logits_t)
            else:
                assert (T == 1)
                logits_t = net(x)
                logits.append(logits_t)

        y = item['y'].to(device).transpose(0, 1)
        logits = torch.concat(logits, dim=0)
        y_hat = torch.sigmoid(logits)

        measurer.add_sample(y, y_hat.detach())

    f1 = measurer.f1()

    wandb.log({
        f'{run_type} f1': f1,
        'step': step, 'epoch': epoch,
    })

    return f1
