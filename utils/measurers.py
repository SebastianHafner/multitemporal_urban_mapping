import torch
from utils import metrics


class Measurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban mapping
        self.TP_csem = self.TN_csem = self.FP_csem = self.FN_csem = 0
        self.TP_flsem = self.TN_flsem = self.FP_flsem = self.FN_flsem = 0

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
        # continuous
        self.TP_csem += torch.sum(y & y_hat).float()
        self.TN_csem += torch.sum(~y & ~y_hat).float()
        self.FP_csem += torch.sum(y_hat & ~y).float()
        self.FN_csem += torch.sum(~y_hat & y).float()
        # first last
        self.TP_csem += torch.sum(y[:, [0, -1]] & y_hat[:, [0, -1]]).float()
        self.TN_csem += torch.sum(~y[:, [0, -1]] & ~y_hat[:, [0, -1]]).float()
        self.FP_csem += torch.sum(y_hat[:, [0, -1]] & ~y[:, [0, -1]]).float()
        self.FN_csem += torch.sum(~y_hat[:, [0, -1]] & y[:, [0, -1]]).float()

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
        for b in range(B):
            self.unsup_tc_values.append(metrics.unsupervised_tc(y_hat[b]))
            self.sup_tc_values.append(metrics.supervised_tc(y[b], y_hat[b]))
            self.sup_tc_urban_values.append(metrics.supervised_tc_urban(y[b], y_hat[b]))

    def is_empty(self):
        assert (len(self.unsup_tc_values) == len(self.sup_tc_values))
        return True if len(self.sup_tc_values) == 0 else False

    def reset(self):
        # urban mapping
        self.TP_csem = self.TN_csem = self.FP_csem = self.FN_csem = 0
        self.TP_flsem = self.TN_flsem = self.FP_flsem = self.FN_flsem = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_cch = self.TN_cch = self.FP_cch = self.FN_cch = 0
        self.TP_flch = self.TN_flch = self.FP_flch = self.FN_flch = 0

        # temporal consistency
        self.unsup_tc_values = []
        self.sup_tc_values = []
        self.sup_tc_urban_values = []


class ChangeMeasurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_cch = self.TN_cch = self.FP_cch = self.FN_cch = 0
        self.TP_flch = self.TN_flch = self.FP_flch = self.FN_flch = 0

    def add_sample(self, y_ch: torch.Tensor, y_hat_ch: torch.Tensor, change_method: str):
        B, T, _, H, W = y_ch.size()

        y_ch = y_ch.bool()
        y_hat_ch = y_hat_ch > self.threshold

        if change_method == 'bitemporal':
            # continuous change
            self.TP_cch += torch.sum(y_ch & y_hat_ch).float()
            self.TN_cch += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_cch += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_cch += torch.sum(~y_hat_ch & y_ch).float()

            # first last change derived from continuous change
            y_ch = torch.sum(y_ch, dim=1) > 0
            y_hat_ch = torch.sum(y_hat_ch, dim=1) > 0
            self.TP_flch += torch.sum(y_ch & y_hat_ch).float()
            self.TN_flch += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_flch += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_flch += torch.sum(~y_hat_ch & y_ch).float()
        elif change_method == 'timeseries':
            y_ch = torch.sum(y_ch, dim=1) > 0
            self.TP_flch += torch.sum(y_ch & y_hat_ch).float()
            self.TN_flch += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_flch += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_flch += torch.sum(~y_hat_ch & y_ch).float()
        else:
            raise Exception('Unknown change method!')

    def reset(self):
        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_cch = self.TN_cch = self.FP_cch = self.FN_cch = 0
        self.TP_flch = self.TN_flch = self.FP_flch = self.FN_flch = 0


class MultiTaskLUNetMeasurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban mapping
        self.TP_flsem = self.TN_flsem = self.FP_flsem = self.FN_flsem = 0

        # urban change | flch -> first to last change
        self.TP_flch = self.TN_flch = self.FP_flch = self.FN_flch = 0

    def add_sample(self, y: torch.Tensor, y_hat_ch: torch.Tensor, y_hat_seg: torch.Tensor):
        B, T, _, H, W = y.size()

        y = y.bool()
        y_hat_ch = y_hat_ch > self.threshold
        y_hat_seg = y_hat_seg > self.threshold

        # urban mapping first last
        self.TP_flsem += torch.sum(y[:, [0, -1]] & y_hat_seg[:, [0, -1]]).float()
        self.TN_flsem += torch.sum(~y[:, [0, -1]] & ~y_hat_seg[:, [0, -1]]).float()
        self.FP_flsem += torch.sum(y_hat_seg[:, [0, -1]] & ~y[:, [0, -1]]).float()
        self.FN_flsem += torch.sum(~y_hat_seg[:, [0, -1]] & y[:, [0, -1]]).float()

        # urban change first last change
        y_ch = ~torch.eq(y[:, -1], y[:, 0])
        self.TP_flch += torch.sum(y_ch & y_hat_ch).float()
        self.TN_flch += torch.sum(~y_ch & ~y_hat_ch).float()
        self.FP_flch += torch.sum(y_hat_ch & ~y_ch).float()
        self.FN_flch += torch.sum(~y_hat_ch & y_ch).float()

    def is_empty(self):
        return True if self.TP_flsem == 0 else False

    def reset(self):
        # urban mapping
        self.TP_flsem = self.TN_flsem = self.FP_flsem = self.FN_flsem = 0

        # urban change | flch -> first to last change
        self.TP_flch = self.TN_flch = self.FP_flch = self.FN_flch = 0