import torch
from utils import metrics


class Measurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban mapping
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

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
        self.TP_seg_cont += torch.sum(y & y_hat).float()
        self.TN_seg_cont += torch.sum(~y & ~y_hat).float()
        self.FP_seg_cont += torch.sum(y_hat & ~y).float()
        self.FN_seg_cont += torch.sum(~y_hat & y).float()
        # first last
        self.TP_seg_cont += torch.sum(y[:, [0, -1]] & y_hat[:, [0, -1]]).float()
        self.TN_seg_cont += torch.sum(~y[:, [0, -1]] & ~y_hat[:, [0, -1]]).float()
        self.FP_seg_cont += torch.sum(y_hat[:, [0, -1]] & ~y[:, [0, -1]]).float()
        self.FN_seg_cont += torch.sum(~y_hat[:, [0, -1]] & y[:, [0, -1]]).float()

        # urban change
        # continuous change
        for t in range(1, T):
            y_ch = ~torch.eq(y[:, t], y[:, t - 1])
            y_hat_ch = ~torch.eq(y_hat[:, t], y_hat[:, t - 1])
            self.TP_ch_cont += torch.sum(y_ch & y_hat_ch).float()
            self.TN_ch_cont += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_ch_cont += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_ch_cont += torch.sum(~y_hat_ch & y_ch).float()
        # first last change
        y_ch = ~torch.eq(y[:, -1], y[:, 0])
        y_hat_ch = ~torch.eq(y_hat[:, -1], y_hat[:, 0])
        self.TP_ch_fl += torch.sum(y_ch & y_hat_ch).float()
        self.TN_ch_fl += torch.sum(~y_ch & ~y_hat_ch).float()
        self.FP_ch_fl += torch.sum(y_hat_ch & ~y_ch).float()
        self.FN_ch_fl += torch.sum(~y_hat_ch & y_ch).float()

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
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

        # temporal consistency
        self.unsup_tc_values = []
        self.sup_tc_values = []
        self.sup_tc_urban_values = []


class ChangeMeasurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

    def add_sample(self, y_ch: torch.Tensor, y_hat_ch: torch.Tensor, change_method: str):
        B, T, _, H, W = y_ch.size()

        y_ch = y_ch.bool()
        y_hat_ch = y_hat_ch > self.threshold

        if change_method == 'bitemporal':
            # continuous change
            self.TP_ch_cont += torch.sum(y_ch & y_hat_ch).float()
            self.TN_ch_cont += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_ch_cont += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_ch_cont += torch.sum(~y_hat_ch & y_ch).float()

            # first last change derived from continuous change
            y_ch = torch.sum(y_ch, dim=1) > 0
            y_hat_ch = torch.sum(y_hat_ch, dim=1) > 0
            self.TP_ch_fl += torch.sum(y_ch & y_hat_ch).float()
            self.TN_ch_fl += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_ch_fl += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_ch_fl += torch.sum(~y_hat_ch & y_ch).float()
        elif change_method == 'timeseries':
            y_ch = torch.sum(y_ch, dim=1) > 0
            self.TP_ch_fl += torch.sum(y_ch & y_hat_ch).float()
            self.TN_ch_fl += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_ch_fl += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_ch_fl += torch.sum(~y_hat_ch & y_ch).float()
        else:
            raise Exception('Unknown change method!')

    def reset(self):
        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0


class MultiTaskLUNetMeasurer(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban mapping
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | flch -> first to last change
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

    def add_sample(self, y: torch.Tensor, y_hat_ch: torch.Tensor, y_hat_seg: torch.Tensor):
        B, T, _, H, W = y.size()

        y = y.bool()
        y_hat_ch = y_hat_ch > self.threshold
        y_hat_seg = y_hat_seg > self.threshold

        # urban mapping first last
        self.TP_seg_fl += torch.sum(y[:, [0, -1]] & y_hat_seg[:, [0, -1]]).float()
        self.TN_seg_fl += torch.sum(~y[:, [0, -1]] & ~y_hat_seg[:, [0, -1]]).float()
        self.FP_seg_fl += torch.sum(y_hat_seg[:, [0, -1]] & ~y[:, [0, -1]]).float()
        self.FN_seg_fl += torch.sum(~y_hat_seg[:, [0, -1]] & y[:, [0, -1]]).float()

        # urban change first last change
        y_ch = ~torch.eq(y[:, -1], y[:, 0])
        self.TP_ch_fl += torch.sum(y_ch & y_hat_ch).float()
        self.TN_ch_fl += torch.sum(~y_ch & ~y_hat_ch).float()
        self.FP_ch_fl += torch.sum(y_hat_ch & ~y_ch).float()
        self.FN_ch_fl += torch.sum(~y_hat_ch & y_ch).float()

    def is_empty(self):
        return True if self.TP_seg_fl == 0 else False

    def reset(self):
        # urban mapping
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | flch -> first to last change
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0


class MeasurerProposed(object):
    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

        # urban mapping
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

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
        self.TP_seg_cont += torch.sum(y & y_hat).float()
        self.TN_seg_cont += torch.sum(~y & ~y_hat).float()
        self.FP_seg_cont += torch.sum(y_hat & ~y).float()
        self.FN_seg_cont += torch.sum(~y_hat & y).float()
        # first last
        self.TP_seg_cont += torch.sum(y[:, [0, -1]] & y_hat[:, [0, -1]]).float()
        self.TN_seg_cont += torch.sum(~y[:, [0, -1]] & ~y_hat[:, [0, -1]]).float()
        self.FP_seg_cont += torch.sum(y_hat[:, [0, -1]] & ~y[:, [0, -1]]).float()
        self.FN_seg_cont += torch.sum(~y_hat[:, [0, -1]] & y[:, [0, -1]]).float()

        # urban change
        # continuous change
        for t in range(1, T):
            y_ch = ~torch.eq(y[:, t], y[:, t - 1])
            y_hat_ch = ~torch.eq(y_hat[:, t], y_hat[:, t - 1])
            self.TP_ch_cont += torch.sum(y_ch & y_hat_ch).float()
            self.TN_ch_cont += torch.sum(~y_ch & ~y_hat_ch).float()
            self.FP_ch_cont += torch.sum(y_hat_ch & ~y_ch).float()
            self.FN_ch_cont += torch.sum(~y_hat_ch & y_ch).float()
        # first last change
        y_ch = ~torch.eq(y[:, -1], y[:, 0])
        y_hat_ch = ~torch.eq(y_hat[:, -1], y_hat[:, 0])
        self.TP_ch_fl += torch.sum(y_ch & y_hat_ch).float()
        self.TN_ch_fl += torch.sum(~y_ch & ~y_hat_ch).float()
        self.FP_ch_fl += torch.sum(y_hat_ch & ~y_ch).float()
        self.FN_ch_fl += torch.sum(~y_hat_ch & y_ch).float()

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
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

        # temporal consistency
        self.unsup_tc_values = []
        self.sup_tc_values = []
        self.sup_tc_urban_values = []