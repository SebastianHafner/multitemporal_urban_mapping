import abc
import torch
from utils import metrics


class AbstractMeasurer(abc.ABC):
    def __init__(self, threshold: float = 0.5, name: str = None):

        self.threshold = threshold
        self.name = name

        # urban mapping
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

        # temporal consistency
        self.unsup_tc_values, self.sup_tc_values, self.sup_tc_urban_values = [], [], []

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError("add_sample method must be implemented in the subclass.")

    def _update_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, attr_name: str):
        y = y.bool()
        y_hat = y_hat > self.threshold

        tp_attr = f'TP_{attr_name}'
        tn_attr = f'TN_{attr_name}'
        fp_attr = f'FP_{attr_name}'
        fn_attr = f'FN_{attr_name}'

        setattr(self, tp_attr, getattr(self, tp_attr) + torch.sum(y & y_hat).float().item())
        setattr(self, tn_attr, getattr(self, tn_attr) + torch.sum(~y & ~y_hat).float().item())
        setattr(self, fp_attr, getattr(self, fp_attr) + torch.sum(y_hat & ~y).float().item())
        setattr(self, fn_attr, getattr(self, fn_attr) + torch.sum(~y_hat & y).float().item())

    def _update_temporal_consistency(self, y_seg: torch.Tensor, y_hat_seg: torch.Tensor):
        y_seg = y_seg.bool()
        y_hat_seg = y_hat_seg > self.threshold
        for b in range(y_hat_seg.size(0)):
            self.unsup_tc_values.append(metrics.unsupervised_tc(y_hat_seg[b]))
            self.sup_tc_values.append(metrics.supervised_tc(y_seg[b], y_hat_seg[b]))
            self.sup_tc_urban_values.append(metrics.supervised_tc_urban(y_seg[b], y_hat_seg[b]))

    @staticmethod
    def _cont_ch_from_seg(y_seg: torch.Tensor, y_hat_seg: torch.Tensor) -> tuple:
        B, T, C, H, W = y_seg.size()
        y_ch = torch.empty((B, T - 1, C, H, W), dtype=torch.bool)
        y_hat_ch = torch.empty((B, T - 1, C, H, W), dtype=torch.bool)
        for t in range(T - 1):
            y_ch[:, t] = torch.ne(y_seg[:, t + 1], y_seg[:, t])
            y_hat_ch[:, t] = torch.ne(y_hat_seg[:, t + 1], y_hat_seg[:, t])
        return y_ch, y_hat_ch

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

    # def is_empty(self):
    #     raise NotImplementedError("is_empty method must be implemented in the subclass.")


class MappingMeasurer(AbstractMeasurer):
    def __init__(self, threshold: float = 0.5, name: str = None):
        super().__init__(threshold, name)

    def add_sample(self, y_seg: torch.Tensor, y_hat_seg: torch.Tensor):

        # urban mapping
        self._update_metrics(y_seg, y_hat_seg, 'seg_cont')
        self._update_metrics(y_seg[:, [0, -1]], y_hat_seg[:, 0, -1], 'seg_fl')

        # urban change
        y_ch_cont, y_hat_ch_cont = self._cont_ch_from_seg(y_seg, y_hat_seg)
        self._update_metrics(y_ch_cont, y_hat_ch_cont, 'ch_cont')

        y_ch_fl = torch.ne(y_seg[:, -1], y_seg[:, 0])
        y_hat_ch_fl = torch.ne(y_hat_seg[:, -1], y_hat_seg[:, 0])
        self._update_metrics(y_ch_fl, y_hat_ch_fl, 'ch_fl')

        self._update_temporal_consistency(y_seg, y_hat_seg)


class ChangeMeasurer(AbstractMeasurer):
    def __init__(self, threshold: float = 0.5, name: str = None):
        super().__init__(threshold, name)

    def add_sample(self, y_ch: torch.Tensor, y_hat_ch: torch.Tensor, change_method: str):

        if change_method == 'bitemporal':
            # continuous change
            self._update_metrics(y_ch, y_hat_ch, 'ch_cont')

            # first last change derived from continuous change
            y_ch_fl = torch.sum(y_ch, dim=1) > 0
            y_hat_ch_fl = torch.sum(y_hat_ch, dim=1) > 0
            self._update_metrics(y_ch_fl, y_hat_ch_fl, 'ch_fl')

        elif change_method == 'timeseries':
            y_ch_fl = torch.sum(y_ch, dim=1) > 0
            self._update_metrics(y_ch_fl, y_hat_ch, 'ch_fl')

        else:
            raise Exception('Unknown change method!')


class MultiTaskMeasurerLimited(AbstractMeasurer):
    def __init__(self, threshold: float = 0.5, name: str = None):
        super().__init__(threshold, name)

    def add_sample(self, y_seg: torch.Tensor, y_hat_seg: torch.Tensor, y_hat_ch_fl: torch.Tensor):

        # urban mapping first last
        self._update_metrics(y_seg[:, [0, -1]], y_hat_seg[:, 0, -1], 'seg_fl')

        # urban change first last change
        y_ch_fl = torch.ne(y_seg[:, -1], y_seg[:, 0])
        self._update_metrics(y_ch_fl, y_hat_ch_fl, 'ch_fl')


class MultiTaskMeasurer(AbstractMeasurer):
    def __init__(self, threshold: float = 0.5, name: str = None):
        super().__init__(threshold, name)

    def add_sample(self, y_seg: torch.Tensor, y_hat_seg: torch.Tensor, y_ch: torch.Tensor, y_hat_ch: torch.Tensor):

        # urban mapping
        self._update_metrics(y_seg, y_hat_seg, 'seg_cont')
        self._update_metrics(y_seg[:, [0, -1]], y_hat_seg[:, 0, -1], 'seg_fl')

        # urban change
        self._update_metrics(y_ch[:, :-1], y_hat_ch[:, :-1], 'ch_cont')
        self._update_metrics(y_ch[:, -1], y_hat_ch[:, -1], 'ch_fl')

        # temporal consistency
        self._update_temporal_consistency(y_seg, y_hat_seg)
