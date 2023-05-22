import torch

EPS = 10e-05


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
    # y_hat (T, C, H, W)
    T = y_hat.size(0)
    sum_tc = 0
    for t in range(1, T):
        sum_tc += iou_tensors(y_hat[t - 1], y_hat[t]).cpu().item()
    return (1 / (T - 1)) * sum_tc


def supervised_tc(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    # y and y_hat (T, C, H, W)
    T = y.size(0)
    consistency = torch.empty((T - 1))
    for t in range(1, T):
        diff_y_hat = ~torch.eq(y_hat[t - 1], y_hat[t])
        cons_y = torch.eq(y[t - 1], y[t])
        # inconsistencies: predictions of two consecutive timestamps disagree and ground truth is consistent
        inconsistencies = diff_y_hat & cons_y
        # ratio of inconsistencies to consistent pixels in the ground truth
        consistency[t - 1] = 1 - torch.sum(inconsistencies) / torch.sum(cons_y)
    return torch.mean(consistency).cpu().item()


# TODO: double check this one
def supervised_tc_urban(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    # y and y_hat (T, C, H, W)
    T = y.size(0)
    consistency_urban = torch.empty((T - 1))
    for t in range(1, T):
        diff_urban_y_hat = (y_hat[t - 1] == 1) != (y_hat[t] == 1)
        diff_urban_y = (y[t - 1] == 1) != (y[t] == 1)
        inconsistencies_urban = diff_urban_y_hat & torch.logical_not(diff_urban_y)
        consistency_urban[t - 1] = 1 - torch.sum(inconsistencies_urban) / torch.sum(torch.logical_not(diff_urban_y))
    return torch.mean(consistency_urban).cpu().item()

