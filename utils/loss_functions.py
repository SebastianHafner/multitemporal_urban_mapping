import torch
import torch.nn as nn
from torch.nn import functional as F


def get_criterion(loss_type, negative_weight: float = 1, positive_weight: float = 1):
    if loss_type == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'CrossEntropyLoss':
        balance_weight = [negative_weight, positive_weight]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        balance_weight = torch.tensor(balance_weight).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=balance_weight)
    elif loss_type == 'SoftDiceLoss':
        criterion = soft_dice_loss
    elif loss_type == 'SoftDiceSquaredSumLoss':
        criterion = soft_dice_squared_sum_loss
    elif loss_type == 'SoftDiceBalancedLoss':
        criterion = soft_dice_loss_balanced
    elif loss_type == 'PowerJaccardLoss':
        criterion = power_jaccard_loss
    elif loss_type == 'MeanSquareErrorLoss':
        criterion = nn.MSELoss()
    elif loss_type == 'IoULoss':
        criterion = iou_loss
    elif loss_type == 'DiceLikeLoss':
        criterion = dice_like_loss
    elif loss_type == 'L2':
        criterion = nn.MSELoss()
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion


def soft_dice_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


# TODO: fix this one
def soft_dice_squared_sum_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


def soft_dice_loss_multi_class(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims = (0, 2, 3)  # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom = (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def soft_dice_loss_multi_class_debug(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims = (0, 2, 3)  # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom = (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    loss_components = 1 - 2 * intersection / denom
    return loss, loss_components


def generalized_soft_dice_loss_multi_class(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-12

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims = (0, 2, 3)  # Batch, height, width
    ysum = y.sum(dim=sum_dims)
    wc = 1 / (ysum ** 2 + eps)
    intersection = ((y * p).sum(dim=sum_dims) * wc).sum()
    denom = ((ysum + p.sum(dim=sum_dims)) * wc).sum()

    loss = 1 - (2. * intersection / denom)
    return loss


def jaccard_like_loss_multi_class(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims = (0, 2, 3)  # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom = (y ** 2 + p ** 2).sum(dim=sum_dims) + (y * p).sum(dim=sum_dims) + eps

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def jaccard_like_loss(input: torch.Tensor, target: torch.Tensor, disable_sigmoid: bool = False):
    input_sigmoid = torch.sigmoid(input) if not disable_sigmoid else input
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps

    return 1 - ((2. * intersection) / denom)


def dice_like_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() + eps

    return 1 - ((2. * intersection) / denom)


def power_jaccard_loss(input: torch.Tensor, target: torch.Tensor, disable_sigmoid: bool = False):
    input_sigmoid = torch.sigmoid(input) if not disable_sigmoid else input
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps

    return 1 - (intersection / denom)


def iou_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_pred = torch.sigmoid(y_logit)
    eps = 1e-6

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum() - intersection + eps

    return 1 - (intersection / union)


def jaccard_like_balanced_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps
    piccard = (2. * intersection) / denom

    n_iflat = 1 - iflat
    n_tflat = 1 - tflat
    neg_intersection = (n_iflat * n_tflat).sum()
    neg_denom = (n_iflat ** 2 + n_tflat ** 2).sum() - (n_iflat * n_tflat).sum()
    n_piccard = (2. * neg_intersection) / neg_denom

    return 1 - piccard - n_piccard


def soft_dice_loss_balanced(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    dice_pos = ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

    negatiev_intersection = ((1 - iflat) * (1 - tflat)).sum()
    dice_neg = (2 * negatiev_intersection) / ((1 - iflat).sum() + (1 - tflat).sum() + eps)

    return 1 - dice_pos - dice_neg


import torch
import numpy as np
import torch.nn as nn
import scipy.ndimage as sp_img
# https://github.com/SebastianHafner/f2f-consistent-semantic-segmentation/blob/master/model/loss.py

def video_loss(output, target, cross_entropy_lambda, consistency_lambda, consistency_function, ignore_class):
    # output: Time, BatchSize, Channels, Height, Width
    # labels: Time, BatchSize, Height, Width
    valid_mask = (target != ignore_class)
    target_select = target.clone()
    target_select[target_select == ignore_class] = 0
    target_select = target_select[:, :, None, :, :].long()

    loss_cross_entropy = torch.tensor([0.0], dtype=torch.float32, device=output.device)
    if cross_entropy_lambda > 0:
        loss_cross_entropy = cross_entropy_lambda * cross_entropy_loss(output, target_select, valid_mask)

    loss_inconsistency = torch.tensor([0.0], dtype=torch.float32, device=output.device)
    if consistency_lambda > 0 and output.shape[0] > 1:
        loss_inconsistency = consistency_lambda * inconsistency_loss(output, target, consistency_function, valid_mask,
                                                                     target_select)

    return loss_cross_entropy, loss_inconsistency


def cross_entropy_loss(output, target_select, valid_mask):
    pixel_loss = torch.gather(output, dim=2, index=target_select).squeeze(dim=2)
    pixel_loss = - torch.log(pixel_loss.clamp(min=1e-10))  # clamp: values smaller than 1e-10 become 1e-10
    pixel_loss = pixel_loss * valid_mask.to(dtype=torch.float32)  # without ignore pixels
    total_loss = pixel_loss.sum()
    return total_loss / valid_mask.sum().to(dtype=torch.float32)  # normalize


def inconsistency_loss(output, target, consistency_function, valid_mask, target_select):
    pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
    valid_mask_sum = torch.tensor([0.0], dtype=torch.float32, device=output.device)
    inconsistencies_sum = torch.tensor([0.0], dtype=torch.float32, device=output.device)

    for t in range(output.shape[0] - 1):
        gt1 = target[t]
        gt2 = target[t + 1]
        valid_mask2 = valid_mask[t] & valid_mask[t + 1]  # valid mask always has to be calculated over 2 imgs

        if consistency_function == 'argmax_pred':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            diff_pred_valid = ((pred1 != pred2) & valid_mask2).to(output.dtype)
        elif consistency_function == 'abs_diff':
            diff_pred_valid = (torch.abs(output[t] - output[t + 1])).sum(dim=1) * valid_mask2.to(output.dtype)
        elif consistency_function == 'sq_diff':
            diff_pred_valid = (torch.pow(output[t] - output[t + 1], 2)).sum(dim=1) * valid_mask2.to(output.dtype)
        elif consistency_function == 'abs_diff_true':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            right_pred_mask = (pred1 == gt1) | (pred2 == gt2)
            diff_pred = torch.abs(output[t] - output[t + 1])
            diff_pred_true = torch.gather(diff_pred, dim=1, index=target_select[t]).squeeze(dim=1)
            diff_pred_valid = diff_pred_true * (valid_mask2 & right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'sq_diff_true':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            right_pred_mask = (pred1 == gt1) | (pred2 == gt2)
            diff_pred = torch.pow(output[t] - output[t + 1], 2)
            diff_pred_true = torch.gather(diff_pred, dim=1, index=target_select[t]).squeeze(dim=1)
            diff_pred_valid = diff_pred_true * (valid_mask2 & right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'sq_diff_true_XOR':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            right_pred_mask = (pred1 == gt1) ^ (pred2 == gt2)
            diff_pred = torch.pow(output[t] - output[t + 1], 2)
            diff_pred_true = torch.gather(diff_pred, dim=1, index=target_select[t]).squeeze(dim=1)
            diff_pred_valid = diff_pred_true * (valid_mask2 & right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'abs_diff_th20':
            th_mask = (output[t] > 0.2) & (output[t + 1] > 0.2)
            diff_pred_valid = (torch.abs((output[t] - output[t + 1]) * th_mask.to(dtype=output.dtype))).sum(
                dim=1) * valid_mask2.to(output.dtype)

        diff_gt_valid = ((gt1 != gt2) & valid_mask2)  # torch.uint8
        diff_gt_valid_dil = sp_img.binary_dilation(diff_gt_valid.cpu().numpy(),
                                                   iterations=2)  # default: 4-neighbourhood
        inconsistencies = diff_pred_valid * torch.from_numpy(np.logical_not(diff_gt_valid_dil).astype(np.uint8)).to(
            output.device, dtype=output.dtype)
        valid_mask_sum += valid_mask2.sum()
        inconsistencies_sum += inconsistencies.sum()

    return inconsistencies_sum / valid_mask_sum