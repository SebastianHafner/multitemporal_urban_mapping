import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, measurers, metrics
import numpy as np
import einops

EPS = 10e-05


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    m = measurers.Measurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE * 2, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            if cfg.MODEL.MAP_FROM_CHANGES:
                logits_seg, logits_ch = net(x)
                y_hat_seg = net.module.continuous_mapping_from_logits(logits_seg, logits_ch)
            else:
                logits_seg = net(x)
                y_hat_seg = torch.sigmoid(logits_seg)

        y_seg = item['y'].to(device)
        m.add_sample(y_seg, y_hat_seg.detach())

    f1_seg_cont = metrics.f1_score(m.TP_seg_cont, m.FP_seg_cont, m.FN_seg_cont)

    wandb.log({
        f'{run_type} f1': f1_seg_cont,
        f'{run_type} f1 seg cont': f1_seg_cont,
        f'{run_type} f1 seg fl': metrics.f1_score(m.TP_seg_fl, m.FP_seg_fl, m.FN_seg_fl),
        f'{run_type} f1 ch cont': metrics.f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont),
        f'{run_type} f1 ch fl': metrics.f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl),
        f'{run_type} unsup_tc': np.mean(m.unsup_tc_values),
        f'{run_type} sup_tc': np.mean(m.sup_tc_values),
        f'{run_type} sup_tc_urban': np.mean(m.sup_tc_urban_values),
        'step': step, 'epoch': epoch,
    })

    return f1_seg_cont


def model_evaluation_ch(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    m = measurers.ChangeMeasurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE * 2, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            logits_ch = net(x)
        y_hat_ch = torch.sigmoid(logits_ch)

        y_ch = item['y_ch'].to(device)
        m.add_sample(y_ch, y_hat_ch.detach(), net.module.change_method)

    if net.module.change_method == 'bitemporal':
        f1_ch_cont = metrics.f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont)
        f1_ch_fl = metrics.f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
        f1 = (f1_ch_cont + f1_ch_fl) / 2
        wandb.log({
            f'{run_type} f1': f1,
            f'{run_type} f1 ch cont': f1_ch_cont,
            f'{run_type} f1 ch fl': f1_ch_fl,
            'step': step, 'epoch': epoch,
        })
        return f1
    elif net.module.change_method == 'timeseries':
        f1_ch_fl = metrics.f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
        wandb.log({
            f'{run_type} f1 ch fl': f1_ch_fl,
            'step': step, 'epoch': epoch,
        })
        return f1_ch_fl
    else:
        raise Exception('Unkown change method!')


def model_evaluation_multitasklunet(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    m = measurers.MultiTaskLUNetMeasurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            logits_ch, logits_seg = net(x)

        y_hat_ch = torch.sigmoid(logits_ch)
        y_hat_seg = torch.sigmoid(logits_seg)

        y = item['y'].to(device)
        m.add_sample(y, y_hat_ch.detach(), y_hat_seg.detach())

    f1_ch_fl = metrics.f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
    f1_seg_fl = metrics.f1_score(m.TP_seg_fl, m.FP_seg_fl, m.FN_seg_fl)
    f1 = (f1_ch_fl + f1_seg_fl) / 2

    wandb.log({
        f'{run_type} f1': f1,
        f'{run_type} f1 ch fl': f1_ch_fl,
        f'{run_type} f1 seg fl': f1_seg_fl,
        'step': step, 'epoch': epoch,
    })

    return f1


def model_evaluation_proposed(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    m = measurers.MeasurerProposed()

    batch_size = int(cfg.TRAINER.BATCH_SIZE) * 2
    dataloader = torch_data.DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            features = net(x)

        # urban mapping
        logits_seg = net.module.outc_seg(einops.rearrange(features, 'b t f h w -> (b t) f h w'))
        logits_seg = einops.rearrange(logits_seg, '(b t) c h w -> b t c h w', b=batch_size)
        y_hat_seg = torch.sigmoid(logits_seg).detach()

        y_hat_ch = []
        for t in range(cfg.DATALOADER.TIMESERIES_LENGTH - 1):
            logits_ch = net.module.outc_ch(features[:, t + 1] - features[:, t])
            y_hat_ch.append(torch.sigmoid(logits_ch).detach())
        logits_ch = net.module.outc_ch(features[:, -1] - features[:, 0])
        y_hat_ch.append(torch.sigmoid(logits_ch).detach())
        y_hat_ch = torch.stack(y_hat_ch)

        y_seg, y_ch = item['y'].to(device), item['y'].to(device)
        m.add_sample(y_seg, y_hat_seg, y_ch, y_hat_ch)

    f1_seg_cont = metrics.f1_score(m.TP_seg_cont, m.FP_seg_cont, m.FN_seg_cont)
    f1_seg_fl = metrics.f1_score(m.TP_seg_fl, m.FP_seg_fl, m.FN_seg_fl)
    f1_ch_cont = metrics.f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont)
    f1_ch_fl = metrics.f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
    f1 = (f1_seg_cont + f1_seg_fl + f1_ch_cont + f1_ch_fl) / 4

    wandb.log({
        f'{run_type} f1': f1,
        f'{run_type} f1 seg cont': f1_seg_cont,
        f'{run_type} f1 seg fl': f1_seg_fl,
        f'{run_type} f1 ch cont': f1_ch_cont,
        f'{run_type} f1 ch fl': f1_ch_fl,
        f'{run_type} unsup_tc': np.mean(m.unsup_tc_values),
        f'{run_type} sup_tc': np.mean(m.sup_tc_values),
        f'{run_type} sup_tc_urban': np.mean(m.sup_tc_urban_values),
        'step': step, 'epoch': epoch,
    })

    return f1
