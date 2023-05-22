import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, measurers, metrics
import numpy as np

EPS = 10e-05


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    m = measurers.Measurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            if cfg.MODEL.MAP_FROM_CHANGES:
                logits_sem, logits_ch = net(x)
                y_hat = net.module.continuous_mapping_from_logits(logits_sem, logits_ch)
            else:
                logits = net(x)
                y_hat = torch.sigmoid(logits)

        y = item['y'].to(device)
        m.add_sample(y, y_hat.detach())

    f1_csem = metrics.f1_score(m.TP_csem, m.FP_csem, m.FN_csem)

    wandb.log({
        f'{run_type} f1': f1_csem,
        f'{run_type} f1_flsem': metrics.f1_score(m.TP_flsem, m.FP_flsem, m.FN_flsem),
        f'{run_type} f1 cch': metrics.f1_score(m.TP_cch, m.FP_cch, m.FN_cch),
        f'{run_type} f1 flch': metrics.f1_score(m.TP_flch, m.FP_flch, m.FN_flch),
        f'{run_type} unsup_tc': np.mean(m.unsup_tc_values),
        f'{run_type} sup_tc': np.mean(m.sup_tc_values),
        f'{run_type} sup_tc_urban': np.mean(m.sup_tc_urban_values),
        'step': step, 'epoch': epoch,
    })

    return f1_csem


def model_evaluation_ch(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)

    net.to(device)
    net.eval()

    m = measurers.ChangeMeasurer()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=0, shuffle=False,
                                       drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)

        with torch.no_grad():
            logits_ch = net(x)
        y_hat_ch = torch.sigmoid(logits_ch)

        y_ch = item['y_ch'].to(device)
        m.add_sample(y_ch, y_hat_ch.detach(), net.module.change_method)

    if net.module.change_method == 'bitemporal':
        f1_cch = metrics.f1_score(m.TP_cch, m.FP_cch, m.FN_cch)
        wandb.log({
            f'{run_type} f1 cch': f1_cch,
            f'{run_type} f1 flch': metrics.f1_score(m.TP_flch, m.FP_flch, m.FN_flch),
            'step': step, 'epoch': epoch,
        })
        return f1_cch
    elif net.module.change_method == 'timeseries':
        f1_flch = metrics.f1_score(m.TP_flch, m.FP_flch, m.FN_flch)
        wandb.log({
            f'{run_type} f1 flch': f1_flch,
            'step': step, 'epoch': epoch,
        })
        return f1_flch
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
            logits_ch, logits_seg1, logits_seg2 = net(x)

        y_hat_ch = torch.sigmoid(logits_ch)
        y_hat_seg1, y_hat_seg2 = torch.sigmoid(logits_seg1), torch.sigmoid(logits_seg2)

        y = item['y'].to(device)
        m.add_sample(y, y_hat_ch.detach(), y_hat_seg1.detach(), y_hat_seg2.detach())

    f1_flch = metrics.f1_score(m.TP_flch, m.FP_flch, m.FN_flch)

    wandb.log({
        f'{run_type} f1 flch': f1_flch,
        f'{run_type} f1_flsem': metrics.f1_score(m.TP_flsem, m.FP_flsem, m.FN_flsem),
        'step': step, 'epoch': epoch,
    })

    return f1_flch
