import torch
import torch.nn as nn
from pathlib import Path
from utils.experiment_manager import CfgNode
from models import unet, segformer, segmenter, unetouttransformer, unetformer, change_baseline_models, changeformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_network(cfg):
    if cfg.MODEL.TYPE == 'unet':
        net = unet.UNet(cfg)
    elif cfg.MODEL.TYPE == 'segmenter':
        net = segmenter.Segmenter(cfg)
    elif cfg.MODEL.TYPE == 'segformer':
        net = segformer.SegFormer(cfg)
    elif cfg.MODEL.TYPE == 'changeformer':
        net = changeformer.ChangeFormerV6(cfg)
    elif cfg.MODEL.TYPE == 'unetouttransformer':
        net = unetouttransformer.UNetOutTransformer(cfg)
    elif cfg.MODEL.TYPE == 'unetouttransformermultitask':
        net = unetouttransformer.MultiTaskUNetOutTransformer(cfg)
    elif cfg.MODEL.TYPE == 'unetouttransformerv4':
        net = unetouttransformer.UNetOutTransformerV4(cfg)
    elif cfg.MODEL.TYPE == 'unetouttransformerv5':
        net = unetouttransformer.UNetOutTransformerV5(cfg)
    elif cfg.MODEL.TYPE == 'unetouttransformerv6':
        net = unetouttransformer.UNetOutTransformerV6(cfg)
    elif cfg.MODEL.TYPE == 'unetformer':
        net = unetformer.UNetFormer(cfg)
    elif cfg.MODEL.TYPE == 'unetformermultitask':
        net = unetformer.MultiTaskUNetFormer(cfg)
    elif cfg.MODEL.TYPE == 'siamdiffunet':
        net = change_baseline_models.SiamDiffUNet(cfg)
    elif cfg.MODEL.TYPE == 'lunet':
        net = change_baseline_models.LUNet(cfg)
    elif cfg.MODEL.TYPE == 'mtlunet':
        net = change_baseline_models.MultiTaskLUNet(cfg)
    elif cfg.MODEL.TYPE == 'changeformer':
        net = changeformer.ChangeFormerV6(cfg)
    else:
        raise Exception(f'Unknown network ({cfg.MODEL.TYPE}).')
    return nn.DataParallel(net)


def save_checkpoint(network, optimizer, epoch: float, cfg: CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(cfg: CfgNode, device: torch.device):
    net = create_network(cfg)
    net.to(device)

    net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'

    checkpoint = torch.load(net_file, map_location=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['epoch']

