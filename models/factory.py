import torch
import torch.nn as nn
from pathlib import Path
from utils.experiment_manager import CfgNode
from models import unet, lunet, transformers, segformer, segmenter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_network(cfg):
    if cfg.MODEL.TYPE == 'unet':
        net = unet.UNet(cfg)
    elif cfg.MODEL.TYPE == 'lunet':
        net = lunet.LUNet(cfg)
    elif cfg.MODEL.TYPE == 'transformer':
        net = transformers.TransformerModel(cfg)
    elif cfg.MODEL.TYPE == 'segmenter':
        net = segmenter.Segmenter(cfg)
    elif cfg.MODEL.TYPE == 'segformer':
        net = segformer.SegFormer(cfg)
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

