import torch
from torch.utils import data as torch_data

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils import datasets, parsers, experiment_manager, spacenet7_helpers, visualization
from utils.experiment_manager import CfgNode
from models import model_factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_change_date(y_ch: torch.Tensor) -> torch.Tensor:
    y_ch = y_ch[:-1].type(torch.int8)  # last change is first to last
    y_ch_t = torch.argmax(y_ch, dim=0)
    no_change = torch.all(y_ch == 0, dim=0)
    y_ch_t = torch.where(no_change, torch.tensor(-1), y_ch_t)
    return torch.add(y_ch_t, 1)


def vis_label(cfg: CfgNode, run_type: str = 'test'):
    tile_size = cfg.AUGMENTATION.CROP_SIZE
    for aoi_id in spacenet7_helpers.get_aoi_ids(cfg, run_type):
        ds = datasets.EvalDataset(cfg, run_type, tiling=tile_size, aoi_id=aoi_id)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        img_t1, img_t2 = np.zeros((1024, 1024, 3), dtype=np.float32), np.zeros((1024, 1024, 3), dtype=np.float32)
        label_ch, label_seg = np.zeros((1024, 1024, 1), dtype=np.uint8), np.zeros((1024, 1024, 1), dtype=np.uint8)

        for index in range(len(ds)):
            item = ds.__getitem__(index)
            i, j = item['i'], item['j']

            x = item['x'].permute(0, 2, 3, 1)
            img_t1[i:i+tile_size, j:j+tile_size] = x[0]
            img_t2[i:i + tile_size, j:j + tile_size] = x[-1]

            y_seg, y_ch = item['y'].permute(0, 2, 3, 1), item['y_ch'].permute(0, 2, 3, 1)
            label_seg[i:i + tile_size, j:j + tile_size] = y_seg[-1]

            y_ch_t = get_change_date(y_ch)
            label_ch[i:i + tile_size, j:j + tile_size] = y_ch_t

        axs[0].imshow(img_t1)
        axs[1].imshow(img_t2)

        cmap = visualization.DateColorMap(x.size(0))
        axs[2].imshow(label_ch, cmap=cmap.get_cmap(), vmin=cmap.get_vmin(), vmax=cmap.get_vmax())

        axs[3].imshow(label_seg, cmap='gray')

        out_file = Path(cfg.PATHS.OUTPUT) / 'label' / f'y_ch_t_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')


def vis_unetouttransformer_v2(cfg: CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    net, *_ = model_factory.load_checkpoint(cfg, device)
    for aoi_id in spacenet7_helpers.get_aoi_ids(cfg, run_type):
        ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE, aoi_id=aoi_id)
        dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

        for step, item in enumerate(dataloader):
            x, y = item['x'].to(device), item['y'].to(device)
            B, T, _, H, W = x.size()

            with torch.no_grad():
                logits_ch, logits_seg = net(x)



if __name__ == '__main__':
    args = parsers.deployement_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    vis_label(cfg)
