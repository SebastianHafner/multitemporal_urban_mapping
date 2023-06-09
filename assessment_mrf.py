import torch
from torch.utils import data as torch_data

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip
from pathlib import Path

from utils import datasets, parsers, experiment_manager, measurers
from utils.experiment_manager import CfgNode
from models import model_factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mrf_unetouttransformer_v2(cfg: CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    net, *_ = model_factory.load_checkpoint(cfg, device)
    net.eval()

    m_vanilla = measurers.MultiTaskLUNetMeasurer()
    m_mrf = measurers.MeasurerMRF()

    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    for step, item in enumerate(dataloader):
        x, y = item['x'].to(device), item['y'].to(device)
        B, T, _, H, W = x.size()

        with torch.no_grad():
            logits_ch, logits_seg = net(x)

        y_hat_ch, y_hat_seg = torch.sigmoid(logits_ch).detach(), torch.sigmoid(logits_seg).detach()
        m_vanilla.add_sample(y, y_hat_ch, y_hat_seg)

        for b in range(B):
            for i in range(H):
                for j in range(W):
                    # Create a Markov Model (MRF)
                    model = MarkovNetwork()

                    # Add nodes and node values
                    for t in range(T):
                        model.add_node(f'N{t}')
                        # Values for node: P(urban=True), P(urban=False)
                        urban_value = float(y_hat_seg[b, t, 0, i, j])
                        factor = DiscreteFactor([f'N{t}'], cardinality=[2], values=[1 - urban_value, urban_value])
                        model.add_factors(factor)

                    # add edges and edge values
                    for t in range(T - 1):
                        model.add_edge(f'N{t}', f'N{t+1}')
                        change_value = float(y_hat_ch[b, t, 0, i, j])
                        # [P(A=False, B=False), P(A=False, B=True), P(A=True, B=False), P(A=True, B=True)]
                        factor = DiscreteFactor([f'N{t}', f'N{t+1}'], cardinality=[2, 2],
                                                values=[1 - change_value, change_value, 0, 1 - change_value])
                        model.add_factors(factor)


                    # Create an instance of BeliefPropagation algorithm
                    bp = BeliefPropagation(model)

                    # Compute the most probable state of the MRF
                    state = bp.map_query()

                    print(state)


if __name__ == '__main__':
    args = parsers.deployement_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    mrf_unetouttransformer_v2(cfg)
