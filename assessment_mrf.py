import torch
from torch.utils import data as torch_data

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

from utils import datasets, parsers, experiment_manager, measurers, metrics, geofiles
from utils.experiment_manager import CfgNode
from models import model_factory

from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mrf_unetouttransformer_v2(cfg: CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    net, *_ = model_factory.load_checkpoint(cfg, device)
    net.eval()

    m_vanilla = measurers.MultiTaskMeasurer(name='vanilla')
    m_mrf = measurers.MappingMeasurer(name='mrf')

    ds = datasets.EvalDataset(cfg, run_type, tiling=cfg.AUGMENTATION.CROP_SIZE, aoi_id='L15-0387E-1276N_1549_3087_13')
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    for step, item in enumerate(tqdm(dataloader)):
        x, y_seg, y_ch = item['x'].to(device), item['y'].to(device), item['y_ch'].to(device)
        B, T, _, H, W = x.size()

        with torch.no_grad():
            logits_ch, logits_seg = net(x)

        y_hat_ch, y_hat_seg = torch.sigmoid(logits_ch).detach(), torch.sigmoid(logits_seg).detach()
        m_vanilla.add_sample(y_seg, y_hat_seg, y_ch, y_hat_ch)

        y_hat_seg_mrf = torch.empty(y_hat_seg.shape, dtype=torch.float32)
        pixels = [(i, j) for i in range(H) for j in range(W)]
        for i, j in pixels:
            for b in range(B):
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
                states_list = [state[f'N{t}'] for t in range(T)]
                y_hat_seg_mrf[b, :, 0, i, j] = torch.Tensor(states_list)

        m_mrf.add_sample(y_seg.cpu(), y_hat_seg_mrf)

    data = {}
    for m in [m_vanilla, m_mrf]:
        data[m.name] = {}
        for attr in ['seg_cont', 'seg_fl', 'ch_cont', 'ch_fl']:
            f1 = metrics.f1_score(getattr(m, f'TP_{attr}'), getattr(m, f'TP_{attr}'), getattr(m, f'TN_{attr}'))
            data[m.name][attr] = f1.item()

    out_file = Path(cfg.PATHS.OUTPUT) / 'assessment' / f'{cfg.NAME}.json'
    geofiles.write_json(out_file, data)


if __name__ == '__main__':
    args = parsers.deployement_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    mrf_unetouttransformer_v2(cfg)
