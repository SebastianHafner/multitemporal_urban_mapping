import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np
import einops
from itertools import combinations

from utils import datasets, loss_factory, evaluation, experiment_manager, parsers
from models import model_factory


def run_training(cfg: experiment_manager.CfgNode):
    net = model_factory.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_factory.get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.TrainDataset(cfg=cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_val = 0
    trigger_times = 0
    stop_training = False

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_seg_set, loss_ch_set, loss_set = [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            features = net(x)

            y = batch['y'].to(device)

            # urban mapping
            logits_seg = net.module.outc_seg(einops.rearrange(features, 'b t f h w -> (b t) f h w'))
            logits_seg = einops.rearrange(logits_seg, '(b t) c h w -> b t c h w', b=cfg.TRAINER.BATCH_SIZE)
            loss_seg = criterion(logits_seg, y)

            loss_ch = torch.tensor([0.0], dtype=torch.float32, device=device)
            timestamp_combinations = combinations(range(cfg.DATALOADER.TIMESERIES_LENGTH), 2)
            for t1, t2 in timestamp_combinations:
                logits_ch = net.module.outc_ch(features[:, t2] - features[:, t1])
                y_ch = torch.ne(y[:, t1], y[:, t2])
                loss_ch += criterion(logits_ch, y_ch)

            loss = loss_seg + loss_ch
            loss.backward()
            optimizer.step()

            loss_seg_set.append(loss_seg.item())
            loss_ch_set.append(loss_ch.item())
            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss_seg': np.mean(loss_seg_set),
                    'loss_ch': np.mean(loss_ch_set),
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_set = [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        _ = evaluation.model_evaluation_proposed(net, cfg, device, 'train', epoch_float, global_step)
        f1_val = evaluation.model_evaluation_proposed(net, cfg, device, 'val', epoch_float, global_step)

        if f1_val <= best_f1_val:
            trigger_times += 1
            if trigger_times > cfg.TRAINER.PATIENCE:
                stop_training = True
        else:
            best_f1_val = f1_val
            wandb.log({
                'best val f1': best_f1_val,
                'step': global_step,
                'epoch': epoch_float,
            })
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            model_factory.save_checkpoint(net, optimizer, epoch, cfg)
            trigger_times = 0

        if stop_training:
            break

    net, *_ = model_factory.load_checkpoint(cfg, device)
    _ = evaluation.model_evaluation_proposed(net, cfg, device, 'test', epoch_float, global_step)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        project=args.wandb_project,
        entity='population_mapping',
        tags=['urban mapping', 'multi-temporal', 'temporal consistency', 'spacenet7', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
