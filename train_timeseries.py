import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg: experiment_manager.CfgNode):
    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

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

    lstm_states_prev = None
    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            # => TimeStep, BatchSize, ...
            x = batch['x'].to(device).transpose(0, 1)
            T = x.shape[0]

            lstm_states = None
            logits = []
            if net.module.is_lstm_net():
                for t in range(T):
                    if t != 0:
                        lstm_states = lstm_states_prev
                    logits_t, lstm_states_prev = net(x[t].unsqueeze(0), lstm_states)
                    logits.append(logits_t)
            else:
                assert(T == 1)
                logits_t = net(x)
                logits.append(logits_t)

            y = batch['y'].to(device).transpose(0, 1)
            logits = torch.concat(logits, dim=0)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        _ = evaluation.model_evaluation(net, cfg, device, 'train', epoch_float, global_step)
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

        if f1_val > best_f1_val:
            best_f1_val = f1_val
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, cfg)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= cfg.TRAINER.PATIENCE:
                stop_training = True

        if stop_training:
            break

    net, *_ = networks.load_checkpoint(cfg, device)
    _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)


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
