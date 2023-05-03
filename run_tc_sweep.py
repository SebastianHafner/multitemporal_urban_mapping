import torch
from torch import optim
from torch.utils import data as torch_data

import timeit

import wandb
import numpy as np

from utils import datasets, loss_factory, evaluation, experiment_manager, parsers
from models import model_factory

# https://github.com/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb
if __name__ == '__main__':
    args = parsers.sweep_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Runnning on device: p', device)


    def run_training(sweep_cfg=None):

        with wandb.init(config=sweep_cfg):
            sweep_cfg = wandb.config

            # make training deterministic
            torch.manual_seed(cfg.SEED)
            np.random.seed(cfg.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            net = model_factory.create_network(cfg)
            net.to(device)
            optimizer = optim.AdamW(net.parameters(), lr=sweep_cfg.lr, weight_decay=0.01)

            criterion_seg = loss_factory.get_criterion(cfg.MODEL.LOSS_TYPE)
            criterion_tc = loss_factory.get_criterion(sweep_cfg.cons_loss_type)

            # reset the generators
            dataset = datasets.TrainDataset(cfg=cfg, run_type='train')
            print(dataset)

            dataloader_kwargs = {
                'batch_size': sweep_cfg.batch_size,
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
                loss_set, loss_seg_set, loss_tc_set = [], [], []

                for i, batch in enumerate(dataloader):
                    net.train()
                    optimizer.zero_grad()

                    x = batch['x'].to(device)
                    logits = net(x)

                    y = batch['y'].to(device)
                    loss_seg = criterion_seg(logits, y)
                    loss_tc = sweep_cfg.cons_lambda * criterion_tc(logits, y)

                    loss = loss_seg + loss_tc

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
                            'loss_seg': np.mean(loss_seg_set),
                            'loss_tc': np.mean(loss_tc_set),
                            'time': time,
                            'step': global_step,
                            'epoch': epoch_float,
                        })
                        start = timeit.default_timer()
                        loss_set, loss_seg_set, loss_tc_set = [], [], []
                    # end of batch

                assert (epoch == epoch_float)
                print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
                # evaluation at the end of an epoch
                _ = evaluation.model_evaluation(net, cfg, device, 'train', epoch_float, global_step)
                f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

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
            _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)


    if args.sweep_id is None:
        # Step 2: Define sweep config
        sweep_config = {
            'method': 'grid',
            'name': cfg.NAME,
            'metric': {'goal': 'maximize', 'name': 'best val f1'},
            'parameters':
                {
                    'lr': {'values': [0.0001, 0.00005, 0.00001]},
                    'batch_size': {'values': [4]},
                    'cons_loss_type': {'values': ['ConsLoss', 'UnsupConsLoss']},
                    'cons_lambda': {'values': [0.01, 0.1, 1, 10, 100]}

                }
        }

        # Step 3: Initialize sweep by passing in config or resume sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project, entity='population_mapping')
        # Step 4: Call to `wandb.agent` to start a sweep
        wandb.agent(sweep_id, function=run_training)
    else:
        # Or resume existing sweep via its id
        # https://github.com/wandb/wandb/issues/1501
        sweep_id = args.sweep_id
        wandb.agent(sweep_id, project=args.wandb_project, function=run_training)