"""
@author: bochengz
@date: 2023/04/14
@email: bochengzeng@bochengz.top
"""
import config as cfg
import pprint
from kogger import Logger
import torch.multiprocessing as mp
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset import ToyDataset
from model import ToyModel
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import save_checkpoint, load_checkpoint


def train(model, data_loader, epochs, loss_func, optimizer, scheduler, checkpoint_path, dtype, local_rank, logger,):
    train_losses = []

    for ii in range(1, epochs+1):
        for batch_idx, (data, label) in enumerate(data_loader):
            # data: [b, n]
            data = torch.tensor(data, dtype=dtype, device=local_rank)
            label = torch.tensor(label, dtype=dtype, device=local_rank)

            output = model(data)
            loss = loss_func(output, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if local_rank == 0 and (ii == 1 or ii % 10 == 0):
            save_checkpoint(model, optimizer, checkpoint_path)
            logger.info('[Train {}/{}] MSE loss: {:.4e}'.format(ii, epochs, loss))

    return train_losses


def main_work(local_rank, config):
    logger = Logger(name='PID %d' % local_rank, file=config['log_file'])
    logger.info('Start')

    os.environ['MASTER_ADDR'] = config['addr']
    os.environ['MASTER_PORT'] = config['port']
    # initialize the process group
    dist.init_process_group("nccl", rank=local_rank, world_size=config['world_size'])

    torch.cuda.set_device(local_rank)

    dataset = ToyDataset(length=1000)
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=config['world_size'],
        rank=local_rank,
        shuffle=config['shuffle']
    )
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=config['shuffle'],
        sampler=sampler,
        batch_size=config['batch_size']
    )

    if local_rank == 0:
        logger.info('Build model...')

    model = ToyModel()
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # load checkpoint
    if config['continuous_train']:
        map_location = {'cuda:0': 'cuda:%d' % local_rank}
        load_checkpoint(model, optimizer, config['checkpoint_path'], map_location=map_location)

    mse_func = torch.nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['steplr_size'], gamma=config['steplr_gamma'])

    # train
    if local_rank == 0:
        logger.info('Train...')

    train_loss = train(
        model=model,
        data_loader=data_loader,
        epochs=config['epochs'],
        loss_func=mse_func,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=config['checkpoint_path'],
        dtype=config['dtype'],
        logger=logger,
        local_rank=local_rank
    )

    if local_rank == 0:
        # plot train loss
        fig = plt.figure()
        train_idx = np.arange(1, config['epochs']+1)
        # plt.plot(train_idx[int(len(train_idx)*0.2):-1], train_loss[int(len(train_idx)*0.2):-1], label='Train')  # 只绘制后0.8
        plt.semilogy(train_idx, train_loss)
        plt.title('Train Loss')
        fig.savefig(config['figs_loss_train'])
        # plt.show()

    dist.destroy_process_group()


if __name__ == '__main__':

    # load and set config
    args = cfg.get_parser().parse_args()
    config = cfg.load_config(yaml_filename=args.filename)
    config = cfg.process_config(config)

    logger = Logger(name='MAIN', file=config['log_file'])
    logger.info('Load config successfully!')
    logger.info(pprint.pformat(config))

    mp.spawn(
        fn=main_work,
        args=(config,),
        nprocs=config['world_size'],
        join=True
    )

    logger.info('Done!')
