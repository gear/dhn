import os
import yaml
import argparse

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from dhn.models import DHN
from dhn.datasets import HomDataLoader, HomDataset
from dhn.utils import *  # Functions start with `get`


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a DHN model on benchmark datasets'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='default.yaml',
        help='Path to train config file'
    )
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config 


def train_one_epoch(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    logger,
    log_step,
    scheduler = None,
    fold = None,
    device = 'cpu',
    **kwargs
    ):
    model.train() 
    local_log_step = 0
    for gdata in dataloader:
        gdata = gdata.to(device)
        optimizer.zero_grad()
        outputs = model(gdata)
        loss = criterion(outputs, gdata.y)
        loss.backward()
        optimizer.step() 
        logger.add_scalar(f'loss/train/{fold}', loss.item(), log_step+local_log_step)
        local_log_step += 1
    if scheduler:
        scheduler.step()
    return log_step + local_log_step


def eval(
    model, 
    dataloader,
    logger, 
    log_step,
    fold = None,
    device = 'cpu',
    **kwargs
    ):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        total = 0 
        correct = 0
        for gdata in dataloader:
            gdata = gdata.to(device)
            outputs = model(gdata)
            predicted = outputs.argmax(1)
            correct += (predicted == gdata.y).sum() 
            total += len(gdata.y)
        accuracy = correct / total
        logger.add_scalar(f'acc/val/{fold}', accuracy, log_step)


def main():
    args = parse_args() 
    config = load_config(args.config)
    logdir = os.path.join(
        config['logging']['path'], 
        config['logging']['experiment']
    )
    logger = SummaryWriter(log_dir=logdir)
    log_step = 0

    device = config['device']

    # Data setup
    dataset = HomDataset(
        name = config['data']['dataset'],
        root_path = config['data']['root_path']
    )
    indices = None
    if config['data']['cross_validation']:
        labels = [dataset[i].y.item() for i in range(len(dataset))]
        k_folds_indices = StratifiedKFold(
            n_splits=10, 
            random_state=config['seed'], 
            shuffle=True
        )
        indices = [*k_folds_indices.split(labels, labels)]
    else:
        train_path = os.path.join(
            config['data']['root_path'],
            config['data']['train_data_path'] 
        )
        val_path = os.path.join(
            config['data']['root_path'],
            config['data']['val_data_path'] 
        )
        tri = np.fromfile(train_path, sep=' ').astype(int)
        vai = np.fromfile(val_path, sep=' ').astype(int)
        indices = [(tri, vai)]

    # Main training loop 
    for fold, (tr_indices, val_indicies) in enumerate(indices):
        train_loader = HomDataLoader(
            [dataset[tri] for tri in tr_indices],
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        val_loader = HomDataLoader(
            [dataset[vai] for vai in val_indicies],
            batch_size=config['training']['batch_size'],
            shuffle=True
        )

        # Model
        model = DHN(
            out_dim = config['model']['out_dim'],
            layers_config = config['model']['layers_config'],
            act_module = get_act_module(config['model']['activation']['name']),
            agg = config['model']['agg'],
            **config['model']['activation']['kwargs']
        ).to(device)

        # Training setup
        criterion_fn = get_criterion(config['training']['loss']['name'])
        criterion = criterion_fn(**config['training']['loss']['kwargs'])
        optimizer_fn = get_optimizer(config['training']['optimizer']['name'])
        optimizer = optimizer_fn(
            params=model.parameters(), 
            **config['training']['optimizer']['kwargs']
            )
        scheduler = None
        if config['training']['lr_scheduling']['name']:
            scheduler_fn = get_lr_scheduler(config['training']['lr_scheduling']['name'])
            scheduler = scheduler_fn(optimizer, **config['training']['lr_scheduling']['kwargs'])

        # Single fold training loop
        for e in tqdm(range(1, config['training']['epochs']+1)):
            log_step = train_one_epoch(
                model = model,
                dataloader = train_loader,
                criterion = criterion,
                optimizer = optimizer,
                logger = logger, 
                log_step = log_step,
                scheduler = scheduler,
                fold = fold,
                device = device
            )
            eval(
                model, 
                dataloader = val_loader,
                logger = logger, 
                log_step = log_step,
                fold = fold,
                device = device
            )


if __name__ == "__main__":
    main()