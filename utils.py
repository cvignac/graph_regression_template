import numpy as np
from torch_geometric.data import Dataset
import wandb
import omegaconf

def split_dataset(dataset: Dataset, train_set_size: int, val_set_size: int):
    """ test_set_size = len(dataset) - train_set_size - val_set_size. """
    train_val_test = np.arange(dataset.len())
    np.random.shuffle(train_val_test)

    train_indices = train_val_test[:train_set_size]
    val_indices = train_val_test[train_set_size: train_set_size + val_set_size]
    test_indices = train_val_test[train_set_size + val_set_size:]

    train_dataset = dataset.index_select(train_indices)
    val_dataset = dataset.index_select(val_indices)
    test_dataset = dataset.index_select(test_indices)

    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.name, 'project': cfg.project_name, 'config': config_dict, 'reinit': True, 'mode': cfg.wandb,
              'settings': wandb.Settings(_disable_stats=True)}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg
