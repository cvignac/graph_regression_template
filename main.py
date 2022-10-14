import os

import torch
import hydra
import numpy as np
import torch_geometric as tg
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch_geometric.datasets import QM9
from hydra.utils import get_original_cwd
import lightning_module
import utils

np.random.seed(0)
torch.manual_seed(0)


@hydra.main(version_base='1.3', config_path='.', config_name='config')
def main(cfg: DictConfig):
    utils.setup_wandb(cfg)

    dataset = QM9(root=os.path.join(get_original_cwd(), 'data'))
    split_datasets = utils.split_dataset(dataset, train_set_size=cfg.train_set_size, val_set_size=cfg.val_set_size)
    lightning_datamodule = tg.data.LightningDataset(train_dataset=split_datasets['train'],
                                                    val_dataset=split_datasets['val'],
                                                    test_dataset=split_datasets['test'],
                                                    batch_size=cfg.batch_size)

    callbacks = []
    if cfg.save_model:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"checkpoints/{cfg.name}",
                                                           filename='{epoch}',
                                                           monitor='val/epoch_mse',
                                                           save_top_k=1,
                                                           mode='min',
                                                           every_n_epochs=1)
        last_ckpt_save = pl.callbacks.ModelCheckpoint(dirpath=f"checkpoints/{cfg.name}",
                                                      filename='last',
                                                      every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(last_ckpt_save)

    use_gpu = torch.cuda.is_available() and cfg.general.gpus > 0
    trainer = pl.Trainer(gradient_clip_val=cfg.gradient_clipping,
                         accelerator='gpu' if use_gpu else 'cpu',
                         devices=1 if use_gpu else None,
                         max_epochs=cfg.n_epochs,
                         check_val_every_n_epoch=cfg.check_val_every_n_epochs,
                         limit_train_batches=cfg.limit_batches,
                         limit_val_batches=cfg.limit_batches,
                         fast_dev_run=cfg.debug,
                         enable_progress_bar=True,
                         log_every_n_steps=cfg.log_every_steps,
                         callbacks=callbacks)

    model = lightning_module.GraphAttentionModule(cfg)

    trainer.fit(model, datamodule=lightning_datamodule)
    # trainer.test(model, datamodule=lightning_datamodule)


if __name__ == '__main__':
    main()
