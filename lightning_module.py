import os
import time

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError

import models


class GraphAttentionModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_loss = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.model = models.SparseGraphTransformer(cfg)

        self.start_epoch_time = None
        self.log_every_steps = cfg.log_every_steps
        self.best_val_loss = torch.inf

    def training_step(self, data, i):
        pred = self.forward(data)
        loss = self.train_loss(pred, data.y)
        # self.train_metrics(pred, data.y)
        if i % self.log_every_steps == 0:
            to_log = {'train_loss/batch_loss': loss.detach()}
            wandb.log(to_log, commit=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, amsgrad=True,
                                weight_decay=self.cfg.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())

    def on_train_epoch_start(self) -> None:
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        # self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        epoch_loss = self.train_loss.compute()
        print(f"Epoch {self.current_epoch}: loss {epoch_loss :.4f}-- {time.time() - self.start_epoch_time :.1f}s")
        wandb.log({'epoch/train_loss': epoch_loss})
        # self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time)
        # self.train_metrics.log_epoch_metrics(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_mse.reset()

        # self.val_metrics.reset()

    def validation_step(self, data, i):
        pred = self.forward(data)
        mse = self.val_mse(pred, data.y)
        # loss = self.compute_val_loss()
        return {'mse': mse}

    def validation_epoch_end(self, outs) -> None:
        epoch_mse = self.val_mse.compute()
        wandb.log({"val/epoch_MSE": epoch_mse}, commit=False)

        print(f"Epoch {self.current_epoch}: Val MSE {epoch_mse:.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        self.log("val/epoch_NLL", epoch_mse)

        if epoch_mse < self.best_val_loss:
            self.best_val_loss = epoch_mse
        print(f'Val loss: {epoch_mse :.2f} -- Best val loss: {self.best_val_loss :.2f}')

    def on_test_epoch_start(self) -> None:
        self.test_mse.reset()

    def test_step(self, data, i):
        pred = self.forward(data)
        mse = self.test_mse(pred, data.y)
        return {'mse': mse}

    def test_epoch_end(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        test_mse = self.test_mse.compute()
        wandb.log({"test/epoch_MSE": test_mse}, commit=False)
        print(f"Epoch {self.current_epoch}: Test MSE {test_mse :.2f}")

    def forward(self, data):
        return self.model(data)
