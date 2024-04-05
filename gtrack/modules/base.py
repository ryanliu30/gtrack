# 3rd party imports
from lightning.pytorch.core import LightningModule
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from gtrack.utils import TracksDataset, collate_fn
from typing import Dict, Any, Optional
from abc import ABC
from abc import abstractmethod

class BaseModule(ABC, LightningModule):
    def __init__(
            self, 
            batch_size: int,
            warmup: Optional[int] = 0,
            lr: Optional[float] = 1e-3,
            patience: Optional[int] = 10,
            factor: Optional[float] = 1,
            dataset_args: Optional[Dict[str, Any]] = {},
        ):
        super().__init__()
        """
        Initialise the Lightning Module
        """
        self.save_hyperparameters()
    
    def _get_dataloader(self):
        dataset = TracksDataset(
            **self.hparams["dataset_args"]
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            collate_fn=collate_fn,
            num_workers=32,
        )
    
    def train_dataloader(self):
        return self._get_dataloader()

    def val_dataloader(self):
        return self._get_dataloader()

    def test_dataloader(self):
        return self._get_dataloader()
    
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"]
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    
    
    @abstractmethod
    def predict(self, x, mask):
        raise NotImplementedError("implement anomaly detection method!")
    
    def training_step(self, batch, batch_idx):
        x, mask, y, _ = batch
        predictions = self.predict(x, mask)
        loss = F.binary_cross_entropy_with_logits(predictions, y)
        self.log("training_loss", loss, on_step=True)
        return loss
    
    def shared_evaluation(self, batch, batch_idx, log=False):
        x, mask, y, _ = batch
        predictions = self.predict(x, mask)
        loss = F.binary_cross_entropy_with_logits(predictions, y)
        scores = torch.sigmoid(predictions).cpu().numpy()
        y = y.cpu().numpy()
        roc_score = roc_auc_score(y, scores)
        accuracy = (y == (scores >= 0.5)).sum() / len(y)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        self.log("validation_auc", roc_score, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become 
        built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()