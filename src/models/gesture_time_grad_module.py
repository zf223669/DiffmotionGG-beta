from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from src import utils
import hydra
import omegaconf
import pyrootutils

log = utils.get_pylogger(__name__)


class GestureTimeGradLightingModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            train_net: torch.nn.Module,
            prediction_net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["train_net"])

        self.train_net = train_net
        self.prediction_net = prediction_net

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.train_acc = Accuracy()
        # self.val_acc = Accuracy()
        # self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        return self.train_net(x, cond)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def train_step(self, batch: Any):
        x = batch["x"]
        cond = batch["cond"]
        likelihoods, mean_loss = self.forward(x, cond)
        return likelihoods, mean_loss

    def training_step(self, batch: Any, batch_idx: int):
        # log.info(f"-------------------training_step----------------")
        likelihoods, loss = self.train_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"train mean_loss: {loss}")
        return loss
        # return {"likelihoods": likelihoods, "mean_loss": mean_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # self.train_acc.reset()
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        likelihoods, loss = self.train_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"val mean_loss: {loss}")
        return {"loss": loss, "likelihoods": likelihoods}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        autoreg_all = batch["autoreg"]
        control_all = batch["control"]
        trainer = self.trainer
        output = self.prediction_net.forward(autoreg_all, control_all, trainer)

    def test_epoch_end(self, outputs: List[Any]):
        # self.test_acc.reset()
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gesture_diffusion_lightningmodule.yaml")
    _ = hydra.utils.instantiate(cfg)
