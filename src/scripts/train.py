import logging
import os
import pathlib
from typing import Final, List

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from src.factory.lit.task import Classifier

logger = logging.getLogger(__name__)


def _get_callbacks(data: pl.LightningDataModule) -> List[pl.Callback]:
    checkpoint_callback: Final = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback: Final = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    return [
        checkpoint_callback,
        # ColaSamplesVisualisationLogger(data),
        early_stopping_callback,
    ]


def _get_logger() -> LightningLoggerBase:
    try:
        return WandbLogger(
            project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"]
        )
    except Exception:
        logger.error("cannot connect to wandb.")
        raise


@hydra.main(config_path="../config", config_name="train")
def main(cfg) -> None:
    OmegaConf.set_readonly(cfg, True)
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # data module
    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    data: Final = instantiate(cfg.data, num_workers=4, root=cwd / "data")

    # task
    num_classes: Final = data.get_num_classes()
    encoder: torch.nn.Module = instantiate(cfg.encoder, num_classes=num_classes)
    classifier: pl.LightningModule = Classifier(
        encoder=encoder,
        num_classes=num_classes,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
    )

    trainer = pl.Trainer(
        callbacks=_get_callbacks(data=data),
        deterministic=cfg.deterministic,
        gpus=1,
        logger=_get_logger(),
        log_every_n_steps=cfg.log_every_n_steps,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        max_epochs=cfg.max_epochs,
    )
    trainer.fit(classifier, data)
    wandb.finish()


if __name__ == "__main__":
    main()
