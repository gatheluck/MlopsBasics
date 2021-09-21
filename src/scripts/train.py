import logging
import os
from typing import Any, Final, List

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from src.factory.lit.callback import SamplesVisualisationLogger
from src.factory.lit.task import Classifier

logger = logging.getLogger(__name__)


class ColaSamplesVisualisationLogger(SamplesVisualisationLogger):
    def __init__(self, datamodule: pl.LightningDataModule) -> None:
        super().__init__(datamodule)

    def _get_wrong_dataframe(
        self, batch: Any, pl_module: pl.LightningModule
    ) -> pd.DataFrame:
        labels: Final = batch["label"]
        sentences: Final = batch["sentence"]
        outputs: Final = pl_module(batch["input_ids"], batch["attention_mask"])
        preds: Final = torch.argmax(outputs.logits, 1)

        df: Final = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        return df[df["Label"] != df["Predicted"]]


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
        ColaSamplesVisualisationLogger(data),
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
    tokenizer: torch.nn.Module = instantiate(cfg.tokenizer)
    data: Final = instantiate(cfg.data, tokenizer=tokenizer)

    # task
    encoder: torch.nn.Module = instantiate(cfg.encoder, num_labels=data.num_classes)
    classifier: pl.LightningModule = Classifier(
        encoder=encoder,
        num_classes=data.num_classes,
        optimizer_cfg=cfg.optimizer,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=_get_logger(),
        callbacks=_get_callbacks(data=data),
        log_every_n_steps=cfg.log_every_n_steps,
        deterministic=cfg.deterministic,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )
    trainer.fit(classifier, data)
    wandb.finish()


if __name__ == "__main__":
    main()
