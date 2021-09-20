import logging
from abc import ABC, abstractmethod
from typing import Any, Final

import pandas as pd
import pytorch_lightning as pl
import wandb

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback, ABC):
    def __init__(self, datamodule: pl.LightningDataModule) -> None:
        super().__init__()

        self.datamodule: Final = datamodule

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_batch: Final = next(iter(self.datamodule.val_dataloader()))
        wrong_df: Final = self._get_wrong_dataframe(val_batch, pl_module)

        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

    @abstractmethod
    def _get_wrong_dataframe(
        self, batch: Any, pl_module: pl.LightningModule
    ) -> pd.DataFrame:
        logger.error("please override the method _get_wrong_dataframe.")
        raise NotImplementedError
