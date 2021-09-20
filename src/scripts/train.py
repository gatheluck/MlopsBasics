import logging
from typing import Any, Final, cast

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.factory.lit.callback import SamplesVisualisationLogger

logger = logging.getLogger(__name__)


class ColaSamplesVisualisationLogger(SamplesVisualisationLogger):
    def __init__(self, datamodule: pl.LightningDataModule) -> None:
        super().__init__(datamodule)

    def _get_wrong_dataframe(
        self, batch: Any, pl_module: pl.LightningModule
    ) -> pd.DataFrame:
        labels: Final = batch["label"]
        sentences: Final = batch["sentence"]
        outputs: Final = pl_module(batch["intput_ids"], batch["attention_mask"])
        preds: Final = torch.argmax(outputs.logits, 1)

        df: Final = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        return df[df["Label"] != df["Predicted"]]


@hydra.main(config_path="../configs", config_name="train")
def main(cfg) -> None:
    OmegaConf.set_readonly(cfg, True)
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # data module
    tokenizer: torch.nn.Module = instantiate(cfg.tokenizer)
    data: Final = instantiate(cfg.data, tokenizer=tokenizer)

    encoder: torch.nn.Module = instantiate(cfg.encoder, num_labels=data.num_classes)


if __name__ == "__main__":
    main()
