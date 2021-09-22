from typing import Final

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class ColaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str = "google/bert_uncased_L-2_H-128_A-2",
        batch_size: int = 64,
        max_length: int = 128,
    ):
        super().__init__()

        self._num_classes: Final = 2
        self.batch_size: Final = batch_size
        self.max_length: Final = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )

    @property
    def num_classes(self) -> int:
        return self._num_classes
