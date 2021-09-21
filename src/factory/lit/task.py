from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from hydra.utils import instantiate
from sklearn.metrics import confusion_matrix


class Classifier(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        num_classes: int,
        optimizer_cfg,
    ) -> None:
        super().__init__()
        self.optimizer_cfg: Final = optimizer_cfg
        self.encoder: torch.nn.Module = encoder

        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None) -> torch.Tensor:  # type: ignore[override]
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        labels: Final = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=labels
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        train_acc = self.train_accuracy_metric(preds, batch["label"])

        # Logging metrics
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels: Final = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=labels
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels: Final = torch.cat([x["labels"] for x in outputs])
        logits: Final = torch.cat([x["logits"] for x in outputs])
        preds: Final = torch.argmax(logits, 1)

        # self.logger.experiment.log(
        #     {
        #         "conf": wandb.plot.confusion_matrix(
        #             probs=logits.numpy(), y_true=labels.numpy()
        #         )
        #     }
        # )

        # wandb.log({"confusion_matrix": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        data = confusion_matrix(labels.numpy(), preds.numpy())
        df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"
        plt.figure(figsize=(7, 4))
        plot = sns.heatmap(
            df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        )  # font size
        self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        self.logger.experiment.log(
            {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        )

    def configure_optimizers(self):
        return {"optimizer": instantiate(self.optimizer_cfg, params=self.parameters())}
