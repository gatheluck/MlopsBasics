import pathlib
from typing import Dict, Final, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from torch.nn.modules.loss import _Loss
from torchmetrics.classification.stat_scores import StatScores

from src.misc.profile import timing


class Classifier(pl.LightningModule):
    """Lightning Module for supervised image classfication.

    Attributes:
        encoder (nn.Module): The encoder to extract feature for classification.
        optimizer_cfg (DictConfig): The config for optimizer.
        scheduler_cfg (DictConfig): The config for sheduler.
        criterion (_Loss): The loss used by optimizer.
        num_classes (int): The number of class.

    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        num_classes: int,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        scheduler_monitor: str = "train/accuracy",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.optimizer_cfg: Final = optimizer_cfg
        self.scheduler_cfg: Final = scheduler_cfg
        self.scheduler_monitor: Final = scheduler_monitor

        self.criterion: Final[_Loss] = torch.nn.CrossEntropyLoss()
        self.metrics: Final = self._get_metrics(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.encoder(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): An input tensor and label.
            batch_index (int): An index of the batch.

        Returns:
            torch.Tensor: A loss tensor.
                If multiple nodes are used for training, return type should be Dict[str, torch.Tensor].

        """
        x, t = batch  # DO NOT need to send GPUs manually.
        logits = self.forward(x)
        loss = self.criterion(logits, t)

        # Logging metrics
        preds: Final = torch.argmax(logits, 1)
        self._log_metrics("train", preds, t, loss.detach(), self.metrics)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The tuple of input tensor and label.
            batch_index (int): The index of batch.

        Returns:
            Dict[str, torch.Tensor]: The dict of log info.

        """
        x, t = batch  # DO NOT need to send GPU manually.
        logits = self.forward(x)
        loss = self.criterion(logits, t)

        # Logging metrics
        preds: Final = torch.argmax(logits, 1)
        self._log_metrics("valid", preds, t, loss.detach(), self.metrics)

        return {"targets": t, "logits": logits.detach()}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:  # type: ignore[override]
        targets: Final = torch.cat([x["targets"] for x in outputs]).cpu()
        logits: Final = torch.cat([x["logits"] for x in outputs]).cpu()
        preds: Final = torch.argmax(logits, 1).cpu()

        # self.logger.experiment.log(
        #     {
        #         "conf": wandb.plot.confusion_matrix(
        #             probs=logits.numpy(), y_true=labels.numpy()
        #         )
        #     }
        # )

        # wandb.log({"confusion_matrix": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        data = confusion_matrix(targets.numpy(), preds.numpy())
        df_cm = pd.DataFrame(data, columns=np.unique(targets), index=np.unique(targets))
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"
        plt.figure(figsize=(7, 4))
        plot = sns.heatmap(
            df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        )  # font size
        self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        self.logger.experiment.log(
            {"roc": wandb.plot.roc_curve(targets.numpy(), logits.numpy())}
        )

    def configure_optimizers(self):
        """setup optimzier and scheduler."""
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        scheduler = {
            "scheduler": instantiate(self.scheduler_cfg, optimizer),
            "monitor": self.scheduler_monitor,
        }

        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def _get_metrics(self, num_classes: int) -> Dict[str, StatScores]:
        return {
            "train/accuracy": torchmetrics.Accuracy(),
            "valid/accuracy": torchmetrics.Accuracy(),
            "valid/precision_micro": torchmetrics.Precision(average="micro"),
            "valid/precision_macro": torchmetrics.Precision(
                average="macro", num_classes=num_classes
            ),
            "valid/recall_micro": torchmetrics.Recall(average="micro"),
            "valid/recall_macro": torchmetrics.Recall(
                average="macro", num_classes=num_classes
            ),
            "valid/f1": torchmetrics.F1(num_classes=num_classes),
        }

    def _log_metrics(
        self,
        stage: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        metrics: Dict[str, StatScores],
    ) -> None:

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True)

        for k, metric in metrics.items():
            if k.startswith(stage):
                value = metric(preds.cpu(), targets.cpu())
                self.log(k, value, prog_bar=True, on_epoch=True)


class ClassificationPredictor:
    def __init__(self, classifier: pl.LightningModule, labels: List[str]) -> None:
        self.classifier: Final = classifier
        self.classifier.eval()
        self.classifier.freeze()

        self.softmax: Final = torch.nn.Softmax(dim=1)
        self.labels: Final = labels

    @timing
    def predict(
        self, x: torch.Tensor, topk: Optional[int] = None
    ) -> List[Dict[str, float]]:
        k: Final = x.size(-1) if topk is None else topk
        logits: Final = self.classifier(x)
        probs: Final = self.softmax(logits)

        predictions = list()
        for prob in probs.split(split_size=1, dim=0):
            value, index = prob.squeeze().topk(k=k)
            _lables = [self.labels[i] for i in index.tolist()]
            _probs = value.tolist()
            predictions.append(dict(zip(_lables, _probs)))

        return predictions

    @timing
    def predict_labels(self, x: torch.Tensor) -> List[str]:
        logits = self.classifier(x)
        predicted_indices: Final[List[int]] = torch.argmax(logits, 1).tolist()

        return [self.labels[i] for i in predicted_indices]


class OnnxClassificationPredictor:
    def __init__(self, model_path: pathlib.Path, processor) -> None:
        self.onnx_session = onnxruntime.InferenceSession(str(model_path))
        self.processor: Final = processor
        self.lables: Final = ["unacceptable", "acceptable"]

    @timing
    def predict(self, input: str):
        input_dict = {"sentence": input}
        processed_input = self.processor.tokenize_data(input_dict)

        onnx_inputs = {
            "input_ids": np.expand_dims(processed_input["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed_input["attention_mask"], axis=0),
        }
        # None will return all the outputs.
        onnx_outputs = self.onnx_session.run(None, onnx_inputs)
        scores = softmax(onnx_outputs[0])[0]

        predictions = list()
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})

        return predictions
