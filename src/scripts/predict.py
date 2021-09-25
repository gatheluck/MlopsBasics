import logging
import pathlib
from typing import Final

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf.omegaconf import OmegaConf

from src.factory.lit.task import ClassificationPredictor, Classifier

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="predict")
def main(cfg) -> None:
    OmegaConf.set_readonly(cfg, True)
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    checkpoint_path: Final = cwd / cfg.checkpoint_path

    data: Final = instantiate(cfg.data, num_workers=4, root=cwd / "data")
    data.prepare_data()
    data.setup()
    labels: Final = data.get_labels()

    classifier: pl.LightningModule = Classifier.load_from_checkpoint(checkpoint_path)
    predictor: Final = ClassificationPredictor(classifier, labels)

    dataloader: Final = data.test_dataloader()
    for i, (x, t) in enumerate(dataloader):
        if i >= 10:
            break
        if i == 0:
            print(predictor.predict_labels(x))
            print(predictor.predict(x, topk=3))
        predictor.predict_labels(x)
        predictor.predict(x, topk=3)


if __name__ == "__main__":
    main()
