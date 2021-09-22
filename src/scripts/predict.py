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

    data: Final = instantiate(cfg.data)

    classifier: pl.LightningModule = Classifier.load_from_checkpoint(checkpoint_path)
    predictor: Final = ClassificationPredictor(classifier, data)

    print(predictor.predict(cfg.sentence))
    sentences = [cfg.sentence] * 10
    for sentence in sentences:
        predictor.predict(sentence)


if __name__ == "__main__":
    main()
