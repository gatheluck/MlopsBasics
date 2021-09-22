import logging
import pathlib
from typing import Final

import hydra
import torch
from hydra.utils import instantiate
from omegaconf.omegaconf import OmegaConf

from src.factory.lit.task import OnnxClassificationPredictor

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="predict_onnx")
def main(cfg) -> None:
    OmegaConf.set_readonly(cfg, True)
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    onnx_model_path: Final = cwd / cfg.onnx_model_path

    # data module
    tokenizer: torch.nn.Module = instantiate(cfg.tokenizer)
    data: Final = instantiate(cfg.data, tokenizer=tokenizer)

    predictor: Final = OnnxClassificationPredictor(onnx_model_path, data)

    print(predictor.predict(cfg.sentence))
    sentences = [cfg.sentence] * 10
    for sentence in sentences:
        predictor.predict(sentence)


if __name__ == "__main__":
    main()
