import logging
import pathlib
from typing import Final

import hydra
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

    data: Final = instantiate(cfg.data, num_workers=4, root=cwd / "data")
    data.prepare_data()
    data.setup()
    labels: Final = data.labels

    predictor: Final = OnnxClassificationPredictor(onnx_model_path, labels)

    dataloader: Final = data.test_dataloader()
    for i, (_x, _) in enumerate(dataloader):
        x = _x.cpu().numpy()
        if i >= 10:
            break
        if i == 0:
            print(predictor.predict_labels(x))
            print(predictor.predict(x, topk=3))
        predictor.predict_labels(x)
        predictor.predict(x, topk=3)


if __name__ == "__main__":
    main()
