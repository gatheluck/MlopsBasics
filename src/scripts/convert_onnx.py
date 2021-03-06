import logging
import pathlib
from typing import Final

import hydra
import torch
from hydra.utils import instantiate

from src.factory.lit.task import Classifier

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="convert_onnx")
def convert_to_onnx(cfg) -> None:
    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    checkpoint_path: Final = cwd / cfg.checkpoint_path
    export_path: Final = pathlib.Path(cfg.export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading pre-trained model from: {checkpoint_path}")
    model: Final = Classifier.load_from_checkpoint(checkpoint_path)

    # data module
    data: Final = instantiate(cfg.data, num_workers=4, root=cwd / "data")
    data.prepare_data()
    data.setup()

    x, _ = next(iter(data.train_dataloader()))
    input_sample = x[0].unsqueeze(0)

    # export the model
    logger.info("Converting the model into ONNX format.")
    torch.onnx.export(
        model,  # model being run
        input_sample,  # model input (or a tuple for multiple inputs)
        str(export_path),  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input"],  # the model's input names
        output_names=["logits"],  # the model's output names
        # NOTE: if dynamic axes don't specified, size of all dim will be fixed.
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "logits": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {str(export_path)}"
    )


if __name__ == "__main__":
    convert_to_onnx()
