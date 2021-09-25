import pathlib
from typing import Final

import fastapi
import numpy as np
from fastapi.responses import JSONResponse
from PIL import Image

import src.app.views.api._lib as _lib
from src.factory.data.cifar10 import Cifar10DataModule
from src.factory.lit.task import OnnxClassificationPredictor


async def health() -> JSONResponse:
    return JSONResponse({"health": "ok"})


async def predict(
    file: fastapi.UploadFile = fastapi.File(...),
) -> JSONResponse:
    """Return topk prediction result.

    Args:
        file (fastapi.UploadFile): A uploaded image file.

    Returns:
        JSONResponse: A topk prediction results.
            Returned dict has predicted lables and its probabilities.

    Raises:
        ValueError: If `file` has unsppported suffix.

    Note:
        Input to the onnx model is np.ndarray whoes shape is (b, c, h, w).

    """
    transform: Final = Cifar10DataModule.get_transform(train=False, to_tensor=False)
    labels: Final = Cifar10DataModule.get_labels()
    onnx_model_path: Final = pathlib.Path("models/model.onnx")
    topk: Final = 3

    # Convert input image to np.ndarray
    _lib.check_suffix(pathlib.Path(file.filename))
    image: Final[Image] = _lib.read_imagefile(await file.read())
    image_np: Final = np.asarray(image)  # (h, w, c)
    image_transformed: Final = _lib.apply_transform(image_np, transform)  # (1, c, h, w)

    predictor: Final = OnnxClassificationPredictor(onnx_model_path, labels)
    predictions: Final = predictor.predict(image_transformed, topk=topk)

    # predictions is List of Dict.
    # Because now batch size is 1, jsut return first element.
    return JSONResponse({"prediction": predictions[0]})


async def predict_label(
    file: fastapi.UploadFile = fastapi.File(...),
) -> JSONResponse:
    """Return topk prediction result.

    Args:
        file (fastapi.UploadFile): A uploaded image file.

    Returns:
        JSONResponse: A predicted label.

    Raises:
        ValueError: If `file` has unsppported suffix.

    Note:
        Input to the onnx model is np.ndarray whoes shape is (b, c, h, w).

    """
    transform: Final = Cifar10DataModule.get_transform(train=False, to_tensor=False)
    labels: Final = Cifar10DataModule.get_labels()
    onnx_model_path: Final = pathlib.Path("models/model.onnx")

    # Convert input image to np.ndarray
    _lib.check_suffix(pathlib.Path(file.filename))
    image: Final[Image] = _lib.read_imagefile(await file.read())
    image_np: Final = np.asarray(image)  # (h, w, c)
    image_transformed: Final = _lib.apply_transform(image_np, transform)  # (1, c, h, w)

    predictor: Final = OnnxClassificationPredictor(onnx_model_path, labels)
    predictions: Final = predictor.predict_labels(image_transformed)

    # predictions is List of str.
    # Because now batch size is 1, jsut return first element.
    return JSONResponse({"prediction": predictions[0]})
