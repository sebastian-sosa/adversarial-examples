from typing import List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module


def load_model_and_labels(model_name: str = "resnet18") -> Tuple[Module, List[str]]:
    """
    Load a pretrained model and its corresponding labels.

    Args:
        model_name (str, optional): The name of the model to load. Defaults to "resnet18".

    Returns:
        Tuple[Module, List[str]]: The loaded model and its labels.
    """
    pass


def load_image(
    image_path: str,
    resize_size: int = 256,
    crop_size: int = 224,
    target_mean: List[float] = [0.485, 0.456, 0.406],
    target_std: List[float] = [0.229, 0.224, 0.225],
) -> Tensor:
    """
    Load an image as a Tensor and preprocess it.

    Args:
        image_path (str): The path to the image.
        resize_size (int, optional): The size to resize the image to. Defaults to 256.
        crop_size (int, optional): The size to crop the image to. Defaults to 224.
        target_mean (List[float], optional): The target mean for normalization. Defaults to [0.485, 0.456, 0.406].
        target_std (List[float], optional): The target standard deviation for normalization. Defaults to [0.229, 0.224, 0.225].

    Returns:
        Tensor: The preprocessed image.
    """
    pass


def predict(model: Module, image_tensor: Tensor) -> Tensor:
    """
    Make a prediction using a model and an image.

    Args:
        model (Module): The model to use for prediction.
        image_tensor (Tensor): The image to predict.

    Returns:
        Tensor: The predicted class label.
    """
    pass


def display_tensor(image_tensor: Tensor) -> None:
    """_summary_

    Args:
        image_tensor (Tensor): _description_
    """
    pass


model, labels = load_model_and_labels()
    """
    Display a tensor as an image.

    Args:
        image_tensor (Tensor): The tensor to display.
    """