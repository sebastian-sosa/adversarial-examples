from typing import Union

from torch import Tensor
from torch.nn import Module


def create_adversarial_example(
    model: Module,
    image_tensor: Tensor,
    target_label: Union[str, int],
    epsilon: float = 0.1,
    num_steps: int = 100,
    alpha: float = 0.01,
) -> Tensor:
    """
    This function creates an adversarial example using the given model, image tensor,
    and target label. The target label can be either the label name or label index
    as defined in the imagenet-labels.json file.

    Args:
        model (Module): The model to be used for creating the adversarial example.
        image_tensor (Tensor): The image tensor to be used for creating the adversarial example.
        target_label (Union[str, int]): The target label for the adversarial example.
            Can be either the label name or label index.
        epsilon (float, optional): The maximum allowed size of the adversarial perturbation.
            Default is 0.1.
        num_steps (int, optional): The number of steps for the gradient descent. Default is 100.
        alpha (float, optional): The step size for the gradient descent. Default is 0.01.

    Returns:
        Tensor: The adversarial example as a tensor.
    """
    pass


model, labels = load_model_and_labels()
image = load_image("imgs/ostrich.jpeg")
indices = predict(model, image)

target_class = torch.tensor([1])  # label in position 1, goldfish
adversarial_example = create_adversarial_example(model, image, target_class)
y_hat = predict(model, adversarial_example)
print(labels[y_hat.item()])
display(to_image(adversarial_example))
