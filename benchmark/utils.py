from typing import Literal, Optional, Union
import numpy as np
import torch
import torchvision.transforms as transforms

from torchvision.datasets import ImageNet, CIFAR10, Caltech256
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


from benchmark.models.hresnet import PoincareBottleneckBlock, PoincareResNet, PoincareResidualBlock
from benchmark.models.resnet import BottleneckBlock, ResNet, ResidualBlock
from torch import nn
from hypll.manifolds.poincare_ball.manifold import PoincareBall


def get_dataset(
    dataset: Literal["imagenet", "cifar10", "caltech256"],
    batch_size: int,
    flatten: bool = False,
    n_samples: int | None = None,
) -> tuple[DataLoader, DataLoader] | DataLoader:
    t = []

    if dataset == "imagenet":
        t += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    elif dataset == "cifar10":
        t += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    elif dataset == "caltech256":
        t += [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Grayscale(),  # Convert all datasets to single channel
            transforms.Normalize(mean=[0.45], std=[0.225]),
        ]
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    if flatten:
        t += [transforms.Lambda(lambda x: torch.flatten(x))]

    transform = transforms.Compose(t)

    if dataset == "imagenet":
        dataset_class = ImageNet
    elif dataset == "cifar10":
        dataset_class = CIFAR10
    elif dataset == "caltech256":
        dataset_class = Caltech256

    dataset_kwargs = dict(root="./data", transform=transform)
    if dataset in ["imagenet", "cifar10"]:
        dataset_kwargs |= dict(train=True)
    if dataset in ["cifar10"]:
        dataset_kwargs |= dict(download=True)

    trainset = dataset_class(**dataset_kwargs)
    if n_samples:
        trainset = Subset(trainset, range(min(n_samples, len(trainset))))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return trainloader


def make_resnet(
    config: str, manifold: Optional[PoincareBall] = None, **model_kwargs
) -> Union["ResNet", PoincareResNet]:
    # Define configurations for different ResNet variants
    channels = {
        "micro": [4, 4, 4, 4],
        "mini": [16, 32, 64, 128],
        "18": [64, 128, 256, 512],
        "34": [64, 128, 256, 512],
        "50": [256, 512, 1024, 2048],
        "101": [256, 512, 1024, 2048],
        "152": [256, 512, 1024, 2048],
    }

    depths = {
        "micro": [1, 1, 1, 1],
        "mini": [1, 1, 1, 1],
        "18": [2, 2, 2, 2],
        "34": [3, 4, 6, 3],
        "50": [3, 4, 6, 3],
        "101": [3, 4, 23, 3],
        "152": [3, 8, 36, 3],
    }

    block_type = "bottleneck" if config in ["50", "101", "152"] else "basic"

    # Validate configuration
    if config not in channels:
        raise ValueError(
            f"Invalid config: {config}. Available options are: {list(channels.keys())}"
        )

    kwargs = model_kwargs | {
        "channel_sizes": channels[config],
        "group_depths": depths[config],
    }

    if manifold is not None:
        kwargs.update(
            {
                "manifold": manifold,
                "block": (
                    PoincareBottleneckBlock if block_type == "bottleneck" else PoincareResidualBlock
                ),
            }
        )
        return PoincareResNet(**kwargs)
    else:
        kwargs.update(
            {
                "block": (BottleneckBlock if block_type == "bottleneck" else ResidualBlock),
            }
        )
        return ResNet(**kwargs)


def parameter_count(model: nn.Module) -> int:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
