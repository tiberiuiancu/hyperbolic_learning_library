from typing import Optional, Union
import torch
import torchvision
import torchvision.transforms as transforms

from benchmark.models.hresnet import PoincareBottleneckBlock, PoincareResNet, PoincareResidualBlock
from benchmark.models.resnet import BottleneckBlock, ResNet, ResidualBlock
from hypll.manifolds.poincare_ball.manifold import PoincareBall


def get_cifar10(batch_size: int = 64, flatten: bool = False):
    t = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if flatten:
        t += [t.Lambda(lambda x: torch.flatten(x))]
    transform = transforms.Compose(t)

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    return trainloader, testloader, classes


def make_resnet(
    config: str = "18", manifold: Optional[PoincareBall] = None
) -> Union["ResNet", PoincareResNet]:
    # Define configurations for different ResNet variants
    channels = {
        "18": [64, 128, 256, 512],
        "34": [64, 128, 256, 512],
        "50": [256, 512, 1024, 2048],
        "101": [256, 512, 1024, 2048],
        "152": [256, 512, 1024, 2048],
    }

    depths = {
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

    kwargs = {
        "channel_sizes": channels[config],
        "group_depths": depths[config],
        "num_classes": 10,
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
