from typing import Optional, Union
import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


from benchmark.models.hresnet import PoincareBottleneckBlock, PoincareResNet, PoincareResidualBlock
from benchmark.models.resnet import BottleneckBlock, ResNet, ResidualBlock
from hypll.manifolds.poincare_ball.manifold import PoincareBall


def get_cifar10(batch_size: int = 64, flatten: bool = False, num_images: Optional[int] = None):
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
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    if num_images:
        trainset = Subset(trainset, range(min(num_images, len(trainset))))
        testset = Subset(testset, range(min(num_images, len(testset))))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    return trainloader, testloader, classes


def get_imagenet(batch_size: int = 64, num_images: int = None):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = ImageNet(root="./data", split="train", transform=transform)
    testset = ImageNet(root="./data", split="val", transform=transform)

    if num_images:
        trainset = Subset(trainset, range(min(num_images, len(trainset))))
        testset = Subset(testset, range(min(num_images, len(testset))))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


def make_resnet(
    config: str = "18", manifold: Optional[PoincareBall] = None
) -> Union["ResNet", PoincareResNet]:
    # Define configurations for different ResNet variants
    channels = {
        "mini": [16, 32, 64, 128],
        "18": [64, 128, 256, 512],
        "34": [64, 128, 256, 512],
        "50": [256, 512, 1024, 2048],
        "101": [256, 512, 1024, 2048],
        "152": [256, 512, 1024, 2048],
    }

    depths = {
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
