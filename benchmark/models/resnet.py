import torch
from torch import nn
from typing import Optional, Union, Type


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        bottleneck_ratio: int = 4,
    ):
        super().__init__()
        mid_channels = out_channels // bottleneck_ratio

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        channel_sizes: list[int],
        group_depths: list[int],
        in_channels: int = 3,
        out_size: int = 10,
        block: Type[Union[ResidualBlock, BottleneckBlock]] = ResidualBlock,
        **kwargs
    ):
        super().__init__()
        assert len(channel_sizes) == len(group_depths)
        assert len(channel_sizes) >= 1, "channel_sizes must have length at least 1"

        self.block = block

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channel_sizes[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channel_sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.groups = nn.ModuleList()
        prev_channels = channel_sizes[0]
        for i in range(len(channel_sizes)):
            stride = 1 if i == 0 else 2
            self.groups.append(
                self._make_group(
                    in_channels=prev_channels,
                    out_channels=channel_sizes[i],
                    depth=group_depths[i],
                    stride=stride,
                )
            )
            prev_channels = channel_sizes[i]

        last_channels = channel_sizes[-1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=last_channels, out_features=out_size)

    def _make_group(
        self, in_channels: int, out_channels: int, depth: int, stride: int = 1
    ) -> nn.Sequential:
        if depth == 0:
            return nn.Sequential()
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            self.block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                self.block(
                    in_channels=out_channels, out_channels=out_channels, stride=1, downsample=None
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for group in self.groups:
            x = group(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
