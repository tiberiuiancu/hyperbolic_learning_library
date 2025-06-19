from typing import Optional, Union, Type

from torch import nn

from hypll import nn as hnn
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.tensors import ManifoldTensor


class PoincareResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = manifold
        self.stride = stride
        self.downsample = downsample

        self.conv1 = hnn.HConvolution2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)
        self.relu = hnn.HReLU(manifold=self.manifold)
        self.conv2 = hnn.HConvolution2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            padding=1,
            bias=False,
        )
        self.bn2 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.manifold.mobius_add(x, residual)
        x = self.relu(x)

        return x


class PoincareBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        bottleneck_ratio: int = 4,
    ):
        super().__init__()
        mid_channels = out_channels // bottleneck_ratio
        self.manifold = manifold
        self.downsample = downsample
        self.relu = hnn.HReLU(manifold=manifold)

        self.conv1 = hnn.HConvolution2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            manifold=manifold,
            bias=False,
        )
        self.bn1 = hnn.HBatchNorm2d(features=mid_channels, manifold=manifold)

        self.conv2 = hnn.HConvolution2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            manifold=manifold,
            bias=False,
        )
        self.bn2 = hnn.HBatchNorm2d(features=mid_channels, manifold=manifold)

        self.conv3 = hnn.HConvolution2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            manifold=manifold,
            bias=False,
        )
        self.bn3 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.manifold.mobius_add(out, residual)
        out = self.relu(out)

        return out


class PoincareResNet(nn.Module):
    def __init__(
        self,
        channel_sizes: list[int],
        group_depths: list[int],
        manifold: PoincareBall,
        block: Type[Union[PoincareResidualBlock, PoincareBottleneckBlock]] = PoincareResidualBlock,
        num_classes: int = 10,
    ):
        super().__init__()
        assert len(channel_sizes) == 4 and len(group_depths) == 4

        self.manifold = manifold
        self.block = block

        self.conv = hnn.HConvolution2d(
            in_channels=3,
            out_channels=channel_sizes[0],
            kernel_size=7,
            stride=2,
            padding=3,
            manifold=manifold,
        )
        self.bn = hnn.HBatchNorm2d(features=channel_sizes[0], manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)
        self.maxpool = hnn.HMaxPool2d(kernel_size=3, stride=2, padding=1, manifold=manifold)

        self.group1 = self._make_group(
            channel_sizes[0], channel_sizes[0], group_depths[0], stride=1
        )
        self.group2 = self._make_group(
            channel_sizes[0], channel_sizes[1], group_depths[1], stride=2
        )
        self.group3 = self._make_group(
            channel_sizes[1], channel_sizes[2], group_depths[2], stride=2
        )
        self.group4 = self._make_group(
            channel_sizes[2], channel_sizes[3], group_depths[3], stride=2
        )

        self.avg_pool = hnn.HAdaptiveAvgPool2d((1, 1), manifold=manifold)
        self.fc = hnn.HLinear(
            in_features=channel_sizes[3], out_features=num_classes, manifold=manifold
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)

        x = self.avg_pool(x)
        x = self.fc(x.squeeze())
        return x

    def _make_group(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = hnn.HConvolution2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                manifold=self.manifold,
            )

        layers = [
            self.block(
                in_channels=in_channels,
                out_channels=out_channels,
                manifold=self.manifold,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                self.block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    manifold=self.manifold,
                )
            )

        return nn.Sequential(*layers)
