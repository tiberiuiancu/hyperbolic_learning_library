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
        use_midpoint: bool = True,
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
        self.bn1 = hnn.HBatchNorm2d(
            features=out_channels, manifold=manifold, use_midpoint=use_midpoint
        )
        self.relu = hnn.HReLU(manifold=self.manifold)
        self.conv2 = hnn.HConvolution2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            padding=1,
            bias=False,
        )
        self.bn2 = hnn.HBatchNorm2d(
            features=out_channels, manifold=manifold, use_midpoint=use_midpoint
        )

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
        use_midpoint: bool = True,
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
        self.bn1 = hnn.HBatchNorm2d(
            features=mid_channels, manifold=manifold, use_midpoint=use_midpoint
        )

        self.conv2 = hnn.HConvolution2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            manifold=manifold,
            bias=False,
        )
        self.bn2 = hnn.HBatchNorm2d(
            features=mid_channels, manifold=manifold, use_midpoint=use_midpoint
        )

        self.conv3 = hnn.HConvolution2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            manifold=manifold,
            bias=False,
        )
        self.bn3 = hnn.HBatchNorm2d(
            features=out_channels, manifold=manifold, use_midpoint=use_midpoint
        )

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
        in_channels: int = 3,
        out_size: int = 10,
        use_midpoint: bool = True,
    ):
        super().__init__()
        assert len(channel_sizes) == len(group_depths)
        assert len(channel_sizes) >= 1, "channel_sizes must have length at least 1"
        self.manifold = manifold
        self.block = block

        self.conv = hnn.HConvolution2d(
            in_channels=in_channels,
            out_channels=channel_sizes[0],
            kernel_size=7,
            stride=2,
            padding=3,
            manifold=manifold,
        )
        self.bn = hnn.HBatchNorm2d(
            features=channel_sizes[0], manifold=manifold, use_midpoint=use_midpoint
        )
        self.relu = hnn.HReLU(manifold=manifold)
        self.maxpool = hnn.HMaxPool2d(kernel_size=3, stride=2, padding=1, manifold=manifold)

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
                    use_midpoint=use_midpoint,
                )
            )
            prev_channels = channel_sizes[i]

        last_channels = channel_sizes[-1]
        self.avg_pool = hnn.HAdaptiveAvgPool2d((1, 1), manifold=manifold)
        self.fc = hnn.HLinear(in_features=last_channels, out_features=out_size, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for group in self.groups:
            x = group(x)

        x = self.avg_pool(x)
        x = self.fc(x.squeeze())
        return x

    def _make_group(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
        use_midpoint: bool = True,
    ) -> nn.Sequential:
        if depth == 0:
            return nn.Sequential()
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
                use_midpoint=use_midpoint,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                self.block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    manifold=self.manifold,
                    stride=1,
                    downsample=None,
                    use_midpoint=use_midpoint,
                )
            )

        return nn.Sequential(*layers)
