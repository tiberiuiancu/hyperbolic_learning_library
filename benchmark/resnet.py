from torch import nn

from benchmark.utils import get_cifar10
from benchmark.profiling import profile_training

from hypll import nn as hnn
from hypll.manifolds.base.manifold import Manifold
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall


class HResNet(nn.Module):
    def __init__(self, manifold: Manifold):
        super().__init__()
        self.conv1 = hnn.HConvolution2d(
            in_channels=3, out_channels=6, kernel_size=5, manifold=manifold
        )
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv2 = hnn.HConvolution2d(
            in_channels=6, out_channels=16, kernel_size=5, manifold=manifold
        )
        self.fc1 = hnn.HLinear(in_features=16 * 5 * 5, out_features=120, manifold=manifold)
        self.fc2 = hnn.HLinear(in_features=120, out_features=84, manifold=manifold)
        self.fc3 = hnn.HLinear(in_features=84, out_features=10, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    trainloader, _, _ = get_cifar10()

    manifold = PoincareBall(c=Curvature(requires_grad=True))
    net = HResNet(manifold)
    profile_training(net, trainloader, manifold=manifold, config_name="hresnet")

    net = ResNet()
    profile_training(net, trainloader, config_name="resnet")
