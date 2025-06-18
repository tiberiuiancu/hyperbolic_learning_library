import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from hypll.manifolds import Manifold
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
import hypll.nn as hnn

from profiling import profile_training


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int = 32 * 32 * 3,
        out_size: int = 10,
        hdims: list[int] | None = None,
        manifold: Manifold | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        hdims = hdims or []
        hdims = [in_size] + hdims + [out_size]

        layers = []
        for i in range(len(hdims) - 1):
            layers.append(self.make_layer(hdims[i], hdims[i + 1], manifold=manifold))
        self.net = nn.Sequential(*layers)

    def make_layer(self, in_dim: int, out_dim: int, manifold: Manifold | None):
        if manifold is None:
            return nn.Linear(in_dim, out_dim)
        else:
            return hnn.HLinear(in_dim, out_dim, manifold=manifold)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    hdims = [2**11, 2**11]

    batch_size = 64

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

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

    # profile mlp
    net = MLP(hdims=hdims)
    profile_training(net, trainloader, config_name="mlp")

    # profile hyperbolic mlp
    manifold = PoincareBall(c=Curvature(requires_grad=True))
    net = MLP(manifold=manifold, hdims=hdims)
    profile_training(net, trainloader, manifold=manifold, config_name="hmlp")
