import torch.nn as nn

from benchmark.utils import get_cifar10
from benchmark.profiling import profile_training

from hypll.manifolds import Manifold
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
import hypll.nn as hnn


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
    trainloader, _, _ = get_cifar10(batch_size, flatten=True)

    # profile mlp
    net = MLP(hdims=hdims)
    profile_training(net, trainloader, config_name="mlp")
f
    # profile hyperbolic mlp
    manifold = PoincareBall(c=Curvature(requires_grad=True))
    net = MLP(manifold=manifold, hdims=hdims)
    profile_training(net, trainloader, manifold=manifold, config_name="hmlp")
