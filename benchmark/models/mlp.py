import torch.nn as nn

from hypll.manifolds import Manifold
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
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            return nn.Sequential(
                hnn.HLinear(in_dim, out_dim, manifold=manifold), hnn.HReLU(manifold=manifold)
            )

    def forward(self, x):
        return self.net(x)
