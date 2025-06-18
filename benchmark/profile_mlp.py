from benchmark.models.mlp import MLP
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from benchmark.utils import get_cifar10
from benchmark.profiling import profile_training


if __name__ == "__main__":
    hdims = [2**11, 2**11]

    batch_size = 64
    trainloader, _, _ = get_cifar10(batch_size, flatten=True)

    # profile mlp
    net = MLP(hdims=hdims)
    profile_training(net, trainloader, config_name="mlp")

    # profile hyperbolic mlp
    manifold = PoincareBall(c=Curvature(requires_grad=True))
    net = MLP(manifold=manifold, hdims=hdims)
    profile_training(net, trainloader, manifold=manifold, config_name="hmlp")
