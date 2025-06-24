from benchmark.models.mlp import MLP
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from benchmark.utils import get_cifar10
from benchmark.profiling import profile_training


if __name__ == "__main__":
    hdims = [2**14]

    batch_size = 64
    trainloader, _, _ = get_cifar10(batch_size, flatten=True)

    # profile mlp
    net = MLP(hdims=hdims)
    profile_training(net, trainloader, config="mlp")

    # profile hyperbolic mlp
    manifold = PoincareBall(c=Curvature(requires_grad=True))
    net = MLP(manifold=manifold, hdims=hdims)
    profile_training(net, trainloader, manifold=manifold, config="hmlp")

    # profile hyperbolic mlp with torch compile
    net = MLP(manifold=manifold, hdims=hdims)
    profile_training(
        net,
        trainloader,
        manifold=manifold,
        compile_model=True,
        compile_optimizer=True,
        config="hmlp_compiled",
    )
