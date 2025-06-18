from benchmark.models.mlp import MLP
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from benchmark.utils import get_cifar10, make_resnet
from benchmark.profiling import profile_training


if __name__ == "__main__":
    batch_size = 64
    trainloader, _, _ = get_cifar10(batch_size, flatten=False)

    # profile mlp
    net = make_resnet(config="18")
    profile_training(net, trainloader, config_name="resnet")

    # profile hyperbolic mlp
    manifold = PoincareBall(c=Curvature(requires_grad=True))
    net = make_resnet(config="18", manifold=manifold)
    profile_training(net, trainloader, manifold=manifold, config_name="hresnet")
