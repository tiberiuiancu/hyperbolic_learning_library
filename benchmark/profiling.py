import os
import torch
import torch.nn as nn

import torch.profiler
from torch.profiler import record_function
from tqdm import tqdm

from benchmark.models.mlp import MLP
from benchmark.utils import make_resnet, get_dataset, parameter_count
from hypll.manifolds import Manifold
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.optim.adam import RiemannianAdam
from hypll.tensors.tangent_tensor import TangentTensor

from typing import Literal
from tap import Tap

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000

torch._dynamo.config.assume_static_by_default = True


def profile_training(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    active: int = 1,
    warmup: int = 1,
    wait: int = 1,
    config: str = "model",
    manifold: Manifold | None = None,
    lr: float = 0.001,
    compile_model: bool = False,
    compile_optimizer: bool = False,
):
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(warmup=warmup, active=active, wait=wait, skip_first=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        model.cuda()
        if compile_model:
            model.compile()

        criterion = nn.CrossEntropyLoss()
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=lr)
            if manifold is None
            else RiemannianAdam(model.parameters(), lr=lr)
        )

        def opt():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        @torch.compile
        def opt_compiled():
            opt()

        # first step is skipped
        prof.step()

        for step, data in tqdm(enumerate(trainloader), total=active + warmup + wait):
            if step >= active + warmup + wait:
                break

            with record_function("move_to_cuda"):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

            if manifold is not None:
                with record_function("move_to_manifold"):
                    # TODO: try to compile this
                    tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold)
                    inputs = manifold.expmap(tangents)

            with record_function("forward_pass"):
                outputs = model(inputs)
                outputs_tensor = outputs if manifold is None else outputs.tensor

            with record_function("criterion"):
                loss = criterion(outputs_tensor, labels)

            with record_function("backward_pass"):
                loss.backward()

            with record_function("optimizer"):
                if compile_optimizer:
                    opt_compiled()
                else:
                    opt()

            prof.step()

    # write trace and memory usage history
    out_dir = f"traces/{config}"
    os.makedirs(out_dir, exist_ok=True)

    prof.export_chrome_trace(f"{out_dir}/{config}_trace.json")
    prof.export_memory_timeline(f"{out_dir}/{config}_mem.html", device="cuda:0")

    try:
        # save detailed memory snapshot
        torch.cuda.memory._dump_snapshot(f"{out_dir}/{config}_mem.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    # stop recording memory
    torch.cuda.memory._record_memory_history(enabled=None)


def get_model(args, in_size, out_size, manifold):
    """
    Create and return a model based on the provided arguments.

    Args:
        args: Parsed arguments containing model configuration.
        in_size: Input size of the model.
        out_size: Output size of the model.
        manifold: The manifold to use, if any.

    Returns:
        A PyTorch model instance.

    Raises:
        ValueError: If the model type specified in args is invalid.
    """
    if args.model == "mlp":
        return MLP(in_size=in_size, out_size=out_size, hdims=args.mlp_hdims, manifold=manifold)
    elif args.model.startswith("resnet"):
        resnet_config = args.model.removeprefix("resnet")
        # Dynamically determine the number of input channels based on the dataset
        # Set in_channels to 1 for single-channel processing
        in_channels = 1
        return make_resnet(
            resnet_config,
            manifold=manifold,
            in_channels=in_channels,
            out_size=out_size,
            use_midpoint=True,
        )
    else:
        raise ValueError(f"Invalid model {args.model}")


class ProfileArgs(Tap):
    model: Literal[
        "resnetmicro",
        "resnetmini",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "mlp",
    ]
    dataset: Literal["imagenet", "cifar10", "caltech256"] = "cifar10"
    hyperbolic: bool = False
    compile_model: bool = False
    compile_optimizer: bool = False
    batch_size: int = 128
    mlp_hdims: list[int] = [2**12]
    curvature: float = 0.1
    use_triton_backend: bool = False
    active: int = 1
    warmup: int = 1
    wait: int = 1

    def configure(self):
        self.add_argument("model")
        self.add_argument("--dataset", "-d")
        self.add_argument("--hyperbolic", "-H", action="store_true")
        self.add_argument("--use_triton_backend", "-t", action="store_true")
        self.add_argument("--compile_model", "-c", action="store_true")
        self.add_argument("--batch_size", "-b")
        self.add_argument("-a", "--active")
        self.add_argument("-w", "--warmup")


if __name__ == "__main__":
    args = ProfileArgs().parse_args()

    trainloader = get_dataset(
        args.dataset,
        args.batch_size,
        flatten=(args.model == "mlp"),
        n_samples=10 * args.batch_size,
    )

    in0, out0 = next(iter(trainloader))
    in_size = torch.flatten(in0, start_dim=1).shape[-1]
    out_size = out0.shape[-1]

    # create manifold
    manifold = (
        PoincareBall(c=Curvature(args.curvature), use_triton_backend=args.use_triton_backend)
        if args.hyperbolic
        else None
    )

    net = get_model(args, in_size, out_size, manifold)

    print(f"Model {args.model} has {parameter_count(net)} parameters")

    config_name = (
        ("h_" if args.hyperbolic else "")
        + ("c_" if args.compile_model else "")
        + ("co_" if args.compile_optimizer else "")
        + (f"t_" if args.use_triton_backend else "")
        + f"{args.model}"
    )
    profile_training(
        model=net,
        trainloader=trainloader,
        config=config_name,
        manifold=manifold,
        compile_model=args.compile_model,
        compile_optimizer=args.compile_optimizer,
        active=args.active,
        warmup=args.warmup,
        wait=args.wait,
    )
