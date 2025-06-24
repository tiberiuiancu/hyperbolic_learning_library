import os
import torch
import torch.nn as nn

import torch.profiler
from torch.profiler import record_function
from tqdm import tqdm

from hypll.manifolds import Manifold
from hypll.optim.adam import RiemannianAdam
from hypll.tensors.tangent_tensor import TangentTensor

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000


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

    model.cuda()
    if compile_model:
        model = torch.compile(model)

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

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(warmup=warmup, active=active, wait=wait),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # todo: allocate mempory after profiler start
        for step, data in tqdm(enumerate(trainloader), total=active + warmup + wait):
            if step >= active + warmup + wait:
                break

            with record_function("move_to_cuda"):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

            if manifold is not None:
                with record_function("move_to_manifold"):
                    # todo: try to compile this
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
