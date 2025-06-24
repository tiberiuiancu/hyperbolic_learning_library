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
    config: str = "model",
    manifold: Manifold | None = None,
    lr: float = 0.001,
):
    model.cuda()

    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

    criterion = nn.CrossEntropyLoss()
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=lr)
        if manifold is None
        else RiemannianAdam(model.parameters(), lr=lr)
    )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(warmup=warmup, active=active, wait=0),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, data in tqdm(enumerate(trainloader), total=active + warmup):
            if step >= active + warmup:
                break

            with record_function("move_to_cuda"):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

            if manifold is not None:
                with record_function("move_to_manifold"):
                    tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold)
                    inputs = manifold.expmap(tangents)

            with record_function("forward_pass"):
                outputs = model(inputs)
                outputs_tensor = outputs if manifold is None else outputs.tensor

            with record_function("criterion"):
                loss = criterion(outputs_tensor, labels)

            with record_function("backward_pass"):
                loss.backward()

            with record_function("step"):
                optimizer.step()

            with record_function("zero_grad"):
                optimizer.zero_grad(set_to_none=True)

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
