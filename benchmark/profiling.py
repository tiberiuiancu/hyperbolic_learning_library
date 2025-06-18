import torch
import torch.nn as nn

import torch.profiler
from torch.profiler import record_function

from hypll.manifolds import Manifold
from hypll.optim.adam import RiemannianAdam
from hypll.tensors.tangent_tensor import TangentTensor

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000


def profile_training(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    active: int = 8,
    warmup: int = 2,
    wait: int = 2,
    config_name: str = "model",
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
        schedule=torch.profiler.schedule(warmup=warmup, active=active, wait=wait),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, data in enumerate(trainloader):
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
    prof.export_chrome_trace(f"traces/{config_name}_trace.json")
    prof.export_memory_timeline(f"traces/{config_name}_mem.html", device="cuda:0")

    try:
        # save detailed memory snapshot
        torch.cuda.memory._dump_snapshot(f"traces/{config_name}_mem.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    # stop recording memory
    torch.cuda.memory._record_memory_history(enabled=None)
