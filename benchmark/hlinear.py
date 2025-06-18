from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.nn import HLinear
from hypll.tensors import TangentTensor
from torchview import draw_graph

import torch
from torch.profiler import profile, record_function, ProfilerActivity


device = "cuda" if torch.cuda.is_available() else "cpu"

in_size = 8192
out_size = 32768

manifold = PoincareBall(c=Curvature(requires_grad=True))
layer = HLinear(in_size, out_size, manifold=manifold).to(device)
inputs = torch.rand((11, 64, in_size), device=device)
tangents = [TangentTensor(data=inputs[i], man_dim=1, manifold=manifold) for i in range(11)]


class LayerWrapper(torch.nn.Module):
    def __init__(self, layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer = layer

    def forward(self, x):
        self._layer(TangentTensor(data=x, man_dim=1, manifold=manifold))


# warmup
for i in range(10):
    layer(tangents[i])


activities = [ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU]
with profile(activities=activities, record_shapes=True) as prof:
    with record_function("layer forward"):
        layer(tangents[-1])

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("traces/hlinear.json")

g = draw_graph(
    LayerWrapper(layer),
    input_size=(inputs[0].shape),
    device="cpu",
    expand_nested=True,
)
g.visual_graph.render("traces/hlinear", format="png", cleanup=True)
