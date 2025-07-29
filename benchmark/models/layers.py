import torch

import torch.nn as nn

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, hyperbolic=False, **kwargs):
        super().__init__()
        self.hyperbolic = hyperbolic
        if hyperbolic:
            # Placeholder for hyperbolic convolution, replace with actual implementation if available
            # For example, you might use a custom HyperbolicConv2d layer
            raise NotImplementedError("Hyperbolic convolution is not implemented.")
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, **kwargs
            )

    def forward(self, x):
        if self.hyperbolic:
            # Implement or call your hyperbolic convolution here
            raise NotImplementedError("Hyperbolic convolution is not implemented.")
        else:
            return self.conv(x)
