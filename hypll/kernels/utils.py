import triton
import triton.language as tl


@triton.jit
def atanh(x):
    return 0.5 * tl.log((1 + x) / (1 - x))


@triton.jit
def tanh(x):
    ex = tl.exp(x)
    exi = tl.exp(-x)
    return (ex - exi) / (ex + exi)
