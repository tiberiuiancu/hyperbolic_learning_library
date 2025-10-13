import triton
import triton.language as tl


@triton.jit
def atanh(x):
    return 0.5 * tl.log((1 + x) / (1 - x))


@triton.jit
def tanh(x):
    e2m = tl.exp(-2 * tl.abs(x))
    t = 1 - 2 / (1 + 1 / e2m)
    return tl.where(x > 0, t, -t)


@triton.jit
def sech_squared(x):
    t = tanh(x)
    return 1 - t * t
