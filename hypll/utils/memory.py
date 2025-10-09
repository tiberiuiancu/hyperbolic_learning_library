import torch


class GPUMemoryPool:
    def __init__(self, device="cuda"):
        self.device = device
        self.buffers = {}
        self.stream = torch.cuda.Stream(device=self.device)

    def alloc(self, shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def get_shared(self, name, shape, dtype=torch.float32):
        key = (name, shape, dtype)
        buf = self.buffers.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = self.alloc(shape, dtype)
            self.buffers[key] = buf
        return buf


gpu_memory_pool = GPUMemoryPool()
