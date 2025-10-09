import torch


class GPUMemoryPool:
    def __init__(self, device="cuda"):
        self.device = device
        self.buffers = {}
        self.stream = torch.cuda.Stream(device=self.device)
        self.zero_buffs = []

    def alloc(self, shape, dtype=torch.float32, zero_buf=False):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        if zero_buf:
            self.zero_buffs.append(buf)
        return buf

    def get_shared(self, name, shape, dtype=torch.float32, zero_buf=False):
        key = (name, shape, dtype)
        buf = self.buffers.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = self.alloc(shape, dtype, zero_buf=zero_buf)
            self.buffers[key] = buf
        return buf

    def zero_async(self):
        with torch.cuda.stream(self.stream):
            for target in self.zero_buffs:
                target.zero_()


gpu_memory_pool = GPUMemoryPool()
