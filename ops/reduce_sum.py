"""Full sum reduction: one fp32 scalar from a 1D tensor.

Memory-bound (1 add per element loaded). The interesting part is how each
impl combines partial sums across blocks: the CUDA kernel uses one atomic
per block, the Triton kernel one atomic per program, PyTorch its own
multi-stage reduction.

Inputs are in [0, 1), so the sum has no cancellation and the relative error
measures accumulation error only. The output is fp32 for every input dtype;
an fp16 result would overflow (65504 max) on all but the smallest size.
"""

import torch

from bench import cuda_ext, registry

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def eager_sum(x):
    return torch.sum(x, dtype=torch.float32)


_compiled = {}


def compiled_sum(x):
    if "sum" not in _compiled:
        _compiled["sum"] = torch.compile(eager_sum, dynamic=False)
    return _compiled["sum"](x)


if HAS_TRITON:

    @triton.jit
    def _sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        # each program sums one BLOCK-sized chunk, then adds it to the output
        offs = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
        tl.atomic_add(out_ptr, tl.sum(x, axis=0))


def triton_sum(x):
    out = torch.zeros((), dtype=torch.float32, device=x.device)
    n = x.numel()
    block = 4096
    _sum_kernel[(triton.cdiv(n, block),)](x, out, n, BLOCK=block, num_warps=8)
    return out


def cuda_sum(x):
    mod = cuda_ext.load_kernel("kb_reduce_sum", ["reduce_sum.cu"])
    return mod.reduce_sum(x)


@registry.register
class ReduceSum(registry.Op):

    name = "reduce_sum"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)

    def configs(self):
        return [{"n": 1 << k} for k in (20, 22, 24, 26, 28)]

    def config_label(self, cfg):
        return "n=2^{}".format(cfg["n"].bit_length() - 1)

    def make_inputs(self, cfg, dtype, device, generator):
        x64 = torch.empty(cfg["n"], dtype=torch.float64, device=device)
        x64.uniform_(0.0, 1.0, generator=generator)
        x, x_ref = registry.cast_pair(x64, dtype)
        del x64
        return (x,), (x_ref,)

    def reference(self, inputs_fp64):
        (x,) = inputs_fp64
        return x.sum()

    def flops(self, cfg):
        return cfg["n"]

    def bytes(self, cfg, dtype):
        itemsize = torch.tensor([], dtype=dtype).element_size()
        return cfg["n"] * itemsize + 4  # read x, write one fp32

    def impls(self, dtype):
        out = {"eager": eager_sum, "compile": compiled_sum, "cuda": cuda_sum}
        if HAS_TRITON:
            out["triton"] = triton_sum
        return out
