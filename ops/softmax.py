"""Row-wise softmax.

    y[i][j] = exp(x[i][j] - max_j x[i][j]) / sum_j exp(x[i][j] - max_j x[i][j])

Memory-bound like RMSNorm, but with two reductions per row (max and sum)
instead of one. The PyTorch baseline is torch.softmax, which is already a
single fused ATen kernel, so unlike RMSNorm there's no easy fusion win here.
Beating it means doing the same work with better memory access.

All impls compute in fp32 and round once at the end.
"""

import torch

from bench import cuda_ext, registry

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def eager_softmax(x):
    # ATen's CUDA softmax accumulates fp16/bf16 inputs in fp32
    return torch.softmax(x, dim=-1)


_compiled = {}


def compiled_softmax(x):
    if "softmax" not in _compiled:
        _compiled["softmax"] = torch.compile(eager_softmax, dynamic=False)
    return _compiled["softmax"](x)


if HAS_TRITON:

    @triton.jit
    def _softmax_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
        # one program per row, the whole row is loaded at once
        row = tl.program_id(axis=0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        # padding lanes read -inf so they become exp(-inf) = 0 in the sum
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)

        shifted = x - tl.max(x, axis=0)
        numerator = tl.exp(shifted)
        y = numerator / tl.sum(numerator, axis=0)

        tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty),
                 mask=mask)


def triton_softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    # the kernel holds a whole row in registers, which stops fitting
    # somewhere past this
    assert n_cols <= 16384, "triton_softmax: row too long"
    block = triton.next_power_of_2(n_cols)
    if block >= 8192:
        num_warps = 16
    elif block >= 2048:
        num_warps = 8
    else:
        num_warps = 4

    _softmax_kernel[(n_rows,)](x, y, x.stride(0), n_cols,
                               BLOCK=block, num_warps=num_warps)
    return y


def cuda_softmax(x):
    mod = cuda_ext.load_kernel("kb_softmax", ["softmax.cu"])
    return mod.softmax(x)


@registry.register
class Softmax(registry.Op):

    name = "softmax"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)

    def configs(self):
        return [
            {"rows": 4096, "cols": 1024},
            {"rows": 4096, "cols": 2048},
            {"rows": 4096, "cols": 4096},
            {"rows": 8192, "cols": 4096},
            {"rows": 2048, "cols": 8192},
            {"rows": 16384, "cols": 2048},
        ]

    def config_label(self, cfg):
        return "{}x{}".format(cfg["rows"], cfg["cols"])

    def make_inputs(self, cfg, dtype, device, generator):
        x64 = torch.empty((cfg["rows"], cfg["cols"]), dtype=torch.float64,
                          device=device)
        # std 3 so rows have a real spread and the max-subtraction matters
        x64.normal_(0.0, 3.0, generator=generator)
        x, x_ref = registry.cast_pair(x64, dtype)
        del x64
        return (x,), (x_ref,)

    def reference(self, inputs_fp64):
        (x,) = inputs_fp64
        return torch.softmax(x, dim=-1)

    def flops(self, cfg):
        # per element: max compare, subtract, exp, sum, divide
        return 5 * cfg["rows"] * cfg["cols"]

    def bytes(self, cfg, dtype):
        itemsize = torch.tensor([], dtype=dtype).element_size()
        # x read once (the CUDA kernel's second read should hit L2), y written once
        return 2 * cfg["rows"] * cfg["cols"] * itemsize

    def impls(self, dtype):
        out = {"eager": eager_softmax, "compile": compiled_softmax,
               "cuda": cuda_softmax}
        if HAS_TRITON:
            out["triton"] = triton_softmax
        return out
