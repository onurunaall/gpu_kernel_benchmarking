"""RMSNorm forward.

    y[i][j] = x[i][j] * rsqrt(mean_j(x[i][j]^2) + eps) * w[j]

All four impls do the same arithmetic: accumulate in fp32, apply the scale in
fp32, cast once at the end. On purpose. If eager used a different precision
policy I'd be comparing two different functions and the error column would
mean nothing.
"""

import functools

import torch

from bench import cuda_ext, registry

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

EPS = 1e-6


def eager_rmsnorm(x, w, eps):
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale * w.float()).to(x.dtype)


_compiled = {}


def compiled_rmsnorm(x, w, eps):
    if "rmsnorm" not in _compiled:
        _compiled["rmsnorm"] = torch.compile(eager_rmsnorm, dynamic=False)
    return _compiled["rmsnorm"](x, w, eps)


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, n_cols, eps,
                        BLOCK: tl.constexpr):
        row = tl.program_id(axis=0)
        x_row = x_ptr + row * stride
        y_row = y_ptr + row * stride

        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < n_cols
            v = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            acc += v * v
        scale = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < n_cols
            v = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(y_row + cols, (v * scale * g).to(y_ptr.dtype.element_ty),
                     mask=mask)


def triton_rmsnorm(x, w, eps):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    block = max(128, min(triton.next_power_of_2(n_cols), 4096))
    if block >= 2048:
        num_warps = 16
    elif block >= 1024:
        num_warps = 8
    else:
        num_warps = 4

    _rmsnorm_kernel[(n_rows,)](x, w, y, x.stride(0), n_cols, eps,
                               BLOCK=block, num_warps=num_warps)
    return y


def cuda_rmsnorm(x, w, eps):
    mod = cuda_ext.load_kernel("kb_rmsnorm", ["rmsnorm.cu"])
    return mod.rmsnorm(x, w, eps)


@registry.register
class RMSNorm(registry.Op):

    name = "rmsnorm"
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
        rows, cols = cfg["rows"], cfg["cols"]
        x64 = torch.empty((rows, cols), dtype=torch.float64, device=device)
        x64.normal_(0.0, 1.0, generator=generator)
        w64 = torch.empty(cols, dtype=torch.float64, device=device)
        w64.normal_(1.0, 0.1, generator=generator)

        x, x_ref = registry.cast_pair(x64, dtype)
        w, w_ref = registry.cast_pair(w64, dtype)
        del x64, w64
        return (x, w), (x_ref, w_ref)

    def reference(self, inputs_fp64):
        x, w = inputs_fp64
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return x * scale * w

    def flops(self, cfg):
        # per element: square, accumulate, mul by scale, mul by weight
        return 4 * cfg["rows"] * cfg["cols"]

    def bytes(self, cfg, dtype):
        itemsize = torch.tensor([], dtype=dtype).element_size()
        rows, cols = cfg["rows"], cfg["cols"]
        # x read once (minimum, second pass should hit L2), w read once,
        # y written once
        return (2 * rows * cols + cols) * itemsize

    def impls(self, dtype):
        out = {
            "eager": functools.partial(eager_rmsnorm, eps=EPS),
            "compile": functools.partial(compiled_rmsnorm, eps=EPS),
            "cuda": functools.partial(cuda_rmsnorm, eps=EPS),
        }
        if HAS_TRITON:
            out["triton"] = functools.partial(triton_rmsnorm, eps=EPS)
        return out
