"""2D transpose, out = x.T as a new contiguous tensor.

Zero FLOPs: every byte is read once and written once, so GB/s against the
copy roof is the only metric. A plain copy runs at the roof; how close a
transpose gets tells you how well it hides the strided side of the access
(rows in, columns out) behind coalesced reads and writes.
"""

import torch

from bench import cuda_ext, registry

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def eager_transpose(x):
    # .t() only swaps strides, .contiguous() is the actual data movement
    return x.t().contiguous()


_compiled = {}


def compiled_transpose(x):
    if "transpose" not in _compiled:
        _compiled["transpose"] = torch.compile(eager_transpose, dynamic=False)
    return _compiled["transpose"](x)


if HAS_TRITON:

    @triton.jit
    def _transpose_kernel(x_ptr, y_ptr, rows, cols, BLOCK: tl.constexpr):
        # each program moves one BLOCK x BLOCK tile
        r = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
        c = tl.program_id(axis=1) * BLOCK + tl.arange(0, BLOCK)

        tile = tl.load(x_ptr + r[:, None] * cols + c[None, :],
                       mask=(r[:, None] < rows) & (c[None, :] < cols))

        # y is (cols, rows): y[c][r] = x[r][c]. tl.trans flips the tile so
        # the store is row-major in y; the compiler stages it through
        # shared memory, like the hand-written CUDA version does explicitly.
        tl.store(y_ptr + c[:, None] * rows + r[None, :], tl.trans(tile),
                 mask=(c[:, None] < cols) & (r[None, :] < rows))


def triton_transpose(x):
    rows, cols = x.shape
    y = torch.empty((cols, rows), dtype=x.dtype, device=x.device)
    block = 64
    grid = (triton.cdiv(rows, block), triton.cdiv(cols, block))
    _transpose_kernel[grid](x, y, rows, cols, BLOCK=block, num_warps=4)
    return y


def cuda_transpose(x):
    mod = cuda_ext.load_kernel("kb_transpose", ["transpose.cu"])
    return mod.transpose(x)


@registry.register
class Transpose(registry.Op):

    name = "transpose"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)

    def configs(self):
        return [
            {"rows": 1024, "cols": 1024},
            {"rows": 2048, "cols": 2048},
            {"rows": 4096, "cols": 4096},
            {"rows": 8192, "cols": 8192},
            {"rows": 4096, "cols": 16384},
            {"rows": 3000, "cols": 5000},  # not a multiple of any tile size
        ]

    def config_label(self, cfg):
        return "{}x{}".format(cfg["rows"], cfg["cols"])

    def make_inputs(self, cfg, dtype, device, generator):
        x64 = torch.empty((cfg["rows"], cfg["cols"]), dtype=torch.float64,
                          device=device)
        x64.uniform_(-1.0, 1.0, generator=generator)
        x, x_ref = registry.cast_pair(x64, dtype)
        del x64
        return (x,), (x_ref,)

    def reference(self, inputs_fp64):
        (x,) = inputs_fp64
        return x.t().contiguous()

    def flops(self, cfg):
        return 0

    def bytes(self, cfg, dtype):
        itemsize = torch.tensor([], dtype=dtype).element_size()
        return 2 * cfg["rows"] * cfg["cols"] * itemsize  # read once, write once

    def impls(self, dtype):
        out = {"eager": eager_transpose, "compile": compiled_transpose,
               "cuda": cuda_transpose}
        if HAS_TRITON:
            out["triton"] = triton_transpose
        return out
