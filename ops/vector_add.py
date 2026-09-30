"""Elementwise add.

Calibration, not a contest. AI is about 1/6 FLOP/byte in fp16, way under the
ridge, so all four should sit near the measured copy bandwidth. If they don't
the harness is broken and nothing after this is trustworthy.
"""

import torch

from bench import cuda_ext, registry

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def eager_add(a, b):
    return torch.add(a, b)


_compiled = {}


def compiled_add(a, b):
    if "add" not in _compiled:
        _compiled["add"] = torch.compile(eager_add, dynamic=False)
    return _compiled["add"](a, b)


if HAS_TRITON:

    @triton.jit
    def _vadd_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(o_ptr + offs, a + b, mask=mask)


def triton_add(a, b):
    out = torch.empty_like(a)
    n = a.numel()
    block = 1024
    _vadd_kernel[(triton.cdiv(n, block),)](a, b, out, n,
                                           BLOCK=block, num_warps=4)
    return out


def cuda_add(a, b):
    mod = cuda_ext.load_kernel("kb_vector_add", ["vector_add.cu"])
    return mod.vector_add(a, b)


@registry.register
class VectorAdd(registry.Op):

    name = "vector_add"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)

    def configs(self):
        # spans from well under L2 to several times L2
        return [{"n": 1 << k} for k in (20, 22, 24, 26, 27, 28)]

    def config_label(self, cfg):
        return "n=2^{}".format(cfg["n"].bit_length() - 1)

    def make_inputs(self, cfg, dtype, device, generator):
        n = cfg["n"]
        a64 = torch.empty(n, dtype=torch.float64, device=device)
        a64.uniform_(-1.0, 1.0, generator=generator)
        b64 = torch.empty(n, dtype=torch.float64, device=device)
        b64.uniform_(-1.0, 1.0, generator=generator)

        a, a_ref = registry.cast_pair(a64, dtype)
        b, b_ref = registry.cast_pair(b64, dtype)
        del a64, b64
        return (a, b), (a_ref, b_ref)

    def reference(self, inputs_fp64):
        a, b = inputs_fp64
        return a + b

    def flops(self, cfg):
        return cfg["n"]

    def bytes(self, cfg, dtype):
        itemsize = torch.tensor([], dtype=dtype).element_size()
        return 3 * cfg["n"] * itemsize  # two reads one write

    def impls(self, dtype):
        out = {"eager": eager_add, "compile": compiled_add, "cuda": cuda_add}
        if HAS_TRITON:
            out["triton"] = triton_add
        return out
