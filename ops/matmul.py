"""Matrix multiply, C = A @ B.

The one compute-bound op. Arithmetic intensity grows with size, so large
sizes sit far above the ridge and TFLOP/s against the measured cuBLAS roof is
the number that matters. Expect:

* eager / compile: cuBLAS. torch.compile hands plain matmuls to cuBLAS too,
  so these two should match, and eager is the compute roof by construction.
* triton: tl.dot uses tensor cores for fp16/bf16, should get close to cuBLAS.
* cuda: the readable tiled kernel on FMA units, far behind for fp16/bf16.

fp32 means real fp32 everywhere by default. torch.matmul follows the harness's
TF32 flag (off unless --tf32), and the Triton kernel reads the same flag, so
all impls do the same arithmetic.
"""

import torch

from bench import cuda_ext, registry

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def eager_matmul(a, b):
    return torch.matmul(a, b)


_compiled = {}


def compiled_matmul(a, b):
    if "matmul" not in _compiled:
        _compiled["matmul"] = torch.compile(eager_matmul, dynamic=False)
    return _compiled["matmul"](a, b)


if HAS_TRITON:

    # the autotuner times each config on the first call for every new
    # (M, N, K) and keeps the fastest. That first call is the correctness
    # check, so the tuning never lands inside the timed region.
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       PRECISION: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
        # each program computes one BLOCK_M x BLOCK_N tile of C
        rows = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                        mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
            b = tl.load(b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                        mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc, input_precision=PRECISION)

        tl.store(c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
                 acc.to(c_ptr.dtype.element_ty),
                 mask=(rows[:, None] < M) & (cols[None, :] < N))


def triton_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # only matters for fp32 inputs: "ieee" is true fp32, "tf32" is what
    # torch.matmul does when TF32 is allowed
    precision = "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1), PRECISION=precision)
    return c


def cuda_matmul(a, b):
    mod = cuda_ext.load_kernel("kb_matmul", ["matmul.cu"])
    return mod.matmul(a, b)


@registry.register
class Matmul(registry.Op):

    name = "matmul"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)

    def configs(self):
        # square sizes, plus one that isn't a multiple of any tile size so
        # the edge handling gets exercised
        return [{"m": n, "n": n, "k": n} for n in (512, 1024, 2048, 3000, 4096, 8192)]

    def config_label(self, cfg):
        return "{}x{}x{}".format(cfg["m"], cfg["n"], cfg["k"])

    def make_inputs(self, cfg, dtype, device, generator):
        a64 = torch.empty((cfg["m"], cfg["k"]), dtype=torch.float64, device=device)
        a64.normal_(0.0, 1.0, generator=generator)
        b64 = torch.empty((cfg["k"], cfg["n"]), dtype=torch.float64, device=device)
        b64.normal_(0.0, 1.0, generator=generator)

        a, a_ref = registry.cast_pair(a64, dtype)
        b, b_ref = registry.cast_pair(b64, dtype)
        del a64, b64
        return (a, b), (a_ref, b_ref)

    def reference(self, inputs_fp64):
        a, b = inputs_fp64
        return a @ b

    def error(self, actual, expected_fp64):
        # outputs are sums of K signed products, some cancel to near zero
        return registry.max_error_vs_largest(actual, expected_fp64)

    def tolerance(self, dtype):
        if dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
            return 5e-3  # TF32 keeps 10 mantissa bits
        return super().tolerance(dtype)

    def flops(self, cfg):
        return 2 * cfg["m"] * cfg["n"] * cfg["k"]

    def bytes(self, cfg, dtype):
        itemsize = torch.tensor([], dtype=dtype).element_size()
        m, n, k = cfg["m"], cfg["n"], cfg["k"]
        return (m * k + k * n + m * n) * itemsize

    def impls(self, dtype):
        out = {"eager": eager_matmul, "compile": compiled_matmul,
               "cuda": cuda_matmul}
        if HAS_TRITON:
            out["triton"] = triton_matmul
        return out
