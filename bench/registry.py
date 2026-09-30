"""Op interface. One file per kernel in ops/, the harness does the rest."""

import abc

import torch

_OPS = {}


def register(cls):
    op = cls()
    if op.name in _OPS:
        raise ValueError("duplicate op name: {}".format(op.name))
    _OPS[op.name] = op
    return cls


def get(name):
    if name not in _OPS:
        raise KeyError("unknown op '{}', have: {}".format(name, sorted(_OPS)))
    return _OPS[name]


def all_ops():
    return dict(_OPS)


class Op(abc.ABC):

    name = None
    supported_dtypes = (torch.float16, torch.float32)

    @abc.abstractmethod
    def configs(self):
        """Size sweep. One size isn't a result, the curve is the result."""

    @abc.abstractmethod
    def config_label(self, cfg):
        """Short label like '4096x4096'."""

    @abc.abstractmethod
    def make_inputs(self, cfg, dtype, device, generator):
        """Return (working_inputs, oracle_inputs).

        Use cast_pair. The oracle has to see the rounded values, not the
        original fp64 ones, otherwise input rounding error gets mixed into
        the kernel error and you can't separate them.
        """

    @abc.abstractmethod
    def reference(self, inputs_fp64):
        """Ground truth in fp64.

        Not PyTorch's fp16 output. For a reduction that's just a different
        summation order, so comparing against it measures agreement, not
        correctness.
        """

    @abc.abstractmethod
    def flops(self, cfg):
        """Useful FLOPs. Count the algorithm, not the instructions."""

    @abc.abstractmethod
    def bytes(self, cfg, dtype):
        """Algorithmic minimum DRAM traffic: inputs read once, output written
        once.

        Easy to reproduce, but a cache-friendly kernel can appear to beat
        100% of DRAM bandwidth against it. The profiler reports measured
        dram__bytes_* next to this and the ratio is the interesting part.
        """

    @abc.abstractmethod
    def impls(self, dtype):
        """name -> callable(*inputs) -> output. Skip any the dtype can't do."""

    def skip_reason(self, cfg, dtype, device):
        return None

    def error(self, actual, expected_fp64):
        """Error of one output against the fp64 oracle. Override when an
        element-wise relative error is the wrong question (see matmul)."""
        return max_rel_error(actual, expected_fp64)

    def tolerance(self, dtype):
        # Loose on purpose. These catch a broken kernel; the reported
        # max_rel_error is the number I actually care about.
        return {
            torch.float16: 2e-2,
            torch.bfloat16: 1e-1,
            torch.float32: 1e-4,
        }.get(dtype, 1e-10)


def cast_pair(x_fp64, dtype):
    """Round to dtype, then promote the rounded value back to fp64."""
    working = x_fp64.to(dtype)
    return working, working.to(torch.float64)


def max_rel_error(actual, expected_fp64):
    """Largest element-wise relative error.

    The denominator never drops below the smallest *normal* number of the
    output dtype. Below that the dtype is subnormal and only has absolute
    precision left: an fp16 RMSNorm output of 3e-7 can only be stored to
    within about 3e-8, which is a 10% "relative error" that says nothing
    about the kernel. With the floor, a correctly rounded result scores at
    most half an ulp everywhere, subnormal range included.
    """
    a = actual.to(torch.float64)
    tiny = torch.finfo(actual.dtype).tiny
    denom = torch.clamp(expected_fp64.abs(), min=tiny)
    return float((a - expected_fp64).abs().div(denom).max().item())


def max_error_vs_largest(actual, expected_fp64):
    """Largest absolute error divided by the largest |expected| value.

    For outputs that are sums of many signed terms (matmul) some elements
    land near zero by cancellation. Their element-wise relative error is
    huge for every implementation, cuBLAS included, so it stops telling a
    correct kernel from a broken one. Scaling by the largest output doesn't
    have that problem.
    """
    a = actual.to(torch.float64)
    largest = float(expected_fp64.abs().max().item())
    return float((a - expected_fp64).abs().max().item()) / max(largest, 1e-300)
