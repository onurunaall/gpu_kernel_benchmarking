// Row-wise softmax:
//   y[i][j] = exp(x[i][j] - max_i) / sum_j(exp(x[i][j] - max_i))
//
// Subtracting the row max keeps exp() from overflowing. The textbook version
// needs three passes over the row: find the max, sum the exponentials,
// write the output. This is the "online softmax" version with two passes:
// the first pass tracks the running max and the running sum together, and
// rescales the sum whenever the max grows:
//
//   new_max = max(old_max, v)
//   sum     = sum * exp(old_max - new_max) + exp(v - new_max)
//
// One block per row, fp32 math regardless of input dtype.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

// partial softmax statistics: sum is relative to max, sum = sum_j exp(x_j - max)
struct MaxSum {
  float max;
  float sum;
};

__device__ MaxSum empty_max_sum() { return MaxSum{-INFINITY, 0.0f}; }

// Moves a partial sum from old_max to new_max. An empty partial (max = -inf)
// contributes nothing; checking for it avoids exp(-inf - -inf) = NaN.
__device__ float rescale(float sum, float old_max, float new_max) {
  return (old_max == -INFINITY) ? 0.0f : sum * expf(old_max - new_max);
}

__device__ MaxSum combine(MaxSum a, MaxSum b) {
  const float new_max = fmaxf(a.max, b.max);
  return MaxSum{new_max,
                rescale(a.sum, a.max, new_max) + rescale(b.sum, b.max, new_max)};
}

// after this every lane of the warp holds the combined value
__device__ MaxSum warp_combine(MaxSum v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    MaxSum other;
    other.max = __shfl_xor_sync(0xffffffffu, v.max, offset);
    other.sum = __shfl_xor_sync(0xffffffffu, v.sum, offset);
    v = combine(v, other);
  }
  return v;
}

// after this every thread of the block holds the combined value
__device__ MaxSum block_combine(MaxSum v) {
  __shared__ MaxSum partials[32];  // one slot per warp, max block is 1024
  __shared__ MaxSum total;

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int n_warps = (blockDim.x + 31) >> 5;

  v = warp_combine(v);
  if (lane == 0) partials[wid] = v;
  __syncthreads();

  if (wid == 0) {
    v = (lane < n_warps) ? partials[lane] : empty_max_sum();
    v = warp_combine(v);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

template <typename scalar_t>
__global__ void softmax_fwd(const scalar_t* __restrict__ x,
                            scalar_t* __restrict__ y,
                            int n_cols) {
  const int64_t row = blockIdx.x;
  const scalar_t* x_row = x + row * static_cast<int64_t>(n_cols);
  scalar_t* y_row = y + row * static_cast<int64_t>(n_cols);

  // pass 1: running max and running sum, per thread, then per block
  MaxSum stats = empty_max_sum();
  for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
    const float v = static_cast<float>(x_row[j]);
    const float new_max = fmaxf(stats.max, v);
    stats.sum = rescale(stats.sum, stats.max, new_max) + expf(v - new_max);
    stats.max = new_max;
  }
  stats = block_combine(stats);

  // pass 2: normalise and write
  const float inv_sum = 1.0f / stats.sum;
  for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
    const float v = static_cast<float>(x_row[j]);
    y_row[j] = static_cast<scalar_t>(expf(v - stats.max) * inv_sum);
  }
}

int block_size_for(int n_cols) {
  if (n_cols >= 4096) return 1024;
  if (n_cols >= 2048) return 512;
  if (n_cols >= 512) return 256;
  return 128;
}

}  // namespace

torch::Tensor softmax(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, cols)");

  auto y = torch::empty_like(x);
  const int64_t n_rows = x.size(0);
  const int n_cols = static_cast<int>(x.size(1));
  if (n_rows == 0 || n_cols == 0) return y;
  TORCH_CHECK(n_rows <= 2147483647LL, "too many rows for a 1D grid");

  const int threads = block_size_for(n_cols);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "softmax", [&] {
        softmax_fwd<scalar_t><<<static_cast<int>(n_rows), threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n_cols);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax, "row-wise softmax, online max/sum, fp32 math");
}
