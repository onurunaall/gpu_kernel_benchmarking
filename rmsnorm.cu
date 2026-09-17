// RMSNorm forward:
//   y[i][j] = x[i][j] * rsqrt(mean_j(x[i][j]^2) + eps) * w[j]
//
// Eager torch does this as a handful of separate kernels (square, mean, add,
// rsqrt, mul, mul), each reading and writing the whole tensor. This reads x
// twice and writes y once. The speedup is from killing DRAM round trips and
// launches, not from out-computing anyone.
//
// About the second read: on a big-L2 part like the 4090 (72 MB) the row
// re-read should mostly hit L2, so measured dram__bytes_read ought to land
// near the analytic minimum rather than 2x it. That gap between analytic and
// measured is the thing worth writing up.
//
// Sum of squares accumulates in float regardless of input dtype. Naive fp16
// accumulation over 4096 elements loses a lot of significand and the fp64
// oracle catches it immediately.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

__inline__ __device__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// result is only valid in thread 0, caller broadcasts through shared
__inline__ __device__ float block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int n_warps = (blockDim.x + 31) >> 5;

  v = warp_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  v = (threadIdx.x < n_warps) ? smem[lane] : 0.0f;
  if (wid == 0) v = warp_sum(v);
  return v;
}

template <typename scalar_t>
__global__ void rmsnorm_fwd(const scalar_t* __restrict__ x,
                            const scalar_t* __restrict__ w,
                            scalar_t* __restrict__ y,
                            int n_cols, float eps) {
  __shared__ float partials[32];  // one slot per warp, max block is 1024
  __shared__ float s_scale;

  const int64_t row = blockIdx.x;
  const scalar_t* x_row = x + row * static_cast<int64_t>(n_cols);
  scalar_t* y_row = y + row * static_cast<int64_t>(n_cols);

  float ss = 0.0f;
  for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
    const float v = static_cast<float>(x_row[j]);
    ss += v * v;
  }

  ss = block_sum(ss, partials);
  if (threadIdx.x == 0) {
    s_scale = rsqrtf(ss / static_cast<float>(n_cols) + eps);
  }
  __syncthreads();

  const float scale = s_scale;
  for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
    const float v = static_cast<float>(x_row[j]) * scale *
                    static_cast<float>(w[j]);
    y_row[j] = static_cast<scalar_t>(v);
  }
}

int block_size_for(int n_cols) {
  if (n_cols >= 4096) return 1024;
  if (n_cols >= 2048) return 512;
  if (n_cols >= 512) return 256;
  return 128;
}

}  // namespace

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, cols)");
  TORCH_CHECK(w.dim() == 1 && w.size(0) == x.size(1), "weight shape mismatch");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "dtype mismatch");

  auto y = torch::empty_like(x);
  const int64_t n_rows = x.size(0);
  const int n_cols = static_cast<int>(x.size(1));
  if (n_rows == 0 || n_cols == 0) return y;

  // TODO: 2D grid if I ever need more rows than this
  TORCH_CHECK(n_rows <= 2147483647LL, "too many rows for a 1D grid");

  const int threads = block_size_for(n_cols);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "rmsnorm", [&] {
        rmsnorm_fwd<scalar_t><<<static_cast<int>(n_rows), threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(), n_cols, static_cast<float>(eps));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "fused RMSNorm forward, fp32 accumulate");
}
