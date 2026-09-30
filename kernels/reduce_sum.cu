// Sum of every element of a 1D tensor, returned as a float32 scalar.
//
// A full reduction has to combine values across blocks, which no single
// block can do alone. Three levels:
//
// 1. each thread sums a strided slice of the input in a register
//    (grid-stride loop, so a fixed-size grid covers any n),
// 2. the block combines its threads' sums with warp shuffles,
// 3. one thread per block atomically adds the block's sum to the output.
//
// The grid is capped at a few blocks per SM, so step 3 is a few hundred
// atomics, not millions. Atomics make the summation order vary from run to
// run, so the last bits of the result aren't deterministic.
//
// Accumulates in fp32 whatever the input dtype. The output is fp32 too: the
// sum of 2^26 values in [0, 1) is ~3e7, far past the fp16 max of 65504.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace {

__inline__ __device__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// result is only valid in thread 0
__inline__ __device__ float block_sum(float v) {
  __shared__ float partials[32];  // one slot per warp, max block is 1024
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int n_warps = (blockDim.x + 31) >> 5;

  v = warp_sum(v);
  if (lane == 0) partials[wid] = v;
  __syncthreads();

  v = (threadIdx.x < n_warps) ? partials[lane] : 0.0f;
  if (wid == 0) v = warp_sum(v);
  return v;
}

template <typename scalar_t>
__global__ void reduce_sum_kernel(const scalar_t* __restrict__ x,
                                  float* __restrict__ out,
                                  int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  float acc = 0.0f;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    acc += static_cast<float>(x[i]);
  }

  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}

}  // namespace

torch::Tensor reduce_sum(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "input must be contiguous");

  // zero-initialised because every block adds into it
  auto out = torch::zeros({}, x.options().dtype(torch::kFloat32));
  const int64_t n = x.numel();
  if (n == 0) return out;

  const int threads = 256;
  const int blocks_per_sm = 4;
  const int64_t max_blocks =
      static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
      blocks_per_sm;
  const int64_t blocks = std::min((n + threads - 1) / threads, max_blocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "reduce_sum", [&] {
        reduce_sum_kernel<scalar_t><<<static_cast<int>(blocks), threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<float>(), n);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum", &reduce_sum, "full sum to an fp32 scalar, one atomic per block");
}
