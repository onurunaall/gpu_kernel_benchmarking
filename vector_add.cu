// Elementwise add. 1 FLOP per 3*sizeof(T) bytes, so this is a pure bandwidth
// test and its job is to confirm the harness agrees with the measured copy
// roof. If this doesn't land near the roof the harness is wrong, not the
// kernel.
//
// Only real trick is 128-bit access. A 32-bit per-thread load issues 4
// sectors per request; a 128-bit load moves the same sectors with a quarter
// of the requests, so less LSU and instruction pressure. Visible in ncu as
// sectors-per-request.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

// 16 bytes whatever the scalar type is, so one thread does exactly one
// 128-bit load per operand
template <typename scalar_t>
struct alignas(16) Vec16 {
  static constexpr int N = 16 / sizeof(scalar_t);
  scalar_t v[N];
};

template <typename scalar_t>
__global__ void vadd_vec(const Vec16<scalar_t>* __restrict__ a,
                         const Vec16<scalar_t>* __restrict__ b,
                         Vec16<scalar_t>* __restrict__ out,
                         int64_t n_vec) {
  constexpr int N = Vec16<scalar_t>::N;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < n_vec; i += stride) {
    Vec16<scalar_t> va = a[i];
    Vec16<scalar_t> vb = b[i];
    Vec16<scalar_t> vo;

#pragma unroll
    for (int k = 0; k < N; ++k) {
      // float accumulate even for half. Costs nothing here and keeps the
      // fp64 error comparison meaningful.
      vo.v[k] = static_cast<scalar_t>(static_cast<float>(va.v[k]) +
                                      static_cast<float>(vb.v[k]));
    }
    out[i] = vo;
  }
}

// leftover elements that don't fill a whole 16-byte vector
template <typename scalar_t>
__global__ void vadd_tail(const scalar_t* __restrict__ a,
                          const scalar_t* __restrict__ b,
                          scalar_t* __restrict__ out,
                          int64_t start, int64_t n) {
  const int64_t i =
      start + blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    out[i] = static_cast<scalar_t>(static_cast<float>(a[i]) +
                                   static_cast<float>(b[i]));
  }
}

}  // namespace

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(a.sizes() == b.sizes(), "size mismatch");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "dtype mismatch");

  auto out = torch::empty_like(a);
  const int64_t n = a.numel();
  if (n == 0) return out;

  const int threads = 256;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      a.scalar_type(), "vector_add", [&] {
        constexpr int VN = Vec16<scalar_t>::N;
        const int64_t n_vec = n / VN;
        const int64_t tail_start = n_vec * VN;

        if (n_vec > 0) {
          // cap the grid so each thread handles several vectors instead of
          // launching millions of blocks that do one load each
          int64_t blocks = (n_vec + threads - 1) / threads;
          const int64_t max_blocks = 65535 * 4;
          if (blocks > max_blocks) blocks = max_blocks;

          vadd_vec<scalar_t><<<blocks, threads, 0, stream>>>(
              reinterpret_cast<const Vec16<scalar_t>*>(a.data_ptr<scalar_t>()),
              reinterpret_cast<const Vec16<scalar_t>*>(b.data_ptr<scalar_t>()),
              reinterpret_cast<Vec16<scalar_t>*>(out.data_ptr<scalar_t>()),
              n_vec);
        }

        if (tail_start < n) {
          const int64_t tail = n - tail_start;
          const int64_t blocks = (tail + threads - 1) / threads;
          vadd_tail<scalar_t><<<blocks, threads, 0, stream>>>(
              a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(), tail_start, n);
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "elementwise add, 128-bit vectorized");
}
