// Matrix multiply C = A @ B, A is (M, K), B is (K, N), all row-major.
//
// Compute-bound, and this kernel is expected to lose to cuBLAS by a wide
// margin: it uses plain FMA units, while cuBLAS (and Triton's tl.dot) use
// tensor cores for fp16/bf16. It's the readable baseline, not a contender.
//
// Two standard tricks, each with one job:
//
// 1. Shared-memory tiling. A block computes a BM x BN tile of C. It walks
//    along K in steps of BK, loading a BM x BK tile of A and a BK x BN tile
//    of B into shared memory once, and every thread reuses them from there.
//    Without it every thread reads its own row of A and column of B from
//    global memory.
//
// 2. Register blocking. Each thread computes a TM x TN patch of C instead
//    of one element. Per step along K it reads TM values of A and TN values
//    of B from shared memory and does TM * TN FMAs with them, so it does
//    more math per shared-memory read.
//
// Accumulates in fp32 whatever the input dtype.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int BM = 64;  // rows of C per block
constexpr int BN = 64;  // columns of C per block
constexpr int BK = 16;  // step along K
constexpr int TM = 4;   // rows of C per thread
constexpr int TN = 4;   // columns of C per thread
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256

template <typename scalar_t>
__global__ void matmul_tiled(const scalar_t* __restrict__ A,
                             const scalar_t* __restrict__ B,
                             scalar_t* __restrict__ C,
                             int M, int N, int K) {
  // A's tile is stored transposed (k-major) so the inner loop reads a
  // column of it contiguously. The +1 padding puts consecutive k rows in
  // different shared-memory banks, otherwise the transposed store is a
  // 16-way bank conflict.
  __shared__ float A_tile[BK][BM + 1];
  __shared__ float B_tile[BK][BN];

  // which TM x TN patch of the block's tile this thread owns
  const int thread_col = threadIdx.x % (BN / TN);
  const int thread_row = threadIdx.x / (BN / TN);

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // cooperative load of A[block_row : +BM, k0 : +BK], zero outside A
    for (int i = threadIdx.x; i < BM * BK; i += THREADS) {
      const int r = i / BK;
      const int k = i % BK;
      const int global_r = block_row + r;
      const int global_k = k0 + k;
      A_tile[k][r] = (global_r < M && global_k < K)
          ? static_cast<float>(A[static_cast<int64_t>(global_r) * K + global_k])
          : 0.0f;
    }
    // cooperative load of B[k0 : +BK, block_col : +BN], zero outside B
    for (int i = threadIdx.x; i < BK * BN; i += THREADS) {
      const int k = i / BN;
      const int c = i % BN;
      const int global_k = k0 + k;
      const int global_c = block_col + c;
      B_tile[k][c] = (global_k < K && global_c < N)
          ? static_cast<float>(B[static_cast<int64_t>(global_k) * N + global_c])
          : 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      float a[TM];
      float b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = A_tile[k][thread_row * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = B_tile[k][thread_col * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
      }
    }
    // everyone has to finish reading the tiles before they're overwritten
    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      const int r = block_row + thread_row * TM + i;
      const int c = block_col + thread_col * TN + j;
      if (r < M && c < N) {
        C[static_cast<int64_t>(r) * N + c] = static_cast<scalar_t>(acc[i][j]);
      }
    }
  }
}

}  // namespace

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions don't match");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "dtype mismatch");
  TORCH_CHECK(a.size(0) < (1 << 30) && a.size(1) < (1 << 30) && b.size(1) < (1 << 30),
              "dimensions must fit in an int");

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));
  auto c = torch::empty({M, N}, a.options());
  if (M == 0 || N == 0) return c;

  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      a.scalar_type(), "matmul", [&] {
        matmul_tiled<scalar_t><<<grid, THREADS, 0, stream>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(), M, N, K);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul, "tiled matmul, shared memory + register blocking");
}
