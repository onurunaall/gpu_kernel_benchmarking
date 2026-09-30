// 2D transpose: out[j][i] = in[i][j], in is (rows, cols), out is (cols, rows).
//
// No arithmetic at all, so this is purely about memory access patterns. The
// naive version reads rows (coalesced) but writes columns, so each warp's
// store touches 32 different cache lines. The fix is the classic one:
//
// 1. a block reads a 32x32 tile with coalesced row reads into shared memory,
// 2. then writes it out, again as coalesced rows, reading the tile by column.
//
// The shared tile is padded to 33 columns. Without the padding, reading a
// column of a 32-wide float tile hits the same shared-memory bank 32 times
// (a 32-way bank conflict). The tile is stored as float for every input
// dtype: fp16/bf16 -> float -> fp16/bf16 is exact, and it keeps the bank
// math identical for all dtypes.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;  // block is 32 x 8 threads, each moves 4 elements

template <typename scalar_t>
__global__ void transpose_tiled(const scalar_t* __restrict__ in,
                                scalar_t* __restrict__ out,
                                int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];

  // read the tile starting at in[blockIdx.y * TILE][blockIdx.x * TILE]
  int col = blockIdx.x * TILE + threadIdx.x;
  int row = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (row + j < rows && col < cols) {
      tile[threadIdx.y + j][threadIdx.x] =
          static_cast<float>(in[static_cast<int64_t>(row + j) * cols + col]);
    }
  }
  __syncthreads();

  // write it to the mirrored position in out, which has `rows` columns
  col = blockIdx.y * TILE + threadIdx.x;
  row = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (row + j < cols && col < rows) {
      out[static_cast<int64_t>(row + j) * rows + col] =
          static_cast<scalar_t>(tile[threadIdx.x][threadIdx.y + j]);
    }
  }
}

}  // namespace

torch::Tensor transpose(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, cols)");
  TORCH_CHECK(x.size(0) < (1 << 30) && x.size(1) < (1 << 30),
              "dimensions must fit in an int");

  const int rows = static_cast<int>(x.size(0));
  const int cols = static_cast<int>(x.size(1));
  auto out = torch::empty({cols, rows}, x.options());
  if (rows == 0 || cols == 0) return out;

  const dim3 block(TILE, BLOCK_ROWS);
  const dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  TORCH_CHECK(grid.y <= 65535, "too many rows for the grid");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "transpose", [&] {
        transpose_tiled<scalar_t><<<grid, block, 0, stream>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), rows, cols);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose", &transpose, "2D transpose through a padded shared tile");
}
