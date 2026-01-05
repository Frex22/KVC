#include "allocator.h"
#include <cstdio>
#include <stdexcept>

namespace kv {

void checkCuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
    throw std::runtime_error("CUDA failure");
  }
}

__global__ void sanity_kernel(int32_t* top, uint32_t* free_count, uint32_t num_blocks) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Set expected initial values (we'll verify on host)
    *top = static_cast<int32_t>(num_blocks);
    *free_count = 0;
  }
}

__global__ void init_free_stack_kernel(uint32_t* free_stack, uint32_t num_blocks) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_blocks) {
    free_stack[idx] = num_blocks - 1u - idx; // LIFO order
  }
}


void init_system(AllocatorSystem& sys) {
  // Compute block counts from pool_bytes
  const std::size_t bytes_per_block = bytes_per_block_k_plus_v();
  if (bytes_per_block == 0) throw std::runtime_error("bytes_per_block is zero");

  // How many physical blocks fit
  const std::uint32_t num_blocks =
      static_cast<std::uint32_t>(sys.cfg.pool_bytes / bytes_per_block);

  if (num_blocks == 0) throw std::runtime_error("pool too small for even one block");

  sys.pool.num_blocks = num_blocks;

  // Allocate K and V pools
  sys.pool.bytes_per_tensor = static_cast<std::size_t>(num_blocks) * bytes_per_block_per_tensor();
  checkCuda(cudaMalloc(&sys.pool.K, sys.pool.bytes_per_tensor), "cudaMalloc K_pool");
  checkCuda(cudaMalloc(&sys.pool.V, sys.pool.bytes_per_tensor), "cudaMalloc V_pool");

  // Allocate allocator metadata
  checkCuda(cudaMalloc(&sys.alloc.free_stack, sizeof(std::uint32_t) * num_blocks), "cudaMalloc free_stack");
  checkCuda(cudaMalloc(&sys.alloc.top, sizeof(std::int32_t)), "cudaMalloc top");
  checkCuda(cudaMalloc(&sys.alloc.free_count, sizeof(std::uint32_t)), "cudaMalloc free_count");

  // Free queue capacity: at most all blocks (simple v0)
  sys.alloc.free_queue_cap = num_blocks;
  checkCuda(cudaMalloc(&sys.alloc.free_queue, sizeof(std::uint32_t) * sys.alloc.free_queue_cap), "cudaMalloc free_queue");

#if KV_DEBUG
  checkCuda(cudaMalloc(&sys.alloc.owner, sizeof(std::int32_t) * num_blocks), "cudaMalloc owner");
  checkCuda(cudaMemset(sys.alloc.owner, 0xFF, sizeof(std::int32_t) * num_blocks), "cudaMemset owner=-1");
#endif

  // Block table sizing
  const std::uint32_t mbs = max_blocks_per_seq(sys.cfg.max_seq_len);
  const std::size_t bt_elems = static_cast<std::size_t>(sys.cfg.batch_size) * mbs;
  checkCuda(cudaMalloc(&sys.block_table, sizeof(std::uint32_t) * bt_elems), "cudaMalloc block_table");
  checkCuda(cudaMemset(sys.block_table, 0xFF, sizeof(std::uint32_t) * bt_elems), "cudaMemset block_table=0xFFFFFFFF");

  checkCuda(cudaMalloc(&sys.seq_block_cursor, sizeof(std::uint32_t) * sys.cfg.batch_size), "cudaMalloc seq_block_cursor");
  checkCuda(cudaMemset(sys.seq_block_cursor, 0, sizeof(std::uint32_t) * sys.cfg.batch_size), "cudaMemset seq_block_cursor=0");

  checkCuda(cudaMalloc(&sys.seq_len, sizeof(std::uint32_t) * sys.cfg.batch_size), "cudaMalloc seq_len");
  checkCuda(cudaMemset(sys.seq_len, 0, sizeof(std::uint32_t) * sys.cfg.batch_size), "cudaMemset seq_len=0");

  // Initialize free stack contents
  const int threads = 256;
  const int blocks = (num_blocks + threads -1) / threads;
  init_free_stack_kernel<<<blocks, threads>>>(sys.alloc.free_stack, num_blocks);
  checkCuda(cudaGetLastError(), "launch init_free_stack_kernel");
  checkCuda(cudaDeviceSynchronize(), "sync init_free_stack_kernel");

  // NOTE: We will initialize free_stack contents in the next step.
  // For now, just run sanity kernel to set top/free_count.
  sanity_kernel<<<1, 32>>>(sys.alloc.top, sys.alloc.free_count, num_blocks);
  checkCuda(cudaGetLastError(), "launch sanity_kernel");
  checkCuda(cudaDeviceSynchronize(), "sync sanity_kernel");
}

void destroy_system(AllocatorSystem& sys) {
  auto free_if = [](void* p) { if (p) cudaFree(p); };

  free_if(sys.pool.K);
  free_if(sys.pool.V);

  free_if(sys.alloc.free_stack);
  free_if(sys.alloc.top);
  free_if(sys.alloc.free_queue);
  free_if(sys.alloc.free_count);
#if KV_DEBUG
  free_if(sys.alloc.owner);
#endif

  free_if(sys.block_table);
  free_if(sys.seq_block_cursor);
  free_if(sys.seq_len);

  sys = AllocatorSystem{};
}

void run_sanity(AllocatorSystem& sys) {
  int32_t h_top = -999;
  uint32_t h_free_count = 999;

  checkCuda(cudaMemcpy(&h_top, sys.alloc.top, sizeof(int32_t), cudaMemcpyDeviceToHost), "memcpy top D2H");
  checkCuda(cudaMemcpy(&h_free_count, sys.alloc.free_count, sizeof(uint32_t), cudaMemcpyDeviceToHost), "memcpy free_count D2H");

  std::printf("[sanity] num_blocks=%u top=%d free_count=%u\n",
              sys.pool.num_blocks, h_top, h_free_count);

  if (h_top != static_cast<int32_t>(sys.pool.num_blocks)) {
    throw std::runtime_error("sanity failed: top != num_blocks");
  }
  if (h_free_count != 0) {
    throw std::runtime_error("sanity failed: free_count != 0");
  
  }
    // Verify free_stack first/last few entries
  uint32_t first8[8]{}, last8[8]{};
  checkCuda(cudaMemcpy(first8, sys.alloc.free_stack, sizeof(first8), cudaMemcpyDeviceToHost),
            "memcpy free_stack first8");
  checkCuda(cudaMemcpy(last8, sys.alloc.free_stack + (sys.pool.num_blocks - 8), sizeof(last8), cudaMemcpyDeviceToHost),
            "memcpy free_stack last8");

  std::printf("[sanity] free_stack first8: ");
  for (auto v : first8) std::printf("%u ", v);
  std::printf("\n");

  std::printf("[sanity] free_stack last8: ");
  for (auto v : last8) std::printf("%u ", v);
  std::printf("\n");
}

} // namespace kv
