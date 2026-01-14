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
    *top = static_cast<int32_t>(num_blocks);
    *free_count = 0;
  }
}

__global__ void init_free_stack_kernel(uint32_t* free_stack, uint32_t num_blocks) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_blocks) {
    // Order B: descending
    free_stack[i] = (num_blocks - 1u - i);
  }
}

__device__ __forceinline__ unsigned lane_id() {
  unsigned id;
  asm("mov.u32 %0, %laneid;" : "=r"(id));
  return id;
}

// One warp per request allocator
__global__ void alloc_blocks_kernel(
    uint32_t* block_table,
    uint32_t max_blocks_per_seq,
    uint32_t* seq_block_cursor,
    const uint32_t* req_new_blocks,
    uint32_t* alloc_failed,
    uint32_t* free_stack,
    int32_t* top,
#if KV_DEBUG
    int32_t* owner,
#endif
    uint32_t num_blocks,
    uint32_t batch_size)
{
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_global = global_thread / 32;
  const uint32_t b = warp_global;

  if (b >= batch_size) return;

  const uint32_t lane = lane_id();

  // If already failed, do nothing
  if (alloc_failed[b]) return;

  const uint32_t k = req_new_blocks[b];
  if (k == 0) return;

  // Per-warp shared state (indexed by warp-in-block)
  const uint32_t warp_in_block = threadIdx.x / 32;
  //const uint32_t warps_per_block = blockDim.x / 32;

  // v0: we expect small warps_per_block (e.g., 4). Keep shared arrays size 8.
  __shared__ int32_t  s_old_top_w[8];
  __shared__ int32_t  s_new_top_w[8];
  __shared__ uint32_t s_fail_w[8];

  if (warp_in_block >= 8) return; // safety if someone changes blockDim later

  // Reserve k blocks from stack via CAS loop (lane 0 of warp)
  if (lane == 0) {
    s_fail_w[warp_in_block] = 0;

    while (true) {
      int32_t old = *top;

      if (old < static_cast<int32_t>(k)) {
        s_fail_w[warp_in_block] = 1;
        s_old_top_w[warp_in_block] = old;
        s_new_top_w[warp_in_block] = old;
        break;
      }

      int32_t desired = old - static_cast<int32_t>(k);
      int32_t prev = atomicCAS(top, old, desired);

      if (prev == old) {
        // Success: reserved k blocks.
        s_old_top_w[warp_in_block] = old;
        s_new_top_w[warp_in_block] = desired;
        break;
      }
      // else retry
    }
  }

  __syncwarp();

  if (s_fail_w[warp_in_block]) {
    if (lane == 0) alloc_failed[b] = 1;
    return;
  }

  const int32_t old_top = s_old_top_w[warp_in_block];
  const int32_t new_top = s_new_top_w[warp_in_block];
  // Popped indices: [new_top .. old_top-1]

  const uint32_t cursor = seq_block_cursor[b];

  // Bounds check: ensure we don't exceed max blocks for the seq
  if (cursor + k > max_blocks_per_seq) {
    if (lane == 0) alloc_failed[b] = 1;
    // NOTE: in v0 we don't rollback reserved blocks; treat as caller bug.
    return;
  }

  // Cooperative write of k block IDs into block table
  for (uint32_t i = lane; i < k; i += 32) {
    const uint32_t stack_idx = static_cast<uint32_t>(new_top + static_cast<int32_t>(i));
    // stack_idx must be < num_blocks; ensured by top logic
    const uint32_t p = free_stack[stack_idx];

    block_table[b * max_blocks_per_seq + (cursor + i)] = p;

#if KV_DEBUG
    // Debug ownership: claim owner[p] = b if it was -1
    int32_t prev_owner = atomicCAS(&owner[p], -1, static_cast<int32_t>(b));
    (void)prev_owner; // if prev_owner != -1 => duplicate allocation bug
#endif
  }

  if (lane == 0) {
    seq_block_cursor[b] = cursor + k;
  }
}

__global__ void end_sequence_enqueue_kernel(
    uint32_t* block_table,
    uint32_t max_blocks_per_seq,
    uint32_t* seq_block_cursor,
    uint32_t* alloc_failed,
    uint32_t* free_queue,
    uint32_t* free_count,
#if KV_DEBUG
    int32_t* owner,
#endif
    uint32_t batch_size) 
    {
      const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
      const uint32_t warp_global = global_thread / 32;
      const uint32_t b = warp_global;
      if (b >= batch_size) return;
      const uint32_t lane = lane_id();

      // If sequence already failed, treat as ended
      const uint32_t n = seq_block_cursor[b];
      if (n == 0) {
        if (lane == 0){
          seq_block_cursor[b] = 0;
          alloc_failed[b] = 0;
        }
        return;

      }

      // Enqueue blocks back to free_queue
      for(uint32_t i = lane; i < n; i += 32) 
      {
        uint32_t p = block_table[b*max_blocks_per_seq + i];
        // Append p to free_queue
        if (p != 0xFFFFFFFF) {
          // valid block
        uint32_t idx = atomicAdd(free_count, 1u);
        free_queue[idx] = p;
#if KV_DEBUG
        // Debug ownership: release owner[p] = -1
        owner[p] = -1;
#endif
        block_table[b*max_blocks_per_seq + i] = 0xFFFFFFFF; // optional clear
 

      }

    }

    __syncwarp();

    if (lane == 0) {
      seq_block_cursor[b] = 0;
      alloc_failed[b] = 0;

    }

  }

/*__global__ void sweep_kernel (
  uint32_t* free_stack,
  int32_t* top,
  uint32_t* free_queue,
  uint32_t* free_count,
  uint32_t num_blocks)

  {
    __shared__ uint32_t s_n;
    __shared__ int32_t s_old_top;

    if (threadIdx.x == 0) {
      s_n = *free_count;
      if (s_n == 0) {
        s_old_top = 0;
      }
      else

     {
        //reserve space in free_stack
        s_old_top = atomicAdd(top, static_cast<int32_t>(s_n));
        //reset free_count
        *free_count = 0;
      }
    }

    __syncthreads();

     const uint32_t n = s_n;
  if (n == 0) return;

  // Bounds safety: old_top + n must not exceed num_blocks
  // (We assume correct usage in v0; later add hard checks.)

  // Copy freed IDs back into stack
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x * gridDim.x) {
    free_stack[static_cast<uint32_t>(s_old_top) + i] = free_queue[i];
  }


  } old implementation skippy*/

__global__ void sweep_kernel(
    uint32_t* free_stack,
    int32_t* top,
    const uint32_t* free_queue,
    uint32_t* free_count,
    uint32_t num_blocks)
{
  __shared__ uint32_t s_n;
  __shared__ int32_t s_old_top;

  if (threadIdx.x == 0) {
    // Snapshot how many frees are pending
    s_n = *free_count;

    if (s_n == 0) {
      s_old_top = 0;
    } else {
      // Reserve exactly s_n slots in the free_stack
      int32_t old = atomicAdd(top, (int32_t)s_n);
      s_old_top = old;

      // Hard bounds guard (v0 safety)
      // If this triggers, your accounting is broken somewhere else.
      if ((uint32_t)old + s_n > num_blocks) {
        // Roll back the top to avoid corrupting memory.
        atomicAdd(top, -(int32_t)s_n);
        s_n = 0;
      } else {
        // Reset free_count (v0 assumes no concurrent enqueues during sweep)
        *free_count = 0;
      }
    }
  }

  __syncthreads();

  const uint32_t n = s_n;
  if (n == 0) return;

  // Copy free_queue[0..n-1] back into free_stack[s_old_top .. s_old_top+n-1]
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = tid; i < n; i += stride) {
    free_stack[(uint32_t)s_old_top + i] = free_queue[i];
  }
}




void init_system(AllocatorSystem& sys) {
  const std::size_t bytes_per_block = bytes_per_block_k_plus_v();
  if (bytes_per_block == 0) throw std::runtime_error("bytes_per_block is zero");

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

  // Deferred free queue (v0: capacity = num_blocks)
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

  // Alloc failure flags (persist once set)
  checkCuda(cudaMalloc(&sys.alloc_failed, sizeof(std::uint32_t) * sys.cfg.batch_size), "cudaMalloc alloc_failed");
  checkCuda(cudaMemset(sys.alloc_failed, 0, sizeof(std::uint32_t) * sys.cfg.batch_size), "cudaMemset alloc_failed=0");

  // Set top and free_count
  sanity_kernel<<<1, 32>>>(sys.alloc.top, sys.alloc.free_count, num_blocks);
  checkCuda(cudaGetLastError(), "launch sanity_kernel");
  checkCuda(cudaDeviceSynchronize(), "sync sanity_kernel");

  // Initialize free_stack
  const int threads = 256;
  const int blocks = (num_blocks + threads - 1) / threads;
  init_free_stack_kernel<<<blocks, threads>>>(sys.alloc.free_stack, num_blocks);
  checkCuda(cudaGetLastError(), "launch init_free_stack_kernel");
  checkCuda(cudaDeviceSynchronize(), "sync init_free_stack_kernel");
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
  free_if(sys.alloc_failed);

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

void test_alloc_step(AllocatorSystem& sys) {
  const uint32_t B = sys.cfg.batch_size;
  const uint32_t M = max_blocks_per_seq(sys.cfg.max_seq_len);

  // Prefill-style: allocate 2 blocks per request (batch=8 in v0)
  uint32_t h_req[8] = {2,2,2,2,2,2,2,2};

  uint32_t* d_req = nullptr;
  checkCuda(cudaMalloc(&d_req, sizeof(uint32_t) * B), "cudaMalloc d_req");
  checkCuda(cudaMemcpy(d_req, h_req, sizeof(uint32_t) * B, cudaMemcpyHostToDevice), "memcpy req H2D");

  // Launch enough warps to cover B requests
  const int threads = 128; // 4 warps per block
  const int total_threads = static_cast<int>(B) * 32;
  const int blocks = (total_threads + threads - 1) / threads;

  alloc_blocks_kernel<<<blocks, threads>>>(
      sys.block_table,
      M,
      sys.seq_block_cursor,
      d_req,
      sys.alloc_failed,
      sys.alloc.free_stack,
      sys.alloc.top,
#if KV_DEBUG
      sys.alloc.owner,
#endif
      sys.pool.num_blocks,
      B
  );
  checkCuda(cudaGetLastError(), "launch alloc_blocks_kernel");
  checkCuda(cudaDeviceSynchronize(), "sync alloc_blocks_kernel");

  // Read back cursors + top + failed
  uint32_t h_cursor[8]{};
  uint32_t h_failed[8]{};
  int32_t h_top = -1;

  checkCuda(cudaMemcpy(h_cursor, sys.seq_block_cursor, sizeof(h_cursor), cudaMemcpyDeviceToHost), "memcpy cursor D2H");
  checkCuda(cudaMemcpy(h_failed, sys.alloc_failed, sizeof(h_failed), cudaMemcpyDeviceToHost), "memcpy failed D2H");
  checkCuda(cudaMemcpy(&h_top, sys.alloc.top, sizeof(h_top), cudaMemcpyDeviceToHost), "memcpy top D2H");

  std::printf("[alloc-test] top=%d (expected %d)\n", h_top,
              static_cast<int>(sys.pool.num_blocks) - static_cast<int>(B * 2));

  for (uint32_t b = 0; b < B; ++b) {
    std::printf("[alloc-test] b=%u cursor=%u failed=%u\n", b, h_cursor[b], h_failed[b]);
  }

  // Optionally inspect first few block_table entries for request 0
  uint32_t h_bt0[8]{};
  checkCuda(cudaMemcpy(h_bt0, sys.block_table + 0 * M, sizeof(h_bt0), cudaMemcpyDeviceToHost),
            "memcpy block_table[0] first8");
  std::printf("[alloc-test] block_table[0] first8: ");
  for (auto v : h_bt0) std::printf("%u ", v);
  std::printf("\n");

  checkCuda(cudaFree(d_req), "cudaFree d_req");
}

void test_free_sweep_reuse(AllocatorSystem& sys) {
  const uint32_t B = sys.cfg.batch_size;
  const uint32_t M = max_blocks_per_seq(sys.cfg.max_seq_len);

  // 1) Allocate 2 blocks per request (reuse your existing test pattern)
  std::printf("\n[reuse-test] allocate phase\n");
  test_alloc_step(sys);

  // 2) End all sequences: enqueue their blocks into free_queue
  std::printf("[reuse-test] end-sequence enqueue\n");
  {
    const int threads = 128; // 4 warps/block
    const int total_threads = static_cast<int>(B) * 32;
    const int blocks = (total_threads + threads - 1) / threads;

    end_sequence_enqueue_kernel<<<blocks, threads>>>(
        sys.block_table,
        M,
        sys.seq_block_cursor,
        sys.alloc_failed,
        sys.alloc.free_queue,
        sys.alloc.free_count,
#if KV_DEBUG
        sys.alloc.owner,
#endif
        B
    );
    checkCuda(cudaGetLastError(), "launch end_sequence_enqueue_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync end_sequence_enqueue_kernel");
  }

  // Read free_count
  uint32_t h_free_count = 0;
  checkCuda(cudaMemcpy(&h_free_count, sys.alloc.free_count, sizeof(h_free_count), cudaMemcpyDeviceToHost),
            "memcpy free_count D2H");
  std::printf("[reuse-test] free_count after enqueue = %u (expected %u)\n", h_free_count, B * 2);

  // 3) Sweep: move free_queue back into free_stack and restore top
  std::printf("[reuse-test] sweep\n");
  {
    // A simple sweep launch (enough threads to copy n entries)
    const int threads = 256;
    const int blocks = 4;
    sweep_kernel<<<blocks, threads>>>(
        sys.alloc.free_stack,
        sys.alloc.top,
        sys.alloc.free_queue,
        sys.alloc.free_count,
        sys.pool.num_blocks
    );
    checkCuda(cudaGetLastError(), "launch sweep_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync sweep_kernel");
  }

  int32_t h_top = -1;
  checkCuda(cudaMemcpy(&h_top, sys.alloc.top, sizeof(h_top), cudaMemcpyDeviceToHost), "memcpy top D2H");
  std::printf("[reuse-test] top after sweep = %d (expected %u)\n", h_top, sys.pool.num_blocks);

  // 4) Allocate again and see if top decreases again (proves reuse)
  std::printf("[reuse-test] allocate again\n");
  test_alloc_step(sys);
  std::printf("[reuse-test] done\n\n");
}

} // namespace kv
