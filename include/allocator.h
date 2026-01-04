#pragma once
#include <cstdint>
#include <cstddef>

#include <cuda_runtime.h>
#include "kv_config.h"

namespace kv {

struct KVPool {
  // Separate pools (you chose S)
  __half* K = nullptr;
  __half* V = nullptr;

  std::uint32_t num_blocks = 0;
  std::size_t   bytes_per_tensor = 0; // size of K pool (same as V)
};

struct BlockAllocator {
  // Free-stack
  std::uint32_t* free_stack = nullptr; // length num_blocks
  std::int32_t*  top = nullptr;        // device scalar: number of free blocks

  // Deferred free queue (append-only for v0)
  std::uint32_t* free_queue = nullptr; // length free_queue_cap
  std::uint32_t* free_count = nullptr; // device scalar

  std::uint32_t  free_queue_cap = 0;

#if KV_DEBUG
  std::int32_t* owner = nullptr; // device array: owner[block] = seq_id or -1
#endif
};

struct AllocatorSystem {
  RuntimeConfig cfg;
  KVPool pool;
  BlockAllocator alloc;

  // Block table: [batch_size * max_blocks_per_seq]
  std::uint32_t* block_table = nullptr;

  // Per-sequence: number of blocks currently allocated (block cursor)
  std::uint32_t* seq_block_cursor = nullptr;

  // Optional: seq_len (tokens) if you want it early
  std::uint32_t* seq_len = nullptr;
};

void checkCuda(cudaError_t e, const char* what);

void init_system(AllocatorSystem& sys);
void destroy_system(AllocatorSystem& sys);

// Tiny sanity kernel to validate we can touch device scalars
void run_sanity(AllocatorSystem& sys);

} // namespace kv
