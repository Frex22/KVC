__device__ __forceinline__ unsigned lane_id() {
  unsigned id;
  asm("mov.u32 %0, %laneid;" : "=r"(id));
  return id;
}

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
    uint32_t num_blocks)
{
  // One warp per request
  const uint32_t warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const uint32_t lane = lane_id();

  // We will launch enough warps to cover batch_size.
  const uint32_t b = warp_global;
  // batch_size passed indirectly by grid sizing in v0; keep it simple.

  // Guard: if b is out of range, return. (Caller ensures grid >= batch_size)
  // We'll pass batch_size via req_new_blocks length; for v0 we pass batch_size to host launch only.
  // We'll implement guard in host: only launch exactly batch warps.
  // Still safe to include a runtime guard if desired (omitted for now).

  // If already failed, do nothing
  if (alloc_failed[b]) return;

  const uint32_t k = req_new_blocks[b];
  if (k == 0) return;

  // Reserve k blocks from stack via CAS loop (lane 0)
  __shared__ int32_t s_old_top[8];   // batch <= 8 v0; safe fixed
  __shared__ int32_t s_new_top[8];
  __shared__ uint32_t s_fail[8];

  if (lane == 0) {
    s_fail[b] = 0;
    while (true) {
      int32_t old = *top;
      if (old < static_cast<int32_t>(k)) {
        s_fail[b] = 1;
        s_old_top[b] = old;
        s_new_top[b] = old;
        break;
      }
      int32_t desired = old - static_cast<int32_t>(k);
      int32_t prev = atomicCAS(top, old, desired);
      if (prev == old) {
        s_old_top[b] = old;
        s_new_top[b] = desired;
        break;
      }
      // else retry
    }
  }
  __syncwarp();

  if (s_fail[b]) {
    if (lane == 0) alloc_failed[b] = 1;
    return;
  }

  // Popped range: indices [new_top .. old_top-1]
  const int32_t old_top = s_old_top[b];
  const int32_t new_top = s_new_top[b];

  // Cursor where to write in block_table
  const uint32_t cursor = seq_block_cursor[b];

  // Bounds check (debug-ish): ensure we don't exceed max_blocks_per_seq
  // For v0 we assume caller doesn't request more than max; later we enforce.
  // We'll still fail-fast if overflow.
  if (cursor + k > max_blocks_per_seq) {
    if (lane == 0) alloc_failed[b] = 1;
    // IMPORTANT: we already reserved blocks; in v0 we accept this as a bug in caller.
    // In later versions, we'd rollback or prevent earlier.
    return;
  }

  // Cooperative write of k block IDs
  for (uint32_t i = lane; i < k; i += 32) {
    const uint32_t stack_idx = static_cast<uint32_t>(new_top + static_cast<int32_t>(i));
    const uint32_t p = free_stack[stack_idx];
    block_table[b * max_blocks_per_seq + (cursor + i)] = p;

#if KV_DEBUG
    // Debug ownership
    int32_t prev_owner = atomicCAS(&owner[p], -1, static_cast<int32_t>(b));
    // If prev_owner != -1, you have a duplicate allocation bug.
    // We cannot assert on device easily here; we'll detect by scanning owner later if needed.
    (void)prev_owner;
#endif
  }

  // Update cursor once (lane 0)
  if (lane == 0) {
    seq_block_cursor[b] = cursor + k;
  }
}
