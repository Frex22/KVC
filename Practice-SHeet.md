#to be practiced by hand until we get a hang of how threads are allocated work.
Here’s your **daily design sheet** — compact, formula-first, and exactly aligned to what we built. Print it or keep it in a note and review it for 10 minutes/day.

---

# Paged KV Cache v0 — Daily Design Sheet

## 0) Constants (compile-time)

* `BLOCK_SIZE_TOKENS = 16`
* `H = NUM_HEADS = 16`
* `D = HEAD_DIM = 128`
* `dtype(K,V) = fp16` → `bytes_per_elem = 2`
* `Tmax = 2048`
* `B = batch = 8` (default scenario)

---

## 1) Sizes (must know cold)

### Per token

* elements per token per tensor = `H * D`
* bytes per token per tensor = `H * D * bytes_per_elem`
* bytes per token (K+V) = `2 * H * D * bytes_per_elem`

**With your values:**

* `H*D = 16*128 = 2048`
* K bytes/token = `2048*2 = 4096 B = 4 KB`
* K+V bytes/token = `8 KB`

### Per block (page)

* tokens per block = `BLOCK_SIZE_TOKENS`
* K bytes/block = `BLOCK_SIZE_TOKENS * H * D * bytes_per_elem`
* V bytes/block = same
* **bytes/block(K+V) = `2 * BLOCK_SIZE_TOKENS * H * D * bytes_per_elem`**

**With your values:**

* K bytes/block = `16 * 4096 = 65536 B = 64 KB`
* **K+V bytes/block = 128 KB**

### Blocks per sequence

* `max_blocks_per_seq = ceil(Tmax / BLOCK_SIZE_TOKENS)`
* With `2048/16 = 128` → **128**

---

## 2) Pool sizing

### Physical blocks in pool

* `num_blocks = floor(pool_bytes / bytes_per_block_kv)`

**With 2GB pool:**

* `2GB / 128KB = 16384` blocks

### Total token capacity in pool

* `pool_token_capacity = num_blocks * BLOCK_SIZE_TOKENS`
* With 16384 blocks: `16384 * 16 = 262,144 tokens` total

---

## 3) Virtual ↔ Physical mapping (the OS analogy)

### Token → (logical block, offset)

For sequence `b` and token index `t`:

* `logical_block = t / BLOCK_SIZE_TOKENS`
* `offset       = t % BLOCK_SIZE_TOKENS`

### Page table (block table)

* `p = block_table[b][logical_block]`
* `p` is a **physical block ID** in `[0 .. num_blocks-1]`

### Actual KV address conceptually

* `K = K_pool[p][offset][h][d]`
* `V = V_pool[p][offset][h][d]`

**Key point:** `p` can be *any free physical block id*. Not contiguous.

---

## 4) Free-stack allocator model (v0)

### Meaning

* `free_stack[0 .. top-1]` holds free physical block IDs
* `top = number of free blocks`
* invariant: `0 ≤ top ≤ num_blocks`

### Pop k blocks (reserve)

If request needs `k` blocks:

* check `old_top >= k`
* set `top = old_top - k` atomically
* reserved indices are:

  * `[new_top .. old_top-1]` where `new_top = old_top - k`

### What gets written

* `cursor = seq_block_cursor[b]` (# blocks already allocated)
* for `i in [0..k-1]`:

  * `block_table[b][cursor+i] = free_stack[new_top + i]`
* update:

  * `seq_block_cursor[b] += k`

### Fail-fast

If `old_top < k`, allocation fails:

* `alloc_failed[b] = 1`
* do not change table/cursor

---

## 5) AtomicCAS meaning (must know)

`atomicCAS(addr, expected, desired)`:

* if `*addr == expected`, write `desired`
* returns the previous value that was in `*addr`

CAS loop for `top`:

* read `old`
* if `old < k` fail
* try `CAS(top, old, old-k)`
* if CAS fails, retry (someone changed top)

---

## 6) Kernel launch math for allocator (warp-per-request)

### Mapping

* `global_thread = blockIdx.x * blockDim.x + threadIdx.x`
* `warp_global   = global_thread / 32`
* `b             = warp_global` (request id)

### Launch sizing formula (general)

If “1 warp per request” and you have `B` requests:

* `total_threads_needed = B * 32`
* choose `threads_per_block` multiple of 32 (e.g., 128)
* `gridDim.x = ceil(total_threads_needed / threads_per_block)`

  * `gridDim.x = (total_threads_needed + threads_per_block - 1) / threads_per_block`

**Example (B=8, threads=128):**

* `total_threads_needed = 256`
* `gridDim.x = ceil(256/128)=2`

---

## 7) Debug invariants (what you assert)

### Allocation invariants

* no duplicate `p` across live sequences
* block_table entries for allocated blocks are valid `p`
* cursor never exceeds `max_blocks_per_seq`

### Owner debug array (debug only)

* `owner[p] = -1` free
* on alloc: `CAS(owner[p], -1, seq_id)` must succeed
* on free: set owner[p] back to -1 (later)

---

## 8) Your daily 10-minute practice

Pick a random:

* `b` in `[0..B-1]`
* `t` in `[0..Tmax-1]`

Compute:

1. `logical_block = t/16`
2. `offset = t%16`
3. `p = block_table[b][logical_block]` (pretend or print)
4. Write the conceptual addresses:

   * `K_pool[p][offset][h][d]`
   * `V_pool[p][offset][h][d]`

Then simulate one allocation:

* choose k=2
* old_top → new_top
* list which `free_stack[new_top..old_top-1]` were taken

---

If you want, I can also give you a **one-page “OS ↔ KV system mapping” diagram** (super memorable).
