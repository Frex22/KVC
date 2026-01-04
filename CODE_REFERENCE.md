# KV Paged Cache - Code Reference Guide

**Purpose**: This document explains key concepts, terminology, and implementation details of the KV-cache paged memory allocator system. Use this as a quick reference when you forget how things work.

---

## 📋 Table of Contents
1. [Quick Glossary](#quick-glossary)
2. [Number Literals & Suffixes](#number-literals--suffixes)
3. [Memory Size Calculations](#memory-size-calculations)
4. [System Architecture](#system-architecture)
5. [Data Structures Explained](#data-structures-explained)
6. [Key Constants](#key-constants)
7. [Memory Layout](#memory-layout)
8. [CUDA Concepts](#cuda-concepts)

---

## Quick Glossary

| Term | Meaning |
|------|---------|
| **KV Cache** | Key-Value cache used in transformer models to store attention keys and values |
| **Block** | Fixed-size unit of memory that holds KV data for multiple tokens |
| **Token** | A single unit in a sequence (e.g., a word or subword in text) |
| **Head** | An attention head in a multi-head attention mechanism |
| **fp16** | 16-bit floating point (half precision) - uses 2 bytes per number |
| **Paged Memory** | Memory divided into fixed-size pages/blocks, like virtual memory in an OS |
| **Device** | The GPU (as opposed to "host" which is the CPU) |
| **Sequence** | A single input to process (e.g., one prompt or conversation) |
| **Batch** | Multiple sequences processed together |

---

## Number Literals & Suffixes

### C++ Integer Type Suffixes

```cpp
// Without suffix - regular int (32-bit signed)
int x = 1024;

// 'u' or 'U' - unsigned int
unsigned int x = 1024u;

// 'll' or 'LL' - long long (64-bit signed)
long long x = 1024ll;

// 'ull' or 'ULL' - unsigned long long (64-bit unsigned)
unsigned long long x = 1024ull;
```

### Why Use `ull` for Memory Calculations?

**Example**: `2ull * 1024ull * 1024ull * 1024ull`

- **Prevents overflow**: Regular int is 32-bit (max ~2.1 billion), but we're calculating gigabytes
- **Type safety**: Memory sizes (`std::size_t`) are typically 64-bit on modern systems
- **Best practice**: Always use `ull` when calculating large memory sizes

**How to read**: "2 gigabytes in bytes, calculated as unsigned long long"

---

## Memory Size Calculations

### Binary Size Units (Powers of 1024)

```
1 KB = 1024 bytes            = 1024¹ bytes
1 MB = 1024 KB = 1,048,576 bytes      = 1024² bytes
1 GB = 1024 MB = 1,073,741,824 bytes  = 1024³ bytes
```

### Common Patterns in Code

```cpp
// 2 GB in bytes
2ull * 1024ull * 1024ull * 1024ull = 2,147,483,648 bytes

// Breaking it down:
1024         // 1 KB
1024 * 1024  // 1 MB
1024³        // 1 GB
2 * 1024³    // 2 GB
```

### Memory Calculation Hierarchy in This Project

```
1 Token in one Head = HEAD_DIM * 2 bytes (fp16)
                    = 128 * 2 = 256 bytes

1 Token across all Heads = NUM_HEADS * HEAD_DIM * 2 bytes
                         = 16 * 128 * 2 = 4,096 bytes = 4 KB

1 Block of K (or V) = BLOCK_SIZE_TOKENS * bytes_per_token
                    = 16 * 4096 = 65,536 bytes = 64 KB

1 Block of K+V = 2 * 64 KB = 128 KB

Total blocks that fit in pool = pool_bytes / bytes_per_block
                              = 2 GB / 128 KB
                              = 16,384 blocks
```

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────┐
│           AllocatorSystem                   │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ RuntimeConfig│  │    KVPool         │   │
│  │   (CPU)      │  │   (GPU memory)    │   │
│  └──────────────┘  │                   │   │
│                    │  • K tensor pool  │   │
│  ┌──────────────┐  │  • V tensor pool  │   │
│  │BlockAllocator│  │  • num_blocks     │   │
│  │  (GPU mem)   │  └──────────────────┘   │
│  │              │                           │
│  │ • free_stack │  ┌──────────────────┐   │
│  │ • free_queue │  │   Block Table     │   │
│  │ • top        │  │   (GPU memory)    │   │
│  │ • free_count │  │                   │   │
│  └──────────────┘  │ Maps seq → blocks │   │
│                    └──────────────────┘   │
└─────────────────────────────────────────────┘
```

### Execution Flow

```
1. main() initializes CUDA device
   ├─ cudaGetDevice(&dev) - gets current GPU ID
   └─ cudaGetDeviceProperties() - queries GPU capabilities

2. Configure system
   ├─ Set pool_bytes (default 2 GB)
   ├─ Set batch_size, max_seq_len
   └─ Calculate derived values

3. init_system()
   ├─ Allocate GPU memory for K and V pools
   ├─ Allocate BlockAllocator metadata
   ├─ Allocate block_table for tracking
   └─ Initialize device counters (top, free_count)

4. run_sanity()
   └─ Verify initialization was correct

5. destroy_system()
   └─ Free all GPU memory
```

---

## Data Structures Explained

### 1. RuntimeConfig (CPU-side configuration)

```cpp
struct RuntimeConfig {
  std::size_t pool_bytes = 2ull * 1024ull * 1024ull * 1024ull; // Total GPU memory budget
  std::uint32_t free_sweep_threshold_blocks = 2048;             // When to trigger cleanup
  std::uint32_t batch_size = 8;                                 // How many sequences at once
  std::uint32_t max_seq_len = 2048;                             // Max tokens per sequence
};
```

**What each field means**:
- `pool_bytes`: Total memory to allocate on GPU for KV storage
- `free_sweep_threshold_blocks`: Deferred free optimization (future use)
- `batch_size`: Number of concurrent sequences being processed
- `max_seq_len`: Maximum length of any sequence in tokens

### 2. KVPool (GPU memory pools)

```cpp
struct KVPool {
  __half* K = nullptr;              // Pointer to Key tensor pool on GPU
  __half* V = nullptr;              // Pointer to Value tensor pool on GPU
  std::uint32_t num_blocks = 0;     // How many blocks fit in the pool
  std::size_t bytes_per_tensor = 0; // Size of K pool (V is same size)
};
```

**Memory organization**:
- K and V are stored in **separate** contiguous arrays (Choice S)
- Each pool is divided into `num_blocks` fixed-size blocks
- Each block stores data for `BLOCK_SIZE_TOKENS` (16) tokens

**Why `__half*`?**
- `__half` is CUDA's 16-bit floating point type (fp16)
- Uses half the memory of fp32 (32-bit float)
- Modern GPUs have fast fp16 hardware

### 3. BlockAllocator (Free block management)

```cpp
struct BlockAllocator {
  std::uint32_t* free_stack = nullptr; // Stack of available block IDs
  std::int32_t*  top = nullptr;        // Pointer to stack top counter (GPU)
  std::uint32_t* free_queue = nullptr; // Queue for deferred frees
  std::uint32_t* free_count = nullptr; // Queue length counter (GPU)
  std::uint32_t  free_queue_cap = 0;   // Max queue size
};
```

**How it works**:
- `free_stack`: Array where `free_stack[0..top-1]` contains available block IDs
- `top`: Number of free blocks (stack grows down from `num_blocks`)
- `free_queue`: Blocks waiting to be freed (v0 simple version)
- Device scalars (`top`, `free_count`) live in GPU memory for fast kernel access

**Example state**:
```
num_blocks = 4
top = 4
free_stack = [0, 1, 2, 3, ...]  // All blocks initially free

After allocating 2 blocks:
top = 2
free_stack = [0, 1, 2, 3, ...]  // blocks 2,3 are now in use
```

### 4. AllocatorSystem (Main container)

```cpp
struct AllocatorSystem {
  RuntimeConfig cfg;                    // Configuration
  KVPool pool;                          // Memory pools
  BlockAllocator alloc;                 // Allocator state
  std::uint32_t* block_table = nullptr; // Sequence → blocks mapping
  std::uint32_t* seq_block_cursor = nullptr; // Blocks per sequence
  std::uint32_t* seq_len = nullptr;     // Tokens per sequence
};
```

**Block Table Layout**:
```
Size: batch_size * max_blocks_per_seq

Example (batch_size=2, max_seq_len=2048, BLOCK_SIZE=16):
max_blocks_per_seq = ceil(2048/16) = 128

block_table layout:
[seq0_block0, seq0_block1, ..., seq0_block127,  // Sequence 0's blocks
 seq1_block0, seq1_block1, ..., seq1_block127]  // Sequence 1's blocks

Access: block_table[seq_id * max_blocks_per_seq + block_idx]
```

---

## Key Constants

### From kv_config.h

```cpp
BLOCK_SIZE_TOKENS = 16  // Each block holds KV for 16 tokens
HEAD_DIM = 128          // Dimension of each attention head
NUM_HEADS = 16          // Number of attention heads
NUM_LAYERS = 1          // Number of transformer layers (v0)
```

**Why BLOCK_SIZE_TOKENS = 16?**
- Trade-off between memory efficiency and allocation granularity
- Smaller blocks = more flexible but more overhead
- Larger blocks = less overhead but more waste
- 16 is a good balance and GPU-friendly (power of 2)

**Memory per token calculation**:
```
One token in attention:
- Has NUM_HEADS independent attention computations
- Each head has HEAD_DIM values
- We store both Key and Value (K and V)

Per token for K: NUM_HEADS * HEAD_DIM * fp16
               = 16 * 128 * 2 bytes
               = 4,096 bytes = 4 KB

Per token for V: same = 4 KB

Per token K+V: 8 KB
```

---

## Memory Layout

### Physical Layout in GPU Memory

```
K Pool (contiguous):
┌─────────┬─────────┬─────────┬─────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ ... │ Block N │
└─────────┴─────────┴─────────┴─────┴─────────┘
  64 KB     64 KB     64 KB           64 KB

V Pool (contiguous):
┌─────────┬─────────┬─────────┬─────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ ... │ Block N │
└─────────┴─────────┴─────────┴─────┴─────────┘
  64 KB     64 KB     64 KB           64 KB
```

### Block Internal Layout (K or V)

```
One Block = 16 tokens of data
Each token = 16 heads * 128 dimensions * 2 bytes (fp16)

Block structure:
┌───────────────────────────────────────┐
│ Token 0:  [h0...h15] * 128 * fp16    │ 4 KB
│ Token 1:  [h0...h15] * 128 * fp16    │ 4 KB
│ Token 2:  [h0...h15] * 128 * fp16    │ 4 KB
│ ...                                   │
│ Token 15: [h0...h15] * 128 * fp16    │ 4 KB
└───────────────────────────────────────┘
Total: 64 KB per block (K or V)
```

---

## CUDA Concepts

### Device vs Host

```cpp
int dev = 0;  // CPU variable storing GPU device ID
cudaGetDevice(&dev);  // Query which GPU is active
```

- **Host**: CPU and system RAM (where main() runs)
- **Device**: GPU and GPU memory (where kernels run)
- **Device ID**: GPUs numbered 0, 1, 2... if multiple GPUs present

### Common CUDA Types

```cpp
__half          // 16-bit floating point (GPU-specific)
cudaError_t     // Error code returned by CUDA functions
cudaDeviceProp  // Struct containing GPU properties (name, memory, compute capability)
```

### Memory Operations

```cpp
// Allocate on GPU
cudaMalloc(&ptr, size);  // Like malloc() but on GPU

// Copy CPU ↔ GPU
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);   // CPU → GPU
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);   // GPU → CPU

// Initialize GPU memory
cudaMemset(ptr, value, size);  // Like memset() but on GPU

// Free GPU memory
cudaFree(ptr);
```

### Kernel Launch

```cpp
kernel_name<<<num_blocks, threads_per_block>>>(args);

// Example:
sanity_kernel<<<1, 32>>>(sys.alloc.top, sys.alloc.free_count, num_blocks);
//             │  │
//             │  └─ 32 threads per block
//             └──── 1 block total (32 threads total)
```

### Error Checking Pattern

```cpp
void checkCuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
    throw std::runtime_error("CUDA failure");
  }
}

// Usage:
checkCuda(cudaMalloc(&ptr, size), "cudaMalloc description");
```

**Why this pattern?**
- CUDA functions return error codes but don't throw exceptions
- We wrap them to convert errors → C++ exceptions
- Makes error handling cleaner and safer

---

## Quick Reference: Common Calculations

### How many blocks fit in memory?

```cpp
num_blocks = pool_bytes / bytes_per_block
           = pool_bytes / (BLOCK_SIZE_TOKENS * NUM_HEADS * HEAD_DIM * 2 * 2)
                                                                      │   │
                                                                      │   └─ K and V
                                                                      └───── fp16 = 2 bytes

With defaults (2 GB pool):
= 2,147,483,648 / (16 * 16 * 128 * 2 * 2)
= 2,147,483,648 / 131,072
= 16,384 blocks
```

### How many blocks does a sequence need?

```cpp
max_blocks_per_seq = ceil(max_seq_len / BLOCK_SIZE_TOKENS)

With defaults (max_seq_len = 2048):
= ceil(2048 / 16)
= 128 blocks per sequence
```

### How much memory per sequence (worst case)?

```cpp
memory_per_seq = max_blocks_per_seq * bytes_per_block
               = 128 * 131,072 bytes
               = 16,777,216 bytes
               = 16 MB per sequence
```

### How many sequences can fit?

```cpp
max_concurrent_sequences = num_blocks / max_blocks_per_seq
                         = 16,384 / 128
                         = 128 sequences

But batch_size limits to 8 in default config
```

---

## Debugging Tips

### Check GPU memory allocation

```cpp
std::printf("Allocated %.2f GB for KV pools\n",
            static_cast<double>(sys.pool.bytes_per_tensor * 2) / (1024.0*1024.0*1024.0));
```

### Verify block calculations

```cpp
std::printf("bytes_per_block=%zu, num_blocks=%u, total=%zu GB\n",
            bytes_per_block,
            num_blocks,
            static_cast<size_t>(bytes_per_block * num_blocks) / (1024*1024*1024));
```

### Common Issues

1. **Out of memory**: Reduce `pool_bytes` or increase GPU memory
2. **Alignment errors**: Ensure sizes are multiples of GPU alignment requirements
3. **Wrong device**: Check `dev` variable matches your intended GPU

---

## File Structure Quick Map

```
include/
  ├─ kv_config.h    → Constants, RuntimeConfig, helper functions
  └─ allocator.h    → Data structures, function declarations

src/
  ├─ main.cu        → Entry point, initialization, sanity check
  └─ allocator.cu   → Implementation of init/destroy/sanity functions

build/
  └─ kv_paged_cache → Compiled executable
```

---

## Version Notes

- **v0**: Initial implementation
- Single layer (`NUM_LAYERS = 1`)
- Simple free queue (append-only)
- Debug mode available via `KV_DEBUG` macro
- Choice S: Separate K and V pools
- Choice A=1: BLOCK_SIZE_TOKENS = 16

---

**Last Updated**: January 2026  
**Remember**: This is a reference document. Update it as the code evolves!
