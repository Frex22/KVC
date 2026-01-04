#pragma once
#include <cstddef>
#include <cstdint>

namespace kv {

// Compile-time constants (your Choice A=1)
static constexpr int BLOCK_SIZE_TOKENS = 16;
static constexpr int HEAD_DIM = 128;
static constexpr int NUM_HEADS = 16;
static constexpr int NUM_LAYERS = 1; // v0

// Runtime config
struct RuntimeConfig {
  std::size_t pool_bytes = 2ull * 1024ull * 1024ull * 1024ull; // 2GB default
  std::uint32_t free_sweep_threshold_blocks = 2048;            // default; tune later
  std::uint32_t batch_size = 8;                                // default scenario
  std::uint32_t max_seq_len = 2048;
};

// Derived sizing helpers (fp16 KV)
static constexpr std::size_t bytes_per_kv_vector_per_token() {
  // NUM_HEADS * HEAD_DIM * fp16(2 bytes)
  return static_cast<std::size_t>(NUM_HEADS) * HEAD_DIM * 2ull;
}

static constexpr std::size_t bytes_per_block_per_tensor() {
  // BLOCK_SIZE tokens of K (or V)
  return static_cast<std::size_t>(BLOCK_SIZE_TOKENS) * bytes_per_kv_vector_per_token();
}

static constexpr std::size_t bytes_per_block_k_plus_v() {
  return 2ull * bytes_per_block_per_tensor();
}

static constexpr std::uint32_t max_blocks_per_seq(std::uint32_t max_seq_len) {
  // ceil(max_seq_len / BLOCK_SIZE_TOKENS) but max_seq_len is divisible here
  return (max_seq_len + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;
}

} // namespace kv
