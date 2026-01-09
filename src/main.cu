#include <cstdio>
#include <cuda_runtime.h>
#include <exception>
#include "allocator.h"
#include "kv_config.h"

int main() {
  try {
    int dev = 0;
    cudaDeviceProp prop{};
    kv::checkCuda(cudaGetDevice(&dev), "cudaGetDevice");
    kv::checkCuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

    std::printf("GPU: %s (SM %d.%d), globalMem=%.2f GB\n",
                prop.name, prop.major, prop.minor,
                static_cast<double>(prop.totalGlobalMem) / (1024.0*1024.0*1024.0));

    kv::RuntimeConfig cfg;
    // cfg.pool_bytes = 2GB default; adjust here if needed.
    // cfg.free_sweep_threshold_blocks = 2048;
    // cfg.batch_size = 8;
    // cfg.max_seq_len = 2048;

    const std::size_t bytes_per_block = kv::bytes_per_block_k_plus_v();
    std::printf("Config: BLOCK_SIZE=%d tokens, H=%d, D=%d, KV fp16\n",
                kv::BLOCK_SIZE_TOKENS, kv::NUM_HEADS, kv::HEAD_DIM);
    std::printf("Derived: bytes/block(K+V)=%.2f KB\n", bytes_per_block / 1024.0);
    std::printf("Pool budget: %.2f GB\n",
                static_cast<double>(cfg.pool_bytes) / (1024.0*1024.0*1024.0));

    kv::AllocatorSystem sys;
    sys.cfg = cfg;

    kv::init_system(sys);
    kv::run_sanity(sys);
    kv::test_alloc_step(sys);

    kv::destroy_system(sys);
    std::printf("Done.\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
}
