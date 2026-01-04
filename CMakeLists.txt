cmake_minimum_required(VERSION 3.24)
project(kv_paged_cache LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reasonable defaults
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable separable compilation if you grow kernels later
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# You can override this at configure time:
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4060 is SM 8.9
endif()

add_executable(kv_paged_cache
  src/main.cu
  src/allocator.cu
)

target_include_directories(kv_paged_cache PRIVATE include)

# Helpful warnings
target_compile_options(kv_paged_cache PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Debug sanit-ish flags for host code (device sanitizers are separate tools)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(kv_paged_cache PRIVATE KV_DEBUG=1)
else()
  target_compile_definitions(kv_paged_cache PRIVATE KV_DEBUG=0)
endif()
