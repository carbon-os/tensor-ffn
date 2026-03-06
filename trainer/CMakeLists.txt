cmake_minimum_required(VERSION 3.22)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "")

project(tensor-ffn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/backend/memory.cpp
    src/backend/cuda_ops.cu
    src/model/rmsnorm.cpp
    src/model/rope.cpp
    src/model/ffn_expert.cu
    src/model/qwen.cu
    src/io/safetensors.cpp
    src/io/tokenizer.cpp
    src/io/corpus.cpp
    src/io/checkpoint.cpp
    src/training/loss.cu
    src/training/optimizer.cu
    src/training/trainer.cpp
)

add_executable(tensor-ffn ${SOURCES})

target_link_libraries(tensor-ffn
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)

target_compile_options(tensor-ffn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler -O3
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -Wall -Wextra
    >
)