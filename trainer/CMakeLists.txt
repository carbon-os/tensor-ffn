cmake_minimum_required(VERSION 3.22)
project(tensor-ffn-trainer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)   # A100; change to 86 for A10/3090

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(trainer/include)

set(SOURCES
    trainer/src/main.cpp
    trainer/src/backend/memory.cpp
    trainer/src/backend/cuda_ops.cu
    trainer/src/model/rmsnorm.cpp
    trainer/src/model/rope.cpp
    trainer/src/model/ffn_expert.cpp
    trainer/src/model/qwen.cpp
    trainer/src/io/safetensors.cpp
    trainer/src/io/tokenizer.cpp
    trainer/src/io/corpus.cpp
    trainer/src/io/checkpoint.cpp
    trainer/src/training/loss.cpp
    trainer/src/training/optimizer.cpp
    trainer/src/training/trainer.cpp
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