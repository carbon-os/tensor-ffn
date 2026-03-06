# BUILD.md — inference

## Requirements
- CMake 3.22+
- CUDA Toolkit 12.x
- GCC 11+ or Clang 14+
- Linux

## Setup vcpkg

Run this once inside `inference/`:
```bash
git clone https://github.com/microsoft/vcpkg --depth 1
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install nlohmann-json
```

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binary lands at `build/tensor-ffn-inference`.

## Run
```bash
# Baseline — base model only
./build/tensor-ffn-inference \
  --model  ../trainer/qwen2.5-0.5b/ \
  --prompt "github.com/gomarkdown/markdown/parser"

# With expert
./build/tensor-ffn-inference \
  --model  ../trainer/qwen2.5-0.5b/ \
  --expert ../trainer/my-expert.bin \
  --prompt "github.com/gomarkdown/markdown/parser"




# Optional flags
#   --max-tokens 200    how many new tokens to generate (default 128)
#   --temperature 0.8   sampling temperature (default 0.0 = greedy)
```