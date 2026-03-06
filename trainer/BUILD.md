# BUILD.md

## Requirements

- CMake 3.22+
- CUDA Toolkit 12.x
- GCC 11+ or Clang 14+
- Linux

## System Dependencies
```bash
apt-get install curl zip unzip tar pkg-config cmake
apt-get update
```

## Setup vcpkg

Clone vcpkg inside the project root and bootstrap it:
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

The binary lands at `build/tensor-ffn`.

> **Tip:** During development on a known GPU, speed up compilation with:
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
> cmake --build build -j$(nproc)
> ```

## GPU Architecture Reference

| GPU              | CUDA Arch |
|------------------|-----------|
| V100             | 70        |
| T4 / RTX 2000    | 75        |
| A100             | 80        |
| A10 / RTX 3000   | 86        |
| ADA RTX 2000/4000| 89        |
| H100             | 90        |

## Quick Test
```bash


export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_LOGDEST_DBG=stdout
export CUBLAS_LOGINFO_DBG=1
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Tokenize
./build/tensor-ffn tokenize \
  --model ./qwen2.5-0.5b/ \
  --in    /root/docs/ \
  --out   ./corpus.bin

# Train
./build/tensor-ffn train \
  --model ./qwen2.5-0.5b/ \
  --data  ./corpus.bin \
  --out   ./my-expert.bin

  

# Inference comparison
./build/tensor-ffn inference --model ./qwen2.5-0.5b/ --prompt "func main()"
./build/tensor-ffn inference --model ./qwen2.5-0.5b/ --expert ./my-expert.bin --prompt "func main()"
```

## Model Weights
```bash
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='./qwen2.5-0.5b/')"
```