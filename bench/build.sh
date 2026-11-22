#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p out

zig cc -O3 -march=native -fno-lto -DNDEBUG -c llama_nvfp4.c -o out/llama_nvfp4.o
zig cc -O3 -march=native -fno-lto -DNDEBUG -c llama_mxfp4.c -o out/llama_mxfp4.o

zig build-exe -femit-bin=out/bench_llama -OReleaseFast -mcpu=native -fno-lto -lc out/llama_nvfp4.o out/llama_mxfp4.o --dep dequant -Mroot=bench_llama.zig -Mdequant=../src/dequant.zig

zig build-exe -femit-bin=out/bench_dequant -OReleaseFast -mcpu=native -fno-lto --dep dequant -Mroot=bench_dequant.zig -Mdequant=../src/dequant.zig
