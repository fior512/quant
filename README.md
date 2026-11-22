
Dequantizer for `.safetensors` in Zig (0.17)

gpt-oss-20b (`openai/gpt-oss-20b`, 12.8 GiB, MXFP4 experts) on Ryzen 5 7600X:

| out_t | 1t | 6t | 12t | GB/s in (6t) | GB/s out (6t) | qw/s (6t) |
|---|---|---|---|---|---|---|
| f32 | 1510 ms | 360 ms | 380 ms | 38.2 | 212.3 | 53.1 G |
| f16 | 1840 ms | 420 ms | 380 ms | 32.8 | 91.0 | 45.5 G |
| bf16 | 3400 ms | 700 ms | 630 ms | 19.7 | 54.6 | 27.3 G |
| f8e4m3 | 3770 ms | 800 ms | 700 ms | 17.2 | 23.9 | 23.9 G |
| f8e8m0 | 1770 ms | 410 ms | 360 ms | 33.6 | 46.6 | 46.6 G |
| i4 | 3390 ms | 720 ms | 640 ms | 19.1 | 13.3 | 26.5 G |
| f4e2m1 | 4210 ms | 890 ms | 790 ms | 15.5 | 10.7 | 21.5 G |

## Usage

```sh
zig build -Doptimize=ReleaseFast -Dcpu=native
./zig-out/bin/dequant ./models/gpt-oss-20b/model*.safetensors f32 8
```

Recognized tensor layouts (`src/safetensors.zig`, `HEADERS`):

| scheme | weight | scale | global scale |
|---|---|---|---|
| NVFP4 (1) | U8 | F8_E4M3 | F32 |
| NVFP4 (2) | U8 | F8_E4M3 | F32 (inv) |
| MXFP4 (3) | U8 | U8 (E8M0) | none |
| INT4 W4A16 (2) | I32 | BF16 | none |
| FP8 (1)(2) | F8_E4M3 | F32 (scalar) | none |
| plain | F32 / F16 / BF16 / F8_E4M3 | none | none |

> 1. https://github.com/NVIDIA/Model-Optimizer
> 2. https://github.com/vllm-project/compressed-tensors
> 3. https://huggingface.co/openai/gpt-oss-20b


## Pipeline

```text
main
  parseJson                         single threaded
  spawn N workers

worker
  incremental atomic counter as shared chunk idx
  read -> dequant -> `doNotOptimizeAway()`

dequant block
  unpack                            lo/hi shuffle
  decode     e_m_ -> f32
  loadScale  e_m_ -> f32            lookup translate
  translate  f32                    RTN | std-affine         
  encode     f32 -> out_t           f32 | f16 | bf16
  pack                              lo/hi shuffle
```
The quantization pipeline is modular through `types.Dtype.RECIPE` and `types.Scheme.RECIPE` which holds type recipies. All threads reuse their buffer, kept via `doNotOptimizeAway`.

## Tests

```sh
zig build test
```

- `dequant: nvfp4`, `dequant: mxfp4`: all 16 E2M1 x 2 scales, bit-exact for f32 / f16 / bf16 (`-0.0` included)
- `safetensors: parseJson`: scale and global scale merged into the weight layer.

## Benchmarks

Setup: Ryzen 5 7600X (Zen 4, 6C/12T), DDR5, Linux 6.18, boost on, EPP balance_performance, zig 0.17. Files warm in page cache. Min of 3 per shard, summed over shards.

```sh
hf download openai/gpt-oss-20b --include "model-*.safetensors" --local-dir models/gpt-oss-20b
./bench/wallmodel.sh models/gpt-oss-20b/model-*.safetensors
```

### Per target

```sh
./bench/target.sh
```

| target | ISA | gpt f32 | gpt f16 | gpt bf16 | kernel nvfp4 ns/w | cyc/block (mca) |
|---|---|---|---|---|---|---|
| x86_64_v1 | SSE2 | 5430 ms | 88330 ms | 7040 ms | 0.2827 | 25.0 |
| x86_64_v2 | SSE4.2 | 5450 ms | 88270 ms | 7040 ms | 0.2909 | 25.0 |
| x86_64_v3 | AVX2+F16C | 2010 ms | 2580 ms | 3340 ms | 0.0785 | 15.0 |
| x86_64_v4 | AVX-512 | 1710 ms | 1930 ms | 2720 ms | 0.0663 | 5.0 |
| native | znver4+GFNI | 1320 ms | 1640 ms | 2330 ms | 0.0439 | 3.0 |

### Thread scaling

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2a78d6, #FF9900, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2"}}}}%%
xychart-beta
    title "gpt-oss-20b, GB/s in vs threads"
    x-axis "threads" ["1", "2", "3", "4", "6", "8", "12"]
    y-axis "GB/s in" 0 --> 45
    line [9.1, 17.4, 24.1, 30.6, 38.2, 39.3, 36.2]
    line [7.5, 14.5, 20.2, 26.0, 32.8, 33.6, 35.3]
    line [4.0, 7.7, 11.1, 14.8, 19.7, 20.5, 22.2]
    line [7.8, 15.0, 21.2, 27.0, 33.6, 36.2, 37.2]
    line [3.7, 7.1, 10.2, 13.5, 17.2, 18.6, 19.9]
    line [4.1, 7.9, 11.4, 15.0, 19.1, 19.7, 21.5]
    line [3.3, 6.3, 9.2, 12.1, 15.6, 16.4, 17.4]
```
- blue: f32
- orange: f16
- green: bf16
- red: f8e8m0
- purple: f8e4m3
- brown: i4
- pink: f4e2m1

### In and Out type permutations

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2a78d6, #FF9900, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2"}}}}%%
xychart-beta
    title "L1 median ns/w, in_t (line) -> out_t (x)"
    x-axis "out_scheme" ["nvfp4", "mxfp4", "i4", "f8e4m3", "f16", "bf16", "f32"]
    y-axis "ns/w" 0 --> 1.44
    line [0.9621, 0.6061, 0.3446, 0.1553, 0.0633, 0.1497, 0.0486]
    line [0.8133, 0.6009, 0.4055, 0.1532, 0.0608, 0.1479, 0.0440]
    line [0.8743, 0.5719, 0.3444, 0.1472, 0.0595, 0.1386, 0.0464]
    line [1.2232, 0.9091, 0.4189, 0.2575, 0.1338, 0.2582, 0.1016]
    line [0.8364, 0.5389, 0.3163, 0.1234, 0.0479, 0.1158, 0.0294]
    line [1.2516, 1.1277, 0.4713, 0.1844, 0.1022, 0.1767, 0.0728]
    line [1.1659, 1.0249, 0.3958, 0.1262, 0.0445, 0.1140, 0.0349]
```

- blue: nvfp4
- orange: mxfp4
- green: i4
- red: f8e4m3
- purple: f16
- brown: bf16
- pink: f32



**Scaling:**

- Reading from userspace cap at 63 GB/s, reached with 3 threads. Then copies start competing for DRAM bus (not on gpt-oss because quantized weight freq is too high)
- Each layer is one work item. A gpt-oss shard has 20 quantized ones. One 253 MiB tensor takes 85 ms on one thread, against 213 ms of ideal work per thread at 6 threads
- `_copy_to_iter` takes 16% of cycles at 1 thread

Caveat: `out_buf` is reused, so output never reaches DRAM. A real f32 consumer writes 8 bytes per byte of quantized input, so it hits the write-bandwidth limit before compute does.

### Kernel vs llama.cpp

Output f32, 1 thread, kernels only (no file read, parse or orchestration). Median ns/weight.

| size | cache | llama.cpp nvfp4 | HEAD nvfp4 | ratio | llama.cpp mxfp4 | HEAD mxfp4 | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 KiB | L1 | 0.2899 | 0.0436 | 6.65x | 0.2656 | 0.0439 | 6.05x |
| 584 KiB | L2 | 0.2924 | 0.0445 | 6.57x | 0.2660 | 0.0446 | 5.96x |
| 18 MiB | L3 | 0.3159 | 0.0455 | 6.94x | 0.2673 | 0.0459 | 5.82x |
| 292 MiB | DRAM | 0.3171 | 0.1427 | 2.22x | 0.2680 | 0.1419 | 1.89x |

```sh
./bench/build.sh
./bench/out/bench_llama nvfp4 && ./bench/out/bench_dequant nvfp4
./bench/out/bench_llama mxfp4 && ./bench/out/bench_dequant mxfp4
```

> llama.cpp kernels from `tests/test-quantize-perf.cpp`, master `ded522f8304554317e7ff8f88698dbbbca5425aa` (2026-11-13). Pinned, 3 trials x 20 s, TSC-timed.

## TODO

- Work item = chunk, not layer
- sort layers for latency tail
- swap `vcvtps2ph`
- more layouts
- gpt-oss-120b (deferred load)
- skip read if unquantized to avoid _copy_to_iter
- tail is dropped
