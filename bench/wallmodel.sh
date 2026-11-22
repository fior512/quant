#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

shards="${*:-models/gpt-oss-20b/model-*.safetensors}"
set -- $shards
first=$1
model_name=$(basename "$(dirname "$first")")
time_file="bench/out/gptoss_time_$$"
prefix="bench/out/release" # private install: a concurrent `zig build` (Debug) cannot swap the binary
bin="$prefix/bin/dequant"

measure_total_ms() {
  type=$1
  threads=$2
  min_s=999999
  i=0
  while [ "$i" -lt 3 ]; do
    /usr/bin/time -f '%e %M' -o "$time_file" "$bin" $shards "$type" "$threads" >/dev/null 2>&1
    t=$(awk '{print $1}' "$time_file")
    min_s=$(awk -v a="$min_s" -v b="$t" 'BEGIN{print (b<a)?b:a}')
    i=$((i+1))
  done
  awk -v s="$min_s" 'BEGIN{printf "%.0f", s*1000}'
}

# build native once
mkdir -p bench/out
zig build -Doptimize=ReleaseFast -Dcpu=native --prefix "$prefix"

# model stats
input_bytes=0
blocks_bytes=0
for f in $shards; do
  s=$(stat -c %s "$f")
  input_bytes=$((input_bytes + s))
  hs=$(od -An -N8 -t u8 "$f" | tr -d ' ')
  b=$(tail -c +9 "$f" | head -c "$hs" | jq '
    . as $root
    | [ to_entries[]
        | .key as $k
        | (if ($k | endswith(".weight_packed"))
             then $k[0:($k|length)-14] + ".weight_scale"
           elif ($k | endswith(".weight"))
             then $k[0:($k|length)-7] + ".weight_scale"
           elif ($k | endswith("_blocks"))
             then $k[0:($k|length)-7] + "_scales"
           else empty
           end) as $scale
        | select($root | has($scale))
        | .value.data_offsets[1] - .value.data_offsets[0]
      ]
    | add // 0
  ')
  blocks_bytes=$((blocks_bytes + b))
done
weights_count=$((blocks_bytes * 2))

echo "# $model_name, min of 3 whole-model runs"
echo "| out_scheme | 1t | 6t | 12t | GB/s in (6t) | GB/s out (6t) | qw/s (6t) |"
echo "|---|---|---|---|---|---|---|"
for type in nvfp4 mxfp4 i4 f8e4m3 f16 bf16 f32; do
  case "$type" in
    f32)                out_bits=32 ;;
    f16 | bf16)         out_bits=16 ;;
    f8e4m3)             out_bits=8 ;;
    nvfp4 | mxfp4 | i4) out_bits=4 ;;
  esac
  t1=$(measure_total_ms "$type" 1)
  t6=$(measure_total_ms "$type" 6)
  t12=$(measure_total_ms "$type" 12)
  out_bytes=$((weights_count * out_bits / 8))
  gb_in=$(awk -v b="$input_bytes" -v t="$t6" 'BEGIN{printf "%.1f", b*1e-6/t}')
  gb_out=$(awk -v b="$out_bytes" -v t="$t6" 'BEGIN{printf "%.1f", b*1e-6/t}')
  qw=$(awk -v w="$weights_count" -v t="$t6" 'BEGIN{printf "%.1f", w*1e-6/t}')
  printf '| %s | %s ms | %s ms | %s ms | %s | %s | %s G |\n' "$type" "$t1" "$t6" "$t12" "$gb_in" "$gb_out" "$qw"
done

echo ""
echo "# xychart: $model_name, GB/s in vs threads, one line per out_scheme"
echo "threads: 1 2 3 4 6 8 12"
for type in nvfp4 mxfp4 i4 f8e4m3 f16 bf16 f32; do
  in_line=""
  for t in 1 2 3 4 6 8 12; do
    ms=$(measure_total_ms "$type" "$t")
    gb_in=$(awk -v b="$input_bytes" -v t="$ms" 'BEGIN{printf "%.1f", b*1e-6/t}')
    in_line="$in_line${in_line:+, }$gb_in"
  done
  printf '%-6s [%s]\n' "$type" "$in_line"
done

rm -f "$time_file"
