#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

size="${1:-L1}"
bench="${BENCH:-bench/out/bench_dequant}"
schemes="nvfp4 mxfp4 i4 f8e4m3 f16 bf16 f32"
out_dir="bench/out"
mkdir -p "$out_dir"

lines_file="$out_dir/matrix_lines_$$"
: > "$lines_file"
max_y=0

for s in $schemes; do
  row=""
  for t in $schemes; do
    echo "== $s -> $t ==" >&2
    "$bench" "$s" "$t" "$size" > "$out_dir/matrix_${s}_${t}.txt" 2>&1
    ns_w=$(awk '/summary/{print $7}' "$out_dir/matrix_${s}_${t}.txt")
    row="${row}${row:+, }$ns_w"
    if awk -v a="$max_y" -v b="$ns_w" 'BEGIN{exit !(b > a)}'; then
      max_y=$ns_w
    fi
  done
  printf '%s [%s]\n' "$s" "$row" >> "$lines_file"
done

max_y=$(awk -v m="$max_y" 'BEGIN{printf "%.2f", m * 1.15}')

echo '```mermaid'
echo '%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2a78d6, #FF9900, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2"}}}}%%'
echo 'xychart-beta'
echo "    title \"$size median ns/w, in_scheme (line) -> out_scheme (x)\""
echo '    x-axis "out_scheme" ["nvfp4", "mxfp4", "i4", "f8e4m3", "f16", "bf16", "f32"]'
echo "    y-axis \"ns/w\" 0 --> $max_y"
while IFS= read -r line; do
  echo "    line ${line#* }"
done < "$lines_file"
echo '```'
echo
echo '- blue: nvfp4'
echo '- orange: mxfp4'
echo '- green: i4'
echo '- red: f8e4m3'
echo '- purple: f16'
echo '- brown: bf16'
echo '- pink: f32'

rm -f "$lines_file"
