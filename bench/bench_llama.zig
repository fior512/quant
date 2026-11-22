const std = @import("std");
const harness = @import("harness.zig");
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

fn readNibble(weights: []const u8, weight_idx: usize) u8 { // element 2k: lo nibble
    return (weights[weight_idx / 2] >> @intCast(4 * (weight_idx % 2))) & 0x0F;
}

// llama.cpp block_nvfp4: 64 weights, 36 B
//   d[s]          e4m3 of w[16s..16s+15]
//   qs[8s + idx]  [w(16s + 8 + idx) << 4 | w(16s + idx)]
const BlockNvfp4 = extern struct { d: [4]u8, qs: [32]u8 };
extern fn dequantize_row_nvfp4(blocks: [*]const BlockNvfp4, out: [*]f32, weight_count: i64) void;

// llama.cpp block_mxfp4: 32 weights, 17 B, align 1
//   e        e8m0 of w[0..31]
//   qs[idx]  [w(16 + idx) << 4 | w(idx)]
const BlockMxfp4 = extern struct { e: u8, qs: [16]u8 };
extern fn dequantize_row_mxfp4(blocks: [*]const BlockMxfp4, out: [*]f32, weight_count: i64) void;

const LlamaNvfp4Kernel = struct {
    pub const SCHEME: generic.Scheme = .nvfp4;
    pub const OUTPUT: generic.Scheme = .f32;

    blocks: []const BlockNvfp4,

    pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !LlamaNvfp4Kernel {
        const blocks = try allocator.alignedAlloc(BlockNvfp4, ALIGN, logical.weight_count / 64);
        for (blocks, 0..) |*block, block_idx| {
            block.d = logical.scales[block_idx * 4 ..][0..4].*;
            for (0..4) |sub_idx| for (0..8) |pair_idx| {
                const weight_idx = block_idx * 64 + sub_idx * 16 + pair_idx;
                const lo = readNibble(logical.weights, weight_idx);
                const hi = readNibble(logical.weights, weight_idx + 8);
                block.qs[sub_idx * 8 + pair_idx] = lo | (hi << 4); // [w(16s + 8 + idx) << 4 | w(16s + idx)]
            };
        }
        return .{ .blocks = blocks };
    }

    pub noinline fn Run(kernel: *const LlamaNvfp4Kernel, out: []u8, out_s: []u8) void {
        _ = out_s;
        const out_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, out));
        dequantize_row_nvfp4(kernel.blocks.ptr, out_f32.ptr, @intCast(out_f32.len));
    }
};

const LlamaMxfp4Kernel = struct {
    pub const SCHEME: generic.Scheme = .mxfp4;
    pub const OUTPUT: generic.Scheme = .f32;

    blocks: []const BlockMxfp4,

    pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !LlamaMxfp4Kernel {
        const blocks = try allocator.alignedAlloc(BlockMxfp4, ALIGN, logical.weight_count / 32);
        for (blocks, 0..) |*block, block_idx| {
            block.e = logical.scales[block_idx];
            for (0..16) |pair_idx| {
                const weight_idx = block_idx * 32 + pair_idx;
                const lo = readNibble(logical.weights, weight_idx);
                const hi = readNibble(logical.weights, weight_idx + 16);
                block.qs[pair_idx] = lo | (hi << 4); // [w(16 + idx) << 4 | w(idx)]
            }
        }
        return .{ .blocks = blocks };
    }

    pub noinline fn Run(kernel: *const LlamaMxfp4Kernel, out: []u8, out_s: []u8) void {
        _ = out_s;
        const out_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, out));
        dequantize_row_mxfp4(kernel.blocks.ptr, out_f32.ptr, @intCast(out_f32.len));
    }
};

pub fn main(init: std.process.Init) !void {
    const args = harness.parseArgs(init.minimal.args.iterate());
    switch (args.out_scheme) {
        .f32 => {},
        else => @panic("Args: llama.cpp output is f32 only"),
    }
    switch (args.scheme) {
        .nvfp4 => try harness.run(LlamaNvfp4Kernel, "llama.cpp", args.size_filter),
        .mxfp4 => try harness.run(LlamaMxfp4Kernel, "llama.cpp", args.size_filter),
        else => @panic("Args: llama.cpp scheme is nvfp4|mxfp4 only"),
    }
}
