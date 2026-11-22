const std = @import("std");
pub const types = @import("types.zig");
const kernels = @import("kernels.zig");

pub const Input = struct {
    weights: []const u8,
    scales: []const u8,
    midpoint: []const u8, // .affine only
    global_scale: f32,
};

pub fn dequant(
    in_scheme: types.Scheme,
    out_scheme: types.Scheme,
    in: Input,
    out_w: []u8,
    out_s: []u8,
) void {
    const translation: types.Scheme.Translation = switch (in.midpoint.len) {
        0 => .RTN,
        else => .affine,
    };
    switch (in_scheme) {
        inline else => |c_scheme| switch (out_scheme) {
            inline else => |c_out_scheme| switch (translation) {
                inline else => |c_translation| block(
                    types.Scheme.RECIPE.get(c_scheme),
                    types.Scheme.RECIPE.get(c_out_scheme),
                    c_translation,
                    in,
                    out_w, // weights
                    out_s, // scales
                ),
            },
        },
    }
}

fn block(
    comptime in_scheme: types.Scheme.Recipe,
    comptime out_scheme: types.Scheme.Recipe,
    comptime translation: types.Scheme.Translation,
    in: Input,
    out_w: []u8,
    out_s: []u8,
) void { // dequant module's main
    // Pipeline: (in_t)unpack() -> decode() -> (f32)translate -> encode() -> (out_t)pack()
    const weight_t = comptime types.Dtype.RECIPE.get(in_scheme.weight);
    const out_t = comptime types.Dtype.RECIPE.get(out_scheme.weight);
    const BLOCK: usize = @max(in_scheme.block, out_scheme.block);
    std.debug.assert(BLOCK % in_scheme.block == 0); // hints
    std.debug.assert(BLOCK % out_scheme.block == 0);
    const Lanes = @Vector(BLOCK, f32);

    // BLOCKs
    const block_size = BLOCK * weight_t.bits / 8; //bytes
    const out_block_size = BLOCK * out_t.bits / 8; //bytes
    const block_count = in.weights.len / block_size;

    for (0..block_count) |idx| {
        // DECODE (int_t -> f32)
        const unpacked = kernels.unpack(
            weight_t,
            BLOCK,
            in.weights[idx * block_size ..][0..block_size],
        );
        const values = kernels.decode(weight_t, BLOCK, unpacked);

        // TRANSLATE (f32)
        const scales = scaleIn(in_scheme, BLOCK, in, idx);
        const weights: Lanes = switch (translation) {
            .affine => blk: { // s(q-z) -> sq + -zs; FMA rewrite
                const midpoint: Lanes = midpointIn(in_scheme, BLOCK, in, idx);
                break :blk @mulAdd(Lanes, values, scales, -midpoint * scales);
            },
            .RTN => values * @as(Lanes, scales),
        };

        // ENCODE (f32 -> out_scheme)
        const encoded = kernels.encode(
            out_t,
            BLOCK,
            weights * scaleOut(out_scheme, BLOCK, weights, out_s, idx),
        );
        out_w[idx * out_block_size ..][0..out_block_size].* = kernels.pack(
            out_t,
            BLOCK,
            encoded,
        );
    }
}

fn scaleIn(
    comptime in_scheme: types.Scheme.Recipe,
    comptime BLOCK: usize,
    in: Input,
    idx: usize,
) @Vector(BLOCK, f32) {
    const scale_t = in_scheme.scale orelse return @splat(in.global_scale);
    const input_t = comptime types.Dtype.RECIPE.get(scale_t);
    const size = input_t.bits / 8;
    const SUB_COUNT = BLOCK / in_scheme.block;

    var scales: [SUB_COUNT]@Vector(in_scheme.block, f32) = undefined;
    inline for (&scales, 0..) |*scale, sub_idx| { // one scale per in_scheme.block
        const block_idx = idx * SUB_COUNT + sub_idx;
        const unpacked = kernels.unpack(input_t, 1, in.scales[block_idx * size ..][0..size]);
        const decoded = kernels.decode(input_t, 1, unpacked)[0];
        scale.* = @splat(decoded * in.global_scale);
    }
    return @bitCast(scales);
}

fn midpointIn(
    comptime in_scheme: types.Scheme.Recipe,
    comptime BLOCK: usize,
    in: Input,
    idx: usize,
) @Vector(BLOCK, f32) {
    const SUB_BLOCK = BLOCK / in_scheme.block;
    var midpoints: [SUB_BLOCK]@Vector(in_scheme.block, f32) = undefined;
    inline for (&midpoints, 0..) |*midpoint, sub_idx| {
        midpoint.* = @splat(@floatFromInt(in.midpoint[idx * SUB_BLOCK + sub_idx]));
    }
    return @bitCast(midpoints);
}

fn scaleOut(
    comptime out_scheme: types.Scheme.Recipe,
    comptime BLOCK: usize,
    weights: @Vector(BLOCK, f32),
    scales: []u8,
    idx: usize,
) @Vector(BLOCK, f32) {
    const scale_t = out_scheme.scale orelse return @splat(1.0);
    const output_t = comptime types.Dtype.RECIPE.get(scale_t);
    const size = output_t.bits / 8;
    const SUB_COUNT = BLOCK / out_scheme.block;

    const blocks: [SUB_COUNT]@Vector(out_scheme.block, f32) = @bitCast(weights);
    var inv_scales: [SUB_COUNT]@Vector(out_scheme.block, f32) = undefined;
    inline for (blocks, &inv_scales, 0..) |sub_block, *inv_scale, sub_idx| { // one scale per out_scheme.block
        const block_idx = idx * SUB_COUNT + sub_idx;
        const block_absmax = @reduce(.Max, @abs(sub_block));
        const encoded = kernels.encode(
            output_t,
            1,
            @as(@Vector(1, f32), @splat(block_absmax / out_scheme.absmax)),
        );
        scales[block_idx * size ..][0..size].* = kernels.pack(output_t, 1, encoded);
        const decoded = kernels.decode(output_t, 1, encoded)[0];
        inv_scale.* = @splat(if (decoded == 0) 0 else 1.0 / decoded);
    }
    return @bitCast(inv_scales);
}
