const std = @import("std");
pub const types = @import("types.zig");
const kernels = @import("kernels.zig");

pub const Input = struct {
    weights: []const u8,
    scales: []const u8,
    midpoint: []const u8, // .affine only
    global_scale: f32,
};


pub fn quant(
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
        inline else => |c_in_scheme| switch (out_scheme) {
            inline else => |c_out_scheme| switch (translation) {
                inline else => |c_translation| chunk(
                    c_in_scheme,
                    c_out_scheme,
                    c_translation,
                    in,
                    out_w,
                    out_s,
                ),
            },
        },
    }
}


fn chunk(
    comptime in_scheme: types.Scheme,
    comptime out_scheme: types.Scheme,
    comptime translation: types.Scheme.Translation,
    in: Input,
    out_w: []u8,
    out_s: []u8,
) void {
    const out_recipe = comptime types.Scheme.RECIPE.get(out_scheme);
    const WEIGHTS_PER_BLOCK = comptime if (out_recipe.scale == null) in_scheme.Block() else @max(in_scheme.Block(), out_scheme.Block());
    const BLOCK_SIZE = comptime in_scheme.WeightsSize(WEIGHTS_PER_BLOCK);
    const SCALES_PER_BLOCK = comptime if (out_recipe.scale == null) 1 else out_scheme.ScalesPerBlock(WEIGHTS_PER_BLOCK);
    const BLOCKS_PER_BATCH = if (out_recipe.scale != null)
        @min(8, @max(1, 16 / SCALES_PER_BLOCK))
    else
        1;

    std.debug.assert(WEIGHTS_PER_BLOCK % in_scheme.Block() == 0);
    if (out_recipe.scale != null) std.debug.assert(WEIGHTS_PER_BLOCK % out_scheme.Block() == 0);
    const block_count = in.weights.len / BLOCK_SIZE;

    // QUANT: SIMD over out_t scale per block
    var block_idx: usize = 0;
    while (block_idx + BLOCKS_PER_BATCH <= block_count) : (block_idx += BLOCKS_PER_BATCH) {
        batch(
            in_scheme,
            out_scheme,
            translation,
            BLOCKS_PER_BATCH,
            in,
            out_w,
            out_s,
            block_idx,
        );
    }

    // QUANT: tail
    while (block_idx < block_count) : (block_idx += 1) {
        batch(in_scheme, out_scheme, translation, 1, in, out_w, out_s, block_idx);
    }
}


fn batch(
    comptime in_scheme: types.Scheme,
    comptime out_scheme: types.Scheme,
    comptime translation: types.Scheme.Translation,
    comptime BLOCKS_PER_BATCH: usize,
    in: Input,
    out_w: []u8,
    out_s: []u8,
    first_block_idx: usize,
) void {
    //pipeline: unpack() -> decode() -> translate -> encode() -> pack()

    // LAYOUT
    const in_recipe = comptime types.Scheme.RECIPE.get(in_scheme);
    const out_recipe = comptime types.Scheme.RECIPE.get(out_scheme);
    const weight_t = comptime types.Dtype.RECIPE.get(in_recipe.weight);
    const out_t = comptime types.Dtype.RECIPE.get(out_recipe.weight);
    const WEIGHTS_PER_BLOCK = comptime if (out_recipe.scale == null) in_scheme.Block() else @max(in_scheme.Block(), out_scheme.Block());
    const Lanes = @Vector(WEIGHTS_PER_BLOCK, f32);
    const block_size = comptime in_scheme.WeightsSize(WEIGHTS_PER_BLOCK);
    const out_block_size = comptime out_scheme.WeightsSize(WEIGHTS_PER_BLOCK);

    // DECODE & ENCODE
    var weights: [BLOCKS_PER_BATCH]Lanes = undefined;
    for (&weights, 0..) |*weight, batch_block_idx| {
        const block_idx = first_block_idx + batch_block_idx;

        // DECODE (int_t -> f32)
        const unpacked = kernels.unpack(
            weight_t,
            WEIGHTS_PER_BLOCK,
            in.weights[block_idx * block_size ..][0..block_size],
        );
        const values = kernels.decode(weight_t, WEIGHTS_PER_BLOCK, unpacked);

        // TRANSLATE (f32)
        const scales = scaleIn(in_scheme, WEIGHTS_PER_BLOCK, in, block_idx);
        weight.* = switch (translation) {
            .affine => blk: { // s(q-z) -> sq + -zs; FMA rewrite
                const midpoint: Lanes = midpointIn(in_scheme, WEIGHTS_PER_BLOCK, in, block_idx);
                break :blk @mulAdd(Lanes, values, scales, -midpoint * scales);
            },
            .RTN => values * @as(Lanes, scales),
        };
    }

    // SCALE (f32 -> out scales)
    const SCALES_PER_BLOCK = comptime if (out_recipe.scale == null) 1 else out_scheme.ScalesPerBlock(WEIGHTS_PER_BLOCK);
    const OUT_BLOCK = comptime if (out_recipe.scale == null) WEIGHTS_PER_BLOCK else out_scheme.Block();
    const inv_scales: [BLOCKS_PER_BATCH][SCALES_PER_BLOCK]f32 = if (out_recipe.scale == null)
        @splat(@splat(1.0))
    else
        @bitCast(scaleOut(out_scheme, WEIGHTS_PER_BLOCK, BLOCKS_PER_BATCH, &weights, out_s, first_block_idx));

    // ENCODE (f32 -> out_scheme)
    for (weights, inv_scales, 0..) |weight, block_scales, batch_block_idx| {
        var inv_scale: [SCALES_PER_BLOCK]@Vector(OUT_BLOCK, f32) = undefined;
        inline for (&inv_scale, block_scales) |*lanes, scale| lanes.* = @splat(scale);
        const encoded = kernels.encode(
            out_t,
            WEIGHTS_PER_BLOCK,
            weight * @as(Lanes, @bitCast(inv_scale)),
        );
        out_w[(first_block_idx + batch_block_idx) * out_block_size ..][0..out_block_size].* = kernels.pack(
            out_t,
            WEIGHTS_PER_BLOCK,
            encoded,
        );
    }
}


fn scaleIn(
    comptime in_scheme: types.Scheme,
    comptime WEIGHTS_PER_BLOCK: usize,
    in: Input,
    block_idx: usize,
) @Vector(WEIGHTS_PER_BLOCK, f32) {
    // resolve scale layout
    const in_recipe = comptime types.Scheme.RECIPE.get(in_scheme);
    const scale_t = in_recipe.scale orelse return @splat(in.global_scale);
    const scale_recipe = comptime types.Dtype.RECIPE.get(scale_t);
    const SCALE_SIZE = comptime in_scheme.ScaleSize();
    const SCALES_PER_BLOCK = comptime in_scheme.ScalesPerBlock(WEIGHTS_PER_BLOCK);

    var scales: [SCALES_PER_BLOCK]@Vector(in_scheme.Block(), f32) = undefined;
    inline for (&scales, 0..) |*scale, block_scale_idx| { // DECODE
        const scale_idx = block_idx * SCALES_PER_BLOCK + block_scale_idx;
        const unpacked = kernels.unpack(
            scale_recipe,
            1,
            in.scales[scale_idx * SCALE_SIZE ..][0..SCALE_SIZE],
        );
        const decoded = kernels.decode(scale_recipe, 1, unpacked)[0];

        scale.* = @splat(decoded * in.global_scale);
    }
    return @bitCast(scales);
}

fn midpointIn(
    comptime in_scheme: types.Scheme,
    comptime WEIGHTS_PER_BLOCK: usize,
    in: Input,
    block_idx: usize,
) @Vector(WEIGHTS_PER_BLOCK, f32) {
    const MIDPOINTS_PER_BLOCK = comptime in_scheme.ScalesPerBlock(WEIGHTS_PER_BLOCK);
    var midpoints: [MIDPOINTS_PER_BLOCK]@Vector(in_scheme.Block(), f32) = undefined;
    inline for (&midpoints, 0..) |*midpoint, block_midpoint_idx| {
        midpoint.* = @splat(
            @floatFromInt(in.midpoint[block_idx * MIDPOINTS_PER_BLOCK + block_midpoint_idx]),
        );
    }
    return @bitCast(midpoints);
}

fn scaleOut(
    comptime out_scheme: types.Scheme,
    comptime WEIGHTS_PER_BLOCK: usize,
    comptime BLOCKS_PER_BATCH: usize,
    weights: *const [BLOCKS_PER_BATCH]@Vector(WEIGHTS_PER_BLOCK, f32),
    scales: []u8,
    first_block_idx: usize,
) @Vector(BLOCKS_PER_BATCH * out_scheme.ScalesPerBlock(WEIGHTS_PER_BLOCK), f32) {
    // SCALE LAYOUT
    const out_recipe = comptime types.Scheme.RECIPE.get(out_scheme);
    const scale_t = out_recipe.scale orelse return @splat(1.0);
    const scale_recipe = comptime types.Dtype.RECIPE.get(scale_t);
    const SCALE_SIZE = comptime out_scheme.ScaleSize();
    const SCALES_PER_BLOCK = comptime out_scheme.ScalesPerBlock(WEIGHTS_PER_BLOCK);
    const SCALE_COUNT = BLOCKS_PER_BATCH * SCALES_PER_BLOCK;
    const ScaleBlock = @Vector(out_scheme.Block(), f32);
    const Scales = @Vector(SCALE_COUNT, f32);

    // REDUCE BLOCK MAGNITUDE
    const scale_blocks: [SCALE_COUNT]ScaleBlock = @bitCast(weights.*);
    var absmax: Scales = undefined;
    inline for (scale_blocks, 0..) |scale_block, scale_idx| absmax[scale_idx] = kernels.absMax(
        out_scheme.Block(),
        scale_block,
    );

    // ENCODE
    const encoded = kernels.encode(
        scale_recipe,
        SCALE_COUNT,
        absmax / @as(Scales, @splat(out_recipe.absmax)),
    );
    const scale_offset = first_block_idx * SCALES_PER_BLOCK * SCALE_SIZE;
    scales[scale_offset..][0 .. SCALE_COUNT * SCALE_SIZE].* = kernels.pack(
        scale_recipe,
        SCALE_COUNT,
        encoded,
    );

    // INVERT
    const decoded = kernels.decode(scale_recipe, SCALE_COUNT, encoded);
    const zero: Scales = @splat(0);
    return @select(f32, decoded == zero, zero, @as(Scales, @splat(1.0)) / decoded);
}
