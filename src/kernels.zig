const std = @import("std");
const types = @import("types.zig");



pub inline fn pack(comptime out_t: types.Dtype.Recipe, comptime count: usize, values: @Vector(count, u32)) [count * out_t.bits / 8]u8 {
    switch (out_t.bits) {
        4 => {
            const Half = @Vector(count/2, u8);
            const unpacked: @Vector(count, u8) = @truncate(values);
            const even, const odd = comptime blk: {
                var even_mask: [count/2]i32 = undefined;
                var odd_mask: [count/2]i32 = undefined;
                for (&even_mask, &odd_mask, 0..) |*even, *odd, idx| {
                    even.* = @intCast(2 * idx);
                    odd.* = @intCast(2 * idx + 1);
                }
                break :blk .{ even_mask, odd_mask };
            };

            const lo: Half = @shuffle(u8, unpacked, undefined, even);
            const hi: Half = @shuffle(u8, unpacked, undefined, odd);
            return lo | (hi << @splat(4));
        },
        else => {
            const Int = @Int(.unsigned, out_t.bits);
            const Bytes = @Vector(count * out_t.bits / 8, u8);
            const unpacked: @Vector(count, Int) = @truncate(values);
            return @as(Bytes, @bitCast(unpacked));
        },
    }
}


pub inline fn unpack(comptime out_t: types.Dtype.Recipe, comptime count: usize, bytes: *const [count * out_t.bits / 8]u8) @Vector(count, @Int(.unsigned, @max(8, out_t.bits))) {
    switch (out_t.bits) {
        4 => {
            const Half = @Vector(count/2, u8);
            const packed_w: Half = bytes.*;

            const lo: Half = packed_w & @as(Half, @splat(0x0F)); // 0x0L, 0x0L, ..
            const hi: Half = packed_w >> @as(Half, @splat(4));   // 0x0H, 0x0H, ..
            const interleaved = comptime blk: {
                var mask: [count]i32 = undefined;
                for(&mask, 0..) |*lane, idx| {
                    const half_idx: i32 = @intCast(idx/2);
                    lane.* = if (idx%2 == 0) half_idx else ~half_idx;   // `~` to sign swap (signed int)
                                                                        // lane.* > 0 -> a
                                                                        // lane.* < 0 -> b
                }
                break :blk mask;
            };
            return @shuffle(u8, lo, hi, interleaved); // 0x0L, 0x0H, 0x0L, 0x0H, ..
                                                                     // lo first bcs little endian
        },
        8 => return bytes.*,
        else => {
            const Int = @Int(.unsigned, out_t.bits);
            const size = out_t.bits / 8;
            var out: @Vector(count, Int) = undefined;
            inline for (0..count) |idx| out[idx] = std.mem.readInt(Int, bytes[idx*size..][0..size], .little);
            return out;
        },
    }
}


pub inline fn decode(comptime out_t: types.Dtype.Recipe, comptime count: usize, values: anytype) @Vector(count, f32) {
    const Lanes = @Vector(count, u32);
    const Floats = @Vector(count, f32);
    const wide: Lanes = values; // 8bits -> 32
    

    const t_size = 1 << out_t.bits;
    if (comptime count == 1 and out_t.bits <= 8) if (!@inComptime()) { // scalar
        const TABLE: [t_size]f32 = comptime tableTranslate(out_t);
        return @splat(TABLE[wide[0]]);
    };
    const reg_units = comptime @min(t_size, std.simd.suggestVectorLength(u32) orelse 1);
    if (comptime t_size <= 2 * reg_units) if (!@inComptime()) { // SIMD Q:fit in 2 registers? (hi,lo)
        const TABLE: [t_size]f32 = comptime tableTranslate(out_t);
        
        const idx = wide & @as(Lanes, @splat(reg_units - 1));

        const TABLE_LO: @Vector(reg_units, f32) = TABLE[0..reg_units].*;
        const TABLE_HI: @Vector(reg_units, f32) = TABLE[t_size - reg_units..].*;

        var lo: Floats = undefined;
        var hi: Floats = undefined;
        inline for (0..count) |lane_idx| {
            lo[lane_idx] = TABLE_LO[idx[lane_idx]];
            hi[lane_idx] = TABLE_HI[idx[lane_idx]];
        }
        return @select(f32, wide >= @as(Lanes, @splat(reg_units)), hi, lo);
    };

    switch (out_t.kind) {
        .float => { // element: [sign | exp | mant]
            // Extract
            const sign = (wide >> @splat(out_t.bits-1)) << @splat(31);
            const abs = wide & @as(Lanes, @splat((1 << (out_t.bits - 1)) - 1));
            const shifted: Floats = @bitCast(abs << @splat(23 - out_t.mant_bits));
            const rescaled: f32 = @bitCast(@as(u32, 254 - @as(u32, out_t.bias)) << 23);
            const magnitude: Lanes = @bitCast(shifted * @as(Floats, @splat(rescaled)));
            return @bitCast(sign | magnitude);
        },
        .exponent => {
            const zero: Lanes = @splat(0);

            const lo = @select(u32, wide == zero, @as(Lanes, @splat(0x00400000)), wide << @splat(23));
            const width_bits = @select(u32, wide == @as(Lanes, @splat(0xFF)), @as(Lanes, @splat(0x7FC00000)), lo);
            return @bitCast(width_bits);
        },
        .native => {
            const Float = std.meta.Float(out_t.bits);
            const Int = @Int(.unsigned, out_t.bits);
            return @floatCast(@as(@Vector(count, Float), @bitCast(@as(@Vector(count, Int), @truncate(wide)))));
        },
        .integer => return @as(Floats, @floatFromInt(wide)) - @as(Floats, @splat(@floatFromInt(out_t.bias))), // x - rescaled
    }
}



pub inline fn encode(comptime out_t: types.Dtype.Recipe, comptime count: usize, values: @Vector(count, f32)) @Vector(count, u32) {
    const Lanes = @Vector(count, u32);
    const Floats = @Vector(count, f32);
    switch (out_t.kind) {
        .native => {
            const Float = std.meta.Float(out_t.bits);
            const Int = @Int(.unsigned, out_t.bits);
            return @as(@Vector(count, Int), @bitCast(@as(@Vector(count, Float), @floatCast(values)))); // f16: vcvtps2ph
        },
        .float => {
            const SHIFT = 23 - out_t.mant_bits; // f32 mantissa
            const width_bits: Lanes = @bitCast(values);
            const sign = (width_bits >> @splat(31)) << @splat(out_t.bits - 1);
            var abs_f = @abs(values);
            if (comptime out_t.expmode != .ieee) abs_f = @min(abs_f, @as(Floats, @splat(out_t.max))); // overflow
            const abs: Lanes = @bitCast(abs_f);

            // ROUND
            const midbit = (abs >> @splat(SHIFT)) & @as(Lanes, @splat(1)); // RTNe
            const rounded = abs +% @as(Lanes, @splat((1 << (SHIFT - 1)) - 1)) +% midbit; // x + (0x8-1) + isOdd
            const normal = (rounded >> @splat(SHIFT)) -% @as(Lanes, @splat(@as(u32, 127 - out_t.bias) << out_t.mant_bits));

            // SUBNORMAL
            const MAGIC: f32 = @bitCast(@as(u32, 127 - @as(u32, out_t.bias) + SHIFT + 1) << 23);
            const subnormal = @as(Lanes, @bitCast(abs_f + @as(Floats, @splat(MAGIC)))) -% @as(Lanes, @splat(@as(u32, @bitCast(MAGIC))));
            
            // ENCODE
            const MIN_NORMAL: f32 = @bitCast(@as(u32, 128 - @as(u32, out_t.bias)) << 23); // 2^(1 - rescaled)
            var code = @select(u32, abs_f < @as(Floats, @splat(MIN_NORMAL)), subnormal, normal);
            if (comptime out_t.expmode == .ieee) code = @min(code, @as(Lanes, @splat(((1 << out_t.exp_bits) - 1) << out_t.mant_bits))); // overflow
            return sign | code;
        },
        .exponent => return (@as(Lanes, @bitCast(values)) >> @splat(23)) & @as(Lanes, @splat(0xFF)), // floor(log2)
        .integer => {
            const shifted = @round(values) + @as(Floats, @splat(@floatFromInt(out_t.bias))); // x + rescaled
            const clamped = @min(@max(shifted, @as(Floats, @splat(0))), @as(Floats, @splat((1 << out_t.bits) - 1)));
            return @intFromFloat(clamped);
        },
    }
}



pub fn tableTranslate(comptime out_t: types.Dtype.Recipe) [1 << out_t.bits]f32 {
    //Produce a lookup table for fast decode
    @setEvalBranchQuota(1_000_000);
    const t_size = 1 << out_t.bits;
    var values: [t_size]u8 = undefined;
    for (&values, 0..) |*element , idx| element.* = @intCast(idx);
    return comptime decode(out_t, t_size, values);
}

pub inline fn absMax(comptime count: usize, values: @Vector(count, f32)) f32 {
    const Ints = @Vector(count, i32);
    const inf_bits: Ints = @splat(@bitCast(std.math.inf(f32)));
    const nan_bits: Ints = @splat(@bitCast(-std.math.nan(f32)));

    const magnitudes: Ints = @bitCast(@abs(values));
    const ordered = @select(i32, magnitudes > inf_bits, nan_bits, magnitudes);
    return @abs(@as(f32, @bitCast(@reduce(.Max, ordered))));
}
