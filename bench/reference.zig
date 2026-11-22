const std = @import("std");
const generic = @import("generic.zig");

const Recipe = generic.Dtype.Recipe;

// decode: value formula, no shared code with src/kernels.zig
pub fn decode(comptime format: Recipe, code: u32) f32 {
    switch (format.kind) {
        .native => return switch (format.bits) {
            16 => @floatCast(@as(f16, @bitCast(@as(u16, @truncate(code))))),
            32 => @bitCast(code),
            else => unreachable,
        },
        .integer => return @as(f32, @floatFromInt(code)) - @as(f32, @floatFromInt(format.bias)), // offset binary
        .exponent => return @floatCast(std.math.ldexp(@as(f64, 1.0), @as(i32, @intCast(code)) - 127)), // 2^(e-127)
        .float => switch (format.bits) {
            16 => return @bitCast(code << 16), // bf16: high u16 of f32
            else => {
                // [s|e..e|m..m] -> (-1)^s * 2^(e-bias) * (1 + m/2^M); e == 0 -> 2^(1-bias) * m/2^M
                const mant_mask = (@as(u32, 1) << @intCast(format.mant_bits)) - 1;
                const exp_mask = (@as(u32, 1) << @intCast(format.exp_bits)) - 1;
                const is_negative = (code >> @intCast(format.bits - 1)) & 1 == 1;
                const exp = (code >> @intCast(format.mant_bits)) & exp_mask;
                const mant: f64 = @floatFromInt(code & mant_mask);
                const mant_scale: f64 = @floatFromInt(@as(u32, 1) << @intCast(format.mant_bits));
                const bias: i32 = format.bias;
                const magnitude: f64 = switch (exp) {
                    0 => std.math.ldexp(mant / mant_scale, 1 - bias),
                    else => std.math.ldexp(1.0 + mant / mant_scale, @as(i32, @intCast(exp)) - bias),
                };
                return @floatCast(if (is_negative) -magnitude else magnitude);
            },
        },
    }
}

pub fn isNanCode(comptime format: Recipe, code: u32) bool {
    const magnitude_mask = (@as(u32, 1) << @intCast(format.bits - 1)) - 1;
    return switch (format.kind) {
        .integer => false,
        .exponent => code == 0xFF,
        .native, .float => switch (format.expmode) {
            .none => false,
            .fn_nan => code & magnitude_mask == magnitude_mask,
            .ieee => std.math.isNan(decode(format, code)),
        },
    };
}

// encode: nearest code, ties -> even code, saturate to `max` (non-ieee)
pub fn encode(comptime format: Recipe, value: f32) u32 {
    switch (format.kind) {
        .native => return switch (format.bits) {
            16 => @as(u16, @bitCast(@as(f16, @floatCast(value)))),
            32 => @bitCast(value),
            else => unreachable,
        },
        .integer => {
            const rounded = std.math.round(value) + @as(f32, @floatFromInt(format.bias)); // half away from zero
            const max: f32 = @floatFromInt((@as(u32, 1) << @intCast(format.bits)) - 1);
            return @intFromFloat(std.math.clamp(rounded, 0, max));
        },
        .exponent => return (@as(u32, @bitCast(value)) >> 23) & 0xFF, // floor(log2 |x|) + 127, subnormal -> 0
        .float => switch (format.bits) {
            16 => { // bf16: RTNe on low u16
                const bits: u32 = @bitCast(value);
                const midbit = (bits >> 16) & 1;
                return @truncate((bits +% 0x7FFF +% midbit) >> 16);
            },
            else => {
                // magnitude code order == value order -> binary search on MAGNITUDES
                const MAGNITUDES = comptime magnitudeTable(format);
                const sign_bit: u32 = if (std.math.signbit(value)) @as(u32, 1) << @intCast(format.bits - 1) else 0;
                const target: f64 = @min(@abs(@as(f64, value)), @as(f64, format.max)); // saturate before nearest

                var lo: usize = 0; // first code with MAGNITUDES[code] >= target
                var hi: usize = MAGNITUDES.len;
                while (lo < hi) {
                    const mid = (lo + hi) / 2;
                    switch (MAGNITUDES[mid] < target) {
                        true => lo = mid + 1,
                        false => hi = mid,
                    }
                }
                const code: usize = switch (lo) {
                    0 => 0,
                    else => blk: {
                        const below_dist = target - MAGNITUDES[lo - 1];
                        const above_dist = MAGNITUDES[lo] - target;
                        break :blk switch (std.math.order(below_dist, above_dist)) {
                            .lt => lo - 1,
                            .gt => lo,
                            .eq => switch ((lo - 1) % 2) { // tie -> even code
                                0 => lo - 1,
                                else => lo,
                            },
                        };
                    },
                };
                return sign_bit | @as(u32, @intCast(code));
            },
        },
    }
}

fn magnitudeTable(comptime format: Recipe) [magnitudeCount(format)]f64 {
    @setEvalBranchQuota(100_000);
    var table: [magnitudeCount(format)]f64 = undefined;
    for (&table, 0..) |*magnitude, code| magnitude.* = decode(format, code);
    return table;
}

fn magnitudeCount(comptime format: Recipe) usize {
    const count = @as(usize, 1) << @intCast(format.bits - 1); // positive values
    return switch (format.expmode) {
        .fn_nan => count - 1, // S.1..1.1..1 is nan
        else => count,
    };
}

pub fn readCode(comptime format: Recipe, bytes: []const u8, weight_idx: usize) u32 {
    return switch (format.bits) {
        4 => (bytes[weight_idx / 2] >> @intCast(4 * (weight_idx % 2))) & 0x0F, // element 2k: lo nibble
        8 => bytes[weight_idx],
        16 => std.mem.readInt(u16, bytes[weight_idx * 2 ..][0..2], .little),
        32 => std.mem.readInt(u32, bytes[weight_idx * 4 ..][0..4], .little),
        else => unreachable,
    };
}

pub fn writeCode(comptime format: Recipe, bytes: []u8, weight_idx: usize, code: u32) void {
    switch (format.bits) {
        4 => bytes[weight_idx / 2] |= @intCast(code << @intCast(4 * (weight_idx % 2))), // `bytes` zeroed by caller
        8 => bytes[weight_idx] = @intCast(code),
        16 => std.mem.writeInt(u16, bytes[weight_idx * 2 ..][0..2], @intCast(code), .little),
        32 => std.mem.writeInt(u32, bytes[weight_idx * 4 ..][0..4], code, .little),
        else => unreachable,
    }
}

// -0 == +0: llama.cpp drops the sign of zero
pub fn isCodeEqual(comptime format: Recipe, want: u32, got: u32) bool {
    const magnitude_mask = (@as(u32, 1) << @intCast(format.bits - 1)) - 1;
    return switch (format.kind) {
        .float, .native => want == got or (want & magnitude_mask == 0 and got & magnitude_mask == 0),
        .integer, .exponent => want == got,
    };
}

pub fn canonicalCode(comptime format: Recipe, code: u32) u32 {
    const magnitude_mask = (@as(u32, 1) << @intCast(format.bits - 1)) - 1;
    return switch (format.kind) {
        .float, .native => switch (code & magnitude_mask) { // -0 -> +0
            0 => 0,
            else => code,
        },
        .integer, .exponent => code,
    };
}
