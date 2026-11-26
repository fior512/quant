const std = @import("std");

// NVFP4 per rows
//  packed: cols/2 bytes, byte = [+nibble= 2k+1] [-nibble= 2k]
//  scales: cols/16 bytes, one E4M3 scale per block of 16 elements (= 8 packed bytes)
//  value= E2M1(nibble) * E4M3(block_scale) * global_scale
//
//
//  safetensors:
//  obj                 dtype       shape
//  "X.weights"         u8          [2,16]  // 32 FP4 values per row/2 per byte = 16
//  "X.weights_scale"   F8_E4M3     [2,2]   // 32 values per row/16 per block = 2
//  "X.weights_scale_2" F32         []      // global_scale
//
// packed bytes (flat, row-major):
//   byte:      0..7        8..15           16..23          24..31
//   row:       0           0               1               1
//   cols:      0..15       16..31          0..15           16..31
//   scale: scales[0]       scales[1]       scales[2]       scales[3]
//
//   high nibble is n+1



//IDEA:
//  bytes, scale, zero
//  
//i4 base -> 2/byte
//  hi = v >> @as(@Vector(size/2, u3), @splat(4)); // 0x0H, 0x0H, 0x0H
//  lo = v &  @as(@Vector(size/2, u3), @splat(0x0F)); // 0x0L, 0x0L, 0x0L
//  q = shuffle(u8, lo, hi, i%2); // 0x0L, 0x0H, 0x0L, 0x0H (L first because little endian)
//
//1) std affine s(q-z)
//  bias = -@as(f32, @floatfromint(z)) * s      // negative because we goes from s(q-z) to qs-zs
//  return FMA(type_o, @FloatFromInt(q), @splat(s), @splat(b))
//
//2) RTN s*q
//  signed = q-8  // [0..15] -> [-8..7]
//  return @as(type_o, @FloatFromInt(sign) * @Splat(s))
//
//
//swap @FloatFromInt for decode E_M_

pub const Dtype = enum { 
    f4e2m1,
    i4,
    f8e4m3,
    f8e8m0,
    f16,
    bf16,
    f32,

    pub const Kind = enum(u8) {
        float,      // sign + exp + mant; decoded through f16
        exponent,   // e8m0; decoded via shift in f32
        native,     // f16|bf16|f32, widen|shift|bitcast
        integer,    // offset value - bias
    };

    pub const ExpMode = enum(u8) {
        ieee,       // exp all ones -> inf|nan; overflow inf
        fn_nan,     // S.FFFF.. -> nan; overflow max
        none,       // finite; overflow max
    };

    pub const Recipe = struct {
        header: ?[]const u8, // "dtype" given as str in header
        max: f32,
        bits: u8,
        exp_bits: u8,
        mant_bits: u8,
        bias: u8,
        kind: Kind, // type of logic
        expmode: ExpMode,
    };

    pub const RECIPE = std.EnumArray(Dtype, Recipe).init(.{
        .f4e2m1 = .{ .header = null, .max = 6.0, .bits = 4, .exp_bits = 2, .mant_bits = 1, .bias = 1, .kind = .float, .expmode = .none}, //"U8"
        .i4     = .{ .header = null, .max = 7.0, .bits = 4, .exp_bits = 0, .mant_bits = 0, .bias = 8, .kind = .integer, .expmode = .none },
        .f8e4m3 = .{ .header = "F8_E4M3", .max = 448.0, .bits = 8, .exp_bits = 4, .mant_bits = 3, .bias = 7, .kind = .float, .expmode = .fn_nan },
        .f8e8m0 = .{ .header = "U8", .max = 0x1p127, .bits = 8, .exp_bits = 8, .mant_bits = 0, .bias = 127, .kind = .exponent, .expmode = .fn_nan },
        .f16    = .{ .header = "F16", .max = 65504.0, .bits = 16, .exp_bits = 5, .mant_bits = 10, .bias = 15, .kind = .native, .expmode = .ieee },
        .bf16   = .{ .header = "BF16", .max = 0x1.FEp127, .bits = 16, .exp_bits = 8, .mant_bits = 7, .bias = 127, .kind = .float, .expmode = .ieee },
        .f32    = .{ .header = "F32", .max = 0x1.FFFFFEp127, .bits = 32, .exp_bits = 8, .mant_bits = 23, .bias = 127, .kind = .native, .expmode = .ieee },
    });

    pub fn NameToDtype(name: []const u8) ?Dtype {
        for (std.enums.values(Dtype)) |dtype| {
            const header = RECIPE.get(dtype).header orelse continue;
            if (std.mem.eql(u8, header, name)) return dtype;
        }
        return null;
    }

};



pub const Scheme = enum {
    nvfp4,
    mxfp4,
    i4,
    f8e4m3,
    f16,
    bf16,
    f32,

    pub const Translation = enum {
        affine, // s(q-z)
        RTN,    // sq
    };

    pub const Recipe = struct {
        absmax: f32,
        block: u32,
        weight: Dtype,
        scale: ?Dtype,
        global_scale: bool,
    };
    
    pub const RECIPE = std.EnumArray(Scheme, Recipe).init(.{
        .nvfp4  = .{ .weight = .f4e2m1, .scale = .f8e4m3, .block = 16, .global_scale = true, .absmax = 6.0 },
        .mxfp4  = .{ .weight = .f4e2m1, .scale = .f8e8m0, .block = 32, .global_scale = false, .absmax = 4.0 },
        .i4     = .{ .weight = .i4, .scale = .bf16, .block = 128, .global_scale = false, .absmax = 7.0 },
        .f8e4m3 = .{ .weight = .f8e4m3, .scale = null, .block = 32, .global_scale = true, .absmax = 448.0 },
        .f16    = .{ .weight = .f16, .scale = null, .block = 32, .global_scale = false, .absmax = 0.0 },
        .bf16   = .{ .weight = .bf16, .scale = null, .block = 32, .global_scale = false, .absmax = 0.0 },
        .f32    = .{ .weight = .f32, .scale = null, .block = 32, .global_scale = false, .absmax = 0.0 },
    });

    pub fn ScaleToScheme(scale: Dtype) ?Scheme {
        for (std.enums.values(Scheme)) |scheme| {
            if (RECIPE.get(scheme).scale == scale) return scheme;
        }
        return null;
    }

    pub fn DtypeToScheme(dtype: Dtype) ?Scheme {
        for (std.enums.values(Scheme)) |scheme| {
            const recipe = RECIPE.get(scheme);
            if (recipe.scale == null and recipe.weight == dtype) return scheme;
        }
        return null;
    }

    pub fn Block(scheme: Scheme) usize {
        return RECIPE.get(scheme).block;
    }

    pub fn WeightsSize(scheme: Scheme, weight_count: usize) usize {
        const bits = Dtype.RECIPE.get(RECIPE.get(scheme).weight).bits;
        std.debug.assert(weight_count * bits % 8 == 0);
        return weight_count * bits / 8;
    }

    pub fn ScalesPerBlock(scheme: Scheme, weight_count: usize) usize {
        std.debug.assert(weight_count % scheme.Block() == 0);
        return weight_count / scheme.Block();
    }

    pub fn BlockSize(scheme: Scheme) usize {
        return scheme.WeightsSize(scheme.Block());
    }

    pub fn ScaleSize(scheme: Scheme) usize {
        const scale_t = RECIPE.get(scheme).scale orelse return 0;
        return Dtype.RECIPE.get(scale_t).bits / 8;
    }

    pub fn ScalesBytes(scheme: Scheme, weight_count: usize) usize {
        return weight_count / RECIPE.get(scheme).block * ScaleSize(scheme);
    }
};












