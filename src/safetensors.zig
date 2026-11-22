const std = @import("std");
const types = @import("types.zig");

// LAYOUT:
// [u64 .little header_size][json, header, header_size][binary data]
// "name": { "dtype": "U8", "shape": [..], "data_offsets": [begin, end) } // absolute = 8+header_size+data_offsets
// can contain "__metadata__" header apparently

pub const Layer = struct {
    // offsets are in file-absolut
    name: []const u8,
    dtype: []const u8,
    shape: []const u64,
    weight_offset: u64,
    weight_size: u64,
    scale_offset: u64,
    midpoint_offset: u64, // 0=RTN
    global_scale: f32,
    scheme: ?types.Scheme, // in_t
};

const Header = struct { // json line
    layer: []const u8,
    weight: []const u8,
    scale: []const u8,
    global: ?[]const u8,
    midpoint: ?[]const u8, // RTN: null
    is_inverse: bool, // global scale may be 1/x
};

const HEADERS = [_]Header{ // format to parse
    .{
        .layer = ".weight",
        .weight = ".weight",
        .scale = ".weight_scale",
        .global = ".weight_scale_2",
        .midpoint = null,
        .is_inverse = false,
    },
    .{
        .layer = ".weight",
        .weight = ".weight_packed",
        .scale = ".weight_scale",
        .global = ".weight_global_scale",
        .midpoint = ".weight_zero_point",
        .is_inverse = true,
    },
    .{
        .layer = "",
        .weight = "_blocks",
        .scale = "_scales",
        .global = null,
        .midpoint = null,
        .is_inverse = false,
    },
};

const Tensor = struct {
    dtype: []const u8 = "",
    shape: []const u64 = &.{},
    data_offsets: [2]u64 = .{ 0, 0 },
};
const Tensors = std.StringArrayHashMapUnmanaged(Tensor);

pub inline fn parseJson(in: std.Io.File, io: std.Io, arena: std.mem.Allocator) ![]Layer {
    var buf: [4096]u8 = undefined;
    var reader = in.reader(io, &buf);

    //HEADER fetch
    const header_size = reader.interface.takeInt(u64, .little) catch return error.InvalidHeader;
    const start = 8 + header_size; // u64 + header
    const bytes = try arena.alloc(u8, header_size); // HEADER
    try reader.interface.readSliceAll(bytes);

    //JSON to name
    const tensors = &(try std.json.parseFromSliceLeaky(
        std.json.ArrayHashMap(Tensor),
        arena,
        bytes,
        .{ .ignore_unknown_fields = true }, // skip optional metadata
    )).map;

    // PARSE
    var layers: std.ArrayList(Layer) = try .initCapacity(arena, tensors.count());
    const isParsed = try arena.alloc(bool, tensors.count());
    @memset(isParsed, false);

    // parse quantized
    for (tensors.keys(), tensors.values()) |name, weight| {
        const layer = try toLayer(
            arena,
            tensors,
            isParsed,
            &reader,
            start,
            name,
            weight,
        ) orelse continue;
        layers.appendAssumeCapacity(layer);
    }

    // parse unnamed
    for (tensors.keys(), tensors.values(), isParsed) |name, tensor, used| {
        if (used or std.mem.eql(u8, name, "__metadata__")) continue;
        layers.appendAssumeCapacity(.{
            .name = name,
            .dtype = tensor.dtype,
            .shape = tensor.shape,
            .weight_offset = start + tensor.data_offsets[0],
            .weight_size = tensor.data_offsets[1] - tensor.data_offsets[0],
            .scale_offset = 0,
            .midpoint_offset = 0,
            .global_scale = 1.0,
            .scheme = if (types.Dtype.NameToDtype(
                tensor.dtype,
            )) |plain_t| types.Scheme.DtypeToScheme(
                plain_t,
            ) else null, // "I64" | "BOOL" | "U8" -> null
        });
    }

    return layers.items;
}

fn toLayer(
    arena: std.mem.Allocator,
    tensors: *const Tensors,
    isParsed: []bool,
    reader: *std.Io.File.Reader,
    start: u64,
    name: []const u8,
    weight: Tensor,
) !?Layer {
    const header: Header = blk: {
        for (HEADERS) |cand| { //shadow header
            if (std.mem.endsWith(u8, name, cand.weight)) break :blk cand;
        }
        return null;
    };
    const prefix = name[0 .. name.len - header.weight.len];
    var name_buf: [512]u8 = undefined;

    // SCALE to SCHEME
    const scale_name = try std.fmt.bufPrint(&name_buf, "{s}{s}", .{ prefix, header.scale });
    const scale_idx = tensors.getIndex(scale_name) orelse return null;
    const scale = tensors.values()[scale_idx];
    const scale_t = types.Dtype.NameToDtype(scale.dtype) orelse return error.UnknownScheme;
    const weight_size = weight.data_offsets[1] - weight.data_offsets[0];
    const scale_size = scale.data_offsets[1] - scale.data_offsets[0];
    const isScalar = scale_size == types.Dtype.RECIPE.get(scale_t).bits / 8;
    const scheme = if (isScalar) //TODO
        types.Scheme.DtypeToScheme(
            types.Dtype.NameToDtype(weight.dtype) orelse return error.UnknownScheme,
        ) orelse return error.UnknownScheme
    else
        types.Scheme.ScaleToScheme(scale_t) orelse return error.UnknownScheme;

    // SCALE LAYOUT
    if (!isScalar) {
        if (weight_size % scheme.BlockSize() != 0) return error.ScaleLayout;
        if (weight_size / scheme.BlockSize() * scheme.ScaleSize() != scale_size) return error.ScaleLayout;
    }

    // GLOBAL SCALE
    var global_idx: ?usize = if (isScalar) scale_idx else null;
    if (header.global) |global_suffix| if (types.Scheme.RECIPE.get(
        scheme,
    ).global_scale and !isScalar) {
        const global_name = try std.fmt.bufPrint(&name_buf, "{s}{s}", .{ prefix, global_suffix });
        global_idx = tensors.getIndex(global_name) orelse return error.MissingGLobalScale;
    };
    var global_scale: f32 = 1.0; // no scale
    if (global_idx) |idx| {
        try reader.seekTo(start + tensors.values()[idx].data_offsets[0]);
        global_scale = @bitCast(try reader.interface.takeInt(u32, .little));
        if (header.is_inverse and !isScalar) global_scale = 1.0 / global_scale; // turn back on
        isParsed[idx] = true;
    }

    // MIDPOINT
    if (header.midpoint) |midpoint_suffix| {
        const midpoint_name = try std.fmt.bufPrint(
            &name_buf,
            "{s}{s}",
            .{ prefix, midpoint_suffix },
        );
        if (tensors.getIndex(midpoint_name) != null) return error.MidpointLayout; // I32 packed along rows not handled
    }
    const midpoint_offset: u64 = 0; // 0=RTN

    // SHAPE
    if (weight.shape.len == 0) return error.ScaleLayout;
    const shape = try arena.dupe(u64, weight.shape);
    const weight_bits = types.Dtype.RECIPE.get(types.Scheme.RECIPE.get(scheme).weight).bits;
    var row_count: u64 = 1;
    for (shape[0 .. shape.len - 1]) |dim| row_count *= dim;
    shape[shape.len - 1] = weight_size * 8 / weight_bits / row_count; // agnostic storage : U8 | I32 | F8_E4M3

    isParsed[scale_idx] = true;
    isParsed[tensors.getIndex(name).?] = true;

    return .{
        .name = try std.fmt.allocPrint(arena, "{s}{s}", .{ prefix, header.layer }),
        .dtype = weight.dtype,
        .shape = shape,
        .weight_offset = start + weight.data_offsets[0],
        .weight_size = weight_size,
        .scale_offset = start + scale.data_offsets[0],
        .midpoint_offset = midpoint_offset,
        .global_scale = global_scale,
        .scheme = scheme,
    };
}
