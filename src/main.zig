const std = @import("std");
const types = @import("types.zig");
const safetensors = @import("safetensors.zig");
const quant = @import("quant.zig");
const kernels = @import("kernels.zig");

const CACHE_LINE = std.atomic.cache_line;
const CHUNK_SIZE = 0x40000; // 256KiB, max layer slice
const SCALES_SIZE = CHUNK_SIZE / 8; // smallest scale block: nvfp4 16 elements -> 8 B -> 1 scale

const MAX_WEIGHTS_PER_BYTE = blk: {
    var max_count: usize = 0;
    for (std.enums.values(types.Scheme)) |scheme| {
        const weight_t = types.Scheme.RECIPE.get(scheme).weight;
        max_count = @max(max_count, 8 / types.Dtype.RECIPE.get(weight_t).bits);
    }
    break :blk max_count;
};

const MAX_OUT_VALUE_SIZE = blk: {
    var max_size: usize = 0;
    for (std.enums.values(types.Dtype)) |dtype| {
        const recipe = types.Dtype.RECIPE.get(dtype);
        max_size = @max(max_size, recipe.bits / 8);
    }
    break :blk max_size;
};
const OUT_SIZE = CHUNK_SIZE * MAX_WEIGHTS_PER_BYTE * MAX_OUT_VALUE_SIZE;

fn worker(shared: *Shared) !void { //TODO rewrite
    var in_w_buf: [CHUNK_SIZE]u8 align(CACHE_LINE) = undefined;
    var in_s_buf: [SCALES_SIZE]u8 align(CACHE_LINE) = undefined;
    var out_w_buf: [OUT_SIZE]u8 align(CACHE_LINE) = undefined;
    var out_s_buf: [SCALES_SIZE]u8 align(CACHE_LINE) = undefined;

    var weight_reader = shared.in.reader(shared.io, &in_w_buf);
    var scales_reader = shared.in.reader(shared.io, &in_s_buf);
    const out_bits: usize = types.Dtype.RECIPE.get(
        types.Scheme.RECIPE.get(shared.out_scheme).weight,
    ).bits; // single weight bits size

    while (true) {
        const layer_idx = shared.next.fetchAdd(1, .monotonic);
        if (layer_idx >= shared.layers.len) return; // done

        const layer = shared.layers[layer_idx];
        const check_scheme = if (layer.scheme == shared.out_scheme) null else layer.scheme;
        try weight_reader.seekTo(layer.weight_offset);
        if (check_scheme != null) try scales_reader.seekTo(layer.scale_offset);

        var rem_size = layer.weight_size;
        while (rem_size > 0) {
            const chunk_size: usize = @intCast(@min(CHUNK_SIZE, rem_size)); //TODO: name
            rem_size -= chunk_size;
            const weights = try weight_reader.interface.take(chunk_size);

            const scheme = check_scheme orelse {
                std.mem.doNotOptimizeAway(weights);
                continue;
            };

            // QUANTIZATION
            const weight_bits: usize = types.Dtype.RECIPE.get(
                types.Scheme.RECIPE.get(scheme).weight,
            ).bits;
            const weight_count = chunk_size * 8 / weight_bits; // weights, scheme-independent
            const scales = try scales_reader.interface.take(
                types.Scheme.ScalesBytes(scheme, weight_count),
            );
            const out_w = out_w_buf[0 .. weight_count * out_bits / 8]; // bytes
            const out_s = out_s_buf[0..types.Scheme.ScalesBytes(shared.out_scheme, weight_count)];

            quant.quant(scheme, shared.out_scheme, .{
                .weights = weights,
                .scales = scales,
                .midpoint = &.{},
                .global_scale = layer.global_scale,
            }, out_w, out_s);

            std.mem.doNotOptimizeAway(out_w); //endpoint
            std.mem.doNotOptimizeAway(out_s); //endpoint
        }
    }
}

const Args = struct {
    path: []const []const u8,
    out_scheme: types.Scheme,
    threads: usize,
};

inline fn flags(gpa: std.mem.Allocator, args: std.process.Args.Iterator) !Args {
    var it = args;
    _ = it.skip();

    var rest: std.ArrayList([]const u8) = .empty;
    defer rest.deinit(gpa);

    while (it.next()) |arg| try rest.append(gpa, arg);
    if (rest.items.len < 3) return error.MissingArguments;

    // SCHEME
    const out_scheme =
        std.meta.stringToEnum(types.Scheme, rest.items[rest.items.len - 2]) orelse
        return error.UnknownOutputScheme;

    // THREADS
    const threads =
        std.fmt.parseInt(usize, rest.items[rest.items.len - 1], 10) catch
            return error.InvalidThreadCount;

    if (threads == 0 or threads > 12) return error.InvalidThreadCount;

    return Args{
        .path = try gpa.dupe([]const u8, rest.items[0 .. rest.items.len - 2]),
        .out_scheme = out_scheme,
        .threads = threads,
    };
}

const Shared = struct {
    next: std.atomic.Value(usize) align(CACHE_LINE) = .init(0), // done/in-progress
    io: std.Io,
    layers: []const safetensors.Layer,
    in: std.Io.File,
    out_scheme: types.Scheme,
};

pub fn main(init: std.process.Init) !void {
    const smp = std.heap.smp_allocator;
    var arena = std.heap.ArenaAllocator.init(smp);
    defer arena.deinit();

    // ARGS
    const args = try flags(init.gpa, init.minimal.args.iterate());
    defer init.gpa.free(args.path);
    const threads: []std.Thread = try init.gpa.alloc(std.Thread, args.threads);
    defer init.gpa.free(threads);

    for (args.path) |path| {
        //FILE
        const in = try std.Io.Dir.cwd().openFile(init.io, path, .{});
        defer in.close(init.io);

        //HEADER MAPING
        const layers = try safetensors.parseJson(
            in,
            init.io,
            arena.allocator(),
        );

        //THREADPOOL
        var shared: Shared = .{
            .io = init.io,
            .layers = layers,
            .in = in,
            .out_scheme = args.out_scheme,
        };
        for (threads) |*thread| thread.* = try std.Thread.spawn(.{}, worker, .{&shared});
        for (threads) |thread| thread.join();
    }
}

// -------------------------------------------------------------------------------------
// TESTS

test "quant: nvfp4" {
    const weights: [16]u8 = @bitCast(
        @as([2][8]u8, @splat(.{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE })),
    );
    const scales = [_]u8{ 0x38, 0x30 }; // e4m3: 1.0, 0.5
    const in: quant.Input = .{
        .weights = &weights,
        .scales = &scales,
        .midpoint = &.{},
        .global_scale = 2.0,
    };

    const want = [32]f32{
        0, 1, 2, 3, 4, 6, 8, 12, -0.0, -1, -2, -3, -4, -6, -8, -12, // e2m1 * 1.0 * 2.0
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6, // e2m1 * 0.5 * 2.0
    };

    var out_f32: [32]f32 = undefined;
    quant.quant(.nvfp4, .f32, in, @ptrCast(&out_f32), &.{});
    for (want, out_f32) |w, got| try std.testing.expectEqual(
        @as(u32, @bitCast(w)),
        @as(u32, @bitCast(got)),
    ); // bits: -0.0 != 0.0

    var out_f16: [32]f16 = undefined;
    quant.quant(.nvfp4, .f16, in, @ptrCast(&out_f16), &.{});
    for (want, out_f16) |w, got| try std.testing.expectEqual(
        @as(u16, @bitCast(@as(f16, @floatCast(w)))),
        @as(u16, @bitCast(got)),
    );

    var out_bf16: [32]u16 = undefined;
    quant.quant(.nvfp4, .bf16, in, @ptrCast(&out_bf16), &.{});
    for (want, out_bf16) |w, got| try std.testing.expectEqual(
        @as(u16, @truncate(@as(u32, @bitCast(w)) >> 16)),
        got,
    ); // exact values: bf16 == high u16
}

test "quant: mxfp4" {
    const weights: [32]u8 = @bitCast(
        @as([4][8]u8, @splat(.{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE })),
    );
    const scales = [_]u8{ 0x80, 0x7F }; // e8m0: 2^1, 2^0
    const in: quant.Input = .{
        .weights = &weights,
        .scales = &scales,
        .midpoint = &.{},
        .global_scale = 1.0,
    };

    const x2 = [16]f32{ 0, 1, 2, 3, 4, 6, 8, 12, -0.0, -1, -2, -3, -4, -6, -8, -12 };
    const x1 = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
    const want = x2 ++ x2 ++ x1 ++ x1;

    var out_f32: [64]f32 = undefined;
    quant.quant(.mxfp4, .f32, in, @ptrCast(&out_f32), &.{});
    for (want, out_f32) |w, got| try std.testing.expectEqual(
        @as(u32, @bitCast(w)),
        @as(u32, @bitCast(got)),
    );

    var out_f16: [64]f16 = undefined;
    quant.quant(.mxfp4, .f16, in, @ptrCast(&out_f16), &.{});
    for (want, out_f16) |w, got| try std.testing.expectEqual(
        @as(u16, @bitCast(@as(f16, @floatCast(w)))),
        @as(u16, @bitCast(got)),
    );

    var out_bf16: [64]u16 = undefined;
    quant.quant(.mxfp4, .bf16, in, @ptrCast(&out_bf16), &.{});
    for (want, out_bf16) |w, got| try std.testing.expectEqual(
        @as(u16, @truncate(@as(u32, @bitCast(w)) >> 16)),
        got,
    );
}

// PARSE JSON
test "safetensors: parseJson" {
    const io = std.testing.io;
    const json =
        \\{"__metadata__":{"format":"pt"},
        \\"fc.weight":{"dtype":"U8","shape":[2,8],"data_offsets":[0,16]},
        \\"fc.weight_scale":{"dtype":"F8_E4M3","shape":[2,1],"data_offsets":[16,18]},
        \\"fc.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[18,22]},
        \\"fc.bias":{"dtype":"F32","shape":[2],"data_offsets":[22,30]}}
    ;
    const data = @as([16]u8, @splat(0x7A)) // fc.weight
        ++ [_]u8{ 0x38, 0x30 } // fc.weight_scale
        ++ [_]u8{ 0x00, 0x00, 0x00, 0x40 } // fc.weight_scale_2 = 2.0
        ++ @as([8]u8, @splat(0)); // fc.bias
    const size: [8]u8 = @bitCast(std.mem.nativeToLittle(u64, json.len));

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.createFile(io, "model.safetensors", .{ .read = true });
    defer path.close(io);
    try path.writePositionalAll(io, &size ++ json ++ &data, 0);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const layers = try safetensors.parseJson(path, io, arena.allocator());
    const start = 8 + json.len;

    try std.testing.expectEqual(
        2,
        layers.len,
    ); // scale + global scale are merged in, __metadata__ skipped

    const fc = layers[0];
    try std.testing.expectEqualStrings("fc.weight", fc.name);
    try std.testing.expectEqual(.nvfp4, fc.scheme.?);
    try std.testing.expectEqualSlices(u64, &.{ 2, 16 }, fc.shape); // 2 weights per packed byte
    try std.testing.expectEqual(start + 0, fc.weight_offset);
    try std.testing.expectEqual(16, fc.weight_size);
    try std.testing.expectEqual(start + 16, fc.scale_offset);
    try std.testing.expectEqual(2.0, fc.global_scale);

    const bias = layers[1];
    try std.testing.expectEqualStrings("fc.bias", bias.name);
    try std.testing.expectEqual(.f32, bias.scheme.?);
    try std.testing.expectEqualSlices(u64, &.{2}, bias.shape);
    try std.testing.expectEqual(start + 22, bias.weight_offset);
    try std.testing.expectEqual(8, bias.weight_size);
}
