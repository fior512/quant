const std = @import("std");
const dequant = @import("dequant");
const harness = @import("harness.zig");
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

fn DequantKernel(comptime scheme: generic.Scheme, comptime out_scheme: generic.Scheme) type {
    return struct {
        pub const SCHEME: generic.Scheme = scheme;
        pub const OUTPUT: generic.Scheme = out_scheme;

        input: dequant.Input,

        pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !@This() {
            const weights = try allocator.alignedAlloc(u8, ALIGN, logical.weights.len);
            @memcpy(weights, logical.weights);
            const scales = try allocator.alignedAlloc(u8, ALIGN, logical.scales.len);
            @memcpy(scales, logical.scales);
            return .{
                .input = .{
                    .weights = weights,
                    .scales = scales,
                    .midpoint = &.{},
                    .global_scale = 1.0,
                },
            };
        }

        pub noinline fn Run(kernel: *const @This(), out: []u8, out_s: []u8) void {
            dequant.dequant(scheme, out_scheme, kernel.input, out, out_s);
        }
    };
}

pub fn main(init: std.process.Init) !void {
    const args = harness.parseArgs(init.minimal.args.iterate());
    switch (args.scheme) {
        inline else => |c_scheme| switch (args.out_scheme) {
            inline else => |c_out_scheme| try harness.run(DequantKernel(c_scheme, c_out_scheme), "dequant  ", args.size_filter),
        },
    }
}
