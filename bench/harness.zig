const std = @import("std");
const linux = std.os.linux;
const generic = @import("generic.zig");
const reference = @import("reference.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

pub const Logical = struct {
    weights: []u8,          // weight_count values of `scheme.weight`, packed like the file (4-bit: element 2k in lo nibble)
    scales: []u8,           // weight_count / `scheme.block` values of `scheme.scale`; empty for plain schemes
    expected: []u8,         // weight_count values of `out_scheme.weight`: decode * scale -> (out scale) -> encode
    expected_scales: []u8,  // weight_count / `out_scheme.block` values of `out_scheme.scale`; empty for plain schemes
    weight_count: usize,
};

const Trial = struct {
    rep_count: u64,
    tsc_ticks: u64,
    wall_ns: u64,
};

pub const Args = struct {
    size_filter: ?[]const u8,
    scheme: generic.Scheme,
    out_scheme: generic.Scheme,
};

pub fn parseArgs(args: std.process.Args.Iterator) Args { // {scheme} [out_scheme (default f32)] [size]
    var it = args;
    _ = it.skip();
    const scheme_name = it.next() orelse @panic("Args: missing scheme (nvfp4|mxfp4|i4|f8e4m3|f16|bf16|f32)");
    const scheme = std.meta.stringToEnum(generic.Scheme, scheme_name) orelse @panic("Args: unknown scheme");

    var out_scheme: generic.Scheme = .f32;
    var size_filter = it.next();
    if (size_filter) |name| if (std.meta.stringToEnum(generic.Scheme, name)) |out_name| {
        out_scheme = out_name;
        size_filter = it.next();
    };
    return .{ .size_filter = size_filter, .scheme = scheme, .out_scheme = out_scheme };
}

pub fn run(comptime Kernel: type, comptime name: []const u8, size_filter: ?[]const u8) !void { // harness module's main
    // Pipeline: [generateLogical() -> Prepare() -> VERIFY -> runTrial() * TRIAL_COUNT -> SUMMARY] * SIZES
    const out_recipe = comptime generic.Scheme.RECIPE.get(Kernel.OUTPUT);
    const out_t = comptime generic.Dtype.RECIPE.get(out_recipe.weight);
    pinCpu(generic.PIN_CPU);
    const tsc_hz = calibrateTscHz();
    std.debug.print("{s}: {t} -> {t}, cpu {d}, tsc {d:.4} GHz, trials {d} x >= {d} s, sleep {d} s\n", .{
        name,
        Kernel.SCHEME,
        Kernel.OUTPUT,
        generic.PIN_CPU,
        tsc_hz / 1e9,
        generic.TRIAL_COUNT,
        generic.TRIAL_MIN_NS / generic.NS_PER_S,
        generic.TRIAL_SLEEP_NS / generic.NS_PER_S,
    });

    var trial_done_count: u32 = 0;
    for (generic.SIZES) |size| {
        if (size_filter) |filter| if (!std.mem.eql(u8, std.mem.trim(u8, size.label, " "), std.mem.trim(u8, filter, " "))) continue;
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const logical = try generateLogical(Kernel.SCHEME, Kernel.OUTPUT, allocator, size.weight_count);
        const kernel = try Kernel.Prepare(allocator, &logical);
        const out = try allocator.alignedAlloc(u8, ALIGN, size.weight_count * out_t.bits / 8);
        const out_s: []u8 = if (logical.expected_scales.len > 0)
            try allocator.alignedAlloc(u8, ALIGN, logical.expected_scales.len)
        else
            try allocator.alloc(u8, 0);

        // VERIFY + pre fault `out`
        Kernel.Run(&kernel, out, out_s);
        for (0..size.weight_count) |weight_idx| {
            const want = reference.readCode(out_t, logical.expected, weight_idx);
            const got = reference.readCode(out_t, out, weight_idx);
            if (!reference.isCodeEqual(out_t, want, got)) std.debug.panic(
                "{s} mismatch @{d}: got 0x{x} vs want 0x{x}",
                .{ name, weight_idx, got, want },
            );
        }
        if (out_recipe.scale) |c_scale_t| {
            const scale_t = comptime generic.Dtype.RECIPE.get(c_scale_t);
            const scale_count = size.weight_count / out_recipe.block;
            for (0..scale_count) |scale_idx| {
                const want = reference.readCode(scale_t, logical.expected_scales, scale_idx);
                const got = reference.readCode(scale_t, out_s, scale_idx);
                if (!reference.isCodeEqual(scale_t, want, got)) std.debug.panic(
                    "{s} scale mismatch @{d}: got 0x{x} vs want 0x{x}",
                    .{ name, scale_idx, got, want },
                );
            }
        }
        const out_hash = hashOutput(out_t, out, size.weight_count);

        const batch_rep_count: u64 = @max(generic.BATCH_WEIGHT_COUNT / size.weight_count, 1);
        std.debug.print(
            "{s} {s} weights {d} batch_reps {d} out_hash {x:0>16}\n",
            .{ name, size.label, size.weight_count, batch_rep_count, out_hash },
        );

        // TRIALs
        var ns_per_weight: [generic.TRIAL_COUNT]f64 = undefined;
        for (&ns_per_weight, 0..) |*trial_ns_per_weight, trial_idx| {
            if (trial_done_count > 0) sleep(generic.TRIAL_SLEEP_NS);
            trial_done_count += 1;

            const trial = runTrial(Kernel, &kernel, out, out_s, batch_rep_count, tsc_hz);

            // REPORT trial
            const weight_total: f64 = @floatFromInt(trial.rep_count * size.weight_count);
            const ticks: f64 = @floatFromInt(trial.tsc_ticks);
            const tsc_ns = ticks / tsc_hz * 1e9;
            trial_ns_per_weight.* = tsc_ns / weight_total;
            std.debug.print("{s} {s} trial {d}  reps {d:>10}  {d:7.4} ticks/w  {d:7.4} ns/w  {d:7.3} Gw/s  {d:7.2} GiB/s out  wall {d:.3} s\n", .{
                name,
                size.label,
                trial_idx,
                trial.rep_count,
                ticks / weight_total,
                trial_ns_per_weight.*,
                weight_total / tsc_ns,
                weight_total * @as(f64, @floatFromInt(out_t.bits)) / 8 / tsc_ns * 1e9 / (1 << 30),
                @as(f64, @floatFromInt(trial.wall_ns)) / 1e9,
            });
        }

        // SUMMARY
        std.mem.sort(f64, &ns_per_weight, {}, std.sort.asc(f64));
        std.debug.print("{s} {s} summary  min {d:7.4}  median {d:7.4}  max {d:7.4} ns/w\n", .{
            name,
            size.label,
            ns_per_weight[0],
            ns_per_weight[generic.TRIAL_COUNT / 2],
            ns_per_weight[generic.TRIAL_COUNT - 1],
        });
    }
}

fn runTrial(
    comptime Kernel: type,
    kernel: *const Kernel,
    out: []u8,
    out_s: []u8,
    batch_rep_count: u64,
    tsc_hz: f64,
) Trial {
    const deadline_ticks: u64 = @intFromFloat(
        @as(f64, @floatFromInt(generic.TRIAL_MIN_NS)) / 1e9 * tsc_hz,
    );
    var rep_count: u64 = 0;

    const wall_start = monotonicNs();
    const tsc_start = tscStart();
    var tsc_now = tsc_start;
    while (tsc_now - tsc_start < deadline_ticks) {
        for (0..batch_rep_count) |_| {
            Kernel.Run(kernel, out, out_s);
            clobberMemory(out);
            clobberMemory(out_s);
        }
        rep_count += batch_rep_count;
        tsc_now = tscEnd();
    }
    const wall_end = monotonicNs();

    return .{
        .rep_count = rep_count,
        .tsc_ticks = tsc_now - tsc_start,
        .wall_ns = wall_end - wall_start,
    };
}

fn generateLogical(
    comptime scheme: generic.Scheme,
    comptime out_scheme: generic.Scheme,
    allocator: std.mem.Allocator,
    weight_count: usize,
) !Logical {
    // Pipeline: WEIGHTS -> SCALES -> VALUES -> (out scales) -> EXPECTED (same prng order as llama.cpp bench)
    const recipe = comptime generic.Scheme.RECIPE.get(scheme);
    const weight_t = comptime generic.Dtype.RECIPE.get(recipe.weight);
    const out_recipe = comptime generic.Scheme.RECIPE.get(out_scheme);
    const out_t = comptime generic.Dtype.RECIPE.get(out_recipe.weight);
    std.debug.assert(weight_count % recipe.block == 0);
    std.debug.assert(weight_count % out_recipe.block == 0);

    var prng: std.Random.DefaultPrng = .init(generic.SEED);
    const random = prng.random();

    // WEIGHTS: random values, nan|inf -> 0
    const weights = try allocator.alignedAlloc(u8, ALIGN, weight_count * weight_t.bits / 8);
    @memset(weights, 0);
    for (0..weight_count) |weight_idx| {
        const code: u32 = random.int(@Int(.unsigned, weight_t.bits));
        const is_finite = !reference.isNanCode(weight_t, code) and !std.math.isInf(reference.decode(weight_t, code));
        reference.writeCode(weight_t, weights, weight_idx, if (is_finite) code else 0);
    }

    // SCALES: values in `SCALE_RANGE`
    var scales: []u8 = &.{};
    if (recipe.scale) |c_scale_t| {
        const scale_t = comptime generic.Dtype.RECIPE.get(c_scale_t);
        const range = comptime generic.SCALE_RANGE.get(c_scale_t);
        const Code = @Int(.unsigned, scale_t.bits);
        const scale_count = weight_count / recipe.block;
        scales = try allocator.alignedAlloc(u8, ALIGN, scale_count * scale_t.bits / 8);
        for (0..scale_count) |scale_idx| {
            reference.writeCode(scale_t, scales, scale_idx, random.intRangeAtMost(Code, range.min, range.max));
        }
    }

    // EXPECTED: encode to out_scheme, with out scales when quantized (global_scale 1.0)
    const expected = try allocator.alignedAlloc(u8, ALIGN, weight_count * out_t.bits / 8);
    @memset(expected, 0);
    var expected_scales: []u8 = &.{};
    if (out_recipe.scale) |c_scale_t| {
        const scale_t = comptime generic.Dtype.RECIPE.get(c_scale_t);
        const scale_count = weight_count / out_recipe.block;
        expected_scales = try allocator.alignedAlloc(u8, ALIGN, scale_count * scale_t.bits / 8);
        @memset(expected_scales, 0);

        const values = try allocator.alloc(f32, weight_count);
        for (0..weight_count) |weight_idx| {
            const value = reference.decode(weight_t, reference.readCode(weight_t, weights, weight_idx));
            const scale: f32 = if (recipe.scale) |c_in_scale_t| blk: {
                const in_scale_t = comptime generic.Dtype.RECIPE.get(c_in_scale_t);
                break :blk reference.decode(in_scale_t, reference.readCode(in_scale_t, scales, weight_idx / recipe.block));
            } else 1.0;
            values[weight_idx] = value * scale;
        }
        for (0..scale_count) |block_idx| {
            var absmax: f32 = 0;
            for (0..out_recipe.block) |i| absmax = @max(absmax, @abs(values[block_idx * out_recipe.block + i]));
            const scale_code = reference.encode(scale_t, absmax / out_recipe.absmax);
            reference.writeCode(scale_t, expected_scales, block_idx, scale_code);
            const scale_value = reference.decode(scale_t, scale_code);
            const inv_scale: f32 = if (scale_value == 0) 0 else 1.0 / scale_value;
            for (0..out_recipe.block) |i| {
                const weight_idx = block_idx * out_recipe.block + i;
                reference.writeCode(out_t, expected, weight_idx, reference.encode(out_t, values[weight_idx] * inv_scale));
            }
        }
    } else {
        for (0..weight_count) |weight_idx| {
            const value = reference.decode(weight_t, reference.readCode(weight_t, weights, weight_idx));
            const scale: f32 = if (recipe.scale) |c_scale_t| blk: {
                const scale_t = comptime generic.Dtype.RECIPE.get(c_scale_t);
                break :blk reference.decode(scale_t, reference.readCode(scale_t, scales, weight_idx / recipe.block));
            } else 1.0;
            reference.writeCode(out_t, expected, weight_idx, reference.encode(out_t, value * scale));
        }
    }

    return .{
        .weights = weights,
        .scales = scales,
        .expected = expected,
        .expected_scales = expected_scales,
        .weight_count = weight_count,
    };
}

fn hashOutput(comptime out_t: generic.Dtype.Recipe, out: []const u8, weight_count: usize) u64 {
    var hasher: std.hash.Wyhash = .init(generic.SEED);
    for (0..weight_count) |weight_idx| {
        const canonical = reference.canonicalCode(out_t, reference.readCode(out_t, out, weight_idx)); // -0 -> +0
        hasher.update(std.mem.asBytes(&canonical));
    }
    return hasher.final();
}

fn calibrateTscHz() f64 {
    const wall_start = monotonicNs();
    const tsc_start = tscStart();
    sleep(generic.TSC_CALIBRATION_NS);
    const tsc_end = tscEnd();
    const wall_end = monotonicNs();
    return @as(
        f64,
        @floatFromInt(tsc_end - tsc_start),
    ) * 1e9 / @as(f64, @floatFromInt(wall_end - wall_start));
}

fn pinCpu(cpu: u32) void {
    var cpu_set: linux.cpu_set_t = @splat(0);
    cpu_set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    linux.sched_setaffinity(
        0,
        &cpu_set,
    ) catch |err| std.debug.panic("sched_setaffinity cpu {d}: {t}", .{ cpu, err });
}

fn sleep(duration_ns: u64) void {
    var request: linux.timespec = .{
        .sec = @intCast(duration_ns / generic.NS_PER_S),
        .nsec = @intCast(duration_ns % generic.NS_PER_S),
    };
    var remaining: linux.timespec = undefined;
    while (linux.errno(linux.nanosleep(&request, &remaining)) == .INTR) request = remaining;
}

fn monotonicNs() u64 {
    var timespec: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &timespec);
    return @as(u64, @intCast(timespec.sec)) * generic.NS_PER_S + @as(u64, @intCast(timespec.nsec));
}

// lfence; rdtsc; lfence -> prior insns retired before read, later insns wait for read
inline fn tscStart() u64 {
    var low: u32 = undefined;
    var high: u32 = undefined;
    asm volatile ("lfence\n\trdtsc\n\tlfence"
        : [low] "={eax}" (low),
          [high] "={edx}" (high),
        :
        : .{ .memory = true });
    return (@as(u64, high) << 32) | low;
}

// rdtscp; lfence -> rdtscp waits prior insns, lfence blocks later insns
inline fn tscEnd() u64 {
    var low: u32 = undefined;
    var high: u32 = undefined;
    asm volatile ("rdtscp\n\tlfence"
        : [low] "={eax}" (low),
          [high] "={edx}" (high),
        :
        : .{ .ecx = true, .memory = true });
    return (@as(u64, high) << 32) | low;
}

// compiler barrier: `out` escapes + memory clobber -> `Run` stores kept
inline fn clobberMemory(out: []u8) void {
    asm volatile (""
        :
        : [out_ptr] "r" (out.ptr),
        : .{ .memory = true });
}
