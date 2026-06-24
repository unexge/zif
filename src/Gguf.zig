header: Header,
tensor_infos: []TensorInfo,
tensor_data: []u8,

// TODO: Mmap.

pub fn parse(gpa: Allocator, io: Io, path: []const u8) !@This() {
    const file = try Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var buffer: [1024]u8 = undefined;
    var reader = file.reader(io, &buffer);

    const header = try Header.parse(gpa, &reader.interface);
    const tensor_infos = try gpa.alloc(TensorInfo, header.tensor_count);
    for (0..header.tensor_count) |i| {
        tensor_infos[i] = try TensorInfo.parse(gpa, &reader.interface);
    }

    const alignment: u32 = if (header.metadata.get("general.alignment")) |value| blk: {
        switch (value) {
            .uint32 => |v| {
                break :blk v;
            },
            else => return error.InvalidAlignment,
        }
    } else 32;

    const pos = reader.logicalPos();
    const align_off = pos + (alignment - (pos % alignment)) % alignment;

    const pad = align_off - pos;
    if (pad > 0) {
        try reader.seekBy(@intCast(pad));
    }

    const remaining = try reader.getSize() - reader.logicalPos();
    const tensor_data = try reader.interface.readAlloc(gpa, remaining);

    return .{ .header = header, .tensor_infos = tensor_infos, .tensor_data = tensor_data };
}

pub fn deinit(self: *@This(), gpa: Allocator) void {
    self.header.deinit(gpa);
    for (self.tensor_infos) |*ti| {
        ti.deinit(gpa);
    }
    gpa.free(self.tensor_infos);
    gpa.free(self.tensor_data);
    self.* = undefined;
}

pub const Header = struct {
    const MAGIC = "GGUF";

    version: u32,
    tensor_count: u64,
    metadata: StringHashMap(Value),

    pub const Value = union(enum) {
        uint8: u8,
        int8: i8,
        uint16: u16,
        int16: i16,
        uint32: u32,
        int32: i32,
        float32: f32,
        uint64: u64,
        int64: i64,
        float64: f64,
        boolean: bool,
        string: []const u8,
        array: []Value,

        fn parse(gpa: Allocator, reader: *Io.Reader, vtype: u32) !@This() {
            return switch (vtype) {
                0 => .{ .uint8 = try reader.takeByte() },
                1 => .{ .int8 = try reader.takeInt(i8, .little) },
                2 => .{ .uint16 = try reader.takeInt(u16, .little) },
                3 => .{ .int16 = try reader.takeInt(i16, .little) },
                4 => .{ .uint32 = try reader.takeInt(u32, .little) },
                5 => .{ .int32 = try reader.takeInt(i32, .little) },
                6 => .{ .float32 = @bitCast(try reader.takeInt(u32, .little)) },
                7 => blk: {
                    const val = try reader.takeByte();
                    const b = switch (val) {
                        0 => false,
                        1 => true,
                        else => return error.InvalidBoolean,
                    };
                    break :blk .{ .boolean = b };
                },
                8 => .{ .string = try parseString(gpa, reader) },
                9 => blk: {
                    const etype = try reader.takeInt(u32, .little);
                    const len = try reader.takeInt(u64, .little);
                    const values = try gpa.alloc(Value, len);
                    for (0..len) |i| {
                        values[i] = try Value.parse(gpa, reader, etype);
                    }
                    break :blk .{ .array = values };
                },
                10 => .{ .uint64 = try reader.takeInt(u64, .little) },
                11 => .{ .int64 = try reader.takeInt(i64, .little) },
                12 => .{ .float64 = @bitCast(try reader.takeInt(u64, .little)) },
                else => return error.UnknownValueType,
            };
        }

        fn deinit(self: *@This(), gpa: Allocator) void {
            switch (self.*) {
                .string => |*str| {
                    gpa.free(str.*);
                },
                .array => |*arr| {
                    for (arr.*) |*val| {
                        val.deinit(gpa);
                    }
                    gpa.free(arr.*);
                },
                else => {},
            }
            self.* = undefined;
        }
    };

    fn parse(gpa: Allocator, reader: *Io.Reader) !@This() {
        var header: [24]u8 = undefined;
        try reader.readSliceAll(&header);

        if (!mem.eql(u8, header[0..4], MAGIC)) {
            return error.NotGguf;
        }

        const version: u32 = @bitCast(header[4..8].*);
        if (version != 3) {
            return error.UnsupportedVersion;
        }

        const metadata_kv_count: u64 = @bitCast(header[16..24].*);
        var metadata: StringHashMap(Value) = .init(gpa);
        try metadata.ensureTotalCapacity(@intCast(metadata_kv_count));
        for (0..metadata_kv_count) |_| {
            const key = try parseString(gpa, reader);
            const vtype = try reader.takeInt(u32, .little);
            const value = try Value.parse(gpa, reader, vtype);
            metadata.putAssumeCapacity(key, value);
        }

        return .{
            .version = version,
            .tensor_count = @bitCast(header[8..16].*),
            .metadata = metadata,
        };
    }

    pub fn getMetadataKey(self: *@This(), comptime T: type, key: []const u8) !?T {
        const val = self.metadata.get(key) orelse return null;
        switch (T) {
            u32 => {
                switch (val) {
                    .uint32 => |*num| {
                        return num.*;
                    },
                    else => return error.ExpectedUint32MetadataValue,
                }
            },
            []const u8 => {
                switch (val) {
                    .string => |*str| {
                        return str.*;
                    },
                    else => return error.ExpectedStringMetadataValue,
                }
            },
            []Value => {
                switch (val) {
                    .array => |*arr| {
                        return arr.*;
                    },
                    else => return error.ExpectedArrayMetadataValue,
                }
            },
            else => @compileError("Unsupported type " + T),
        }
    }

    fn deinit(self: *@This(), gpa: Allocator) void {
        var iter = self.metadata.iterator();
        while (iter.next()) |*entry| {
            gpa.free(entry.key_ptr.*);
            entry.value_ptr.deinit(gpa);
        }
        self.metadata.deinit();
        self.* = undefined;
    }
};

pub const TensorType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, support has been removed
    // Q4_3 = 5, support has been removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // Q4_0_4_4 = 31, support has been removed from gguf files
    // Q4_0_4_8 = 32,
    // Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    // IQ4_NL_4_4 = 36,
    // IQ4_NL_4_8 = 37,
    // IQ4_NL_8_8 = 38,
    MXFP4 = 39, // MXFP4 (1 block)
    COUNT = 40,
};

pub const TensorInfo = struct {
    name: []const u8,
    dimensions: []u64,
    tensor_type: TensorType,
    offset: u64,

    fn parse(gpa: Allocator, reader: *Io.Reader) !@This() {
        const name = try parseString(gpa, reader);
        if (name.len > 64) return error.InvalidTensorName;

        const n_dimensions = try reader.takeInt(u32, .little);
        if (n_dimensions > 4) return error.InvalidTensorDimensions;
        const dimensions = try gpa.alloc(u64, n_dimensions);
        for (0..n_dimensions) |i| {
            dimensions[i] = try reader.takeInt(u64, .little);
        }

        const tensor_type = enums.fromInt(TensorType, try reader.takeInt(u32, .little)) orelse return error.InvalidTensorType;
        const offset = try reader.takeInt(u64, .little);

        return .{
            .name = name,
            .dimensions = dimensions,
            .tensor_type = tensor_type,
            .offset = offset,
        };
    }

    fn deinit(self: *@This(), gpa: Allocator) void {
        gpa.free(self.name);
        gpa.free(self.dimensions);
        self.* = undefined;
    }
};

fn parseString(gpa: Allocator, reader: *Io.Reader) ![]const u8 {
    const len = try reader.takeInt(u64, .little);
    const buf = try gpa.alloc(u8, len);
    try reader.readSliceAll(buf);
    return buf;
}

const std = @import("std");
const mem = std.mem;
const enums = std.enums;
const Allocator = mem.Allocator;
const StringHashMap = std.StringHashMap;
const Io = std.Io;
