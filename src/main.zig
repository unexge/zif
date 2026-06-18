pub fn main(init: Init) !void {
    var args = init.minimal.args.iterate();
    _ = args.next() orelse return error.MissingProgname;

    const subcommand = args.next() orelse return error.MissingSubcommand;

    if (mem.eql(u8, subcommand, "dump")) {
        const filename = args.next() orelse return error.MissingFilename;
        try dump(&init, filename);
    } else {
        return error.UnknownSubcommand;
    }
}

fn dump(init: *const Init, filename: []const u8) !void {
    var file = try Gguf.parse(init.gpa, init.io, filename);
    defer file.deinit(init.gpa);

    std.debug.print("Header ========\n", .{});
    for (file.header.metadata) |kv| {
        switch (kv.value) {
            .array => |*arr| {
                std.debug.print("{s} => array with {d} elements\n", .{ kv.key, arr.len });
                for (0..5) |j| {
                    std.debug.print("\t{any}\n", .{arr.*[j]});
                }
            },
            .string => |*str| {
                std.debug.print("{s} => {s}\n", .{ kv.key, str.* });
            },
            else => {
                std.debug.print("{s} => {any}\n", .{ kv.key, kv.value });
            },
        }
    }

    std.debug.print("Tensor Infos ========\n", .{});
    for (file.tensor_infos) |tensor| {
        std.debug.print("{s} => type={?s} dims={any} \n", .{ tensor.name, std.enums.tagName(Gguf.TensorType, tensor.tensor_type), tensor.dimensions });
    }

    std.debug.print("Tensor Data ========\n", .{});
    std.debug.print("{d} bytes\n", .{file.tensor_data.len});
}

const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const Init = std.process.Init;
const zif = @import("zif");
const Gguf = zif.Gguf;
