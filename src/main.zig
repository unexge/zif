pub fn main(init: Init) !void {
    var args = init.minimal.args.iterate();
    _ = args.next() orelse return error.MissingProgname;

    const subcommand = args.next() orelse return error.MissingSubcommand;

    if (mem.eql(u8, subcommand, "dump")) {
        const filename = args.next() orelse return error.MissingFilename;
        try dump(&init, filename);
    } else if (mem.eql(u8, subcommand, "run")) {
        const filename = args.next() orelse return error.MissingFilename;
        try run(&init, filename);
    } else {
        return error.UnknownSubcommand;
    }
}

fn dump(init: *const Init, filename: []const u8) !void {
    var file = try Gguf.parse(init.gpa, init.io, filename);
    defer file.deinit(init.gpa);

    std.debug.print("Header ========\n", .{});

    var keys = try init.gpa.alloc([]const u8, file.header.metadata.count());
    defer init.gpa.free(keys);

    var key_iter = file.header.metadata.keyIterator();
    var i: usize = 0;
    while (key_iter.next()) |key| {
        keys[i] = key.*;
        i += 1;
    }
    mem.sortUnstable([]const u8, keys, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.lessThan);

    for (keys) |key| {
        const val = file.header.metadata.get(key).?;
        switch (val) {
            .array => |*arr| {
                std.debug.print("{s} => array with {d} elements\n", .{ key, arr.len });
                for (0..5) |j| {
                    std.debug.print("\t{any}\n", .{arr.*[j]});
                }
            },
            .string => |*str| {
                std.debug.print("{s} => {s}\n", .{ key, str.* });
            },
            else => {
                std.debug.print("{s} => {any}\n", .{ key, val });
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

fn run(init: *const Init, filename: []const u8) !void {
    var file = try Gguf.parse(init.gpa, init.io, filename);
    defer file.deinit(init.gpa);

    var model = try Qwen3.init(init.gpa, &file);
    defer model.deinit(init.gpa);

    const input =
        \\<|im_start|>user
        \\What is the capital of France?<|im_end|>
        \\<|im_start|>assistant
    ;
    const output = try model.forward(init.gpa, input);
    std.debug.print("< {s}\n", .{input});
    std.debug.print("> {s}\n", .{output});
}

const std = @import("std");
const mem = std.mem;
const sort = std.sort;
const Io = std.Io;
const Init = std.process.Init;
const zif = @import("zif");
const Gguf = zif.Gguf;
const Qwen3 = zif.Qwen3;
