pub fn Tensor(comptime T: type) type {
    const type_info = @typeInfo(T);
    comptime {
        switch (type_info) {
            .int => {},
            .float => {},
            else => @compileError("A tensor must be a float or an integer"),
        }
    }
    const is_float = type_info == .float;

    return struct {
        gpa: Allocator,

        buf: []T,
        shape: []const usize,
        stride: []const usize,

        // Whether this tensor owns the backing buffer,
        // and if so it will free it on `deinit`.
        owned: bool = false,

        pub fn init(gpa: Allocator, buf: []T, shape: []const usize) !@This() {
            const expected_len = Tensor(T).calculateLen(shape);
            if (buf.len != expected_len) return error.InvalidSize;
            const stride = try Tensor(T).calculateStride(gpa, shape);
            return .{ .gpa = gpa, .buf = buf, .shape = shape, .stride = stride };
        }

        pub fn arange(gpa: Allocator, start: T, end: T, step: T) !@This() {
            if (start >= end or ((end - start) <= step)) return error.InvalidRange;
            const total_steps = (end - start) / step;
            const len: usize = if (comptime is_float) @intFromFloat(total_steps) else @intCast(total_steps);
            const buf = try gpa.alloc(T, len);
            var curr = start;
            for (0..len) |i| {
                buf[i] = curr;
                curr += step;
            }
            const shape = try gpa.alloc(usize, 1);
            shape[0] = len;
            var x: Tensor(T) = try .init(gpa, buf, shape);
            x.owned = true;
            return x;
        }

        pub fn deinit(self: *@This()) void {
            if (self.owned) {
                self.gpa.free(self.buf);
                self.gpa.free(self.shape);
            }
            self.gpa.free(self.stride);
            self.* = undefined;
        }

        pub fn get(self: *@This(), dim: []const usize) !@This() {
            if (dim.len > self.shape.len) return error.OutOfRange;
            var shape = self.shape;
            var offset: usize = 0;
            for (0.., dim) |i, d| {
                const sd = self.shape[i];
                const st = self.stride[i];
                if (d > sd) return error.OutOfRange;
                offset += d * st;
                shape = shape[1..];
            }
            const len = Tensor(T).calculateLen(shape);
            return .init(self.gpa, self.buf[offset..(offset + len)], shape);
        }

        pub fn view(self: *@This(), view_shape: []const usize) !@This() {
            const curr_total = Tensor(T).calculateLen(self.shape);
            const new_total = Tensor(T).calculateLen(view_shape);
            if (curr_total != new_total) return error.SizeMismatch;

            return .init(self.gpa, self.buf, view_shape);
        }

        // pub fn split(self: *@This(), sections: []const usize, dim: usize) !struct {@This(), @This()} {
        //     if (self.shape.len < dim) return error.OutOfRange;
        //     const d = self.shape[dim];
        //     const total_split_size = blk: {
        //         var total: usize = 0;
        //         for (sections) |s| total += s;
        //         break :blk total;
        //     };
        //     if (d != total_split_size) return error.SplitSumNotEqualToTotalDim;

        //     const s = self.stride[dim];
        //     var pos: usize = 0;

        //     const first: Tensor(T) = .init(self.gpa, self.buf[pos], self.shape);

        //     // for (sections) |end| {
        //     //     const offset = pos * s;

        //     //     std.debug.print("view: {d}");

        //     //     pos = end;
        //     // }
        // }

        fn calculateStride(gpa: Allocator, shape: []const usize) ![]usize {
            if (shape.len == 0) return error.EmptyShape;

            const stride = try gpa.alloc(usize, shape.len);
            stride[stride.len - 1] = 1;

            for (0..(stride.len - 1)) |i| {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
            return stride;
        }

        fn calculateLen(shape: []const usize) usize {
            var total: usize = 1;
            for (shape) |d| total *= d;
            return total;
        }
    };
}

test "arange" {
    var x: Tensor(f32) = try .arange(testing.allocator, 0.0, 5.0, 1.0);
    defer x.deinit();
    var y: Tensor(u64) = try .arange(testing.allocator, 0, 10, 2);
    defer y.deinit();

    try testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 3.0, 4.0 }, x.buf);
    try testing.expectEqualSlices(u64, &.{ 0, 2, 4, 6, 8 }, y.buf);
}

test "stride" {
    const buf: []const u8 = &.{ 0, 1, 2, 3, 4, 5 };
    var x: Tensor(u8) = try .init(testing.allocator, @constCast(buf), &.{ 2, 3 });
    defer x.deinit();

    try testing.expectEqualSlices(usize, &.{ 2, 3 }, x.shape);
    try testing.expectEqualSlices(usize, &.{ 3, 1 }, x.stride);
}

test "get" {
    const buf: []const u8 = &.{ 0, 1, 2, 3, 4, 5 };
    var x: Tensor(u8) = try .init(testing.allocator, @constCast(buf), &.{ 2, 3 });
    defer x.deinit();

    var y = try x.get(&.{0});
    defer y.deinit();
    try testing.expectEqualSlices(usize, &.{3}, y.shape);
    try testing.expectEqualSlices(usize, &.{1}, y.stride);
    try testing.expectEqualSlices(u8, &.{ 0, 1, 2 }, y.buf);

    var z = try x.get(&.{1});
    defer z.deinit();
    try testing.expectEqualSlices(usize, &.{3}, z.shape);
    try testing.expectEqualSlices(usize, &.{1}, z.stride);
    try testing.expectEqualSlices(u8, &.{ 3, 4, 5 }, z.buf);
}

test "view" {
    var x: Tensor(f32) = try .arange(testing.allocator, 0.0, 6.0, 1.0);
    defer x.deinit();

    {
        var y = try x.view(&.{ 2, 3 });
        defer y.deinit();
        try testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 }, y.buf);
        try testing.expectEqualSlices(usize, &.{ 2, 3 }, y.shape);
        try testing.expectEqualSlices(usize, &.{ 3, 1 }, y.stride);

        var z = try y.get(&.{0});
        defer z.deinit();
        try testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0 }, z.buf);
    }

    {
        var y = try x.view(&.{ 3, 2 });
        defer y.deinit();
        try testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 }, y.buf);
        try testing.expectEqualSlices(usize, &.{ 3, 2 }, y.shape);
        try testing.expectEqualSlices(usize, &.{ 2, 1 }, y.stride);

        var z = try y.get(&.{0});
        defer z.deinit();
        try testing.expectEqualSlices(f32, &.{ 0.0, 1.0 }, z.buf);
    }
}

// test "split" {
//     var raw: Tensor(f16) = try .arange(testing.allocator, 0.0, 12.0, 1.0);
//     defer raw.deinit();

//     var x = try raw.view(&.{ 4, 3 });
//     defer x.deinit();

//     var iter = x.split(&.{ 1, 2 }, 1);
//     const y = iter.next().?;
//     defer y.deinit();
//     const z = iter.next().?;
//     defer z.deinit();
//     try testing.expect(iter.next() == null);

//     try testing.expectEqualSlices(usize, &.{ 4, 1 }, y.shape);
//     try testing.expectEqualSlices(usize, &.{ 3, 1 }, y.stride);
//     try testing.expectEqualSlices(f16, &.{ 0.0, 3.0, 6.0, 9.0 }, y.buf);

//     try testing.expectEqualSlices(usize, &.{ 4, 2 }, z.shape);
//     try testing.expectEqualSlices(usize, &.{ 3, 1 }, z.stride);
//     // try testing.expectEqualSlices(f16, &.{}, z.buf);
// }

const std = @import("std");
const testing = std.testing;
const mem = std.mem;
const Allocator = mem.Allocator;
