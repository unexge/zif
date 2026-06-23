tokenizer: BytePairEncoding,

pub fn init(gpa: Allocator, model: *Gguf) !@This() {
    const tokenizer: BytePairEncoding = try .init(gpa, model);

    return .{ .tokenizer = tokenizer };
}

pub fn deinit(self: *@This(), gpa: Allocator) void {
    self.tokenizer.deinit(gpa);
    self.* = undefined;
}

pub fn forward(self: *@This(), gpa: Allocator, input: []const u8) ![]const u8 {
    var window = input[0..];
    while (true) {
        const split_len, const tokens = try self.tokenizer.next(gpa, window) orelse break;
        window = window[split_len..];
        std.debug.print("tokens: {any}\n", .{tokens});
        gpa.free(tokens);
    }
    return input;
}

const BytePairEncoding = struct {
    // Holds concatenated pairs to their ranks
    merges: StringHashMap(usize),
    // Holds tokens to their indexes
    tokens: StringHashMap(usize),
    // These control tokens shouldn't be merged/splitted
    control_tokens: StringHashMap(usize),

    fn init(gpa: Allocator, model: *Gguf) !@This() {
        const tokenizer_model = try model.header.getMetadataKey([]const u8, "tokenizer.ggml.model") orelse return error.MissingTokenizerModel;
        if (!mem.eql(u8, tokenizer_model, "gpt2")) {
            return error.UnsupportedTokenizerModel;
        }

        const tokenizer_pre = try model.header.getMetadataKey([]const u8, "tokenizer.ggml.pre") orelse return error.MissingTokenizerPre;
        if (!mem.eql(u8, tokenizer_pre, "qwen2")) {
            return error.UnsupportedTokenizerPre;
        }

        const control_token_indexes = blk: {
            const raw = try model.header.getMetadataKey([]Gguf.Header.Value, "tokenizer.ggml.token_type") orelse return error.MissingTokenizerTokenTypes;
            var control_token_indexes: ArrayList(u32) = .empty;
            try control_token_indexes.ensureTotalCapacity(gpa, 32);

            for (0.., raw) |i, token_type| {
                switch (token_type) {
                    .int32 => |*num| {
                        if (num.* == 3) {
                            try control_token_indexes.append(gpa, @intCast(i));
                        }
                    },
                    else => return error.InvalidTokenizerTokenTypes,
                }
            }
            break :blk control_token_indexes;
        };
        const tokens, const control_tokens = blk: {
            const raw = try model.header.getMetadataKey([]Gguf.Header.Value, "tokenizer.ggml.tokens") orelse return error.MissingTokenizerTokens;
            var tokens: StringHashMap(usize) = .init(gpa);
            try tokens.ensureTotalCapacity(@intCast(raw.len));

            for (0.., raw) |i, token| {
                switch (token) {
                    .string => |*str| {
                        tokens.putAssumeCapacity(str.*, i);
                    },
                    else => return error.InvalidTokenizerMerges,
                }
            }

            var control_tokens: StringHashMap(usize) = .init(gpa);
            try control_tokens.ensureTotalCapacity(@intCast(control_token_indexes.items.len));
            for (control_token_indexes.items) |i| {
                const token = raw[i].string;
                control_tokens.putAssumeCapacity(token, i);
            }

            break :blk .{ tokens, control_tokens };
        };
        const merges = blk: {
            const raw = try model.header.getMetadataKey([]Gguf.Header.Value, "tokenizer.ggml.merges") orelse return error.MissingTokenizerMerges;
            var merges: StringHashMap(usize) = .init(gpa);
            try merges.ensureTotalCapacity(@intCast(raw.len));

            for (0.., raw) |i, merge| {
                switch (merge) {
                    .string => |*str| {
                        const key = try mem.replaceOwned(u8, gpa, str.*, " ", "");
                        merges.putAssumeCapacity(key, i);
                    },
                    else => return error.InvalidTokenizerMerges,
                }
            }
            break :blk merges;
        };

        return .{
            .tokens = tokens,
            .control_tokens = control_tokens,
            .merges = merges,
        };
    }

    fn deinit(self: *@This(), gpa: Allocator) void {
        {
            var iter = self.merges.keyIterator();
            while (iter.next()) |k| {
                gpa.free(k.*);
            }
        }
        self.tokens.deinit();
        self.control_tokens.deinit();
        self.merges.deinit();
        self.* = undefined;
    }

    // Find next token(s) from given input and return end of current token's position on input and list of tokens
    fn next(self: *@This(), gpa: Allocator, input: []const u8) !?struct { usize, []usize } {
        // Get first split based on RegEx pattern of Qwen2/GPT-2
        const split_len = currSplitLen(input);
        if (split_len == 0) return null;
        const split = input[0..split_len];

        // This might be start of a control token, find until |>, and do a lookup
        if (mem.eql(u8, "<|", split)) {
            var control_token_pos = split_len;
            const control_token_end = while (true) {
                const len = currSplitLen(input[control_token_pos..]);
                if (len == 0) {
                    // Reached end of input and couldn't find |>, fallback to regular handling
                    break null;
                }

                if (mem.eql(u8, "|>", input[control_token_pos..(control_token_pos + len)])) {
                    // Found end of possible control token
                    break control_token_pos + len;
                }

                // Continue looking
                control_token_pos += len;
            };
            if (control_token_end) |e| {
                const control_token = input[0..e];
                if (self.control_tokens.get(control_token)) |token| {
                    // It is a control token, return the token without trying merges
                    const tokens = try gpa.alloc(usize, 1);
                    tokens[0] = token;
                    return .{ e, tokens };
                }
            }
        }

        // Do byte-encoding to turn some bytes into Utf8 points
        const byte_encoded = try gpa.alloc(u8, split_len * 4);
        defer gpa.free(byte_encoded);
        var byte_encoded_len: usize = 0;
        for (split) |c| {
            const cp = byteEncoder(c);
            const n = try unicode.utf8Encode(cp, byte_encoded[byte_encoded_len..]);
            byte_encoded_len += n;
        }
        const token = byte_encoded[0..byte_encoded_len];

        // Now split into Utf8 chunks
        // TODO: Can we avoid allocating here?
        var chunks = try gpa.alloc([]const u8, token.len);
        defer gpa.free(chunks);
        var view: unicode.Utf8View = try .init(token);
        var iter = view.iterator();
        var ij: usize = 0;
        while (iter.nextCodepointSlice()) |c| {
            chunks[ij] = c;
            ij += 1;
        }
        chunks.len = ij;

        // Now try to do merges based on merge ranks for each consecutive pairs, see
        // https://github.com/openai/tiktoken/blob/08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa/tiktoken/_educational.py#L83-L116
        while (chunks.len > 1) {
            var min_rank: usize = math.maxInt(usize);
            assert(self.merges.count() < min_rank);
            var min_rank_idx: usize = undefined;

            for (0.., chunks[1..]) |i, next_chunk| {
                const curr_chunk = chunks[i];

                const merge_key = curr_chunk.ptr[0..(curr_chunk.len + next_chunk.len)];
                const rank = self.merges.get(merge_key) orelse continue;
                if (rank < min_rank) {
                    min_rank = rank;
                    min_rank_idx = i;
                }
            }

            if (min_rank == math.maxInt(usize)) {
                // No merges left
                break;
            }

            // Found a merge at min_rank_idx..min_rank_idx+1, merge them and shift everything to left after min_rank_idx+1
            chunks[min_rank_idx] = chunks[min_rank_idx].ptr[0..(chunks[min_rank_idx].len + chunks[min_rank_idx + 1].len)];
            for ((min_rank_idx + 1).., chunks[(min_rank_idx + 2)..]) |i, chunk| {
                chunks[i] = chunk;
            }
            chunks.len -= 1;
        }

        // Now all chunks are merged, try to decode each chunk into it's corresponding token index
        const tokens = try gpa.alloc(usize, chunks.len);
        for (0.., chunks) |i, chunk| {
            const bpe_token = self.tokens.get(chunk) orelse return error.UnknownToken;
            tokens[i] = bpe_token;
        }

        return .{ split_len, tokens };
    }

    // Inlined version of https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L8-L28
    fn byteEncoder(b: u8) u21 {
        return switch (b) {
            0 => 'Ā',
            1 => 'ā',
            2 => 'Ă',
            3 => 'ă',
            4 => 'Ą',
            5 => 'ą',
            6 => 'Ć',
            7 => 'ć',
            8 => 'Ĉ',
            9 => 'ĉ',
            10 => 'Ċ',
            11 => 'ċ',
            12 => 'Č',
            13 => 'č',
            14 => 'Ď',
            15 => 'ď',
            16 => 'Đ',
            17 => 'đ',
            18 => 'Ē',
            19 => 'ē',
            20 => 'Ĕ',
            21 => 'ĕ',
            22 => 'Ė',
            23 => 'ė',
            24 => 'Ę',
            25 => 'ę',
            26 => 'Ě',
            27 => 'ě',
            28 => 'Ĝ',
            29 => 'ĝ',
            30 => 'Ğ',
            31 => 'ğ',
            32 => 'Ġ',
            33 => '!',
            34 => '"',
            35 => '#',
            36 => '$',
            37 => '%',
            38 => '&',
            39 => '\'',
            40 => '(',
            41 => ')',
            42 => '*',
            43 => '+',
            44 => ',',
            45 => '-',
            46 => '.',
            47 => '/',
            48 => '0',
            49 => '1',
            50 => '2',
            51 => '3',
            52 => '4',
            53 => '5',
            54 => '6',
            55 => '7',
            56 => '8',
            57 => '9',
            58 => ':',
            59 => ';',
            60 => '<',
            61 => '=',
            62 => '>',
            63 => '?',
            64 => '@',
            65 => 'A',
            66 => 'B',
            67 => 'C',
            68 => 'D',
            69 => 'E',
            70 => 'F',
            71 => 'G',
            72 => 'H',
            73 => 'I',
            74 => 'J',
            75 => 'K',
            76 => 'L',
            77 => 'M',
            78 => 'N',
            79 => 'O',
            80 => 'P',
            81 => 'Q',
            82 => 'R',
            83 => 'S',
            84 => 'T',
            85 => 'U',
            86 => 'V',
            87 => 'W',
            88 => 'X',
            89 => 'Y',
            90 => 'Z',
            91 => '[',
            92 => '\\',
            93 => ']',
            94 => '^',
            95 => '_',
            96 => '`',
            97 => 'a',
            98 => 'b',
            99 => 'c',
            100 => 'd',
            101 => 'e',
            102 => 'f',
            103 => 'g',
            104 => 'h',
            105 => 'i',
            106 => 'j',
            107 => 'k',
            108 => 'l',
            109 => 'm',
            110 => 'n',
            111 => 'o',
            112 => 'p',
            113 => 'q',
            114 => 'r',
            115 => 's',
            116 => 't',
            117 => 'u',
            118 => 'v',
            119 => 'w',
            120 => 'x',
            121 => 'y',
            122 => 'z',
            123 => '{',
            124 => '|',
            125 => '}',
            126 => '~',
            127 => 'ġ',
            128 => 'Ģ',
            129 => 'ģ',
            130 => 'Ĥ',
            131 => 'ĥ',
            132 => 'Ħ',
            133 => 'ħ',
            134 => 'Ĩ',
            135 => 'ĩ',
            136 => 'Ī',
            137 => 'ī',
            138 => 'Ĭ',
            139 => 'ĭ',
            140 => 'Į',
            141 => 'į',
            142 => 'İ',
            143 => 'ı',
            144 => 'Ĳ',
            145 => 'ĳ',
            146 => 'Ĵ',
            147 => 'ĵ',
            148 => 'Ķ',
            149 => 'ķ',
            150 => 'ĸ',
            151 => 'Ĺ',
            152 => 'ĺ',
            153 => 'Ļ',
            154 => 'ļ',
            155 => 'Ľ',
            156 => 'ľ',
            157 => 'Ŀ',
            158 => 'ŀ',
            159 => 'Ł',
            160 => 'ł',
            161 => '¡',
            162 => '¢',
            163 => '£',
            164 => '¤',
            165 => '¥',
            166 => '¦',
            167 => '§',
            168 => '¨',
            169 => '©',
            170 => 'ª',
            171 => '«',
            172 => '¬',
            173 => 'Ń',
            174 => '®',
            175 => '¯',
            176 => '°',
            177 => '±',
            178 => '²',
            179 => '³',
            180 => '´',
            181 => 'µ',
            182 => '¶',
            183 => '·',
            184 => '¸',
            185 => '¹',
            186 => 'º',
            187 => '»',
            188 => '¼',
            189 => '½',
            190 => '¾',
            191 => '¿',
            192 => 'À',
            193 => 'Á',
            194 => 'Â',
            195 => 'Ã',
            196 => 'Ä',
            197 => 'Å',
            198 => 'Æ',
            199 => 'Ç',
            200 => 'È',
            201 => 'É',
            202 => 'Ê',
            203 => 'Ë',
            204 => 'Ì',
            205 => 'Í',
            206 => 'Î',
            207 => 'Ï',
            208 => 'Ð',
            209 => 'Ñ',
            210 => 'Ò',
            211 => 'Ó',
            212 => 'Ô',
            213 => 'Õ',
            214 => 'Ö',
            215 => '×',
            216 => 'Ø',
            217 => 'Ù',
            218 => 'Ú',
            219 => 'Û',
            220 => 'Ü',
            221 => 'Ý',
            222 => 'Þ',
            223 => 'ß',
            224 => 'à',
            225 => 'á',
            226 => 'â',
            227 => 'ã',
            228 => 'ä',
            229 => 'å',
            230 => 'æ',
            231 => 'ç',
            232 => 'è',
            233 => 'é',
            234 => 'ê',
            235 => 'ë',
            236 => 'ì',
            237 => 'í',
            238 => 'î',
            239 => 'ï',
            240 => 'ð',
            241 => 'ñ',
            242 => 'ò',
            243 => 'ó',
            244 => 'ô',
            245 => 'õ',
            246 => 'ö',
            247 => '÷',
            248 => 'ø',
            249 => 'ù',
            250 => 'ú',
            251 => 'û',
            252 => 'ü',
            253 => 'ý',
            254 => 'þ',
            255 => 'ÿ',
        };
    }

    // This function implements splitting logic of Qwen/GPT-2 RegExp pattern:
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
    pub fn currSplitLen(input: []const u8) usize {
        var i: usize = 0;
        const len = input.len;
        if (len == 0) return 0;

        var c = input[i];

        if (c == '\'') {
            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if ((i + 1 <= len) and (input[i + 1] == 's' or
                input[i + 1] == 'S' or
                input[i + 1] == 't' or
                input[i + 1] == 'T' or
                input[i + 1] == 'm' or
                input[i + 1] == 'M' or
                input[i + 1] == 'd' or
                input[i + 1] == 'D'))
            {
                return i + 1;
            } else if ((i + 2 <= len) and (((input[i + 1] == 'r' or input[i + 1] == 'R') and
                (input[i + 2] == 'e' or input[i + 2] == 'E')) or
                ((input[i + 1] == 'v' or input[i + 1] == 'V') and
                    (input[i + 2] == 'e' or input[i + 2] == 'E')) or
                ((input[i + 1] == 'l' or input[i + 1] == 'L') and
                    (input[i + 2] == 'l' or input[i + 2] == 'L'))))
            {
                return i + 2;
            }
        }

        // [^\\r\\n\\p{L}\\p{N}]?\\p{L}+
        if (c != '\n' and c != '\r' and unicodeLetter(input[i..]) == null and unicodeNumber(input[i..]) == null) {
            i += 1;
        }
        if (i == len - 1) return len;
        c = input[i];
        const letterStart: usize = i;
        while (unicodeLetter(input[i..])) |llen| {
            i += llen;
            if (i >= len) return len;
            c = input[i];
        }
        if (i > letterStart) return i; // Found some unicode letters

        // \\p{N}
        if (unicodeNumber(input[i..])) |llen| {
            // Found a unicode number
            return i + llen;
        }

        // (space)?[^\\s\\p{L}\\p{N}]+[\\r\\n]*
        if (c == ' ') {
            if (i == len - 1) return len;
            i += 1;
            c = input[i];
        }
        var foundSomeUnicodeOrWhitespace = false;
        while (!ascii.isWhitespace(c) and unicodeLetter(input[i..]) == null and unicodeNumber(input[i..]) == null) {
            if (i == len - 1) return len;
            i += 1;
            c = input[i];
            foundSomeUnicodeOrWhitespace = true;
        }
        if (foundSomeUnicodeOrWhitespace) {
            while (c == '\n' or c == '\r') {
                if (i == len - 1) return len;
                i += 1;
                c = input[i];
            }
            return i;
        }

        // \\s*[\\r\\n]+
        while (ascii.isWhitespace(c)) {
            if (i == len - 1) return len;
            i += 1;
            c = input[i];
        }
        const newlineStart: usize = i;
        while (c == '\n' or c == '\r') {
            if (i == len - 1) return len;
            i += 1;
            c = input[i];
        }
        if (i > newlineStart) return i; // Found at least one newline

        // \\s+(?!\\S) and \\s+
        while (ascii.isWhitespace(c)) {
            if (i == len - 1) return len;
            i += 1;
            c = input[i];
        }
        return i;
    }

    // Simulates \\p{L} RegExp pattern
    fn unicodeLetter(input: []const u8) ?usize {
        if (input.len == 0) return null;
        const size = unicode.utf8ByteSequenceLength(input[0]) catch return null;
        if (size > input.len) return null;
        if (!zg.general_categories.isLetter(unicode.utf8Decode(input[0..size]) catch return null)) return null;
        return size;
    }

    // Simulates \\p{N} RegExp pattern
    fn unicodeNumber(input: []const u8) ?usize {
        if (input.len == 0) return null;
        const size = unicode.utf8ByteSequenceLength(input[0]) catch return null;
        if (size > input.len) return null;
        if (!zg.general_categories.isNumber(unicode.utf8Decode(input[0..size]) catch return null)) return null;
        return size;
    }
};

const std = @import("std");
const mem = std.mem;
const unicode = std.unicode;
const ascii = std.ascii;
const math = std.math;
const assert = std.debug.assert;
const Allocator = mem.Allocator;
const StringHashMap = std.StringHashMap;
const StaticStringMap = std.StaticStringMap;
const ArrayList = std.ArrayList;
const Gguf = @import("Gguf.zig");
const zg = @import("zg");
