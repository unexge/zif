# zif

## Parse GGUF file
```shell
$ zig build run -Doptimize=ReleaseSafe -- /path/to/Qwen3-0.6B-Q8_0.gguf
Header ========
general.architecture => qwen3
general.type => model
general.name => Qwen3 0.6B Instruct
...
general.quantization_version => .{ .uint32 = 2 }
general.file_type => .{ .uint32 = 7 }
Tensor Infos ========
output_norm.weight => type=F32 dims={ 1024 }
token_embd.weight => type=Q8_0 dims={ 1024, 151936 }
...
blk.27.ffn_gate.weight => type=Q8_0 dims={ 1024, 3072 }
blk.27.ffn_norm.weight => type=F32 dims={ 1024 }
blk.27.ffn_up.weight => type=Q8_0 dims={ 1024, 3072 }
Tensor Data ========
633495580 bytes
```
