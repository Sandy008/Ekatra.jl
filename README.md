# Ekatra.jl

**A zero-copy GGUF parser and tokenizer bridge for Julia.**

Ekatra.jl provides a production-grade, memory-safe interface for reading GGUF
model files.  It parses the binary format via `Mmap`, exposes metadata and
tensor data lazily through zero-copy pointer views, and includes a tokenizer
bridge with a pure-Julia fallback.

---

## Features

| Feature | Status |
|---|---|
| GGUF v1/v2/v3 binary parsing | ✅ |
| Zero-copy tensor access via `Mmap` | ✅ |
| Full metadata KV dictionary | ✅ |
| Type-driven quantization dispatch | ✅ |
| Native BPE/SentencePiece tokenizer | ✅ |
| Tokenizers.jl bridge (optional extension) | ✅ |
| Validation mode | ✅ |
| Thread-safe read access | ✅ |

## Installation

```julia
pkg> add https://github.com/Sandy008/Ekatra.jl
```

## Quick start

```julia
using Ekatra

# Open a GGUF file (zero-copy memory map, no tensor data read yet)
model = load_gguf("mistral-7b-instruct.Q4_K_M.gguf")

# Read metadata
println(get_metadata(model, "general.name"))
println(get_metadata(model, "general.architecture"))

# List tensor names
for name in list_tensors(model)
    ti = tensor_info(model, name)
    println("  $(name): dims=$(ti.dims), dtype=$(ti.dtype)")
end

# Zero-copy tensor view (no data copied to Julia heap)
t = get_tensor(model, "token_embd.weight")
println("raw bytes: $(Ekatra.nbytes(t))")
println("ptr: $(Ekatra.raw_pointer(t))")

# Tokenizer
tok = get_tokenizer(model)
ids = encode(tok, "Hello, world!")
txt = decode(tok, ids)
println(txt)

# Always close when done (also registered as a finalizer)
close(model)
```

## Optional: Tokenizers.jl bridge

When [Tokenizers.jl](https://github.com/JuliaHub/Tokenizers.jl) is installed,
Ekatra automatically loads a bridge extension that delegates encoding/decoding
to the full HuggingFace tokenizer embedded in the GGUF file:

```julia
using Ekatra, Tokenizers   # extension loads automatically
model = load_gguf("model.gguf")
tok   = get_tokenizer(model)   # → TokenizersBridgeTokenizer
```

## API reference

| Function | Description |
|---|---|
| `load_gguf(path; lazy_tensors, load_tokenizer, validate)` | Open and parse a GGUF file |
| `list_tensors(model)` | Return all tensor names |
| `tensor_info(model, name)` | Return `TensorInfo` for a named tensor |
| `get_tensor(model, name)` | Return a zero-copy `QuantizedTensor` |
| `get_metadata(model, key)` | Read a metadata value by key |
| `get_tokenizer(model)` | Return the associated tokenizer |
| `encode(tokenizer, text)` | Encode a string to token IDs |
| `decode(tokenizer, tokens)` | Decode token IDs to a string |
| `close(model)` | Release the memory map |
| `isopen(model)` | Check whether the model is still open |

## Design principles

- **Zero-Copy** — tensor data is never copied into Julia-managed memory.
- **No Semantic Interpretation** — only structure, storage, and metadata are exposed.
- **Type-Driven Quantization** — every quantization scheme is a distinct Julia type enabling future dispatch for optimized kernels.
- **Safety** — pointer validity is enforced via lifecycle checks; access after `close` throws immediately.
- **Thread-safe reads** — no mutable global state; all mutation is confined to construction.

