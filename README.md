# Ekatra.jl — Zero-copy GGUF parser and tokenizer bridge for Julia

**Status: v0.1 — Phase 1 complete**

| | |
|---|---|
| ✅ | GGUF binary parsing (v1 / v2 / v3) |
| ✅ | Zero-copy tensor access via `Mmap` |
| ✅ | Quantization-aware type system (27 GGML types) |
| ✅ | Full metadata KV dictionary |
| ✅ | Native BPE/SentencePiece tokenizer |
| ✅ | Tokenizers.jl bridge (optional extension) |
| ✅ | Validation mode & thread-safe reads |
| 🚧 | Inference runtime — Phase 2 planned |

---

## Why Ekatra?

Most Julia ML tools copy model weights into managed arrays before doing anything
useful.  For large quantized models this is both slow and wasteful.

**Ekatra takes a different approach:**

- **Zero-copy by default.** Tensor data is exposed as raw `Ptr{UInt8}` pointers
  directly into the OS page cache via `Mmap`.  No bytes are allocated on the
  Julia heap until you explicitly ask for them.  A 7 B model loads in
  milliseconds regardless of size.

- **HPC-grade access patterns.** Metadata lookup is O(1) with respect to model
  size.  Partial loading (metadata-only, tokenizer-only) is a first-class
  option, not an afterthought.  Pointer validity is enforced at every call site
  so there are no silent use-after-free bugs.

- **Julia-native ecosystem.** Quantization schemes are real Julia types, not
  integer tags.  This means dispatch — future dequantization kernels, SIMD
  optimizations, or GPU transfers can all be written as ordinary Julia methods
  without touching the parser.  The Tokenizers.jl bridge loads automatically
  via Julia's package-extension mechanism when the package is present.

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

**Expected output** (stories15M-q4_0.gguf):

```
GGUFModel(open, path="stories15M-q4_0.gguf", tensors=27, metadata=19)
  Architecture : llama
  Name         : stories15M
  ...
  token_embd.weight   dims=(2048, 32000)   dtype=Q4_0
  blk.0.attn_q.weight dims=(2048, 2048)    dtype=Q4_0
  ...
✓ Data is memory-mapped — no copy into Julia heap.
```

### Run the built-in example script

```bash
# Automatically downloads stories15M (~18 MB) on first run
julia --project=. examples/basic_usage.jl

# Or point it at your own model
julia --project=. examples/basic_usage.jl /path/to/model.gguf
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

## Testing

### Unit tests (no internet required)

```bash
julia --project=. test/runtests.jl
```

### Real-model tests (downloads on first run, then cached)

```bash
# Tiny model (~18 MB) — fast, good for CI
julia --project=. test/real_models/test_tiny_model.jl

# Quantized model (~100 MB) — validates K-quant types
julia --project=. test/real_models/test_quantized_model.jl
```

Models are downloaded from HuggingFace into `test/real_models/cache/` (git-ignored).
Set `EKATRA_SKIP_NETWORK_TESTS=true` to skip downloads in offline environments.

### Cross-validation with Python `gguf`

```bash
pip install gguf
python test/real_models/cross_validate.py test/real_models/cache/stories15M-q4_0.gguf
```

The script validates that metadata keys, tensor shapes, and dtype assignments match
the reference Python implementation, and writes a machine-readable JSON report.

**Example output:**
```
============================================================
Cross-validation: stories15M-q4_0.gguf  (19.1 MB)
============================================================

── Metadata ──────────────────────────────────────────────
  general.architecture: 'llama'
  general.name: 'stories15M'
  llama.context_length: 2048
  ...
  Total metadata keys: 19

── Tensors ────────────────────────────────────────────────
  Total tensors: 27
  [  0] token_embd.weight                          shape=[2048, 32000]  dtype=Q4_0
  [  1] blk.0.attn_norm.weight                     shape=[2048]         dtype=F32
  ...

── Quantization type distribution ─────────────────────────
  Q4_0          25 tensors
  F32            2 tensors

✓ Cross-validation PASSED for stories15M-q4_0.gguf
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

---

## Roadmap

### ✅ Phase 1 — GGUF parsing (complete, v0.1)

- [x] Full GGUF v1/v2/v3 binary parser
- [x] Zero-copy tensor access via `Mmap`
- [x] Quantization-aware type system (27 GGML types)
- [x] Full metadata KV dictionary
- [x] Native BPE/SentencePiece tokenizer fallback
- [x] Tokenizers.jl bridge extension
- [x] Validation mode & lifecycle safety
- [x] Real-model tests + Python cross-validation

### 🚧 Phase 2 — Inference runtime (planned)

- [ ] KV cache management (zero-allocation ring buffers)
- [ ] Batched prompt processing
- [ ] Dequantization kernels (Q4_K, Q8_0, F16 → F32) via Julia dispatch
- [ ] GGML_jll integration for hardware-accelerated ops
- [ ] CUDA / Metal backend hooks

### 🔭 Phase 3 — Ecosystem tools & integrations (future)

- [ ] `serve(model)` — HTTP inference endpoint
- [ ] LangChain.jl / Agents.jl integration
- [ ] LoRA / adapter weight loading
- [ ] Streaming decode API
- [ ] Model benchmarking suite

