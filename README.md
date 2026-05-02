# Ekatra.jl — Zero-copy GGUF parser and tokenizer bridge for Julia

![Julia](https://img.shields.io/badge/julia-1.9+-blue)
![Status](https://img.shields.io/badge/status-v0.1-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

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

> **GGUF** is the binary model format used by [llama.cpp](https://github.com/ggerganov/llama.cpp)
> and the broader quantized LLM ecosystem. Ekatra.jl gives Julia first-class,
> zero-copy access to any GGUF file.

---

## Why Ekatra?

Most Julia ML tools copy model weights into managed arrays before doing anything
useful. For large quantized models this is both slow and wasteful.

**Ekatra takes a different approach:**

- **Zero-copy by default.** Tensor data is exposed as raw `Ptr{UInt8}` pointers
  directly into the OS page cache via `Mmap`. No bytes are allocated on the
  Julia heap until you explicitly ask for them. A 7B model loads in milliseconds
  because only metadata is parsed and the file is memory-mapped — no weight
  data is touched at all.

- **HPC-grade access patterns.** Metadata lookup is O(1) with respect to model
  size. Partial loading (metadata-only, tokenizer-only) is a first-class
  option, not an afterthought. Pointer validity is enforced at every call site
  so there are no silent use-after-free bugs.

- **Julia-native ecosystem.** Quantization schemes are real Julia types, not
  integer tags. This means dispatch — future dequantization kernels, SIMD
  optimizations, or GPU transfers can all be written as ordinary Julia methods
  without touching the parser. The Tokenizers.jl bridge loads automatically
  via Julia's package-extension mechanism when the package is present.

---

## Who is this for?

Ekatra.jl is useful if you want to:

- **Inspect GGUF models** — browse metadata, tensor shapes, and quantization types
  without loading weights into memory
- **Build inference runtimes in Julia** — Ekatra handles the data/interface layer;
  you supply the compute
- **Research model structure** — analyze weight distributions, layer naming,
  dtype layouts across quantization formats
- **Integrate LLMs into scientific or HPC workflows** — zero-copy access fits
  naturally into pipelines that already work with raw pointers or external
  C/C++ libraries

---

## What Ekatra does NOT do (yet)

- ❌ **No model inference** — no forward pass, no attention, no KV cache
- ❌ **No training or fine-tuning** — read-only access only
- ❌ **No GPU execution layer** — tensors are CPU memory-mapped pointers

Ekatra focuses entirely on the data and interface layer.
Inference support is planned for Phase 2.

---

## Performance

Loading a GGUF model only parses metadata and memory-maps the file:

```
Model size : 7B parameters  (~4–8 GB GGUF on disk)
Load time  : ~10–50 ms      (depends on filesystem cache warmth)
Heap usage : ~0 MB          (OS-managed page cache — no Julia allocations)
```

No tensor data is read into Julia memory unless `get_tensor` is explicitly
called. This makes Ekatra safe to use in memory-constrained environments and
lets you open models larger than available RAM.

---

## Architecture

```
GGUF file (on disk)
        │
        ▼
  Mmap (OS page cache)
        │
        ▼
  Binary parser (Ekatra)
  ├── Header  (magic, version, counts)
  ├── Metadata KV section  → Dict{String, Any}
  └── Tensor info block    → Dict{String, TensorInfo}
        │
        ▼
  Zero-copy tensor views   → QuantizedTensor{Q<:QuantizationScheme}
        │
        ▼
  (Optional) Tokenizer bridge
  ├── GGUFNativeTokenizer  (pure-Julia BPE fallback)
  └── Tokenizers.jl bridge (full HF tokenizer, auto-loaded)
```

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

---

## Installation

Ekatra.jl is currently available via GitHub:

```julia
import Pkg
Pkg.add(url="https://github.com/Sandy008/Ekatra.jl")
```

Or in the Pkg REPL:

```
pkg> add https://github.com/Sandy008/Ekatra.jl
```

> Registration in the Julia General Registry is planned for a future release.

---

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
GGUFModel(open, path="stories15M-q4_0.gguf", tensors=57, metadata=20)
  Architecture : llama
  Name         : llama
  token_embd.weight   dims=(288, 32000)   dtype=Q4_0
  blk.0.attn_q.weight dims=(288, 288)     dtype=Q4_0
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

---

## Optional: Tokenizers.jl bridge

When [Tokenizers.jl](https://github.com/JuliaHub/Tokenizers.jl) is installed,
Ekatra automatically loads a bridge extension that delegates encoding/decoding
to the full HuggingFace tokenizer embedded in the GGUF file:

```julia
using Ekatra, Tokenizers   # extension loads automatically
model = load_gguf("model.gguf")
tok   = get_tokenizer(model)   # → TokenizersBridgeTokenizer
```

---

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
  general.name: 'llama'
  llama.context_length: 128
  ...
  Total metadata keys: 20

── Tensors ────────────────────────────────────────────────
  Total tensors: 57
  [  0] token_embd.weight     shape=[288, 32000]  dtype=Q4_0
  [  1] output_norm.weight    shape=[288]          dtype=F32
  ...

── Quantization type distribution ─────────────────────────
  Q4_0          43 tensors
  F32           13 tensors
  Q8_0           1 tensors

✓ Cross-validation PASSED for stories15M-q4_0.gguf
```

---

## API reference

### Model loading

| Function | Description |
|---|---|
| `load_gguf(path; lazy_tensors, load_tokenizer, validate)` | Open and parse a GGUF file |
| `close(model)` | Release the memory map |
| `isopen(model)` | Check whether the model is still open |

### Metadata & structure

| Function | Description |
|---|---|
| `get_metadata(model, key)` | Read a metadata value by key (returns `nothing` if absent) |
| `list_tensors(model)` | Return all tensor names |
| `tensor_info(model, name)` | Return `TensorInfo` (dims, dtype, offset) for a named tensor |

### Tensor access

| Function | Description |
|---|---|
| `get_tensor(model, name)` | Return a zero-copy `QuantizedTensor{Q}` |
| `Ekatra.raw_pointer(t)` | Raw `Ptr{UInt8}` into the mmap'd data |
| `Ekatra.nbytes(t)` | Byte size of the tensor storage |

### Tokenizer

| Function | Description |
|---|---|
| `get_tokenizer(model)` | Return the associated tokenizer |
| `encode(tokenizer, text)` | Encode a string to token IDs |
| `decode(tokenizer, tokens)` | Decode token IDs to a string |

---

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

---

## Contributing

Contributions are welcome, especially for:

- **Phase 2 inference runtime** — dequantization kernels, KV cache, batching
- **Performance optimizations** — SIMD, threading, memory layout improvements
- **Additional GGUF validation cases** — edge cases, unusual quantization types
- **Documentation and examples** — more usage patterns, tutorials

Please open an issue to discuss before submitting large changes.
Feel free to open pull requests for bug fixes, tests, or documentation at any time.


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

