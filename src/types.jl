"""
Core types for Ekatra.jl.
"""

# ──────────────────────────────────────────────
# Quantization scheme types
# ──────────────────────────────────────────────

"""Abstract base for all GGML quantization schemes."""
abstract type QuantizationScheme end

struct F32   <: QuantizationScheme end
struct F16   <: QuantizationScheme end
struct Q4_0  <: QuantizationScheme end
struct Q4_1  <: QuantizationScheme end
struct Q5_0  <: QuantizationScheme end
struct Q5_1  <: QuantizationScheme end
struct Q8_0  <: QuantizationScheme end
struct Q8_1  <: QuantizationScheme end
struct Q2_K  <: QuantizationScheme end
struct Q3_K  <: QuantizationScheme end
struct Q4_K  <: QuantizationScheme end
struct Q5_K  <: QuantizationScheme end
struct Q6_K  <: QuantizationScheme end
struct Q8_K  <: QuantizationScheme end
struct IQ2_XXS <: QuantizationScheme end
struct IQ2_XS  <: QuantizationScheme end
struct IQ3_XXS <: QuantizationScheme end
struct IQ1_S   <: QuantizationScheme end
struct IQ4_NL  <: QuantizationScheme end
struct IQ3_S   <: QuantizationScheme end
struct IQ2_S   <: QuantizationScheme end
struct IQ4_XS  <: QuantizationScheme end
struct I8    <: QuantizationScheme end
struct I16   <: QuantizationScheme end
struct I32   <: QuantizationScheme end
struct BF16  <: QuantizationScheme end
struct UnknownQuantization <: QuantizationScheme
    type_id::UInt32
end

# ──────────────────────────────────────────────
# GGUF KV value types (enum mirrors)
# ──────────────────────────────────────────────

const GGUF_TYPE_UINT8   = UInt32(0)
const GGUF_TYPE_INT8    = UInt32(1)
const GGUF_TYPE_UINT16  = UInt32(2)
const GGUF_TYPE_INT16   = UInt32(3)
const GGUF_TYPE_UINT32  = UInt32(4)
const GGUF_TYPE_INT32   = UInt32(5)
const GGUF_TYPE_FLOAT32 = UInt32(6)
const GGUF_TYPE_BOOL    = UInt32(7)
const GGUF_TYPE_STRING  = UInt32(8)
const GGUF_TYPE_ARRAY   = UInt32(9)
const GGUF_TYPE_UINT64  = UInt32(10)
const GGUF_TYPE_INT64   = UInt32(11)
const GGUF_TYPE_FLOAT64 = UInt32(12)

# ──────────────────────────────────────────────
# Abstract tensor
# ──────────────────────────────────────────────

"""Abstract base type for all GGUF tensors."""
abstract type AbstractGGUFTensor end

# ──────────────────────────────────────────────
# TensorInfo — parsed from the tensor info block
# ──────────────────────────────────────────────

"""
Metadata for a single tensor as stored in the GGUF tensor info block.

Fields:
- `name`   : tensor name
- `dims`   : shape tuple (Julia `Dims`)
- `dtype`  : raw GGML type enum value
- `offset` : byte offset relative to the data section base
"""
struct TensorInfo
    name::String
    dims::Dims
    dtype::UInt32
    offset::UInt64
end

# ──────────────────────────────────────────────
# QuantizedTensor — zero-copy view into mmap'd data
# ──────────────────────────────────────────────

"""
A zero-copy, typed view of raw tensor bytes in a memory-mapped GGUF file.

The pointer `ptr` is valid only while the parent `GGUFModel` is open.
"""
struct QuantizedTensor{Q<:QuantizationScheme} <: AbstractGGUFTensor
    ptr::Ptr{UInt8}
    dims::Dims
    nbytes::Int
    offset::UInt64
end

# ──────────────────────────────────────────────
# Tokenizer abstractions
# ──────────────────────────────────────────────

"""Abstract tokenizer interface."""
abstract type AbstractTokenizer end

# ──────────────────────────────────────────────
# GGUFModel — top-level handle
# ──────────────────────────────────────────────

"""
Top-level handle for an open GGUF model file.

- `path`         : absolute path to the file
- `metadata`     : key → Any dictionary of all KV pairs
- `tensor_index` : name → TensorInfo
- `data_offset`  : byte offset in the file where the data section begins
- `mmap_handle`  : memory-mapped byte array (the full file)
- `is_closed`    : whether `close(model)` has been called
- `tokenizer`    : optional tokenizer (Nothing or AbstractTokenizer subtype)
"""
mutable struct GGUFModel
    path::String
    metadata::Dict{String, Any}
    tensor_index::Dict{String, TensorInfo}
    data_offset::UInt64
    mmap_handle::Vector{UInt8}   # Mmap.mmap result
    is_closed::Bool
    tokenizer::Union{Nothing, AbstractTokenizer}
end
