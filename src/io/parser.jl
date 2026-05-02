"""
GGUF binary parser: header, KV metadata, and tensor info block.
"""

using Logging

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

const GGUF_MAGIC         = 0x46554747  # 'GGUF' little-endian UInt32
const GGUF_VERSION_MIN   = UInt32(1)
const GGUF_VERSION_MAX   = UInt32(3)

# Default alignment used when the key is absent in metadata
const GGUF_DEFAULT_ALIGNMENT = UInt32(32)

# ──────────────────────────────────────────────
# Header parsing
# ──────────────────────────────────────────────

struct GGUFHeader
    version::UInt32
    tensor_count::UInt64
    metadata_kv_count::UInt64
end

"""
    parse_header(r) -> GGUFHeader

Parse and validate the 4-byte magic + version + counts.
"""
function parse_header(r::GGUFBinaryReader)::GGUFHeader
    magic = read_uint32(r)
    if magic != GGUF_MAGIC
        throw(GGUFIOError(
            @sprintf("Invalid GGUF magic: expected 0x%08X, got 0x%08X", GGUF_MAGIC, magic)
        ))
    end

    version = read_uint32(r)
    if version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX
        @warn "GGUF version $(version) is outside the tested range " *
              "[$(GGUF_VERSION_MIN), $(GGUF_VERSION_MAX)]. Proceeding with best effort."
    end

    tensor_count       = read_uint64(r)
    metadata_kv_count  = read_uint64(r)

    @debug "GGUF header: version=$(version), tensors=$(tensor_count), " *
           "metadata_kv=$(metadata_kv_count)"

    return GGUFHeader(version, tensor_count, metadata_kv_count)
end

# ──────────────────────────────────────────────
# KV value parsing
# ──────────────────────────────────────────────

"""
    parse_kv_value(r, value_type) -> Any

Dispatch on the GGUF value-type enum and return a Julia value.
Unknown types are skipped with a warning and `nothing` is returned.
"""
function parse_kv_value(r::GGUFBinaryReader, value_type::UInt32)
    if value_type == GGUF_TYPE_UINT8
        return read_uint8(r)
    elseif value_type == GGUF_TYPE_INT8
        return read_int8(r)
    elseif value_type == GGUF_TYPE_UINT16
        return read_uint16(r)
    elseif value_type == GGUF_TYPE_INT16
        return read_int16(r)
    elseif value_type == GGUF_TYPE_UINT32
        return read_uint32(r)
    elseif value_type == GGUF_TYPE_INT32
        return read_int32(r)
    elseif value_type == GGUF_TYPE_FLOAT32
        return read_float32(r)
    elseif value_type == GGUF_TYPE_BOOL
        return read_bool(r)
    elseif value_type == GGUF_TYPE_STRING
        return read_string(r)
    elseif value_type == GGUF_TYPE_UINT64
        return read_uint64(r)
    elseif value_type == GGUF_TYPE_INT64
        return read_int64(r)
    elseif value_type == GGUF_TYPE_FLOAT64
        return read_float64(r)
    elseif value_type == GGUF_TYPE_ARRAY
        return parse_array_value(r)
    else
        @warn "Unknown GGUF value type $(value_type) at position $(current_position(r)); skipping."
        return nothing
    end
end

"""
    parse_array_value(r) -> Vector

Read an array KV entry:  element_type (UInt32) + count (UInt64) + elements.
"""
function parse_array_value(r::GGUFBinaryReader)
    elem_type = read_uint32(r)
    count     = read_uint64(r)
    n         = Int(count)
    out       = Vector{Any}(undef, n)
    for i in 1:n
        out[i] = parse_kv_value(r, elem_type)
    end
    return out
end

# ──────────────────────────────────────────────
# Full KV section
# ──────────────────────────────────────────────

"""
    parse_metadata(r, kv_count) -> Dict{String,Any}

Parse `kv_count` key-value pairs.  Unknown value types are stored as `nothing`
with a WARN log; the loop never aborts on an unknown type.
"""
function parse_metadata(r::GGUFBinaryReader, kv_count::UInt64)::Dict{String, Any}
    meta = Dict{String, Any}()
    for _ in 1:kv_count
        key        = read_string(r)
        value_type = read_uint32(r)
        value      = parse_kv_value(r, value_type)
        if value !== nothing
            meta[key] = value
        end
        @debug "  KV: $(key) [type=$(value_type)] = $(repr(value))"
    end
    return meta
end

# ──────────────────────────────────────────────
# Tensor info block
# ──────────────────────────────────────────────

"""
    parse_tensor_info(r, tensor_count) -> Dict{String, TensorInfo}

Parse `tensor_count` tensor info entries and return a name → TensorInfo index.
"""
function parse_tensor_info(r::GGUFBinaryReader, tensor_count::UInt64)::Dict{String, TensorInfo}
    index = Dict{String, TensorInfo}()
    for _ in 1:tensor_count
        name       = read_string(r)
        n_dims_raw = read_uint32(r)
        n_dims     = Int(n_dims_raw)

        if n_dims < 0 || n_dims > 8
            throw(GGUFTensorError(
                "Tensor '$(name)' has invalid n_dimensions=$(n_dims)"
            ))
        end

        dims_vec = Vector{Int}(undef, n_dims)
        for i in 1:n_dims
            dims_vec[i] = Int(read_uint64(r))
        end

        dtype  = read_uint32(r)
        offset = read_uint64(r)

        dims_tuple = Tuple(dims_vec)::Dims
        index[name] = TensorInfo(name, dims_tuple, dtype, offset)
        @debug "  Tensor: $(name), dims=$(dims_tuple), dtype=$(dtype), offset=$(offset)"
    end
    return index
end

# ──────────────────────────────────────────────
# Alignment helper
# ──────────────────────────────────────────────

"""
    compute_data_offset(raw_offset, alignment) -> UInt64

Round `raw_offset` up to the next multiple of `alignment`.
"""
function compute_data_offset(raw_offset::Int, alignment::UInt32)::UInt64
    a = Int(alignment)
    return UInt64(a * cld(raw_offset, a))
end

# ──────────────────────────────────────────────
# Top-level parse entry point (used by api.jl)
# ──────────────────────────────────────────────

"""
    parse_gguf(buf; validate=false) -> (metadata, tensor_index, data_offset)

Parse the entire GGUF structural header from byte buffer `buf` (a memory-mapped
`Vector{UInt8}`).  Returns:

- `metadata`     : Dict{String,Any}  — all KV pairs
- `tensor_index` : Dict{String,TensorInfo} — tensor info
- `data_offset`  : UInt64 — byte offset of the data section in `buf`
"""
function parse_gguf(buf::Vector{UInt8}; validate::Bool=false)
    r      = GGUFBinaryReader(buf)
    hdr    = parse_header(r)
    meta   = parse_metadata(r, hdr.metadata_kv_count)
    tindex = parse_tensor_info(r, hdr.tensor_count)

    # The data section begins at the next aligned position after tensor info.
    alignment = UInt32(get(meta, "general.alignment", GGUF_DEFAULT_ALIGNMENT))
    data_off  = compute_data_offset(current_position(r) - 1, alignment)

    if validate
        _validate_tensors(tindex, data_off, length(buf))
    end

    @info "GGUF loaded: $(hdr.tensor_count) tensors, $(hdr.metadata_kv_count) metadata entries, " *
          "data section @ byte $(data_off)"

    return meta, tindex, data_off
end

# ──────────────────────────────────────────────
# Validation helpers (used when validate=true)
# ──────────────────────────────────────────────

function _validate_tensors(
    tindex::Dict{String, TensorInfo},
    data_off::UInt64,
    file_size::Int,
)
    for (name, ti) in tindex
        nbytes = _tensor_nbytes(ti)
        start_byte = data_off + ti.offset
        end_byte   = start_byte + nbytes - 1
        if end_byte >= UInt64(file_size)
            throw(GGUFTensorError(
                "Tensor '$(name)' extends beyond file: " *
                "offset=$(start_byte), size=$(nbytes), file_size=$(file_size)"
            ))
        end
    end
end

"""Compute the byte size of a tensor from its TensorInfo (approximate; exact for unquantized)."""
function _tensor_nbytes(ti::TensorInfo)::UInt64
    n_elems = prod(ti.dims; init=1)
    bpe     = _bytes_per_element(ti.dtype)
    return UInt64(cld(n_elems * bpe, 1))
end

"""Return bytes-per-element (or bytes-per-block for block-quantized types)."""
function _bytes_per_element(dtype::UInt32)::Float64
    # Common cases; block-quantized types use a conservative estimate.
    dtype == 0  && return 4.0   # F32
    dtype == 1  && return 2.0   # F16
    dtype == 2  && return 0.5   # Q4_0  (4 bits/elem)
    dtype == 3  && return 0.625 # Q4_1
    dtype == 6  && return 0.625 # Q5_0
    dtype == 7  && return 0.75  # Q5_1
    dtype == 8  && return 1.0   # Q8_0
    dtype == 9  && return 1.0   # Q8_1
    dtype == 10 && return 0.25  # Q2_K
    dtype == 11 && return 0.375 # Q3_K
    dtype == 12 && return 0.5   # Q4_K
    dtype == 13 && return 0.625 # Q5_K
    dtype == 14 && return 0.75  # Q6_K
    dtype == 15 && return 1.0   # Q8_K
    dtype == 30 && return 2.0   # BF16
    return 1.0                  # conservative fallback
end
