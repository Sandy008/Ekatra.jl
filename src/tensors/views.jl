"""
Zero-copy tensor views into the memory-mapped GGUF data section.
"""

# ──────────────────────────────────────────────
# Constructor helpers
# ──────────────────────────────────────────────

"""
    make_tensor(model, ti) -> QuantizedTensor{Q}

Build a `QuantizedTensor` for the given `TensorInfo` using the mmap buffer
from `model`.  The type parameter `Q` is resolved at construction time via
`ggml_type_to_scheme`.

Throws `GGUFClosedError` if the model has been closed.
Throws `GGUFTensorError` if the tensor extends beyond the buffer.
"""
function make_tensor(model::GGUFModel, ti::TensorInfo)::AbstractGGUFTensor
    _check_open(model)

    scheme = ggml_type_to_scheme(ti.dtype)
    abs_offset = model.data_offset + ti.offset   # byte offset into mmap buffer

    n_elems = prod(ti.dims; init=1)
    nbytes  = _tensor_byte_size(ti.dtype, n_elems)

    if abs_offset + UInt64(nbytes) > UInt64(length(model.mmap_handle))
        throw(GGUFTensorError(
            "Tensor '$(ti.name)' is out of bounds: " *
            "offset=$(abs_offset), size=$(nbytes), file_size=$(length(model.mmap_handle))"
        ))
    end

    ptr = pointer(model.mmap_handle, Int(abs_offset) + 1)
    return _make_typed_tensor(scheme, ptr, ti.dims, nbytes, ti.offset)
end

# Internal dispatch so the type parameter Q is known at compile time.
function _make_typed_tensor(
    ::Q,
    ptr::Ptr{UInt8},
    dims::Dims,
    nbytes::Int,
    offset::UInt64,
) where {Q<:QuantizationScheme}
    return QuantizedTensor{Q}(ptr, dims, nbytes, offset)
end

# ──────────────────────────────────────────────
# Byte-size computation
# ──────────────────────────────────────────────

"""
    _tensor_byte_size(dtype, n_elems) -> Int

Compute the number of bytes occupied by `n_elems` elements of the given
GGML dtype.  Block-quantized types use the standard GGML block sizes.
"""
function _tensor_byte_size(dtype::UInt32, n_elems::Int)::Int
    # Unquantized
    dtype == 0  && return n_elems * 4           # F32
    dtype == 1  && return n_elems * 2           # F16
    dtype == 30 && return n_elems * 2           # BF16
    dtype == 24 && return n_elems               # I8
    dtype == 25 && return n_elems * 2           # I16
    dtype == 26 && return n_elems * 4           # I32

    # Block-quantized — GGML block size is 256 elements for *_K and 32 for Q4_0 etc.
    # Sizes taken from ggml/src/ggml.c
    dtype == 2  && return cld(n_elems, 32)  * 18   # Q4_0:  32 elems → 18 bytes
    dtype == 3  && return cld(n_elems, 32)  * 20   # Q4_1:  32 → 20
    dtype == 6  && return cld(n_elems, 32)  * 22   # Q5_0:  32 → 22
    dtype == 7  && return cld(n_elems, 32)  * 24   # Q5_1:  32 → 24
    dtype == 8  && return cld(n_elems, 32)  * 34   # Q8_0:  32 → 34
    dtype == 9  && return cld(n_elems, 32)  * 40   # Q8_1:  32 → 40
    dtype == 10 && return cld(n_elems, 256) * 84   # Q2_K
    dtype == 11 && return cld(n_elems, 256) * 110  # Q3_K
    dtype == 12 && return cld(n_elems, 256) * 144  # Q4_K
    dtype == 13 && return cld(n_elems, 256) * 176  # Q5_K
    dtype == 14 && return cld(n_elems, 256) * 210  # Q6_K
    dtype == 15 && return cld(n_elems, 256) * 288  # Q8_K

    # IQ types — conservative estimate: 1 byte/element
    return n_elems
end

# ──────────────────────────────────────────────
# Pointer access (with lifetime guard)
# ──────────────────────────────────────────────

"""
    raw_pointer(t::QuantizedTensor) -> Ptr{UInt8}

Return the raw pointer to the tensor data.

⚠️  The pointer is only valid while the parent `GGUFModel` is alive and open.
"""
raw_pointer(t::QuantizedTensor) = t.ptr

"""
    nbytes(t::QuantizedTensor) -> Int

Return the number of bytes occupied by the tensor data.
"""
nbytes(t::QuantizedTensor) = t.nbytes

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────

function Base.show(io::IO, t::QuantizedTensor{Q}) where {Q}
    print(io, "QuantizedTensor{$(Q)}(dims=$(t.dims), nbytes=$(t.nbytes), " *
              "offset=0x$(string(t.offset; base=16)))")
end
