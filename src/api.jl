"""
Public API for Ekatra.jl.
"""

# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

"""Throw GGUFClosedError if the model has been closed."""
@inline function _check_open(model::GGUFModel)
    if model.is_closed
        throw(GGUFClosedError("Operation on a closed GGUFModel ('$(model.path)')"))
    end
end

# ──────────────────────────────────────────────
# load_gguf
# ──────────────────────────────────────────────

"""
    load_gguf(path; lazy_tensors=true, load_tokenizer=true, validate=false) -> GGUFModel

Open and parse a GGUF model file at `path`.

Keyword arguments:
- `lazy_tensors`    : when `true` (default), tensor data is accessed on demand
                      via zero-copy pointers; no data is read into Julia memory.
- `load_tokenizer`  : when `true` (default), attempt to build a tokenizer from
                      the embedded metadata.
- `validate`        : when `true`, verify alignment and tensor bounds after parsing.

Returns a `GGUFModel` handle.  Call `close(model)` when done.
"""
function load_gguf(
    path::AbstractString;
    lazy_tensors::Bool   = true,
    load_tokenizer::Bool = true,
    validate::Bool       = false,
)::GGUFModel
    abspath_str = abspath(String(path))
    isfile(abspath_str) || throw(GGUFIOError("File not found: $(abspath_str)"))

    @debug "Opening GGUF file: $(abspath_str)"

    # Memory-map the entire file (read-only, no copy into Julia heap)
    buf = open(abspath_str, "r") do fh
        Mmap.mmap(fh, Vector{UInt8}, (filesize(fh),); grow=false)
    end

    meta, tindex, data_off = parse_gguf(buf; validate=validate)

    tok = nothing
    if load_tokenizer
        tok = _try_build_tokenizer(meta)
    end

    model = GGUFModel(abspath_str, meta, tindex, data_off, buf, false, tok)

    # Register a finalizer so the mmap is reclaimed even if the user forgets to
    # call close().  This is a safety net only.
    finalizer(model) do m
        if !m.is_closed
            finalize(m.mmap_handle)
            m.is_closed = true
        end
    end

    return model
end

# ──────────────────────────────────────────────
# list_tensors
# ──────────────────────────────────────────────

"""
    list_tensors(model) -> Vector{String}

Return the names of all tensors in the model, in no particular order.
"""
function list_tensors(model::GGUFModel)::Vector{String}
    _check_open(model)
    return collect(keys(model.tensor_index))
end

# ──────────────────────────────────────────────
# tensor_info
# ──────────────────────────────────────────────

"""
    tensor_info(model, name) -> TensorInfo

Return the `TensorInfo` for the named tensor.

Throws `GGUFTensorError` if no such tensor exists.
"""
function tensor_info(model::GGUFModel, name::AbstractString)::TensorInfo
    _check_open(model)
    ti = get(model.tensor_index, String(name), nothing)
    ti === nothing && throw(GGUFTensorError("Tensor '$(name)' not found in model"))
    return ti
end

# ──────────────────────────────────────────────
# get_tensor
# ──────────────────────────────────────────────

"""
    get_tensor(model, name) -> QuantizedTensor{Q}

Return a zero-copy `QuantizedTensor` view for the named tensor.

The returned tensor borrows the memory-mapped buffer; it must not be used after
`close(model)`.

Throws `GGUFTensorError` if the tensor does not exist or is out of bounds.
"""
function get_tensor(model::GGUFModel, name::AbstractString)::AbstractGGUFTensor
    ti = tensor_info(model, name)
    return make_tensor(model, ti)
end

# ──────────────────────────────────────────────
# get_metadata
# ──────────────────────────────────────────────

"""
    get_metadata(model, key) -> Any

Retrieve a metadata value by its GGUF key.

Returns `nothing` if the key is absent (does NOT throw).
"""
function get_metadata(model::GGUFModel, key::AbstractString)
    _check_open(model)
    return get(model.metadata, String(key), nothing)
end

# ──────────────────────────────────────────────
# get_tokenizer
# ──────────────────────────────────────────────

"""
    get_tokenizer(model) -> AbstractTokenizer

Return the tokenizer associated with `model`.

Throws `GGUFTokenizerError` if no tokenizer was loaded (e.g. the file does not
contain tokenizer metadata, or `load_tokenizer=false` was passed to `load_gguf`).
"""
function get_tokenizer(model::GGUFModel)::AbstractTokenizer
    _check_open(model)
    model.tokenizer === nothing &&
        throw(GGUFTokenizerError("No tokenizer available for model '$(model.path)'"))
    return model.tokenizer
end

# ──────────────────────────────────────────────
# encode / decode (delegated to tokenizer)
# ──────────────────────────────────────────────

"""
    encode(tokenizer::AbstractTokenizer, text::String) -> Vector{Int}

Encode `text` into token IDs using `tokenizer`.
"""
encode(tok::AbstractTokenizer, text::String) =
    error("encode not implemented for $(typeof(tok))")

"""
    decode(tokenizer::AbstractTokenizer, tokens::Vector{Int}) -> String

Decode `tokens` back to a string using `tokenizer`.
"""
decode(tok::AbstractTokenizer, tokens::Vector{Int}) =
    error("decode not implemented for $(typeof(tok))")

# ──────────────────────────────────────────────
# close
# ──────────────────────────────────────────────

"""
    close(model::GGUFModel)

Release all resources held by `model`, including the memory-mapped file.
After calling `close`, any previously returned `QuantizedTensor` pointers are
invalid and must not be dereferenced.
"""
function Base.close(model::GGUFModel)
    model.is_closed && return
    finalize(model.mmap_handle)
    model.is_closed = true
    @debug "Closed GGUFModel: $(model.path)"
    return nothing
end

# ──────────────────────────────────────────────
# isopen
# ──────────────────────────────────────────────

"""
    isopen(model::GGUFModel) -> Bool

Return `true` if the model file is still open.
"""
Base.isopen(model::GGUFModel) = !model.is_closed

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────

function Base.show(io::IO, model::GGUFModel)
    status = model.is_closed ? "closed" : "open"
    n_tensors = length(model.tensor_index)
    n_meta    = length(model.metadata)
    print(io, "GGUFModel($(status), path=\"$(model.path)\", " *
              "tensors=$(n_tensors), metadata=$(n_meta))")
end

# ──────────────────────────────────────────────
# Internal: tokenizer construction
# ──────────────────────────────────────────────

function _try_build_tokenizer(meta::Dict{String, Any})
    # First, try the Tokenizers.jl bridge (extension, may be nothing)
    tok = build_tokenizer(meta)
    tok !== nothing && return tok

    # Fallback: pure-Julia native tokenizer
    try
        return GGUFNativeTokenizer(meta)
    catch e
        if e isa GGUFTokenizerError
            @warn "Could not build native tokenizer: $(e.msg)"
            return nothing
        end
        rethrow(e)
    end
end
