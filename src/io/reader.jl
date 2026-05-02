"""
GGUFBinaryReader — low-level, little-endian binary reader for GGUF files.

All reads are bounds-checked and advance an internal cursor.
Strings are validated as UTF-8 before being returned.
"""

# ──────────────────────────────────────────────
# Reader struct
# ──────────────────────────────────────────────

"""
Stateful cursor over a byte buffer (typically a memory-mapped file).

Thread-safety: a single `GGUFBinaryReader` must not be shared across threads
because the `pos` field is mutable. Create one reader per thread or protect
access externally.
"""
mutable struct GGUFBinaryReader
    buf::Vector{UInt8}   # the underlying Mmap buffer (no copy)
    pos::Int             # 1-based current position
    len::Int             # total byte length
end

"""
    GGUFBinaryReader(buf::Vector{UInt8})

Construct a reader positioned at the start of `buf`.
"""
GGUFBinaryReader(buf::Vector{UInt8}) = GGUFBinaryReader(buf, 1, length(buf))

# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

"""Ensure `n` bytes are available at the current cursor position."""
@inline function _check_available(r::GGUFBinaryReader, n::Int)
    if r.pos + n - 1 > r.len
        throw(GGUFIOError(
            "Unexpected end of file: need $(n) byte(s) at position $(r.pos), " *
            "but only $(r.len - r.pos + 1) byte(s) remain."
        ))
    end
end

# ──────────────────────────────────────────────
# Primitive readers (little-endian)
# ──────────────────────────────────────────────

"""Read a single `UInt8` from the buffer."""
function read_uint8(r::GGUFBinaryReader)::UInt8
    _check_available(r, 1)
    v = r.buf[r.pos]
    r.pos += 1
    return v
end

"""Read a single `Int8` from the buffer."""
function read_int8(r::GGUFBinaryReader)::Int8
    return reinterpret(Int8, read_uint8(r))
end

"""Read a little-endian `UInt16`."""
function read_uint16(r::GGUFBinaryReader)::UInt16
    _check_available(r, 2)
    v = UInt16(r.buf[r.pos]) | (UInt16(r.buf[r.pos+1]) << 8)
    r.pos += 2
    return v
end

"""Read a little-endian `Int16`."""
function read_int16(r::GGUFBinaryReader)::Int16
    return reinterpret(Int16, read_uint16(r))
end

"""Read a little-endian `UInt32`."""
function read_uint32(r::GGUFBinaryReader)::UInt32
    _check_available(r, 4)
    v = UInt32(r.buf[r.pos])       |
        (UInt32(r.buf[r.pos+1]) << 8)  |
        (UInt32(r.buf[r.pos+2]) << 16) |
        (UInt32(r.buf[r.pos+3]) << 24)
    r.pos += 4
    return v
end

"""Read a little-endian `Int32`."""
function read_int32(r::GGUFBinaryReader)::Int32
    return reinterpret(Int32, read_uint32(r))
end

"""Read a little-endian `UInt64`."""
function read_uint64(r::GGUFBinaryReader)::UInt64
    _check_available(r, 8)
    v = UInt64(r.buf[r.pos])       |
        (UInt64(r.buf[r.pos+1]) << 8)  |
        (UInt64(r.buf[r.pos+2]) << 16) |
        (UInt64(r.buf[r.pos+3]) << 24) |
        (UInt64(r.buf[r.pos+4]) << 32) |
        (UInt64(r.buf[r.pos+5]) << 40) |
        (UInt64(r.buf[r.pos+6]) << 48) |
        (UInt64(r.buf[r.pos+7]) << 56)
    r.pos += 8
    return v
end

"""Read a little-endian `Int64`."""
function read_int64(r::GGUFBinaryReader)::Int64
    return reinterpret(Int64, read_uint64(r))
end

"""Read a little-endian IEEE-754 `Float32`."""
function read_float32(r::GGUFBinaryReader)::Float32
    return reinterpret(Float32, read_uint32(r))
end

"""Read a little-endian IEEE-754 `Float64`."""
function read_float64(r::GGUFBinaryReader)::Float64
    return reinterpret(Float64, read_uint64(r))
end

"""Read a `Bool` stored as a single byte (`0x00` = false, anything else = true)."""
function read_bool(r::GGUFBinaryReader)::Bool
    return read_uint8(r) != 0x00
end

# ──────────────────────────────────────────────
# String reader
# ──────────────────────────────────────────────

"""
    read_string(r) -> String

Read a length-prefixed UTF-8 string.  The wire format is:

    UInt64  length_in_bytes
    UInt8[length_in_bytes]  utf8_bytes

Throws:
- `GGUFIOError`     if the buffer is too short.
- `GGUFStringError` if the byte sequence is not valid UTF-8.
"""
function read_string(r::GGUFBinaryReader)::String
    nbytes = read_uint64(r)

    # Guard against obviously corrupt values (> 256 MiB per key)
    if nbytes > 0x10000000
        throw(GGUFStringError(
            "Implausibly large string length: $(nbytes) bytes at position $(r.pos - 8)"
        ))
    end

    n = Int(nbytes)
    _check_available(r, n)

    raw = r.buf[r.pos : r.pos + n - 1]
    r.pos += n

    # UTF-8 validation — String() accepts arbitrary bytes; we must check explicitly.
    s = String(raw)
    if !isvalid(s)
        throw(GGUFStringError(
            "Invalid UTF-8 string at byte offset $(r.pos - n - 8)"
        ))
    end
    return s
end

# ──────────────────────────────────────────────
# Raw bytes reader (zero-copy slice helper)
# ──────────────────────────────────────────────

"""
    current_position(r) -> Int

Return the current 1-based cursor position.
"""
current_position(r::GGUFBinaryReader) = r.pos

"""
    skip_bytes(r, n)

Advance the cursor by `n` bytes without reading them.
"""
function skip_bytes(r::GGUFBinaryReader, n::Int)
    _check_available(r, n)
    r.pos += n
end
