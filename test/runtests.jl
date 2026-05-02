using Ekatra
using Test

# ──────────────────────────────────────────────
# Helper: build a minimal valid GGUF byte buffer
# ──────────────────────────────────────────────

"""
Construct a minimal GGUF v3 binary in memory with:
  - 1 metadata KV pair  (string "general.name" = "test")
  - 1 tensor            (name="weight", shape=(4,), F32, offset=0)
"""
function make_minimal_gguf(; tensor_name="weight", dtype=UInt32(0))
    buf = UInt8[]

    function write_le!(v::UInt32)
        push!(buf, (v >> 0) & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff)
    end
    function write_le!(v::UInt64)
        for i in 0:7; push!(buf, (v >> (8*i)) & 0xff); end
    end
    function write_str!(s::String)
        bs = codeunits(s)
        write_le!(UInt64(length(bs)))
        append!(buf, bs)
    end

    # magic "GGUF"
    append!(buf, UInt8[0x47, 0x47, 0x55, 0x46])
    write_le!(UInt32(3))          # version
    write_le!(UInt64(1))          # tensor_count
    write_le!(UInt64(1))          # metadata_kv_count

    # KV: "general.name" (type=STRING=8) = "test"
    write_str!("general.name")
    write_le!(UInt32(8))          # GGUF_TYPE_STRING
    write_str!("test")

    # Tensor info: name, n_dims=1, dim=4, dtype, offset=0
    write_str!(tensor_name)
    write_le!(UInt32(1))          # n_dimensions
    write_le!(UInt64(4))          # dim[0]
    write_le!(dtype)              # dtype (F32=0)
    write_le!(UInt64(0))          # offset

    # Pad to alignment boundary (32 bytes default)
    alignment = 32
    current   = length(buf)
    padded    = alignment * cld(current, alignment)
    append!(buf, zeros(UInt8, padded - current))

    # Data section: 4 × F32 = 16 bytes
    for i in 1:4
        fi = Float32(i)
        ui = reinterpret(UInt32, fi)
        push!(buf, (ui >> 0) & 0xff, (ui >> 8) & 0xff, (ui >> 16) & 0xff, (ui >> 24) & 0xff)
    end

    return buf
end

# ──────────────────────────────────────────────
# GGUFBinaryReader tests
# ──────────────────────────────────────────────

@testset "GGUFBinaryReader" begin
    # 15 bytes total: 1 (uint8) + 2 (uint16) + 4 (uint32) + 8 (uint64) = 15
    buf = UInt8[0x01, 0x02, 0x03, 0x04,
                0x05, 0x06, 0x07, 0x08,
                0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]
    r = Ekatra.GGUFBinaryReader(buf)

    @testset "uint8" begin
        @test Ekatra.read_uint8(r) == 0x01
    end

    @testset "uint16 little-endian" begin
        # consume bytes 2,3 → 0x0302
        @test Ekatra.read_uint16(r) == 0x0302
    end

    @testset "uint32 little-endian" begin
        # consume bytes 4,5,6,7 → 0x07060504
        @test Ekatra.read_uint32(r) == 0x07060504
    end

    @testset "uint64 little-endian" begin
        # consume bytes 8..15 → 0x0f0e0d0c0b0a0908
        @test Ekatra.read_uint64(r) == 0x0f0e0d0c0b0a0908
    end

    @testset "bounds check" begin
        @test_throws Ekatra.GGUFIOError Ekatra.read_uint8(r)
    end

    @testset "float32" begin
        # 1.0f0 in LE bytes
        fbytes = reinterpret(UInt8, [1.0f0])
        r2 = Ekatra.GGUFBinaryReader(Vector{UInt8}(fbytes))
        @test Ekatra.read_float32(r2) == 1.0f0
    end

    @testset "bool" begin
        r3 = Ekatra.GGUFBinaryReader(UInt8[0x00, 0x01, 0xff])
        @test Ekatra.read_bool(r3) == false
        @test Ekatra.read_bool(r3) == true
        @test Ekatra.read_bool(r3) == true
    end
end

@testset "read_string" begin
    function make_str_buf(s)
        bs = codeunits(s)
        n  = UInt64(length(bs))
        out = UInt8[]
        for i in 0:7; push!(out, (n >> (8*i)) & 0xff); end
        append!(out, bs)
        return out
    end

    r = Ekatra.GGUFBinaryReader(make_str_buf("hello"))
    @test Ekatra.read_string(r) == "hello"

    r2 = Ekatra.GGUFBinaryReader(make_str_buf("αβγ"))
    @test Ekatra.read_string(r2) == "αβγ"

    # Invalid UTF-8
    invalid = UInt8[]
    n = UInt64(2)
    for i in 0:7; push!(invalid, (n >> (8*i)) & 0xff); end
    push!(invalid, 0xff, 0xfe)   # not valid UTF-8
    r3 = Ekatra.GGUFBinaryReader(invalid)
    @test_throws Ekatra.GGUFStringError Ekatra.read_string(r3)
end

# ──────────────────────────────────────────────
# Parser tests
# ──────────────────────────────────────────────

@testset "parse_gguf round-trip" begin
    buf = make_minimal_gguf()
    meta, tindex, data_off = Ekatra.parse_gguf(buf)

    @test meta["general.name"] == "test"
    @test haskey(tindex, "weight")
    @test tindex["weight"].dims == (4,)
    @test tindex["weight"].dtype == UInt32(0)
    @test tindex["weight"].offset == UInt64(0)
    @test data_off >= UInt64(0)
end

@testset "parse_gguf validate=true" begin
    buf = make_minimal_gguf()
    # Should not throw for a well-formed buffer
    meta, tindex, data_off = Ekatra.parse_gguf(buf; validate=true)
    @test !isempty(meta)
    @test !isempty(tindex)
end

@testset "parse_gguf bad magic" begin
    buf = make_minimal_gguf()
    buf[1] = 0x00   # corrupt the magic
    @test_throws Ekatra.GGUFIOError Ekatra.parse_gguf(buf)
end

# ──────────────────────────────────────────────
# Quantization mapping tests
# ──────────────────────────────────────────────

@testset "ggml_type_to_scheme" begin
    @test Ekatra.ggml_type_to_scheme(UInt32(0))  isa Ekatra.F32
    @test Ekatra.ggml_type_to_scheme(UInt32(1))  isa Ekatra.F16
    @test Ekatra.ggml_type_to_scheme(UInt32(12)) isa Ekatra.Q4_K
    @test Ekatra.ggml_type_to_scheme(UInt32(99)) isa Ekatra.UnknownQuantization
    @test Ekatra.ggml_type_to_scheme(UInt32(99)).type_id == UInt32(99)
end

@testset "quantization_name" begin
    @test Ekatra.quantization_name(Ekatra.F32())  == "F32"
    @test Ekatra.quantization_name(Ekatra.Q4_K()) == "Q4_K"
    @test Ekatra.quantization_name(Ekatra.UnknownQuantization(UInt32(42))) == "Unknown(42)"
end

# ──────────────────────────────────────────────
# load_gguf via temp file
# ──────────────────────────────────────────────

@testset "load_gguf" begin
    buf = make_minimal_gguf()
    tmp = tempname() * ".gguf"
    try
        write(tmp, buf)

        @testset "basic load" begin
            model = load_gguf(tmp; load_tokenizer=false)
            @test isopen(model)
            @test get_metadata(model, "general.name") == "test"
            @test "weight" in list_tensors(model)
            close(model)
            @test !isopen(model)
        end

        @testset "double close is safe" begin
            model = load_gguf(tmp; load_tokenizer=false)
            close(model)
            @test_nowarn close(model)   # idempotent
        end

        @testset "access after close throws" begin
            model = load_gguf(tmp; load_tokenizer=false)
            close(model)
            @test_throws Ekatra.GGUFClosedError list_tensors(model)
        end

        @testset "get_tensor" begin
            model = load_gguf(tmp; load_tokenizer=false)
            t = get_tensor(model, "weight")
            @test t isa Ekatra.QuantizedTensor{Ekatra.F32}
            @test t.dims == (4,)
            @test Ekatra.nbytes(t) == 16
            close(model)
        end

        @testset "missing tensor" begin
            model = load_gguf(tmp; load_tokenizer=false)
            @test_throws Ekatra.GGUFTensorError get_tensor(model, "nonexistent")
            close(model)
        end

        @testset "validate mode" begin
            model = load_gguf(tmp; load_tokenizer=false, validate=true)
            @test isopen(model)
            close(model)
        end

    finally
        isfile(tmp) && rm(tmp)
    end
end

@testset "load_gguf file not found" begin
    @test_throws Ekatra.GGUFIOError load_gguf("/nonexistent/path/model.gguf")
end

# ──────────────────────────────────────────────
# GGUFNativeTokenizer tests
# ──────────────────────────────────────────────

@testset "GGUFNativeTokenizer" begin
    meta = Dict{String, Any}(
        "tokenizer.ggml.tokens"          => ["<unk>", "<s>", "</s>", "▁hello", "▁world", "!"],
        "tokenizer.ggml.bos_token_id"    => 1,
        "tokenizer.ggml.eos_token_id"    => 2,
        "tokenizer.ggml.unknown_token_id"=> 0,
        "tokenizer.ggml.model"           => "llama",
    )
    tok = Ekatra.GGUFNativeTokenizer(meta)

    @testset "vocab size" begin
        @test length(tok.vocab) == 6
    end

    @testset "exact encode" begin
        ids = encode(tok, "▁hello")
        @test ids == [3]   # 0-based index of "▁hello"
    end

    @testset "decode" begin
        txt = decode(tok, [1, 3, 4, 2])   # <s> hello world </s>
        @test occursin("hello", txt)
        @test occursin("world", txt)
    end

    @testset "unknown token" begin
        ids = encode(tok, "xyz")
        @test all(==(0), ids)
    end

    @testset "missing vocab throws" begin
        @test_throws Ekatra.GGUFTokenizerError Ekatra.GGUFNativeTokenizer(Dict{String,Any}())
    end
end

# ──────────────────────────────────────────────
# Error types
# ──────────────────────────────────────────────

@testset "Error types" begin
    for E in [Ekatra.GGUFVersionError, Ekatra.GGUFTensorError,
              Ekatra.GGUFTokenizerError, Ekatra.GGUFStringError,
              Ekatra.GGUFIOError, Ekatra.GGUFClosedError]
        e = E("test message")
        @test e isa Ekatra.GGUFError
        @test occursin("test message", sprint(showerror, e))
    end
end
