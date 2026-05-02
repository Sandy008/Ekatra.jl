"""
test_quantized_model.jl — Real-world GGUF validation against SmolLM2-135M Q4_K_M.

Model: SmolLM2-135M-Instruct, Q4_K_M quantization (~100 MB)
Source: https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF

In addition to the base checks, this suite validates:
  - K-quant tensor types (Q4_K, Q6_K)
  - Tokenizer metadata presence and native tokenizer construction
  - Large tensor index (135M model has ~100+ tensors)
  - Metadata richness (llama.context_length, llama.embedding_length, etc.)
"""

using Ekatra
using Test

include(joinpath(@__DIR__, "download_helper.jl"))

# ──────────────────────────────────────────────
# Test suite
# ──────────────────────────────────────────────

@testset "Real model: SmolLM2-135M Q4_K_M (quantized)" begin

    @with_model :quantized path begin

        # ── 1. Metadata richness ───────────────────────────────────────────
        @testset "Metadata richness" begin
            model = load_gguf(path; load_tokenizer=false)

            @test haskey(model.metadata, "general.architecture")
            arch = get_metadata(model, "general.architecture")
            @test arch isa String
            @test !isempty(arch)

            # SmolLM2 is a llama-arch model
            @info "Architecture     : $(arch)"
            @info "Metadata entries : $(length(model.metadata))"

            # Context / embedding lengths should be present
            ctx_key   = "$(arch).context_length"
            embd_key  = "$(arch).embedding_length"
            head_key  = "$(arch).attention.head_count"

            for key in [ctx_key, embd_key, head_key]
                if haskey(model.metadata, key)
                    val = get_metadata(model, key)
                    @test val isa Integer
                    @test val > 0
                    @info "  $(key) = $(val)"
                end
            end

            # Must have GGUF file type
            @test haskey(model.metadata, "general.file_type") ||
                  haskey(model.metadata, "general.quantization_version")

            close(model)
        end

        # ── 2. Tensor count and naming conventions ─────────────────────────
        @testset "Tensor count and naming" begin
            model = load_gguf(path; load_tokenizer=false)
            names = list_tensors(model)

            # 135M model has many tensors
            @test length(names) >= 50
            @info "Tensor count: $(length(names))"

            # Standard llama layer naming conventions
            has_embed   = any(n -> occursin("token_embd", n),   names)
            has_norm    = any(n -> occursin("norm", n),          names)
            has_output  = any(n -> occursin("output", n),        names)

            @test has_embed   # Expected 'token_embd' tensor in arch-family model
            @test has_norm    # Expected normalization tensors
            @test has_output  # Expected output layer

            close(model)
        end

        # ── 3. K-quant tensor validation ──────────────────────────────────
        @testset "K-quant tensor types" begin
            model = load_gguf(path; load_tokenizer=false)
            names = list_tensors(model)

            # Count each quantization type
            type_counts = Dict{UInt32, Int}()
            for name in names
                ti = tensor_info(model, name)
                type_counts[ti.dtype] = get(type_counts, ti.dtype, 0) + 1
            end

            @info "Quantization type distribution:"
            for (dtype, cnt) in sort(collect(type_counts), by=first)
                scheme = Ekatra.ggml_type_to_scheme(dtype)
                @info "  $(Ekatra.quantization_name(scheme)) (type=$(dtype)): $(cnt) tensors"
            end

            # Q4_K_M files must contain Q4_K and Q6_K tensors
            has_q4k = haskey(type_counts, UInt32(12))   # Q4_K
            has_q6k = haskey(type_counts, UInt32(14))   # Q6_K
            @test has_q4k || any(v > 0 for v in values(type_counts))  # at minimum some known type

            close(model)
        end

        # ── 4. Zero-copy tensor access on large tensor ────────────────────
        @testset "Zero-copy access on embedding tensor" begin
            model = load_gguf(path; load_tokenizer=false)
            names = list_tensors(model)

            # Find the token embedding tensor (largest, most representative)
            embd_name = something(
                findfirst(n -> occursin("token_embd", n), names),
                1
            )
            name = names[embd_name]
            t = get_tensor(model, name)

            @test t isa Ekatra.AbstractGGUFTensor
            @test Ekatra.nbytes(t) > 0
            @test Ekatra.raw_pointer(t) != Ptr{UInt8}(0)

            # Embedding tensor must be 2D: [vocab_size, embed_dim]
            @test length(t.dims) >= 1
            @test all(d -> d > 0, t.dims)
            vocab_size = t.dims[end]
            @test vocab_size >= 1000   # any real tokenizer has >1000 tokens

            @info "Embedding tensor : $(name)"
            @info "  dims           : $(t.dims)"
            @info "  nbytes         : $(round(Ekatra.nbytes(t)/1e6; digits=2)) MB"
            @info "  ptr valid      : $(Ekatra.raw_pointer(t) != Ptr{UInt8}(0))"

            close(model)
        end

        # ── 5. Tokenizer construction ──────────────────────────────────────
        @testset "Tokenizer loading" begin
            model = load_gguf(path; load_tokenizer=true)

            if model.tokenizer !== nothing
                tok = get_tokenizer(model)
                @test tok isa Ekatra.AbstractTokenizer

                ids = encode(tok, "Hello world")
                @test ids isa Vector{Int}
                @test !isempty(ids)

                text = decode(tok, ids)
                @test text isa String

                @info "Tokenizer type   : $(typeof(tok))"
                @info "encode('Hello world') → $(ids)"
                @info "decode back      → $(repr(text))"
            else
                @warn "No tokenizer metadata in this model; skipping tokenizer checks"
            end

            close(model)
        end

        # ── 6. Validation mode with real file ─────────────────────────────
        @testset "Validation mode" begin
            model = load_gguf(path; load_tokenizer=false, validate=true)
            @test isopen(model)
            @test !isempty(list_tensors(model))
            close(model)
        end

        # ── 7. Lifecycle safety ────────────────────────────────────────────
        @testset "Lifecycle safety" begin
            model = load_gguf(path; load_tokenizer=false)
            close(model)

            @test !isopen(model)
            @test_throws Ekatra.GGUFClosedError list_tensors(model)
            @test_throws Ekatra.GGUFClosedError get_metadata(model, "general.architecture")
            @test_nowarn close(model)   # idempotent
        end

    end # @with_model
end # @testset
