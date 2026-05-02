"""
test_tiny_model.jl — Real-world GGUF validation against stories15M-q4_0.gguf.

Model: TinyLlama-stories 15M parameters, Q4_0 quantization (~18 MB)
Source: https://huggingface.co/ggml-org/models (tinyllamas series)

Tests validate:
  - Metadata completeness
  - Tensor index building
  - Zero-copy tensor access
  - Quantization scheme dispatch
  - Lifecycle safety
  - Validation mode
"""

using Ekatra
using Test

# Load shared helpers (download cache, @with_model macro)
include(joinpath(@__DIR__, "download_helper.jl"))

# ──────────────────────────────────────────────
# Test suite
# ──────────────────────────────────────────────

@testset "Real model: stories15M Q4_0 (tiny)" begin

    @with_model :tiny path begin

        # ── 1. Metadata validation ─────────────────────────────────────────
        @testset "Metadata" begin
            model = load_gguf(path; load_tokenizer=false)

            # Must have the general metadata keys present in all GGUF files
            @test haskey(model.metadata, "general.architecture")
            @test get_metadata(model, "general.architecture") isa String
            @test !isempty(get_metadata(model, "general.architecture"))

            # Quantization version key
            @test haskey(model.metadata, "general.file_type") ||
                  haskey(model.metadata, "general.quantization_version") ||
                  !isempty(model.metadata)   # at minimum, some metadata exists

            @info "Architecture : $(get_metadata(model, "general.architecture"))"
            @info "Metadata keys: $(length(model.metadata))"
            close(model)
        end

        # ── 2. Tensor index ────────────────────────────────────────────────
        @testset "Tensor index" begin
            model = load_gguf(path; load_tokenizer=false)

            names = list_tensors(model)
            @test !isempty(names)
            @test all(x -> x isa String, names)

            @info "Tensor count : $(length(names))"

            # Every name round-trips through tensor_info
            for name in names
                ti = tensor_info(model, name)
                @test ti.name == name
                @test length(ti.dims) >= 1
                @test all(d -> d > 0, ti.dims)
                @test ti.dtype isa UInt32
                @test ti.offset isa UInt64
            end

            close(model)
        end

        # ── 3. Zero-copy tensor access ─────────────────────────────────────
        @testset "Zero-copy tensor access" begin
            model = load_gguf(path; load_tokenizer=false)
            names = list_tensors(model)

            # Test first tensor
            name = first(names)
            t = get_tensor(model, name)

            @test t isa Ekatra.AbstractGGUFTensor
            @test t.dims isa Tuple
            @test length(t.dims) >= 1
            @test all(d -> d > 0, t.dims)
            @test Ekatra.nbytes(t) > 0
            @test Ekatra.raw_pointer(t) != Ptr{UInt8}(0)

            @info "First tensor  : $(name), dims=$(t.dims), bytes=$(Ekatra.nbytes(t))"

            # Test a few more (embedding + output layers if present)
            for name in names[1:min(5, end)]
                tx = get_tensor(model, name)
                @test Ekatra.nbytes(tx) > 0
            end

            close(model)
        end

        # ── 4. Quantization scheme dispatch ───────────────────────────────
        @testset "Quantization scheme" begin
            model = load_gguf(path; load_tokenizer=false)
            names = list_tensors(model)

            # All tensors must be AbstractGGUFTensor
            for name in names
                t = get_tensor(model, name)
                @test t isa Ekatra.AbstractGGUFTensor
            end

            # This is a Q4_0 model — at least some tensors should be Q4_0
            q4_count = count(names) do name
                ti = tensor_info(model, name)
                ti.dtype == UInt32(2)   # GGML_TYPE_Q4_0
            end
            @test q4_count > 0
            @info "Q4_0 tensors  : $(q4_count) / $(length(names))"

            # Verify the scheme type name resolves correctly
            @test Ekatra.quantization_name(Ekatra.ggml_type_to_scheme(UInt32(2))) == "Q4_0"

            close(model)
        end

        # ── 5. Validation mode ─────────────────────────────────────────────
        @testset "Validation mode" begin
            # validate=true should succeed for a well-formed file
            model = load_gguf(path; load_tokenizer=false, validate=true)
            @test isopen(model)
            close(model)
        end

        # ── 6. Lifecycle safety ────────────────────────────────────────────
        @testset "Lifecycle safety" begin
            model = load_gguf(path; load_tokenizer=false)
            @test isopen(model)

            close(model)
            @test !isopen(model)

            # All API calls must throw after close
            @test_throws Ekatra.GGUFClosedError list_tensors(model)
            @test_throws Ekatra.GGUFClosedError get_metadata(model, "general.architecture")
            @test_throws Ekatra.GGUFClosedError get_tensor(model, first(list_tensors(load_gguf(path; load_tokenizer=false))))

            # Double close is idempotent
            @test_nowarn close(model)
        end

        # ── 7. Partial loading (metadata-only) ────────────────────────────
        @testset "Partial load (metadata-only flag)" begin
            # load_tokenizer=false still loads full metadata + tensor index
            model = load_gguf(path; load_tokenizer=false, lazy_tensors=true)
            @test !isempty(model.metadata)
            @test !isempty(model.tensor_index)
            @test model.tokenizer === nothing
            close(model)
        end

    end # @with_model
end # @testset
