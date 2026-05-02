#!/usr/bin/env julia
"""
basic_usage.jl — Minimal end-to-end usage example for Ekatra.jl.

Run with any GGUF file:

    julia --project=. examples/basic_usage.jl path/to/model.gguf

If no path is given, the script downloads the tiny stories15M test model
(~18 MB) into a local cache and uses that.
"""

using Ekatra
using Downloads
using Printf

# ──────────────────────────────────────────────
# 0. Resolve model path
# ──────────────────────────────────────────────

const DEMO_URL  = "https://huggingface.co/ggml-org/models/resolve/main/" *
                  "tinyllamas/stories15M-q4_0.gguf"
const CACHE_DIR = joinpath(@__DIR__, ".model_cache")
const DEMO_FILE = joinpath(CACHE_DIR, "stories15M-q4_0.gguf")

function get_demo_model()::String
    if length(ARGS) >= 1
        p = ARGS[1]
        isfile(p) || error("File not found: $p")
        return abspath(p)
    end

    # Auto-download the tiny demo model
    mkpath(CACHE_DIR)
    if !isfile(DEMO_FILE) || filesize(DEMO_FILE) < 10_000_000
        @info "Downloading demo model (~18 MB)…"
        Downloads.download(DEMO_URL, DEMO_FILE; timeout=300.0)
        @info "Download complete: $(round(filesize(DEMO_FILE)/1e6; digits=1)) MB"
    end
    return DEMO_FILE
end

path = get_demo_model()

# ──────────────────────────────────────────────
# 1. Open the GGUF file
# ──────────────────────────────────────────────

println("\n── Loading model ─────────────────────────────────────────────")
model = load_gguf(path)
println(model)

# ──────────────────────────────────────────────
# 2. Read metadata
# ──────────────────────────────────────────────

println("\n── Metadata ──────────────────────────────────────────────────")

arch = get_metadata(model, "general.architecture")
name = get_metadata(model, "general.name")

println("  Name         : ", something(name, "(not set)"))
println("  Architecture : ", something(arch, "(not set)"))

# Print all scalar metadata (skip large arrays like the vocabulary)
println("\n  All metadata keys:")
for (k, v) in sort(collect(model.metadata); by=first)
    v_str = v isa AbstractVector ? "<array[$(length(v))]>" : repr(v)
    println("    $(rpad(k, 45)) = $(v_str)")
end

# ──────────────────────────────────────────────
# 3. Tensor index
# ──────────────────────────────────────────────

println("\n── Tensors ───────────────────────────────────────────────────")
names = list_tensors(model)
println("  Total tensors: $(length(names))\n")

for name in sort(names)[1:min(10, end)]
    ti = tensor_info(model, name)
    scheme = Ekatra.ggml_type_to_scheme(ti.dtype)
    @printf("  %-50s  dims=%-20s  dtype=%s\n",
            name, string(ti.dims), Ekatra.quantization_name(scheme))
end
length(names) > 10 && println("  … and $(length(names) - 10) more")

# ──────────────────────────────────────────────
# 4. Zero-copy tensor access
# ──────────────────────────────────────────────

println("\n── Zero-copy tensor access ───────────────────────────────────")
first_name = first(sort(names))
t = get_tensor(model, first_name)

println("  Tensor name : $(first_name)")
println("  Dims        : $(t.dims)")
println("  Bytes       : $(Ekatra.nbytes(t))  ($(round(Ekatra.nbytes(t)/1e6; digits=3)) MB)")
println("  Pointer     : $(Ekatra.raw_pointer(t))")
println("  Type        : $(typeof(t))")
println()
println("  ✓ Data is memory-mapped — no copy into Julia heap.")

# ──────────────────────────────────────────────
# 5. Tokenizer
# ──────────────────────────────────────────────

println("\n── Tokenizer ─────────────────────────────────────────────────")
if model.tokenizer !== nothing
    tok = get_tokenizer(model)
    println("  Tokenizer type : $(typeof(tok))")

    test_text = "Once upon a time"
    ids  = encode(tok, test_text)
    back = decode(tok, ids)

    println("  Input  : $(repr(test_text))")
    println("  IDs    : $(ids)")
    println("  Decoded: $(repr(back))")
else
    println("  (No tokenizer metadata in this model file)")
end

# ──────────────────────────────────────────────
# 6. Clean up
# ──────────────────────────────────────────────

println("\n── Cleanup ───────────────────────────────────────────────────")
close(model)
println("  Model closed. isopen → $(isopen(model))")
println("\n✓ Done.\n")
