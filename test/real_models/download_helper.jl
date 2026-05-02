"""
download_helper.jl — Test model registry and auto-download helpers.

Models are cached in test/real_models/cache/ and never committed to git.
Tests that require network access are skipped if:
  - EKATRA_SKIP_NETWORK_TESTS=true env var is set
  - or the download fails (e.g. CI without internet)
"""

using Downloads
using Logging

# ──────────────────────────────────────────────
# Cache location (relative to this file)
# ──────────────────────────────────────────────

const CACHE_DIR = joinpath(@__DIR__, "cache")

# ──────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────

"""Known test models with their download URLs and expected properties."""
const TEST_MODELS = Dict(
    # ~18 MB · TinyLlama stories 15 M params · Q4_0 quantization
    # Source: https://huggingface.co/ggml-org/models
    :tiny => (
        filename = "stories15M-q4_0.gguf",
        url      = "https://huggingface.co/ggml-org/models/resolve/main/" *
                   "tinyllamas/stories15M-q4_0.gguf",
        min_size = 15_000_000,  # bytes — sanity lower-bound
        quant    = "Q4_0",
    ),

    # ~100 MB · SmolLM2-135M-Instruct · Q4_K_M quantization
    # Source: https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF
    :quantized => (
        filename = "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        url      = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF" *
                   "/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf",
        min_size = 90_000_000,
        quant    = "Q4_K_M",
    ),
)

# ──────────────────────────────────────────────
# Network skip guard
# ──────────────────────────────────────────────

"""
    skip_network_tests() -> Bool

Return `true` if network tests should be skipped.
Controlled by the `EKATRA_SKIP_NETWORK_TESTS` environment variable.
"""
function skip_network_tests()::Bool
    v = get(ENV, "EKATRA_SKIP_NETWORK_TESTS", "false")
    return lowercase(strip(v)) in ("true", "1", "yes")
end

# ──────────────────────────────────────────────
# Download helper
# ──────────────────────────────────────────────

"""
    ensure_model(key::Symbol) -> Union{String, Nothing}

Return the local path to the test model identified by `key`.
Downloads the model on first call; returns `nothing` if the download fails.

# Example
```julia
path = ensure_model(:tiny)
path === nothing && return   # skip test if unavailable
```
"""
function ensure_model(key::Symbol)::Union{String, Nothing}
    haskey(TEST_MODELS, key) ||
        error("Unknown test model key: $(key). Valid keys: $(keys(TEST_MODELS))")

    info = TEST_MODELS[key]
    mkpath(CACHE_DIR)
    dest = joinpath(CACHE_DIR, info.filename)

    # Already cached and has reasonable size → use it
    if isfile(dest) && filesize(dest) >= info.min_size
        @info "Using cached model: $(info.filename) ($(round(filesize(dest)/1e6; digits=1)) MB)"
        return dest
    end

    # Attempt download
    @info "Downloading $(info.filename) from HuggingFace (~$(round(info.min_size/1e6; digits=0)) MB expected)..."
    try
        Downloads.download(info.url, dest; timeout = 600.0)
    catch e
        @warn "Download failed for $(info.filename): $(e)"
        isfile(dest) && rm(dest; force=true)
        return nothing
    end

    # Sanity-check size
    if filesize(dest) < info.min_size
        @warn "Downloaded file looks too small ($(filesize(dest)) bytes), removing."
        rm(dest; force=true)
        return nothing
    end

    @info "Download complete: $(info.filename) ($(round(filesize(dest)/1e6; digits=1)) MB)"
    return dest
end

# ──────────────────────────────────────────────
# Convenience macro: skip test if model unavailable
# ──────────────────────────────────────────────

"""
    @with_model key var body

Download model `key`, bind its path to `var`, and execute `body`.
The test is skipped (with a WARN log) if:
  - network tests are disabled, or
  - the download fails.

# Example
```julia
@with_model :tiny path begin
    model = load_gguf(path)
    @test isopen(model)
    close(model)
end
```
"""
macro with_model(key, var, body)
    quote
        if skip_network_tests()
            @warn "Skipping real-model test (EKATRA_SKIP_NETWORK_TESTS=true)"
        else
            local _path = ensure_model($(esc(key)))
            if _path === nothing
                @warn "Skipping real-model test: model $($(esc(key))) unavailable"
            else
                local $(esc(var)) = _path
                $(esc(body))
            end
        end
    end
end
