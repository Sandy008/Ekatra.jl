"""
Tokenizers.jl bridge (loaded as a package extension when Tokenizers.jl is available).

When Tokenizers.jl is not available, `build_tokenizer` falls back to the
GGUFNativeTokenizer implemented in native.jl.
"""

"""
    build_tokenizer(metadata) -> AbstractTokenizer

Attempt to construct a Tokenizers.jl-backed tokenizer from `metadata`.
Returns `nothing` if the required keys are absent; the caller should fall back
to `GGUFNativeTokenizer`.

This function is intentionally a stub — the real implementation lives in
`ext/EkatraTokenizersExt.jl` and is loaded automatically by Julia's extension
mechanism when Tokenizers.jl is available.
"""
function build_tokenizer(metadata::Dict{String, Any})
    # Without Tokenizers.jl loaded, we cannot build a bridge tokenizer.
    # Return nothing so the caller falls back to GGUFNativeTokenizer.
    return nothing
end
