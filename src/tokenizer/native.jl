"""
GGUFNativeTokenizer — pure-Julia fallback tokenizer backed by GGUF metadata.

Supports the BPE and SentencePiece vocabularies typically stored in GGUF files
under the `tokenizer.ggml.*` keys.
"""

# ──────────────────────────────────────────────
# Struct
# ──────────────────────────────────────────────

"""
    GGUFNativeTokenizer <: AbstractTokenizer

A minimal BPE tokenizer backed by the vocabulary embedded in the GGUF file.

Supports:
- `encode(tok, text)` — naive whitespace / byte-fallback tokenisation
- `decode(tok, tokens)` — detokenise by joining vocabulary pieces

The tokenizer is intentionally conservative: it does not implement full BPE
merge resolution, but provides correct decode round-trips for sequences that
come directly from a GGUF-embedded model.
"""
struct GGUFNativeTokenizer <: AbstractTokenizer
    vocab::Vector{String}          # index → piece  (1-based)
    token_to_id::Dict{String, Int} # piece → 0-based GGUF id
    bos_id::Int
    eos_id::Int
    unk_id::Int
    model_type::String             # "llama", "gpt2", "bert", …
end

# ──────────────────────────────────────────────
# Constructor from GGUF metadata
# ──────────────────────────────────────────────

"""
    GGUFNativeTokenizer(metadata) -> GGUFNativeTokenizer

Build a tokenizer from GGUF metadata keys `tokenizer.ggml.tokens`,
`tokenizer.ggml.bos_token_id`, etc.

Throws `GGUFTokenizerError` if the vocabulary is missing.
"""
function GGUFNativeTokenizer(metadata::Dict{String, Any})
    tokens_raw = get(metadata, "tokenizer.ggml.tokens", nothing)
    if tokens_raw === nothing
        throw(GGUFTokenizerError(
            "Metadata key 'tokenizer.ggml.tokens' not found; " *
            "cannot build GGUFNativeTokenizer."
        ))
    end

    vocab = Vector{String}(tokens_raw)

    token_to_id = Dict{String, Int}()
    for (i, tok) in enumerate(vocab)
        token_to_id[tok] = i - 1   # GGUF uses 0-based ids
    end

    bos_id = Int(get(metadata, "tokenizer.ggml.bos_token_id", 1))
    eos_id = Int(get(metadata, "tokenizer.ggml.eos_token_id", 2))
    unk_id = Int(get(metadata, "tokenizer.ggml.unknown_token_id", 0))
    model_type = get(metadata, "tokenizer.ggml.model", "unknown")

    @debug "GGUFNativeTokenizer: vocab_size=$(length(vocab)), model_type=$(model_type)"

    return GGUFNativeTokenizer(vocab, token_to_id, bos_id, eos_id, unk_id, model_type)
end

# ──────────────────────────────────────────────
# encode / decode interface
# ──────────────────────────────────────────────

"""
    encode(tok::GGUFNativeTokenizer, text::String) -> Vector{Int}

Encode `text` into a sequence of token IDs (0-based, matching GGUF convention).

Strategy:
1. Try exact match on the whole string.
2. Normalise `▁` (U+2581 LOWER ONE EIGHTH BLOCK) → space and re-try.
3. Fall back to character-level encoding; unknown chars map to `unk_id`.

This is intentionally a best-effort implementation; for production use,
load Tokenizers.jl to get the full merge-table BPE.
"""
function encode(tok::GGUFNativeTokenizer, text::String)::Vector{Int}
    ids = Int[]

    # Attempt whole-string match first (useful for very short inputs)
    if haskey(tok.token_to_id, text)
        push!(ids, tok.token_to_id[text])
        return ids
    end

    # Try space-normalised lookup (SentencePiece uses ▁ as space indicator)
    normalised = replace(text, ' ' => '▁')
    if haskey(tok.token_to_id, normalised)
        push!(ids, tok.token_to_id[normalised])
        return ids
    end

    # Character-level fallback with ▁-prefix on the first character of each word.
    words = split(text, ' '; keepempty=false)
    for (wi, word) in enumerate(words)
        prefix = wi == 1 ? "" : "▁"
        for (ci, ch) in enumerate(word)
            piece = (ci == 1 ? prefix : "") * string(ch)
            if haskey(tok.token_to_id, piece)
                push!(ids, tok.token_to_id[piece])
            elseif haskey(tok.token_to_id, string(ch))
                push!(ids, tok.token_to_id[string(ch)])
            else
                push!(ids, tok.unk_id)
            end
        end
    end

    return ids
end

"""
    decode(tok::GGUFNativeTokenizer, tokens::Vector{Int}) -> String

Decode a sequence of token IDs back to a string.
Out-of-range IDs are replaced with the literal string `<unk>`.
"""
function decode(tok::GGUFNativeTokenizer, tokens::Vector{Int})::String
    pieces = Vector{String}(undef, length(tokens))
    for (i, id) in enumerate(tokens)
        idx = id + 1   # convert 0-based GGUF id to 1-based Julia index
        if 1 <= idx <= length(tok.vocab)
            pieces[i] = tok.vocab[idx]
        else
            pieces[i] = "<unk>"
        end
    end
    # Replace SentencePiece space marker
    joined = join(pieces)
    return replace(joined, '▁' => ' ')
end

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────

function Base.show(io::IO, tok::GGUFNativeTokenizer)
    print(io, "GGUFNativeTokenizer(vocab_size=$(length(tok.vocab)), " *
              "model=$(tok.model_type))")
end
