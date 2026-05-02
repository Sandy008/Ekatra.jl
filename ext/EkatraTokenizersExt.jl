"""
EkatraTokenizersExt — package extension that bridges Ekatra.jl with Tokenizers.jl.

Loaded automatically by Julia when both Ekatra and Tokenizers are imported.
"""
module EkatraTokenizersExt

using Ekatra
import Tokenizers

"""
TokenizersBridgeTokenizer wraps a Tokenizers.jl tokenizer object.
"""
struct TokenizersBridgeTokenizer <: Ekatra.AbstractTokenizer
    tokenizer::Any   # Tokenizers.Tokenizer
end

"""
    Ekatra.build_tokenizer(metadata) -> TokenizersBridgeTokenizer or nothing

When Tokenizers.jl is loaded, attempt to build a tokenizer from the embedded
`tokenizer_config` JSON key (if present).  Falls back to `nothing` so
GGUFNativeTokenizer is used instead.
"""
function Ekatra.build_tokenizer(metadata::Dict{String, Any})
    # GGUF files may embed a full Tokenizers JSON under various keys.
    for key in ("tokenizer.huggingface.json", "tokenizer_config")
        json_str = get(metadata, key, nothing)
        if json_str isa String
            try
                tok = Tokenizers.from_str(json_str)
                @debug "EkatraTokenizersExt: loaded tokenizer from metadata key '$(key)'"
                return TokenizersBridgeTokenizer(tok)
            catch e
                @warn "EkatraTokenizersExt: failed to parse tokenizer from '$(key)': $(e)"
            end
        end
    end
    return nothing
end

"""
    Ekatra.encode(tok::TokenizersBridgeTokenizer, text::String) -> Vector{Int}
"""
function Ekatra.encode(tok::TokenizersBridgeTokenizer, text::String)::Vector{Int}
    enc = Tokenizers.encode(tok.tokenizer, text)
    return Int.(Tokenizers.get_ids(enc))
end

"""
    Ekatra.decode(tok::TokenizersBridgeTokenizer, tokens::Vector{Int}) -> String
"""
function Ekatra.decode(tok::TokenizersBridgeTokenizer, tokens::Vector{Int})::String
    return Tokenizers.decode(tok.tokenizer, UInt32.(tokens))
end

end # module EkatraTokenizersExt
