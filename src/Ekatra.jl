"""
    Ekatra

A zero-copy GGUF parser and tokenizer bridge for Julia.

## Quick start

```julia
using Ekatra

model = load_gguf("model.gguf")

# Inspect metadata
println(get_metadata(model, "general.name"))

# List tensors
for name in list_tensors(model)
    ti = tensor_info(model, name)
    println(name, " ", ti.dims)
end

# Zero-copy tensor access
t = get_tensor(model, "token_embd.weight")

# Tokenizer
tok = get_tokenizer(model)
ids = encode(tok, "Hello world")
txt = decode(tok, ids)

close(model)
```
"""
module Ekatra

using Mmap
using Logging
using Printf: @sprintf

# ──────────────────────────────────────────────
# Core definitions (order matters)
# ──────────────────────────────────────────────

include("errors.jl")
include("types.jl")

include("io/reader.jl")
include("io/parser.jl")

include("tensors/quantization.jl")
include("tensors/views.jl")

include("tokenizer/bridge.jl")
include("tokenizer/native.jl")

include("api.jl")

# ──────────────────────────────────────────────
# Public API exports
# ──────────────────────────────────────────────

# Errors
export GGUFError, GGUFVersionError, GGUFTensorError, GGUFTokenizerError,
       GGUFStringError, GGUFIOError, GGUFClosedError

# Types
export QuantizationScheme,
       F32, F16, BF16,
       Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
       Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
       IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS,
       I8, I16, I32,
       UnknownQuantization,
       AbstractGGUFTensor, QuantizedTensor,
       TensorInfo,
       AbstractTokenizer,
       GGUFModel

# Quantization helpers
export ggml_type_to_scheme, quantization_name

# Tensor access
export raw_pointer, nbytes

# Tokenizer
export GGUFNativeTokenizer, encode, decode

# API
export load_gguf, list_tensors, tensor_info, get_tensor,
       get_metadata, get_tokenizer

end # module Ekatra
