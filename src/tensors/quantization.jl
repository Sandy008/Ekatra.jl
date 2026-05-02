"""
Quantization scheme mapping: GGML enum → QuantizationScheme type.

This module is intentionally minimal — it only provides the type dispatch
mapping, not any dequantization logic.
"""

# ──────────────────────────────────────────────
# GGML type enum constants (mirrors ggml.h)
# ──────────────────────────────────────────────

const GGML_TYPE_F32     = UInt32(0)
const GGML_TYPE_F16     = UInt32(1)
const GGML_TYPE_Q4_0    = UInt32(2)
const GGML_TYPE_Q4_1    = UInt32(3)
# 4, 5 are legacy / unused
const GGML_TYPE_Q5_0    = UInt32(6)
const GGML_TYPE_Q5_1    = UInt32(7)
const GGML_TYPE_Q8_0    = UInt32(8)
const GGML_TYPE_Q8_1    = UInt32(9)
const GGML_TYPE_Q2_K    = UInt32(10)
const GGML_TYPE_Q3_K    = UInt32(11)
const GGML_TYPE_Q4_K    = UInt32(12)
const GGML_TYPE_Q5_K    = UInt32(13)
const GGML_TYPE_Q6_K    = UInt32(14)
const GGML_TYPE_Q8_K    = UInt32(15)
const GGML_TYPE_IQ2_XXS = UInt32(16)
const GGML_TYPE_IQ2_XS  = UInt32(17)
const GGML_TYPE_IQ3_XXS = UInt32(18)
const GGML_TYPE_IQ1_S   = UInt32(19)
const GGML_TYPE_IQ4_NL  = UInt32(20)
const GGML_TYPE_IQ3_S   = UInt32(21)
const GGML_TYPE_IQ2_S   = UInt32(22)
const GGML_TYPE_IQ4_XS  = UInt32(23)
const GGML_TYPE_I8      = UInt32(24)
const GGML_TYPE_I16     = UInt32(25)
const GGML_TYPE_I32     = UInt32(26)
const GGML_TYPE_BF16    = UInt32(30)

# ──────────────────────────────────────────────
# Val-dispatched mapping (extensible)
# ──────────────────────────────────────────────

"""
    ggml_type_to_scheme(::Val{N}) -> QuantizationScheme

Map a GGML type enum value to the corresponding `QuantizationScheme` instance.
Returns `UnknownQuantization(N)` for unrecognised type IDs.

To extend the mapping, define a new method:

```julia
Ekatra.ggml_type_to_scheme(::Val{42}) = MyCustomScheme()
```
"""
ggml_type_to_scheme(::Val{0})  = F32()
ggml_type_to_scheme(::Val{1})  = F16()
ggml_type_to_scheme(::Val{2})  = Q4_0()
ggml_type_to_scheme(::Val{3})  = Q4_1()
ggml_type_to_scheme(::Val{6})  = Q5_0()
ggml_type_to_scheme(::Val{7})  = Q5_1()
ggml_type_to_scheme(::Val{8})  = Q8_0()
ggml_type_to_scheme(::Val{9})  = Q8_1()
ggml_type_to_scheme(::Val{10}) = Q2_K()
ggml_type_to_scheme(::Val{11}) = Q3_K()
ggml_type_to_scheme(::Val{12}) = Q4_K()
ggml_type_to_scheme(::Val{13}) = Q5_K()
ggml_type_to_scheme(::Val{14}) = Q6_K()
ggml_type_to_scheme(::Val{15}) = Q8_K()
ggml_type_to_scheme(::Val{16}) = IQ2_XXS()
ggml_type_to_scheme(::Val{17}) = IQ2_XS()
ggml_type_to_scheme(::Val{18}) = IQ3_XXS()
ggml_type_to_scheme(::Val{19}) = IQ1_S()
ggml_type_to_scheme(::Val{20}) = IQ4_NL()
ggml_type_to_scheme(::Val{21}) = IQ3_S()
ggml_type_to_scheme(::Val{22}) = IQ2_S()
ggml_type_to_scheme(::Val{23}) = IQ4_XS()
ggml_type_to_scheme(::Val{24}) = I8()
ggml_type_to_scheme(::Val{25}) = I16()
ggml_type_to_scheme(::Val{26}) = I32()
ggml_type_to_scheme(::Val{30}) = BF16()

# Fallback for unknown type IDs
ggml_type_to_scheme(v::Val{N}) where {N} = UnknownQuantization(UInt32(N))

"""
    ggml_type_to_scheme(dtype::UInt32) -> QuantizationScheme

Dynamic dispatch version — wraps `dtype` in `Val` and calls the typed method.
"""
function ggml_type_to_scheme(dtype::UInt32)
    return ggml_type_to_scheme(Val(Int(dtype)))
end

# ──────────────────────────────────────────────
# Human-readable name helper
# ──────────────────────────────────────────────

quantization_name(::F32)   = "F32"
quantization_name(::F16)   = "F16"
quantization_name(::Q4_0)  = "Q4_0"
quantization_name(::Q4_1)  = "Q4_1"
quantization_name(::Q5_0)  = "Q5_0"
quantization_name(::Q5_1)  = "Q5_1"
quantization_name(::Q8_0)  = "Q8_0"
quantization_name(::Q8_1)  = "Q8_1"
quantization_name(::Q2_K)  = "Q2_K"
quantization_name(::Q3_K)  = "Q3_K"
quantization_name(::Q4_K)  = "Q4_K"
quantization_name(::Q5_K)  = "Q5_K"
quantization_name(::Q6_K)  = "Q6_K"
quantization_name(::Q8_K)  = "Q8_K"
quantization_name(::IQ2_XXS) = "IQ2_XXS"
quantization_name(::IQ2_XS)  = "IQ2_XS"
quantization_name(::IQ3_XXS) = "IQ3_XXS"
quantization_name(::IQ1_S)   = "IQ1_S"
quantization_name(::IQ4_NL)  = "IQ4_NL"
quantization_name(::IQ3_S)   = "IQ3_S"
quantization_name(::IQ2_S)   = "IQ2_S"
quantization_name(::IQ4_XS)  = "IQ4_XS"
quantization_name(::I8)  = "I8"
quantization_name(::I16) = "I16"
quantization_name(::I32) = "I32"
quantization_name(::BF16) = "BF16"
quantization_name(u::UnknownQuantization) = "Unknown($(u.type_id))"
