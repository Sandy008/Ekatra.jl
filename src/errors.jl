"""
Error hierarchy for Ekatra.jl GGUF parsing.
"""

abstract type GGUFError <: Exception end

"""
Thrown when the GGUF file version is unsupported.
"""
struct GGUFVersionError <: GGUFError
    msg::String
end

"""
Thrown when a tensor cannot be parsed, accessed, or is out of bounds.
"""
struct GGUFTensorError <: GGUFError
    msg::String
end

"""
Thrown when the tokenizer metadata is missing, invalid, or unsupported.
"""
struct GGUFTokenizerError <: GGUFError
    msg::String
end

"""
Thrown when a string in the GGUF file is not valid UTF-8 or exceeds size limits.
"""
struct GGUFStringError <: GGUFError
    msg::String
end

"""
Thrown when the file cannot be opened, memory-mapped, or is truncated.
"""
struct GGUFIOError <: GGUFError
    msg::String
end

"""
Thrown when a closed GGUFModel is accessed.
"""
struct GGUFClosedError <: GGUFError
    msg::String
end

Base.showerror(io::IO, e::GGUFVersionError)   = print(io, "GGUFVersionError: ",   e.msg)
Base.showerror(io::IO, e::GGUFTensorError)    = print(io, "GGUFTensorError: ",    e.msg)
Base.showerror(io::IO, e::GGUFTokenizerError) = print(io, "GGUFTokenizerError: ", e.msg)
Base.showerror(io::IO, e::GGUFStringError)    = print(io, "GGUFStringError: ",    e.msg)
Base.showerror(io::IO, e::GGUFIOError)        = print(io, "GGUFIOError: ",        e.msg)
Base.showerror(io::IO, e::GGUFClosedError)    = print(io, "GGUFClosedError: ",    e.msg)
