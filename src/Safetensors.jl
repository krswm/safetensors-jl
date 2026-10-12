# Specification of the Safetensors format:
# https://github.com/safetensors/safetensors#format

module Safetensors

export get_permuted_tensor!, read_safetensors

using JSON

mutable struct LazySafetensors
    header::JSON.LazyValue{Vector{UInt8}}
    byte_buffer::Vector{UInt8}
    cache::Dict{String,Array}
end

function read_safetensors(path)
    header = undef
    byte_buffer = undef
    open(path) do file
        header = begin
            header_size = ltoh(read(file, UInt64))
            JSON.lazy(read(file, header_size))
        end

        byte_buffer = read(file)
    end

    cache = Dict{String,Array}()

    LazySafetensors(header, byte_buffer, cache)
end

function get_permuted_tensor!(lazy_safetensors, name)
    if haskey(lazy_safetensors.cache, name)
        lazy_safetensors.cache[name]
    else
        info = lazy_safetensors.header[name]

        if info.dtype[] ≠ "F32"
            error("dtype not supported")
        end

        begin_offset, end_offset = info.data_offsets[]
        buffer = IOBuffer(lazy_safetensors.byte_buffer[(begin_offset + 1):end_offset])
        tensor = [read(buffer, Float32) for _ = 1:prod(info.shape[])]
        tensor = reshape(tensor, Tuple(reverse(info.shape[])))

        lazy_safetensors.cache[name] = tensor

        tensor
    end
end

end
