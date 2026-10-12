# Specification of the Safetensors format:
# https://github.com/safetensors/safetensors#format

module Safetensors

export get_permuted_tensor!, read_safetensors

using JSON

mutable struct Safetensors_
    header::Any
    byte_buffer::Any
    cache::Any
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

    Safetensors_(header, byte_buffer, cache)
end

function get_permuted_tensor!(safetensors, name) end

end
