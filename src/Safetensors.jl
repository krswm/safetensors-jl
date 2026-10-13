# Specification of the Safetensors format:
# https://github.com/safetensors/safetensors#format

module Safetensors

export get_permuted_tensors

using JSON

function get_permuted_tensors(path)
    open(path) do file
        header = begin
            header_size = ltoh(read(file, UInt64))
            JSON.parse(read(file, header_size))
        end

        metadata = undef

        byte_buffer = read(file)

        tensors = Dict{String, Array}()

        for (key, value) ∈ header
            if key == "__metadata__"
                metadata = convert(Dict{String, String}, value)
            else
                function get_tensor(T)
                    begin_offset, end_offset = value.data_offsets
                    buffer = IOBuffer(byte_buffer[(begin_offset + 1):end_offset])
                    tensor = [read(buffer, T) for _ = 1:prod(value.shape)]
                    tensor = reshape(tensor, Tuple(reverse(value.shape)))
                end

                if value.dtype == "F16"
                    tensors[key] = get_tensor(Float16)
                elseif value.dtype == "F32"
                    tensors[key] = get_tensor(Float32)
                elseif value.dtype == "F64"
                    tensors[key] = get_tensor(Float64)
                elseif value.dtype == "I8"
                    tensors[key] = get_tensor(Int8)
                elseif value.dtype == "U8"
                    tensors[key] = get_tensor(UInt8)
                elseif value.dtype == "I16"
                    tensors[key] = get_tensor(Int16)
                elseif value.dtype == "U16"
                    tensors[key] = get_tensor(UInt16)
                elseif value.dtype == "I32"
                    tensors[key] = get_tensor(Int32)
                elseif value.dtype == "U32"
                    tensors[key] = get_tensor(UInt32)
                elseif value.dtype == "I64"
                    tensors[key] = get_tensor(Int64)
                elseif value.dtype == "U64"
                    tensors[key] = get_tensor(UInt64)
                end
            end
        end

        tensors
    end
end

#=
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
=#

end
