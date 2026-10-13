# Specification of the Safetensors format:
# https://github.com/safetensors/safetensors#format

module Safetensors

export get_permuted_tensors

using JSON

dtype_table = Dict(
    "F16" => Float16,
    "F32" => Float32,
    "F64" => Float64,
    "I8" => Int8,
    "U8" => UInt8,
    "I16" => Int16,
    "U16" => UInt16,
    "I32" => Int32,
    "U32" => UInt32,
    "I64" => Int64,
    "U64" => UInt64,
)

function get_permuted_tensors(path)
    open(path) do file
        header = begin
            header_size = ltoh(read(file, UInt64))
            JSON.parse(read(file, header_size))
        end


        byte_buffer = read(file)

        metadata = undef
        tensors = Dict{String,Array}()

        for (key, value) ∈ header
            if key == "__metadata__"
                metadata = convert(Dict{String,String}, value)
            else
                function get_tensor(T)
                    begin_offset, end_offset = value.data_offsets
                    buffer = IOBuffer(byte_buffer[(begin_offset+1):end_offset])
                    tensor = [read(buffer, T) for _ = 1:prod(value.shape)]
                    tensor = reshape(tensor, Tuple(reverse(value.shape)))
                end

                if haskey(dtype_table, value.dtype)
                    tensors[key] = get_tensor(dtype_table[value.dtype])
                end
            end
        end

        metadata, tensors
    end
end

end
