"""
    Safetensors

This library reads tensors from a [Safetensors](https://github.com/safetensors/safetensors) file.
"""
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

"""
    get_permuted_tensors(path)

Get tensors from the Safetensors file at `path`.

Return `tensors, metadata` where:

- `tensors` is a `Dict{String,Array}` mapping tensor names to tensors.
- `metadata` is a `Dict{String,String}`.

The returned tensors are permuted.
That is, if the Safetensors file has a tensor:

    0 1 2
    3 4 5

you will get a tensor:

    0 3
    1 4
    2 5
"""
function get_permuted_tensors(path)
    open(path) do file
        header = begin
            header_size = ltoh(read(file, UInt64))
            JSON.parse(read(file, header_size))
        end

        byte_buffer = read(file)

        tensors = Dict{String,Array}()
        metadata = undef

        for (key, value) ∈ header
            if key == "__metadata__"
                metadata = convert(Dict{String,String}, value)
            else
                begin_offset, end_offset = value.data_offsets
                buffer = IOBuffer(byte_buffer[(begin_offset+1):end_offset])

                if haskey(dtype_table, value.dtype)
                    T = dtype_table[value.dtype]
                    values = [ltoh(read(buffer, T)) for _ = 1:prod(value.shape)]

                    tensors[key] = reshape(values, Tuple(reverse(value.shape)))
                elseif value.dtype == "BF16"
                    # The bit-layout of bfloat16: saaa_aaaa_afff_ffff
                    # The bit-layout of float32:  saaa_aaaa_afff_ffff_ffff_ffff_ffff_ffff

                    # When you append sixteen zero bits
                    # at the end of the bit-layout of your bfloat16 number
                    # and interpret it as a bit-layout of a float32 number,
                    # you will get the identical numerical value.

                    # Get the bit-layout of bfloat16 numbers in the file
                    # interpreting them as uint16.
                    values = [ltoh(read(buffer, UInt16)) for _ = 1:prod(value.shape)]

                    # Convert them into uint32 to make a room for bit shifting.
                    values = convert(Vector{UInt32}, values)

                    # Append sixteen zero bits.
                    values = values .<< 16

                    # Interpret the numbers as float32.
                    values = reinterpret.(Float32, values)

                    tensors[key] = reshape(values, Tuple(reverse(value.shape)))
                end
            end
        end

        tensors, metadata
    end
end

end
