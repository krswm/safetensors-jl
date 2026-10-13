using Test

using Safetensors

@testset begin
    tensors, metadata = get_permuted_tensors("tensors.safetensors")
    @test tensors == Dict(
        "f16_tensor" => reshape(convert(Vector{Float16}, 0:23), 4, 3, 2),
        "f32_tensor" => reshape(convert(Vector{Float32}, 0:23), 4, 3, 2),
        "f64_tensor" => reshape(convert(Vector{Float64}, 0:23), 4, 3, 2),
        "i8_tensor" => reshape(convert(Vector{Int8}, 0:23), 4, 3, 2),
        "u8_tensor" => reshape(convert(Vector{UInt8}, 0:23), 4, 3, 2),
        "i16_tensor" => reshape(convert(Vector{Int16}, 0:23), 4, 3, 2),
        "u16_tensor" => reshape(convert(Vector{UInt16}, 0:23), 4, 3, 2),
        "i32_tensor" => reshape(convert(Vector{Int32}, 0:23), 4, 3, 2),
        "u32_tensor" => reshape(convert(Vector{UInt32}, 0:23), 4, 3, 2),
        "i64_tensor" => reshape(convert(Vector{Int64}, 0:23), 4, 3, 2),
        "u64_tensor" => reshape(convert(Vector{UInt64}, 0:23), 4, 3, 2),
    )
    @test metadata == Dict("metadata_key" => "metadata_value")
end
