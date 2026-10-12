using Test

using Safetensors

@testset begin
    safetensors = read_safetensors("tensors.safetensors")
    expected = reshape(convert(Vector{Float32}, 0:23), 4, 3, 2)
    tensor = get_permuted_tensor!(safetensors, "tensor")
    @test tensor == expected
    tensor = get_permuted_tensor!(safetensors, "tensor")
    @test tensor == expected
end
