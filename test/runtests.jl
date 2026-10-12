using Test

using Safetensors

@testset begin
    a = 3
    @test true
    @test 2 == 2
    @test a == 3

    open("tensors.safetensors") do file
        read(file, UInt64) |> println
    end
end
