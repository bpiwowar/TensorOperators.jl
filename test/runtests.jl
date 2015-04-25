using TensorOperators
using Base.Test


# @test 1 == 1

l2cost = TensorOperators.L2Cost()

x = rand(23, 100)
z = rand(23, 5)
m = TensorOperators.LinearOperator{CPUDevice, Float64}(100, 5)
epsilon = 5e-5

init!(m)
# optimizer = StochasticGradient(1e-3)
optimizer = RProp(1e-3)
println(typeof(optimizer))

@time for i = 1:100
    init_gradient!(m)

    # Forward
    y = forward!(m, x)
    e = forward!(l2cost, (y, z))
    println("Cost = $e")

    # Backward
    backward!(m, x, backward!(l2cost, (y, z)))
    optimize!(optimizer, m)
end

include("bhsm.jl")
