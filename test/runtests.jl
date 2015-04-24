using TensorOperators
using Base.Test


# @test 1 == 1

l2cost = TensorOperators.L2Cost()

x = rand(23, 100)
z = rand(23, 5)
m = TensorOperators.LinearOperator{CPUDevice, Float64}(100, 5)
epsilon = 5e-5

init(m)
sgd = StochasticGradient(1e-3)

@time for i = 1:100
    initGradient(m)

    # Forward
    y = forward(m, x)
    e = forward(l2cost, (y, z))
    println("Cost = $e")

    # Backward
    backward(m, x, backward(l2cost, (y, z)))
    optimize(sgd, m)
end

include("bhsm.jl")
