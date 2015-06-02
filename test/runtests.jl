using TensorOperators
using Base.Test


# @test 1 == 1

l2cost = L2Cost()

x = rand(23, 100)
z = rand(23, 5)
m = LinearLayer(cpu, Float64, 100, 5)
epsilon = 5e-5

init!(m)
optimizer = StochasticGradient(1e-3)
#optimizer = RProp(1e-3)

init_gradient!(m)

# Forward
y = forward!(m, x)
e = forward!(l2cost, (y, z))
# println("Cost = $e")

# Backward
backward!(m, x, backward!(l2cost, (y, z)))
optimize!(optimizer, m)


# include("bhsm.jl")

include("linear.jl")
include("convolution.jl")
