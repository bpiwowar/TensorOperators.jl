using TensorOperators
using Base.Test

z = TensorOperators.MatrixParameters(cpu, Float64, 5, 3)

m = TensorOperators.LinearLayer(cpu, Float64, 100, 5)

x = rand(23, 100)
z = rand(23, 5)

y = forward!(m, x)
@test x == m.output
