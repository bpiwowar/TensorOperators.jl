using TensorOperators
using Base.Test

TO = TensorOperators

z = TO.arrayParameters(cpu, Float64, 5, 3)
z = TO.matrixParameters(cpu, Float64, 5, 3)

m = TO.LinearLayer(cpu, Float64, 100, 5)

x = rand(23, 100)
z = rand(23, 5)

y = forward!(m, x)
@test x == m.output
