using TensorOperators
using Base.Test

TO = TensorOperators

z = TO.arrayParameters(cpu, Float64, 5, 3)
z = TO.matrixParameters(cpu, Float64, 5, 3)

m = TO.LinearLayer(cpu, Float64, 3, 5)

x = rand(4, 3)
z = rand(4, 5)

init!(m)
# Computes with module and directly
y = forward!(m, x)
y2 = broadcast(+, m.bias.values, x * m.weight.values)

@test_approx_eq y2 m.output
