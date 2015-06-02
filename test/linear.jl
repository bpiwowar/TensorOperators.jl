using TensorOperators
using Base.Test


m = LinearLayer(cpu, Float64, 3, 5)
init!(m)

# Computes with module and directly

x = rand(4, 3)
z = rand(4, 5)


y = forward!(m, x)
y2 = broadcast(+, m.bias.values, x * m.weight.values)

@test_approx_eq y2 m.output
