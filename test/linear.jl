using TensorOperators
using Base.Test


input_size = 40
output_size = 50
sample_size = 300

m = LinearLayer(cpu, Float64, input_size, output_size)
parameters, gradient = linearize_parameters!(cpu, Float64, m)
init!(m)

# Computes with module and directly

x = randn(sample_size, input_size)

# --- Test forward

y = forward!(m, x)
yp = broadcast(+, m.bias.values, x * m.weight.values)

@test_approx_eq yp m.output

# --- Test of gradient (input)

dx = randn(sample_size, input_size)

y = copy(y)
∂x_∂y = copy(compute_inputgradient!(m, x, eye(output_size)))
@test_approx_eq_eps ∂x_∂y transpose(m.weight.values) (1e-16 * length(m.weight.values))

# Approximate gradient
epsilon = 1e-5
dx = epsilon * randn(size(x))
xdx = x + dx
delta = forward!(m, xdx) - y
@test_approx_eq_eps(dx * transpose(∂x_∂y) - delta, zeros(delta), 1e-16 * length(delta))


# --- Test of gradient (parameters)

init_gradient!(m)
∂x_∂y = copy(compute_inputgradient!(m, x, eye(output_size)))
