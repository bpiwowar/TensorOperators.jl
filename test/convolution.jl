using TensorOperators
using Base.Test
using ArrayViews

input_framesize = 5
output_framesize = 4
kW = 2
dW = 1

m = TemporalConvolution(cpu, Float64, input_framesize, output_framesize, kW, dW)
init!(m)


x = rand(5, 10)

y = forward!(m, x)

# Compute the output

output_length = div(size(x, 2) - kW, dW) + 1
y2 = zeros(output_framesize, output_length)
for i in 1:output_length
    xi = flatten_view(x[:, i:(i+kW-1)])
    y2[:, i] = m.bias.values + m.weight.values * xi
end

@test_approx_eq y y2