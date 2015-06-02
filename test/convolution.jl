using TensorOperators
using Base.Test


input_framesize = 5
output_framesize = 4
kW = 2
dW = 1

m = TemporalConvolution(cpu, Float64, input_framesize, output_framesize, kW, dW)
init!(m)


x = rand(4, 5)

forward!(m, x)