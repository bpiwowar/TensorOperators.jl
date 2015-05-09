using TensorOperators
using Base.Test


m = TemporalConvolution{Float64}(cpu, 5, 7, 3, 1)
