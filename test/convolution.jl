using TensorOperators
using Base.Test


m = TemporalConvolution{CPUDevice, Float64}(5, 7, 3, 1)
