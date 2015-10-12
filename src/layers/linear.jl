#=

      Linear Layer

=#

export LinearLayer

type LinearLayer{D<:Device, F<:Float} <: Layer
  # Device
  device::Device

  # The parameters
  weight::matrixParameters(D,F)
  bias::matrixParameters(D,F)

  # State
  output::denseRealMatrix(D,F)
  input_gradient::denseRealMatrix(D,F)

  function LinearLayer(device, weight, bias)
    new(device, weight, bias, array(device, F, 0, 0), array(device, F, 0, 0))
  end
end

"Initialize a linear module with the input and output size"
function LinearLayer{D<:Device, F<:Float}(d::D, ::Type{F}, inputSize::Int64, outputSize::Int64)
  LinearLayer{D,F}(d, matrixParameters(d, F, inputSize, outputSize), matrixParameters(d, F, 1, outputSize))
end

function forward!{D<:Device, F<:Float}(linear::LinearLayer{D,F}, input::DenseMatrix)
  @ensuresize linear.output, size(input,1), size(linear.weight.values, 2)

  gemm!('N', 'N', 1., input, linear.weight.values, 0., linear.output)
  broadcast!(+, linear.output, linear.output, linear.bias.values)
end

function compute_inputgradient!{D<:Device, F<:Float}(linear::LinearLayer{D,F}, input::DenseMatrix, gradOutput::DenseMatrix)
  @ensuresize linear.input_gradient, size(gradOutput,1), size(linear.weight.values, 1)

  gemm!('N', 'T', 1., gradOutput, linear.weight.values, 0., linear.input_gradient)
end

function update_gradient!{D<:Device, F<:Float}(linear::LinearLayer{D,F}, input::DenseMatrix, gradOutput::DenseMatrix, scale::F=1.)
  if !isnull(linear.weight.gradient)
    gemm!('T', 'N', scale, input, gradOutput, 0., get(linear.weight.gradient))
  end

  if !isnull(linear.bias.gradient)
    axpy!(scale, sum(gradOutput, 1), get(linear.bias.gradient))
  end
end
