#=

      Linear Layer

=#

export LinearLayer

type LinearLayer{D<:Device, F<:Float} <: Layer
  # Device
  device::Device

  # The parameters
  weight::MatrixParameters{D,F}
  bias::MatrixParameters{D,F}

  # State
  output::RealMatrix
  grad_input::RealMatrix

end


# @doc doc"Initialize a linear module with the input and output size" ->
function LinearLayer{D<:Device, F<:Float}(d::D, ::Type{F}, inputSize::Int64, outputSize::Int64)
  LinearLayer{D,F}(
    d,
    matrixParameters(d, F, inputSize, outputSize), 
    matrixParameters(d, F, 1, outputSize), 
    array(d, F, 0, 0), 
    array(d, F, 0, 0)
  )
end

function forward!{D<:Device, F<:Float}(linear::LinearLayer{D,F}, input::RealMatrix)
  @ensuresize linear.output, size(input,1), size(linear.weight.values, 2)
  gemm!('N', 'N', 1., input, linear.weight.values, 0., linear.output)
  broadcast!(+, linear.output, linear.output, linear.bias.values)
end

function compute_inputgradient!{D<:Device, F<:Float}(linear::LinearLayer{D,F}, input::RealMatrix, gradOutput::RealMatrix)
  grad_inputSize = (size(gradOutput,1), size(linear.weight.values, 2))
  if size(linear.grad_input) != grad_inputSize
    linear.grad_input = array(D,F, grad_inputSize...)
  end
  gemm!('N', 'N', 1., gradOutput, linear.weight.values, 0., linear.grad_input)
end

function update_gradient!{D<:Device, F<:Float}(linear::LinearLayer{D,F}, input::RealMatrix, gradOutput::RealMatrix, scale::F=1.)
  if !isnull(linear.weight.gradient)
    gemm!('T', 'N', scale, input, gradOutput, 0., get(linear.weight.gradient))
  end

  if !isnull(linear.bias.gradient)
    axpy!(scale, sum(gradOutput, 1), get(linear.bias.gradient))
  end
end
