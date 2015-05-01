#=

      Linear Operator

=#

type LinearOperator{D<:Device, F<:Float} <: Operator
  # The parameters
  weight::MatrixParameters{D,F}
  bias::MatrixParameters{D,F}

  # State
  output::RealMatrix
  grad_input::RealMatrix

  # @doc doc"Initialize a linear module with the input and output size" ->
  function LinearOperator(inputSize::Int64, outputSize::Int64)
    new(MatrixParameters{D,F}(inputSize, outputSize), MatrixParameters{D,F}(1, outputSize), array(D,F,0,0), array(D,F,0,0))
  end
end


function forward!{D<:Device, F<:Float}(linear::LinearOperator{D,F}, input::RealMatrix)
  ensuresize!(linear.output, size(input,1), size(linear.weight.values, 2))
  # outputSize = (size(input,1), size(linear.weight.values, 2))
  # if size(linear.output) != outputSize
  #   linear.output = array(D,F, outputSize...)
  # end

  gemm!('N', 'N', 1., input, linear.weight.values, 0., linear.output)
  broadcast!(+, linear.output, linear.output, linear.bias.values)
end

function compute_inputgradient!{D<:Device, F<:Float}(linear::LinearOperator{D,F}, input::RealMatrix, gradOutput::RealMatrix)
  grad_inputSize = (size(gradOutput,1), size(linear.weight.values, 2))
  if size(linear.grad_input) != grad_inputSize
    linear.grad_input = array(D,F, grad_inputSize...)
  end
  gemm!('N', 'N', 1., gradOutput, linear.weight.values, 0., linear.grad_input)
end

function update_gradient!{D<:Device, F<:Float}(linear::LinearOperator{D,F}, input::RealMatrix, gradOutput::RealMatrix, scale::F=1.)
  gemm!('T', 'N', scale, input, gradOutput, 0., linear.weight.gradient)
  axpy!(scale, sum(gradOutput, 1), linear.bias.gradient)
end
