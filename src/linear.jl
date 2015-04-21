#=

      Linear Module

=#

# Parameters
type LinearModuleParameters{F<:Float, D<:Device}
  weight::RealMatrix
  bias::RealVector

  LinearModuleParameters(inputSize::Int64, outputSize::Int64) = new(array(T, outputSize, inputSize), array(T, 1, outputSize))
end

# Type definition
type LinearModule{F<:Float, D<:Device} <: Module
  # The parameters
  parameters::LinearModuleParameters{F,D}

  # The gradient of the parameters
  gradient::LinearModuleParameters{F,D}

  # State
  output::RealMatrix
  gradInput::RealMatrix

  # @doc doc"Initialize a linear module with the input and output size" ->
  function LinearModule(inputSize::Int64, outputSize::Int64)
    new(LinearModuleParameters(inputSize, outputSize), LinearModuleParameters(0,0), array(T,0,0), array(T,0,0))
  end
end


function forward{F<:Float, D<:Device}(linear::LinearModule{F,D}, input::RealMatrix)
    return broadcast!(linear.output, +, input * transpose(linear.parameters.weight), linear.bias)
end

function updateGradInput{F<:Float, D<:Device}(linear::LinearModule{F,D}, input::RealMatrix, gradOutput::RealMatrix)
    linear.gradInput = gradOutput * linear.parameters.weight
    return linear.gradInput
end

function accGradParameters{F<:Float, D<:Device}(linear::LinearModule{F,D}, input::RealMatrix, gradOutput::RealMatrix, scale::F=1.)
    linear.gradient.weight += scale * transpose(gradOutput) * input
    linear.gradient.bias += scale * sum(gradOutput, 1)
end

function initGradient{F<:Float, D<:Device}(linear::LinearModule{F,D})
    args = tuple((D,F,)..., size(linear.weight)...)
    linear.gradient.weight = zeros(args...)

    args = tuple((D,F,)..., size(linear.bias)...)
    linear.gradient.bias = zeros(args...)
end
