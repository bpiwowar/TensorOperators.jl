module TensorOperators

VERSION < v"0.4-" && using Docile

import Base.rand
import Base.zeros

# --- Real types

abstract _Real

abstract _CudaNumber <: _Real
type _CudaFloat <: _CudaNumber end
type _CudaDouble <: _CudaNumber end

abstract CPUNumber <: _Real

immutable CPUFloat <: CPUNumber
end

immutable CPUDouble <: CPUNumber
end


# --- Matrix

const RealArray = Union(Array)
const RealMatrix = Union(Matrix)
const RealVector = Union(Vector)

matrixOf(realType) =  error("Cannot compute matrix of $realType")

# --- CPU

cpuType(T::Type{CPUFloat}) = Float32
cpuType(T::Type{CPUDouble}) = Float64

matrixOf{T<:CPUNumber}(realType::Type{T}) = Array{cpuType(T), 2}

array{T<:CPUNumber}(realType::Type{T}, dims::Int64...) = Array(cpuType(CPUDouble), dims...)
rand{T<:CPUNumber}(realType::Type{T}, dims::Int64...) = rand(cpuType(T), dims...)
zeros{T<:CPUNumber}(realType::Type{T}, dims::Int64...) = zeros(cpuType(T), dims...)

# --- Modules ----


abstract Module

@doc doc"Computes the gradient wrt the parameters and the input. Returns the gradient wrt to the inputs" ->
function backward(theModule::Module, input, gradOutput, scale::Float64=1.)
    accGradParameters(theModule, input, gradOutput, scale)
    return updateGradInput(theModule, input, gradOutput)
end

# --- Sequence

type SequenceModule
    list::Array{Module}

    SequenceModule() = new([])
end

function push!(sequence::SequenceModule, a_module::Module)
    push!(sequence.list, a_module)
end

function forward(sequence::SequenceModule, input)
    for a_module = sequence.list
        forward(a_module, input)
    end
end

# --- Linear Module

type LinearModule{T<:_Real} <: Module
    # @doc doc"The weight" ->
    weight
    bias

    output
    gradWeight
    gradBias
    gradInput

    # @doc doc"Initialize a linear module with the input and output size" ->
    LinearModule(inputSize::Int64, outputSize::Int64) = new(rand(T, outputSize, inputSize), rand(T, 1, outputSize), array(T), array(T), array(T))
end


function forward{T<:_Real}(linear::LinearModule{T}, input::RealMatrix)
    return linear.output = broadcast(+, input * transpose(linear.weight), linear.bias)
end

# Compute the gradient wrt to the input
function updateGradInput{T<:_Real}(linear::LinearModule{T}, input::RealMatrix, gradOutput::RealMatrix)
    linear.gradInput = gradOutput * linear.weight
    return linear.gradInput
end

function accGradParameters{T<:_Real}(linear::LinearModule{T}, input::RealMatrix, gradOutput::RealMatrix, scale::Float64=1.)
    linear.gradWeight += scale * transpose(gradOutput) * input
    linear.gradBias += scale * sum(gradOutput, 1)
end

function initGradient{T<:_Real}(linear::LinearModule{T})
    args = tuple((T,)..., size(linear.weight)...)
    linear.gradWeight = zeros(args...)

    args = tuple((T,)..., size(linear.bias)...)
    linear.gradBias = zeros(args...)
end

# --- L2 Cost

type L2Cost <: Module
    output
    gradInput
    L2Cost() = new()
end


forward(cost::L2Cost, inputs) = cost .output = .5 * norm(inputs[1] - inputs[2])
backward(cost::L2Cost, inputs, scale=1.) = cost.gradInput = scale * (inputs[1] - inputs[2])


# --- Code


include("bhsm.jl")

end
