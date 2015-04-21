module TensorOperators

VERSION < v"0.4-" && using Docile

import Base.rand
import Base.zeros

# --- Devices and types

typealias Float FloatingPoint

abstract Device
immutable CPUDevice <: Device end
immutable CudaDevice <: Device end

export CPUDevice, CudaDevice

const RealArray = Union(Array)
const RealMatrix = Union(Matrix)
const RealVector = Union(Vector)

# matrixOf(realType) =  error("Cannot compute matrix of $realType")

# --- CPU

#matrixOf{T<:Float}() = Array{cpuType(T), 2}

array{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = Array(F, dims...)
rand{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = rand(F, dims...)
zeros{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = zeros(F, dims...)

# --- CUDA



# --- Modules ----

abstract Module

@doc doc"Computes the gradient wrt the parameters and the input. Returns the gradient wrt to the inputs" ->
function backward(theModule::Module, input, gradOutput, scale::Float64=1.)
    accGradParameters(theModule, input, gradOutput, scale)
    return updateGradInput(theModule, input, gradOutput)
end


# --- Include modules

# Include the containers
include("containers.jl")

# Module
include("linear.jl")
include("bhsm.jl")

# Transfer functions

# Costs
include("cost.jl")

end
