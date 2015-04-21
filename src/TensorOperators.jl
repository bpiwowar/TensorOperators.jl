module TensorOperators

VERSION < v"0.4-" && using Docile

import Base.rand
import Base.zeros

# --- Devices and types

typealias Float FloatingPoint

abstract Device
immutable CPUDevice <: Device end
immutable CudaDevice <: Device end

export CPUDevice
export CudaDevice

const RealArray = Union(Array)
const RealMatrix = Union(Matrix)
const RealVector = Union(Vector)

arrayOf(D, F, dims::Integer) =  error("Cannot compute matrix of ($D, $F)")

vectorOf(D,F) = arrayOf(D, F, 1)
matrixOf(D,F) = arrayOf(D, F, 2)

# --- CPU

arrayOf{F}(::Type{CPUDevice}, ::Type{F}, dims::Integer) = Array{F, dims}
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
