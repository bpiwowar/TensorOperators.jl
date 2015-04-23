module TensorOperators

VERSION < v"0.4-" && using Docile
using Base.LinAlg.BLAS

import Base.rand
import Base.zeros
import Base.push!

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

# Why should I need this???
matrixOf{F}(::Type{CPUDevice}, ::Type{F}) = Array{F, 2}
vectorOf{F}(::Type{CPUDevice}, ::Type{F}) = Array{F, 1}

arrayOf{F}(::Type{CPUDevice}, ::Type{F}, dims::Int64) = Array{F, dims}
array{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = Array(F, dims...)
rand{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = rand(F, dims...)
zeros{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = zeros(F, dims...)

# --- CUDA


# --- Parameters

abstract Parameters

type ArrayParameters{D<:Device, F<:Float, N} <: Parameters
  values::RealArray
  gradient::RealArray

  function ArrayParameters(dims::Int64...)
    @assert length(dims) == N
    new(array(D,F,dims...), array(D,F,dims...))
  end
end

typealias VectorParameters{D<:Device, F<:Float} ArrayParameters{D, F, 1}
typealias MatrixParameters{D<:Device, F<:Float} ArrayParameters{D, F, 2}

const parametersMap = Dict{Any, Any}()


# Should be m::Module
# getParameters(m) = error("Type $(super(typeof(m))) not handled")
function getParameters(m)
    function compute()
        p = Any[]
        for field = fieldnames(m)
            v = getfield(m, field)
            if typeof(v) <: Parameters
                push!(p, v)
            end
        end
        p
    end

    return get!(compute, parametersMap, m)
end

# --- Modules ----

abstract Module

@doc doc"Computes the gradient wrt the parameters and the input. Returns the gradient wrt to the inputs" ->
function backward(m::Module, input, gradOutput, scale::Float64=1.)
    accGradParameters(m, input, gradOutput, scale)
    return updateGradInput(m, input, gradOutput)
end


initGradient(m::Array) = fill!(m, 0.)

initGradient(p::ArrayParameters) = fill!(p.gradient, 0.)

# Generic function for initializing the gradient
function initGradient(m::Module)
    x = getParameters(m)
    for v in x
        initGradient(v)
    end
end


# --- Include modules

# Optimization
include("optimization.jl")

# Include the containers
include("containers.jl")

# Module
include("linear.jl")
include("bhsm.jl")

# Transfer functions

# Costs
include("cost.jl")


end
