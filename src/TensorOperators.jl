module TensorOperators

VERSION < v"0.4-" && using Docile
using Base.LinAlg.BLAS

import Base.rand
import Base.zeros
import Base.push!

abstract Operator

# --- Some useful macros

macro stabilize(ex)
    @assert ex.head == :(=)
    @assert length(ex.args) == 2
    @assert ex.args[1].head == :(::)

    value = ex.args[2]
    valueType = ex.args[1].args[2]

    quote
        if isa($value, $valueType)
            $ex
        else
            Base.error("Cannot stabilize type to $valueType for $value")
        end
    end

end


# --- Devices and types

typealias Float FloatingPoint

abstract Device

const RealArray = Union(Array)
const RealMatrix = Union(Matrix)
const RealVector = Union(Vector)

arrayOf(D, F, ::Integer) =  error("Cannot compute matrix of ($D, $F)")

vectorOf(D,F) = arrayOf(D, F, 1)
matrixOf(D,F) = arrayOf(D, F, 2)

# --- CPU

immutable CPUDevice <: Device end
export CPUDevice

arrayOf{F}(::Type{CPUDevice}, ::Type{F}, dims::Int64) = Array{F, dims}
array{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = Array(F, dims...)
rand{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = rand(F, dims...)
zeros{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64...) = zeros(F, dims...)

# --- CUDA

immutable CudaDevice <: Device end

export CudaDevice

# --- Parameters

abstract Parameters

type ArrayParameters{D<:Device, F<:Float, N} <: Parameters
  values::RealArray
  gradient::RealArray

  function ArrayParameters(dims::Int64...)
    @assert length(dims) == N::Int
    new(array(D,F,dims...), array(D,F,dims...))
  end
end

typealias VectorParameters{D<:Device, F<:Float} ArrayParameters{D, F, 1}
typealias MatrixParameters{D<:Device, F<:Float} ArrayParameters{D, F, 2}

const parametersMap = Dict{Any, Any}()

function getParameters{T<:Operator}(t::Type{T})
    function compute()
        p = Any[]
        for field = fieldnames(t)
            if fieldtype(t, field) <: Parameters
                push!(p, field)
            end
        end
        p
    end

    return get!(compute, parametersMap, t)
end

function getParameters(m::Operator)
    function _it()
        for field in getParameters(typeof(m))
            produce(m.(field))
        end
    end
    Task(_it)

end

# --- Operators ----


@doc doc"Computes the gradient wrt the parameters and the input. Returns the gradient wrt to the inputs" ->
function backward(m::Operator, input, gradOutput, scale::Float64=1.)
    accGradParameters(m, input, gradOutput, scale)
    return updateGradInput(m, input, gradOutput)
end


initGradient(p::ArrayParameters) = fill!(p.gradient, 0.)

# Generic function for initializing the gradient
function initGradient(m::Operator)
    for v in getParameters(m)
        initGradient(v)
    end
end


init(p::ArrayParameters) = randn!(p.values)
function init(m::Operator)
    for v in getParameters(m)
        init(v)
    end
end


export initGradient, forward, backward, init

# --- Includes

# Optimization
include("optimization.jl")

# Container modules
include("containers.jl")

# Operators
include("linear.jl")
include("bhsm.jl")

# Transfer functions

# Costs
include("cost.jl")


end
