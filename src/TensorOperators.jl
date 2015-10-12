module TensorOperators

using Base.LinAlg.BLAS
using ArrayViews

import Base: rand, randn!, rand!, zeros, push!, length, size


# --- Devices and types

typealias Float AbstractFloat

abstract Device

denseRealArray(::TypeVar) = Union{DenseArray{Float64}, DenseArray{Float32}}
denseRealArray(dim::Int) = Union{DenseArray{Float32,dim}, DenseArray{Float64, dim}}
const DenseRealTensor = denseRealArray(TypeVar(:N))
const DenseRealMatrix = denseRealArray(2)
const DenseRealVector = denseRealArray(1)


denseTensor(::TypeVar, F::TypeVar, N::Union{TypeVar,Int}) = DenseArray{F, N}
denseTensor{D<:Device}(::Type{D}, F, ::Int) =  error("Cannot compute type of dense matrix of ($D, $F)")

denseMatrix(D, F) = denseTensor(D, F, 2)
denseVector(D, F) = denseTensor(D, F, 1)

denseRealMatrix(::TypeVar, ::TypeVar) = DenseRealMatrix
denseRealMatrix{F<:AbstractFloat}(D, ::Type{F}) = denseMatrix(D, F)
denseRealVector(::TypeVar, ::TypeVar) = DenseRealVector
denseRealVector{F<:AbstractFloat}(D, ::Type{F}) = denseVector(D, F)

const IntVector = Array{Int, 1}

arrayOf{D<:Device}(::Type{D}, F, ::Integer) =  error("Cannot compute matrix of ($D, $F)")

vectorOf{D<:Device}(::Type{D}, F) = arrayOf(D, F, 1)
matrixOf{D<:Device}(::Type{D}, F) = arrayOf(D, F, 2)

# --- CPU

immutable CPUDevice <: Device end
export CPUDevice, cpu

cpu = CPUDevice()

arrayOf{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64) = DenseArray{F, dims}

denseTensor{F}(::Type{CPUDevice}, ::Type{F}, dims::Int64) = DenseArray{F, dims}

array{F<:Float}(d::CPUDevice, ::Type{F}, dims::Int64...) = Array(F, dims...)
rand{F<:Float}(d::CPUDevice, ::Type{F}, dims::Int64...) = rand(F, dims...)
zeros{F<:Float}(d::CPUDevice, ::Type{F}, dims::Int64...) = zeros(F, dims...)

# --- Layers ----

abstract Layer

"""Computes the gradient wrt the parameters and the input. Returns the gradient wrt to the inputs"""
function backward!(m::Layer, input, gradOutput, scale::Float64=1.)
    update_gradient!(m, input, gradOutput, scale)
    return compute_inputgradient!(m, input, gradOutput)
end


""" Generic function for initializing the gradient """
function init_gradient!(m::Layer)
    for v in parameters(m)
        if !isnull(v.gradient)
            init_gradient!(v)
        end
    end
end


function init!(m::Layer)
    for v in parameters(m)
        init!(v)
    end
end


export init_gradient!, forward!, backward!, init!, update_gradient!, compute_inputgradient!



# --- Includes

# Misc
include("utils.jl")
include("parameters.jl")

#
include("cuda.jl")

# Optimization
include("optimization.jl")

# Container modules
include("containers.jl")

# Layers
include("layers/linear.jl")
include("layers/convolution.jl")
include("layers/bhsm.jl")
include("layers/lookup.jl")

# Transfer functions
include("transfer.jl")

# Costs
include("cost.jl")

end
