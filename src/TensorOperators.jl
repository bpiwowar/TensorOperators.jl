module TensorOperators

if VERSION < v"0.4-"
    using Docile

    include("compat-0.3.jl")
end

using Base.LinAlg.BLAS
using ArrayViews

import Base: rand, randn!, rand!, zeros, push!, length, size


# --- Devices and types

typealias Float FloatingPoint

abstract Device

const RealArray = Union(DenseArray)
const RealMatrix = Union(DenseMatrix)
const RealVector = Union(DenseVector)

arrayOf{D<:Device}(::Type{D}, F, ::Integer) =  error("Cannot compute matrix of ($D, $F)")

vectorOf{D<:Device}(::Type{D}, F) = arrayOf(D, F, 1)
matrixOf{D<:Device}(::Type{D}, F) = arrayOf(D, F, 2)

# --- CPU

immutable CPUDevice <: Device end
export CPUDevice, cpu

cpu = CPUDevice()

arrayOf{F<:Float}(::Type{CPUDevice}, ::Type{F}, dims::Int64) = DenseArray{F, dims}

array{F<:Float}(d::CPUDevice, ::Type{F}, dims::Int64...) = Array(F, dims...)
rand{F<:Float}(d::CPUDevice, ::Type{F}, dims::Int64...) = rand(F, dims...)
zeros{F<:Float}(d::CPUDevice, ::Type{F}, dims::Int64...) = zeros(F, dims...)

# --- CUDA

immutable CudaDevice <: Device end

export CudaDevice


# --- Layers ----

abstract Layer

@doc doc"Computes the gradient wrt the parameters and the input. Returns the gradient wrt to the inputs" ->
function backward!(m::Layer, input, gradOutput, scale::Float64=1.)
    update_gradient!(m, input, gradOutput, scale)
    return compute_inputgradient!(m, input, gradOutput)
end


# Generic function for initializing the gradient
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


export init_gradient!, forward!, backward!, init!



# --- Includes

# Misc
include("utils.jl")
include("parameters.jl")

# Optimization
include("optimization.jl")

# Container modules
include("containers.jl")

# Layers
include("layers/linear.jl")
include("layers/convolution.jl")
include("layers/bhsm.jl")

# Transfer functions
include("transfer.jl")

# Costs
include("cost.jl")


end
