module TensorOperators

if VERSION < v"0.4-"
    using Docile

    include("compat-0.3.jl")
end

using Base.LinAlg.BLAS

import Base: rand, randn!, rand!, zeros, push!, length


# --- Devices and types

typealias Float FloatingPoint

abstract Device

const RealArray = Union(DenseArray)
const RealMatrix = Union(DenseMatrix)
const RealVector = Union(DenseVector)

arrayOf(::Device, F, ::Integer) =  error("Cannot compute matrix of ($D, $F)")

vectorOf(d::Device, F) = arrayOf(F, d, 1)
matrixOf(d::Device, F) = arrayOf(F, d, 2)

# --- CPU

immutable CPUDevice <: Device end
export CPUDevice, cpu

cpu = CPUDevice()

arrayOf{F}(d::Type{CPUDevice}, ::Type{F}, dims::Int64) = DenseArray{F, dims}
array{F<:Float}(d::Type{CPUDevice}, ::Type{F}, dims::Int64...) = Array(d, F, dims...)
rand{F<:Float}(d::Type{CPUDevice}, ::Type{F}, dims::Int64...) = rand(d, F, dims...)
zeros{F<:Float}(d::Type{CPUDevice}, ::Type{F}, dims::Int64...) = zeros(d, F, dims...)

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




# --- Parameters

include("parameters.jl")


# --- Useful methods

@doc doc"Ensure that the size of the of the array is at least dims

The inner storage might be preserved
" ->
function ensuresize!{D}(m::DenseArray{D}, dims::UInt...)
end

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

@doc "Fill an array with random number from a uniform distribution" ->
function uniform!{F<:Float}(a::DenseArray{F}, min::F, max::F)
  r::F = max - min
  @inbounds for i = 1:length(a)
    a[i] = rand(F) * r + min
  end
end

@doc "Fill an array with random number from a gaussian distribution" ->
function randn!{F<:Float}(a::DenseArray{F}, mu::F, sigma::F)
  @inbounds for i = 1:length(a)
    a[i] = randn(F) * sigma + mu
  end
end

# --- Includes

# Optimization
include("optimization.jl")

# Container modules
include("containers.jl")

# Layers
include("base.jl")
include("convolution.jl")
include("bhsm.jl")

# Transfer functions
include("transfer.jl")

# Costs
include("cost.jl")


end
