"Base type for all parameters"
abstract Parameters

export linearize_parameters!



# --- Parameter types

"Layer parameters - contains the values, the gradient and optimization specific information"
type ArrayParameters{D<:Device, F<:Float, N} <: Parameters
  # The parameters
  values::denseTensor(D,F,N)

  # Gradient, or null if the gradient
  gradient::Nullable{denseTensor(D,F,N)}

  # FIXME: remove? should not be part of operators?
  optimization_state::Nullable{Any} # Used by an optimization method to store any parameter related information


  ArrayParameters(values, gradient) = new(values, gradient, Nullable())

  function ArrayParameters(d::D, ::Type{F}, dims::Int64...)
    @assert length(dims) == N
    new(array(d, F, dims...), Nullable(array(d, F, dims...)), Nullable())
  end
end


typealias VectorParameters{D<:Device, F<:Float} ArrayParameters{D, F, 1}
typealias MatrixParameters{D<:Device, F<:Float} ArrayParameters{D, F, 2}

matrixParameters(D::TypeVar, F::TypeVar) = ArrayParameters{D,F,2}
matrixParameters{D<:Device, F<:Float}(::Type{D}, ::Type{F}) = ArrayParameters{D,F,2}

vectorParameters(D::TypeVar, F::TypeVar) = ArrayParameters{D,F,1}
vectorParameters{D<:Device, F<:Float}(::Type{D}, ::Type{F}) = ArrayParameters{D,F,1}

arrayParameters{D<:Device, F<:Float}(d::D, ::Type{F}, dims::Int64...) =
  ArrayParameters{D,F,length(dims)}(d, F, dims...)


matrixParameters{D<:Device, F<:Float}(d::D, ::Type{F}, dims::Int64...) =
  ArrayParameters{D,F,2}(d, F, dims...)

vectorParameters{D<:Device, F<:Float}(d::D, ::Type{F}, dims::Int64...) =
  ArrayParameters{D,F,1}(d, F, dims...)



length(m::ArrayParameters) = length(m.values)
size(m::ArrayParameters) = size(m.values)

# Caches the parameter fields for each type
const parametersMap = Dict{Type, Array{Symbol}}()

function _parameters(layer::Layer)
  return get!(parametersMap, typeof(layer)) do
    p = Any[]
    for field in fieldnames(layer)
        if fieldtype(layer, field) <: Parameters
            push!(p, field)
        end
    end
    p
  end
end

function parameters(m::Layer)
    function _it()
        for field in _parameters(m)
            produce(m.(field))
        end
    end
    Task(_it)
end




# --- Methods for parameters

init!(p::ArrayParameters) = randn!(p.values)
init_gradient!(p::ArrayParameters) = fill!(get(p.gradient), 0.)


"""Linearize parameters and their gradients

- Parameters with null gradient are ignored
- All parameters should have the same storage and type

"""
function linearize_parameters!{F<:Float, D<:Device}(d::D, ::Type{F}, m::Layer, copy::Bool = false)
  @assert(!copy, "not implemented")

  # Compute the total size needed
  totalsize::Int = 0
  for p in parameters(m)
    if typeof(p) <: ArrayParameters && !isnull(p.gradient)
      @assert typeof(p) <: ArrayParameters{D,F}
      totalsize += length(p)
    end
  end

  # Allocate the array
  values = array(d, F, totalsize)
  gradient = array(d, F, totalsize)

  offset::UInt = 0
  for p in parameters(m)
    if typeof(p) <: ArrayParameters && !isnull(p.gradient)
      psize = length(p.values)
      r = offset:(offset+psize-1)

      p.values = reshape_view(view(values, r), size(p.values))
      p.gradient = Nullable(reshape_view(view(gradient, r), size(get(p.gradient))))

      offset += psize
    end
  end

  return values, gradient
end


