abstract Parameters

@doc doc"Layer parameters - contains the values, the gradient and optimization specific information" ->
type ArrayParameters{D<:Device, F<:Float, N} <: Parameters
  # The parameters
  values::RealArray

  # Gradient, or null if the gradient
  gradient::Nullable{RealArray}

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

init!(p::ArrayParameters) = randn!(p.values)
init_gradient!(p::ArrayParameters) = fill!(get(p.gradient), 0.)


@doc """Linearize parameters and their gradients

- Parameters with null gradient are ignored
- All parameters should have the same storage and type

""" ->
function linearize_parameters!(D::Device, F::Float, m::Layer, copy::Bool = false)
  # Compute the total size needed
  size::Uint = 0
  for p in parameters(m)
    if typeof(p) <: ArrayParameters && !isnull(m.gradient)
      @assert typeof(p) <: ArrayParameter{D,F}
      size += length(m)
    end
  end

  # Allocate the array
  values = array(D, F, size)
  gradient = array(D, F, size)

  offset::UInt = 0
  for p in parameters(m)
    if typeof(p) <: ArrayParameters && !isnull(m.gradient)
      copy()
      size = length(m.values)
      m.values = reshape_view(view(values, offset:(offset+size-1)), size(m))
      m.gradient = reshape_view(view(gradient, offset:(offset+size-1)), size(m))
      offset += size
    end
  end

  return values, gradient
end


