function ensuresize{T,D}(array::Array{T,D}, dims::Int...)
  @assert(ndims(array) == length(dims), "$(size(array)) is different to $(dims)")

  if size(array) == dims
    return array
  end

  return Array(T, dims...)
end

macro ensuresize(ex)
  array = ex.args[1]
  other = ex.args[2:length(ex.args)]

  expr = Expr(:call, :ensuresize, array, other...)
  return esc(Expr(:(=), array, expr))
end



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


    esc(quote
      if !isa($value, $valueType)
        Base.error("Cannot stabilize type to $valueType for $value")
      end
      $ex
    end)

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


# ---

function randn!{T,N}(x::DenseArray{T,N})
  @inbounds for i in 1:length(x)
    x[i] = randn()
  end
end