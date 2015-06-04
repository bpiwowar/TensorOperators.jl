
export LookupLayer

type LookupLayer{D<:Device, F<:Float} <: Layer
  # Device
  device::Device

  # The parameters
  table::matrixParameters(D,F)

  # State
  output::denseRealMatrix(D,F)

  LookupLayer(device, table) = new(device, table, array(device, F, 0, 0))
end

@doc "Initialize a lookup table" ->
function LookupLayer{D<:Device, F<:Float}(d::D, ::Type{F}, size::Int64, vectorsize::Int64)
  LookupLayer{D,F}(d, matrixParameters(d, F, size, vectorsize))
end

function forward!{D<:Device, F<:Float}(lookup::LookupLayer{D,F}, input::IntVector)
  @ensuresize lookup.output, size(input), size(lookup.weight.values, 2)

  # Just copy the output
  for i in 1:size(input)
    lookup.output[i, :] = lookup[input[i], :]
  end
end

function update_gradient!{D<:Device, F<:Float}(lookup::LookupLayer{D,F}, input::IntVector, gradOutput::DenseMatrix, scale::F=1.)
  if !isnull(lookup.table.gradient)
    for i in 1:size(input)
      lookup.output[i, :] = lookup[input[i], :]
    end
  end
end
