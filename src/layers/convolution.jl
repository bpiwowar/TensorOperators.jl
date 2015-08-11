#=

      Temporal convolution

=#

export TemporalConvolution

type TemporalConvolution{D<:Device, F<:Float} <: Layer
  kW::UInt

  dW::UInt

  input_framesize::UInt
  output_framesize::UInt

  weight::matrixParameters(D,F)

  bias::vectorParameters(D,F)

  # State
  output::denseRealMatrix(D,F)
  grad_input::denseRealMatrix(D,F)

  TemporalConvolution(device, kw, dW, input_framesize, output_framesize, weight, bias) =
    new(kw, dW, input_framesize, output_framesize, weight, bias, array(device, F, 0, 0), array(device, F, 0, 0))
end

# @doc doc"Creates a new temporal convolution operator

# input_framesize The size of the inputs vectors
# output_framesize The size of the output vectors
# kW The kernel width - how many inputs are taken into account
# dW The move width - the step size
# " ->
function TemporalConvolution{D<:Device, F<:Float}(device::D, ::Type{F}, input_framesize::UInt, output_framesize::UInt, kW::UInt, dW::Int=1)
  weight = matrixParameters(device, F, output_framesize, input_framesize * kW)
  bias = vectorParameters(device, F, output_framesize)

  self = TemporalConvolution{D,F}(device, kW, dW, input_framesize, output_framesize, weight, bias)

  return self
end

function init!{D<:Device, F<:Float}(s::TemporalConvolution{D,F}, stdv=Nullable{F}())
   if isnull(stdv)
      stdv = 1/ sqrt(s.kW * s.input_framesize)
   else
      stdv = stdv * math.sqrt(3)
   end

    uniform!(s.weight.values, -stdv, stdv)
    uniform!(s.bias.values, -stdv, stdv)
end

# Case where the input is (input_size, nb_samples)
function forward!{D<:Device,F<:Float}(m::TemporalConvolution{D,F}, input::DenseArray{F, 2})
  # Prepare
  nInputFrame = size(input, 2)
  nOutputFrame = div(nInputFrame - m.kW, m.dW) + 1

  output = @ensuresize m.output, nOutputFrame, m.output_framesize

  weight = m.weight.values

  # Compute the convolution
  pos::Int = 1 # Position in the input

  for k = 1:nOutputFrame
    outputview = unsafe_view(output, :, k)
    copy!(outputview, m.bias.values)
    inputview = flatten_view(view(input, :, pos:(pos + m.kW - 1)))

    BLAS.gemv!('N', one(F), weight, inputview, one(F), outputview)

    # Advance the window
    pos += m.dW
  end

  output
end

function compute_inputgradient!{D<:Device, F<:Float}(m::TemporalConvolution{D,F}, input::DenseRealMatrix, gradOutput::DenseRealMatrix)
 # Prepare
  nInputFrame = size(input, 2)
  nOutputFrame = div(nInputFrame - m.kW, m.dW) + 1

  grad_input = @ensuresize m.grad_input, (size(input, 1), size(input, 2))

  weight = m.weight.values

  # Compute the convolution
  pos::Int = 1 # Position in the input

  for k = 1:nOutputFrame
    outputview = unsafe_view(output, :, k)
    copy!(outputview, m.bias.values)
    inputview = flatten_view(view(input, :, pos:(pos + m.kW - 1)))

    BLAS.gemv!('N', one(F), weight, inputview, one(F), grad_input)

    # Advance the window
    pos += m.dW
  end
end

function update_gradient!{D<:Device, F<:Float}(m::TemporalConvolution{D,F}, input::DenseRealMatrix, gradOutput::DenseRealMatrix, scale::F=1.)
  @assert false
end


