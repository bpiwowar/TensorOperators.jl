#=

      Temporal convolution

=#

export TemporalConvolution

type TemporalConvolution{D<:Device, F<:Float} <: Layer
  kW::UInt

  dW::UInt

  input_framesize::UInt
  output_framesize::UInt

  weight::MatrixParameters{D,F}

  bias::VectorParameters{D,F}

  padding::MatrixParameters{D,F}

  # State
  output::RealMatrix
  grad_input::RealMatrix

end

# @doc doc"Creates a new temporal convolution operator

# input_framesize The size of the inputs vectors
# output_framesize The size of the output vectors
# kW The kernel width - how many inputs are taken into account
# dW The move width - the step size
# padding An input matrix used for padding. Must have input_framesize rows
# " ->
function TemporalConvolution{D<:Device, F<:Float}(device::D, ::Type{F}, input_framesize::UInt, output_framesize::UInt, kW::UInt, dW::Int=1, padding=Nullable{RealMatrix}())
  @assert(isnull(padding) || size(padding, 1) == input_framesize, "Padding should be null or have the same number of rows than the input")

  if isnull(padding)
    padding = matrixParameters(device, F, input_framesize, 0)
  else
    padding = MatrixParameters{D,F}(array(device, F, input_framesize, 0), array(device, F, input_framesize, size(padding, 0)))
  end

  @assert(size(padding.values, 2) < kW, "Kernel width should be greater than the padding size")

  kW = kW
  dW = dW

  weight = matrixParameters(device, F, output_framesize, input_framesize * kW)
  bias = vectorParameters(device, F, output_framesize)

  self = TemporalConvolution{D,F}(kW, dW, input_framesize, output_framesize, weight, bias, padding, array(device, F, 0, 0), array(device, F, 0, 0))

  reset!(self)
  return self
end

function reset!{D<:Device, F<:Float}(s::TemporalConvolution{D,F}, stdv=Nullable{F}())
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
  paddingsize = size(m.padding.values, 2)
  nOutputFrame = div(nInputFrame - m.kW + paddingsize, m.dW) + 1

  output = @ensuresize m.output, m.output_framesize, nOutputFrame

  # Type stability
  @stabilize padding::matrixOf(D, F) = m.padding.values
  @stabilize weight::matrixOf(D, F) = m.weight.values

  # Compute the convolution
  pos::Int = 1-paddingsize # Position in the input
  inputwidth::Int = 0

  for k = 1:nOutputFrame
    output[:, k] = m.bias.values
    outputview = unsafe_view(output, :, k)

    # Deals with the padding
    inputwidth = m.kW
    d::Int = 0
    if pos < 0
      d = paddingsize - pos
      inputwidth -= d
      gemm!('N', 'T', one(F), padding, unsave_view(weight, :, 1:paddingsize), one(F), outputview)
    end

    # Add the result of the convolution for this output
    weightview = unsafe_view(weight, :, (d*m.input_framesize+1):((d+inputwidth)*m.input_framesize))
    inputview = flatten_view(view(input, :, pos:(pos + inputwidth - 1)))

    BLAS.gemv!('N', one(F), weightview, inputview, one(F), outputview)

    # Advance the window
    pos += m.dW
  end

  output
end

function compute_inputgradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix)
end

function update_gradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix, scale::F=1.)
end


