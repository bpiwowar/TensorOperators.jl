#=

      Temporal convolution

=#

type TemporalConvolution{D<:Device, F<:Float} <: Operator
  @doc "Kernel width"
  kW::UInt

  @doc "Delta"
  dW::UInt

  @doc "The convolution matrix"
  weight::MatrixParameters

  @doc "Real vector"
  bias::MatrixVector

  @doc doc"Used for padding"
  padding::MatrixParameters
end


local TemporalConvolution, parent = torch.class('nn.TemporalConvolution', 'nn.Module')

@doc doc"Creates a new temporal convolution operator

input_framesize The size of the inputs vectors
output_framesize The size of the output vectors
kW The kernel width - how many inputs are taken into account
dW The move width - the step size
padding An input matrix used for padding. Must have input_framesize rows
"
function TemporalConvolution{D<:Device, F<:Float}(input_framesize, output_framesize, kW, dW::UInt=1, padding=Nullable{RealMatrix})
  self = new()

  @assert(isnull(padding) || size(padding, 1) == input_framesize, "Padding should be null or have the same number of rows than the input")
  if isnull(padding)
    self.padding = new MatrixParameters{D,F}(array(D, F, input_framesize, 0), array(D, F, input_framesize, 0))
  else
    self.padding = new MatrixParameters{D,F}(array(D, F, input_framesize, 0), array(D, F, input_framesize, size(padding, 0)))
  end

  @assert(size(padding, 2) < kW, "Kernel width should be greater than the padding size")

  self.input_framesize = input_framesize
  self.output_framesize = output_framesize
  self.kW = kW
  self.dW = dW

  self.weight = MatrixParameters{D,F}(output_framesize, input_framesize * kW)
  self.bias = VectorParameters{D, F}(output_framesize)

  reset!(self)
  self
end

function reset!{D<:Device, F<:Float}(s::TemporalConvolution{D,F}, stdv=Nullable{F})
   if isnull(stdv) then
      stdv = 1/math.sqrt(self.kW*self.input_framesize)
   else
      stdv = stdv * math.sqrt(3)
   end

    uniform!(s.weight.values, -stdv, stdv)
    uniform!(s.bias.values, -stdv, stdv)
end


function forward!{D<:Device,F<:Float}(m::TemporalConvolution, input::DenseArray{F, 2})

  # Prepare
  nInputFrame = size(input, 1)
  paddingsize = size(m.padding, 2)
  nOutputFrame = div(nInputFrame - kW + paddingsize, dW) + 1
  output = m.output
  ensuresize!(output, nOutputFrame, outputFrameSize)

  # Type stability
  @stabilize padding::matrixOf(D,F) = m.padding.values
  @stabilize weight::matrixOf(D, F) = m.weight.values

  # Copy bias first
  for k = 1:nOutputFrame
    output[k, :] = bias
  end

  # Compute the convolution
  pos::Int = -paddingsize # Position in the input
  for k = 1:nOutputFrame
    outputview = unsafe_view(output, :, k)

    # Deals with the padding
    inputwidth::Int = kW
    d::Int = 0
    if pos < 0
      d = paddingsize - pos
      inputwidth -= d
      gemm!('N', 'T', one(F), padding, unsave_view(weight, :, 1:paddingsize), one(F), outputview)
    end

    # Add the result of the convolution for this output
    inputview = unsafe_view(input, :, pos:(pos + inputwidth))
    weightview = unsafe_view(weight, :, (d+1):inputwidth)
    gemm!('N', 'T', one(F), inputview, weightview, one(F), outputview)

    # Advance the window
    pos += dW
  end
end

function compute_inputgradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix)
end

function update_gradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix, scale::F=1.)
end


