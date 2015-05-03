#=

      Temporal convolution

=#

type TemporalConvolution{D<:Device, F<:Float} <: Operator
  @doc "Kernel width"
  kW::UInt

  @doc "Delta"
  dW::UInt

  @doc "The convolution matrix"
  weight::RealMatrix

  @doc "Real vector"
  bias::RealVector

  @doc doc"Used for padding"
  padding::RealMatrix
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

  @assert isnull(padding) || size(padding, 1) == input_framesize
  if isnull(padding)
    padding = array(D, F, input_framesize, 0)
  end

  self.input_framesize = input_framesize
  self.output_framesize = output_framesize
  self.kW = kW
  self.dW = dW

  self.weight = MatrixParameters{D,F}(output_framesize, input_framesize * kW)
  self.bias = VectorParameters{D, F}(output_framesize)
  self.padding = padding

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
  nOutputFrame = div(nInputFrame - kW + size(m.padding, 2), dW) + 1
  output = m.output
  ensuresize!(output, nOutputFrame, outputFrameSize)

  # Copy bias first
  for k = 1:nOutputFrame
    output[k, :] = bias
  end

  # Compute the convolution
  pos::Int = 0
  for k = 1:nOutputFrame
    nFrame = div(nInputFrame - k * dW - kW, inputFrameStride) + 1;
    nOutputFrame -= nFrame
    outputWindow = unsafe_view(output, :, k)

    inputwidth::Int = kW
    if pos < size(padding, 2)
      d = size(padding, 2) - pos
      inputWidth -= size(padding, 2) - pos
      gemm!('N', 'T', 1, inputWindow, unsave_view(weight, :, 1:d), 1, outputWindow)

    end

    inputWindow = unsafe_view(input, :, pos:(pos + inputwidth))
    gemm!('N', 'T', 1, inputWindow, weight[:, (1:inputWidth)], 1, outputWindow)

    # Advance the window
    pos += kW
  end
end

function compute_inputgradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix)
end

function update_gradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix, scale::F=1.)
end


