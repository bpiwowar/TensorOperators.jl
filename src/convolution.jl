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
dW The move width - how much 
"
function TemporalConvolution{D<:Device, F<:Float}(input_framesize, output_framesize, kW, dW::UInt=1)
  self = new()

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
  const dimS::Int = 0 # sequence dimension
  const dimF::Int = 1 # feature dimension

  nInputFrame = size(input, dimS)
  nOutputFrame = div(nInputFrame - kW, dW) + 1
  output = m.output
  ensuresize!(output, nOutputFrame, outputFrameSize)

  # Copy bias first
  for k = 1:nOutputFrame
    output[k, :] = bias
  end

  for k = 1:nOutputFrame
    outputFrameStride = div(m.kW - 1, m.dW) + 1
    inputFrameStride = outputFrameStride * dW
    nFrame = div(nInputFrame - k * dW - kW, inputFrameStride) + 1;
    nOutputFrame -= nFrame

    inputWindow = unsafe_view(input, :, ((k - 1) * dW):((k - 1) * dW + kW))
    outputWindow = unsafe_view(output, :, k)

    gemm!('N', 'T', 1, inputWindow, weight, 1, outputWindow)
  end
end

function forward!{D<:Device,F<:Float}(m::TemporalConvolution, input::DenseArray{F, 3})
  nInputFrame::Int
  nOutputFrame::Int
  k::Int, i::Int

  const dimS::Int = 1 # sequence dimension
  const dimF::Int = 2 # feature dimension

  @assert(size(input, 1) >= m.kW, "input sequence smaller than kernel size")

  # input = THTensor_(newContiguous)(input)
  MAKE_CONTIGUOUS

  nInputFrame = input->size[dimS]
  nOutputFrame = (nInputFrame - kW) / dW + 1

  nBatchFrame = size(input, 1)


  resize!(output, nBatchFrame, nOutputFrame, output_framesize)

  for i = 1:nBatchFrame
    outputSample = slice(output, i, :, :)    # THTensor_(select)(outputSample, output, 0, i)
    inputSample = slice(input, i, :, :)     # THTensor_(select)(inputSample, input, 0, i)
    nOutputSampleFrame = nOutputFrame

    # Copy the bias first to all the outputs
    for k = 1:nOutputFrame
      outputWindow = slicedim(outputSample, 1, k) # THTensor_(select)(outputWindow, outputSample, 0, k)
      copy!(outputWindow, bias)                   # THTensor_(copy)(outputWindow, bias)
    end

    for k=1:nOutputSampleFrame
      outputFrameStride = div(kW - 1, dW) + 1
      inputFrameStride = outputFrameStride * dW
      nFrame = div(nInputFrame - k * dW - kW, inputFrameStride) + 1
      nOutputSampleFrame -= nFrame

      THTensor_(setStorage2d)(inputWindow, inputSample->storage,
                              inputSample->storageOffset+k*dW*inputSample->size[1], # offset
                              nFrame, inputFrameStride*inputSample->size[1], # size, stride
                              kW*inputSample->size[1], 1) # size, stride

      THTensor_(setStorage2d)(outputWindow, outputSample->storage, 
                              outputSample->storageOffset + k*outputSample->size[1],
                              nFrame, outputFrameStride*outputSample->size[1],
                              outputSample->size[1], 1)

      # THTensor_(transpose)(weight, NULL, 0, 1)
      # THTensor_(addmm)(outputWindow, 1, outputWindow, 1, inputWindow, weight)
      # THTensor_(transpose)(weight, NULL, 0, 1)
      gemm!('N', 'T', 1, inputWindow, weight, 1, outputWindow)
    end
  end
end


function compute_inputgradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix)
   if not(isnull(self.gradInput)) then
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);  
  int kW = luaT_getfieldcheckint(L, 1, "kW")
    int dW = luaT_getfieldcheckint(L, 1, "dW")
    long nInputFrame
    long nOutputFrame

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor)
    THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor)

  THTensor *gradOutputWindow
    THTensor *gradInputWindow
    long k, i

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (gradOutput->nDimension == 3) 
  {
    dimS = 1
        dimF = 2
      }
  
  nInputFrame = input->size[dimS]
    nOutputFrame = gradOutput->size[dimS]

  gradOutputWindow = THTensor_(new)()
    gradInputWindow = THTensor_(new)()

  THTensor_(resizeAs)(gradInput, input)
    THTensor_(zero)(gradInput)

 
    THTensor *gradOutputSample = THTensor_(new)()
        THTensor *gradInputSample = THTensor_(new)()
        int nBatchFrame = input->size[0]

    for i = 1:nBatchFrame
      THTensor_(select)(gradOutputSample, gradOutput, 0, i)
            THTensor_(select)(gradInputSample, gradInput, 0, i)
            int nOutputSampleFrame = nOutputFrame

      # ouch
      for k =  1:nOutputSampleFrame
        long outputFrameStride = (kW-1)/dW+1
                long inputFrameStride = outputFrameStride*dW
                long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1
                nOutputSampleFrame -= nFrame

        THTensor_(setStorage2d)(gradOutputWindow, gradOutputSample->storage,
                                gradOutputSample->storageOffset + k*gradOutputSample->size[1],
                                nFrame, outputFrameStride*gradOutputSample->size[1],
                                gradOutputSample->size[1], 1)

        THTensor_(setStorage2d)(gradInputWindow, gradInputSample->storage,
                                gradInputSample->storageOffset+k*dW*gradInputSample->size[1],
                                nFrame, inputFrameStride*gradInputSample->size[1],
                                kW*gradInputSample->size[1], 1)

        gemm!('N', 'N', 1, gradOutputWindow, weight, 1, gradOutputWindow)

        THTensor_(addmm)(gradInputWindow, 1, gradInputWindow, 1, gradOutputWindow, weight)
      end
    end


  return 1
       end
end

function update_gradient!{D<:Device, F<:Float}(linear::TemporalConvolution{D,F}, input::RealMatrix, gradOutput::RealMatrix, scale::F=1.)
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);  
  real scale = luaL_optnumber(L, 4, 1)
    int kW = luaT_getfieldcheckint(L, 1, "kW")
    int dW = luaT_getfieldcheckint(L, 1, "dW")
    long nInputFrame
    long nOutputFrame

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor)
    THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor)

  

  THTensor *gradOutputWindow
    THTensor *inputWindow
    long k, i

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (gradOutput->nDimension == 3) 
  {
    dimS = 1
        dimF = 2
      }
  
  nInputFrame = input->size[dimS]
    nOutputFrame = gradOutput->size[dimS]

  input = THTensor_(newContiguous)(input)
    gradOutputWindow = THTensor_(new)()
    inputWindow = THTensor_(new)()

    THTensor *gradOutputSample = THTensor_(new)()
        THTensor *inputSample = THTensor_(new)()
        int nBatchFrame = input->size[0]

    for(i = 0; i < nBatchFrame; i++)
    {
      THTensor_(select)(gradOutputSample, gradOutput, 0, i)
            THTensor_(select)(inputSample, input, 0, i)
            int nOutputSampleFrame = nOutputFrame

      # bias first
      for k=1:nOutputFrame
        THTensor_(select)(gradOutputWindow, gradOutputSample, 0, k)
        THTensor_(cadd)(gradBias, gradBias, scale, gradOutputWindow)
      end

      # ouch
      for k = 1:nOutputSampleFrame
        outputFrameStride = (kW-1)/dW+1
        long inputFrameStride = outputFrameStride*dW
        long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1
        nOutputSampleFrame -= nFrame

        THTensor_(setStorage2d)(inputWindow, inputSample->storage,
                                inputSample->storageOffset+k*dW*inputSample->size[1],
                                nFrame, inputFrameStride*inputSample->size[1],
                                kW*inputSample->size[1], 1)

        THTensor_(setStorage2d)(gradOutputWindow, gradOutputSample->storage, 
                                gradOutputSample->storageOffset + k*gradOutputSample->size[1],
                                nFrame, outputFrameStride*gradOutputSample->size[1],
                                gradOutputSample->size[1], 1)

        THTensor_(transpose)(gradOutputWindow, NULL, 0, 1)
                THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutputWindow, inputWindow)
                THTensor_(transpose)(gradOutputWindow, NULL, 0, 1)
      end
    }
  

end


