#=

  Computes f(x,y) for each entry

=#
export ComponentWiseBiFunction

type ComponentWiseBiFunction{D<:Device, F<:Float} <: Layer
  output::denseRealMatrix(D,F)
  left_gradient::denseRealMatrix(D,F)
  right_gradient::denseRealMatrix(D,F)

  ComponentWiseBiFunction() = new(array(device, F, 0, 0), array(device, F, 0, 0))
end


function forward!{D<:Device, F<:Float}(linear::ComponentWiseBiFunction{D,F}, x::DenseMatrix, y::DenseMatrix)
    @assert size(x) == size(y)
    @ensuresize
end

function compute_inputgradient!{D<:Device, F<:Float}(linear::ComponentWiseBiFunction{D,F}, input::DenseMatrix, gradOutput::DenseMatrix)
  @ensuresize linear.input_gradient, (size(gradOutput,1), size(linear.weight.values, 1))

  gemm!('N', 'T', 1., gradOutput, linear.weight.values, 0., linear.input_gradient)
end
