@doc doc"Hyperbolic tangent transfer layer" ->
type TanH
    output::DenseRealTensor
    grad_input::DenseRealTensor
end

function forward!(m::TanH, input::DenseMatrix)
    self.output = tanh(input)
end

function compute_inputgradient!(m::TanH, grad_output::DenseMatrix)
    @simd for i = eachindex(grad_output.values)
        self.grad_input[i] = (one(F) - m.output[i] * m.output[i]) * grad_output[i]
    end
    self.grad_input
end