@doc doc"Hyperbolic tangent transfer layer" ->
type TanH
    output::RealArray
    grad_input::RealArray
end

function forward!(m::TanH, input::RealMatrix)
    self.output = tanh(input)
end

function compute_inputgradient!(m::TanH, grad_output::RealMatrix)
    @simd for i = eachindex(grad_output.values)
        self.grad_input[i] = (one(F) - m.output[i] * m.output[i]) * grad_output[i]
    end
end