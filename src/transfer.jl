@doc doc"Hyperbolic tangent transfer layer"
type TanH{D<:Device, F<:Float}
    output::RealArray
    grad_input::RealArray
end

function forward!{D<:Device, F<:Float}(m::TanH{D,F}, input::RealMatrix)
    self.output = tanh(input)
end

function compute_inputgradient!{D<:Device, F<:Float}(m::TanH{D,F}, grad_output::RealMatrix)
    @simd for i = eachindex(grad_output.values)
        self.grad_input[i] = (one(F) - m.output[i] * m.output[i]) * grad_output[i]
    end
end