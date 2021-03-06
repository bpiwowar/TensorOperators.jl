# --- L2 Cost

type L2Cost <: Layer
    output
    grad_input
    L2Cost() = new()
end
export L2Cost

# Defines forward & backward
forward!(cost::L2Cost, inputs) = cost.output = .5 * norm(inputs[1] - inputs[2])
backward!(cost::L2Cost, inputs, scale=1.) = cost.grad_input = scale * (inputs[1] - inputs[2])
