
# module optimization

export optimize

abstract Optimizer

@doc doc"Modifies "
function optimize!(s::Optimizer, m::Operator)
  for v in parameters(m)
    optimize!(s, v)
  end
end

# --- Stochastic Gradient
#= A plain implementation of SGD
ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.learningRates`      : vector of individual learning rates

=#

type StochasticGradient
    learningRate::Float64
end

function optimize!(s::StochasticGradient, p::ArrayParameters)
    axpy!(-s.learningRate, p.gradient, p.values)
end



export StochasticGradient

# end