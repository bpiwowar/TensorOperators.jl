
# module optimization

export optimize!

abstract Optimizer

@doc doc"Modifies "
function optimize!(s::Optimizer, m::Operator)
  for v in parameters(m)
    optimize!(s, v)
  end
end

# -# Stochastic Gradient
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

type StochasticGradient <: Optimizer
    learningRate::Float64
    # learningRateDecay::Float64

    # weithDecay::Float64
    # momentum::Float64
    
    # @doc doc"dampening for momentum" ->
    # dampening::Float64

    # @doc doc"Whether we use Nesterov momentum"
    # nesterov::Bool

    StochasticGradient(learningRate) = new(learningRate)
end

function optimize!(s::StochasticGradient, p::ArrayParameters)
    # TODO : Implement momentum
    axpy!(-s.learningRate, p.gradient, p.values)
end


@doc doc"RProp optimizer
- `state.stepsize`    : initial step size, common to all components
- `state.etaplus`     : multiplicative increase factor, > 1 (default 1.2)
- `state.etaminus`    : multiplicative decrease factor, < 1 (default 0.5)
- `state.stepsizemax` : maximum stepsize allowed (default 50)
- `state.stepsizemin` : minimum stepsize allowed (default 1e-6)
- `state.niter`       : number of iterations (default 1)    
"
immutable RProp <: Optimizer
    @doc "Initial step size"
    stepsize::Float64

    etaplus::Float64
    etaminus::Float64

    stepsizemax::Float64
    stepsizemin::Float64


    function RProp()
        self = new()
        self.stepsize = 0.1
        self.etaplus = 1.2
        self.etaminus = 0.5
        self.stepsizemax = 50.0
        self.stepsizemin = 1e-6
        self
    end

    RProp(stepsize) = new(stepsize, 1.2, 0.5, 50., 1e-6)
    RProp() = new(0.1)
end

type RPropState
    delta
    stepsize

    @doc doc"11 = zero, 01 = negative, 00 = positive, 11 = unassigned"
    sign::BitArray{2}   

    function RPropState(l::Int, stepsize::Float64)
        self = new()
        self.delta = zeros(l)
        self.stepsize = fill(stepsize, l)
        self.sign = BitArray(l, 2)
        self
    end
end


const RProp_POSITIVE = bitpack([false, false])
const RProp_NEGATIVE = bitpack([false, true])
const RProp_ZERO = bitpack([true, true])

function optimize!(s::RProp, p::ArrayParameters)
    # init temp storage
    if isnull(p.optimization_state)
        p.optimization_state = Nullable(RPropState(length(p.values), s.stepsize))
    end

    state = get(p.optimization_state)::RPropState

    for i in eachindex(p.gradient)
        # Compute the new step size
        sign = p.gradient[i] > 0 ? RProp_POSITIVE : (p.gradient[i] < 0 ? RProp_NEGATIVE : RProp_ZERO)

        if sign != RProp_ZERO 
            if sign == state.sign[i]
                state.stepsize[i] = min(state.stepsize[i] * s.etaplus, s.stepsizemax)
            else
                state.stepsize[i] = max(state.stepsize[i] * s.etaminus, s.stepsizemin)
            end            
        end

        # Update weight
        p.values[i] -= state.stepsize[i] * p.gradient[i]
    end

    println(sum(state.stepsize) / length(p.gradient))

end

export StochasticGradient, RProp

# end