
# module optimization
# TODO:
# (1) make a wrapper for Optim package

export optimize!, StochasticGradient, RProp


abstract Optimizer

@doc doc"Loop over parameters and call the optimizer for each parameter set with non null gradient" ->
function optimize!(s::Optimizer, m::Layer)
  for v in parameters(m)
    if !isnull(v.gradient)
        optimize!(s, v)
    end
  end
end


@doc doc"Stochatic gradient" ->
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
    axpy!(-s.learningRate, get(p.gradient), p.values)
end




#
# RProp optimizer
#

@doc doc"RProp optimizer
- stepsize    : initial step size, common to all components
- etaplus     : multiplicative increase factor, > 1 (default 1.2)
- etaminus    : multiplicative decrease factor, < 1 (default 0.5)
- stepsizemax : maximum stepsize allowed (default 50)
- stepsizemin : minimum stepsize allowed (default 1e-6)
- niter       : number of iterations (default 1)
" ->
immutable RProp <: Optimizer
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
    # initialize auxiliary storage
    if isnull(p.optimization_state)
        p.optimization_state = Nullable(RPropState(length(p.values), s.stepsize))
    end

    state = get(p.optimization_state)::RPropState
    gradient = get(p.gradient)

    @inbounds for i in eachindex(gradient)
        # Compute the new step size
        sign = gradient[i] > 0 ? RProp_POSITIVE : (gradient[i] < 0 ? RProp_NEGATIVE : RProp_ZERO)

        if sign != RProp_ZERO
            if sign == state.sign[i]
                state.stepsize[i] = min(state.stepsize[i] * s.etaplus, s.stepsizemax)
            else
                state.stepsize[i] = max(state.stepsize[i] * s.etaminus, s.stepsizemin)
            end
        end

        # Update weight
        p.values[i] -= state.stepsize[i] * gradient[i]
    end

end

