 type _TreeState
   leaveIx::Int
   innerIx::Int
 end

 # a,b is the span of the subtree
 # terminates when a == b
 function construct_tree(parents::Vector{Int}, a::Int, b::Int)
   function _construct_tree(parents, a, b, state::_TreeState)
       #println("Split $a and $b with state $state")
       if a == b then
           state.leaveIx = state.leaveIx + 1
           return state.leaveIx
       end

       # Construct subtrees
       local split = rand(0:(b - a - 1))
       local leftIx = _construct_tree(parents, a, a + split, state)
       local rightIx = _construct_tree(parents, a + split + 1, b, state)

       state.innerIx = state.innerIx + 1
       # println("Node $(state.innerIx) ($a-$b), children $leftIx and $rightIx")
       parents[leftIx] = -state.innerIx
       parents[rightIx] = state.innerIx
       return state.innerIx
   end

   parents[size(parents, 1)] = 0
   return _construct_tree(parents, a, b, _TreeState(0, div(size(parents, 1) + 1, 2)))
 end


 #######

function generate_bhs(nleaves, input_size)
  parents = Array(Int, 2 * nleaves - 1)
  construct_tree(parents, 1, nleaves)
  hs = BinaryHierarchicalSoftmax{CPUDevice, Float64}(input_size, parents)
end

input_size = 100
hs = generate_bhs(200000, input_size)
init_gradient!(hs)

words = reshape([i for i=1:hs.nleaves], hs.nleaves, 1)
x = randn(1, input_size)
input = repmat(x, hs.nleaves)
output = forward!(hs, Any[input, words])
@assert abs(sum(exp(output)) - 1) / hs.nleaves <= 1e-16

# -- Timeit

ninput = 10000
input = randn(ninput, input_size)
targets = rand(1:hs.nleaves, ninput, 1)
println("Size of input = $(size(input)), # leaves = $(hs.nleaves)")
for i=1:10
  @time forward!(hs, Any[input, targets])
end


gradOutput = fill(-1., (ninput, 1)) # We want to increase sum(o) => decrease -sum(o)
sgd = StochasticGradient(1e-3)

inputs = Any[input, targets]
@time for i = 1:10
  init_gradient!(hs)
  forward!(hs, inputs)
  backward!(hs, inputs, gradOutput)
  optimize!(sgd, hs)
  println("Cost = $(sum(hs.output))")
end



