#=

Binary Hierarchical SoftMax

A hierarchical softmax with a binary tree

B. Piwowarski
April 2015

=#

type BinaryHierarchicalSoftmaxUpdates
  offsets::Array{Int}
  nodes::Array{Int}
  values::RealVector
end

# Holds the different structures
type BinaryHierarchicalSoftmax{T<:_Real} <: Module
  # List of parents, using the sign for denoting which branch
  parents::Vector{Int}

  # The number of leaves
  nleaves::Int

  # The depth of each leave
  depth::Vector{Int}

  weight::RealMatrix
  bias::RealVector

  output::RealMatrix
  gradInput::RealVector

  gradWeight::RealMatrix
  gradBias::RealVector

  updates::BinaryHierarchicalSoftmaxUpdates

  function BinaryHierarchicalSoftmax(inputSize::Int, parents::Vector{Int})
    self = new()

    self.parents = parents
    self.nleaves = div(size(parents, 1) + 1, 2)

    nbInnerNodes = size(parents, 1) - nleaves

    @assert(sum(parents) == 0, "Tree is not balanced")
    @assert(parents[size(parents,1)] == 0, "Root is not the last node")

    # Computes the depth at each inner node
    # (this is useful to store the updates)
    function compute_depth(ix::Int)
        if self.depth[ix] == -1 then
           local pix = abs(self.parents[ix + nleaves])
           if pix == 0 then
              self.depth[ix] = 0
           else
              self.depth[ix] = compute_depth(pix - nleaves) + 1
           end
        end
        return self.depth[ix]
     end

     self.depth = fill(-1, nbInnerNodes)
     for ix = 1:nbInnerNodes
        compute_depth(ix)
     end


     # Initialize the different fields

     self.weight = array(T, nbInnerNodes, inputSize)
     self.bias = array(T, nbInnerNodes)

     self.gradWeight = array(T, nbInnerNodes, inputSize)
     self.gradBias = array(T, nbInnerNodes)

     #  No gradient for the targets
     self.gradInput = { array(T), 0 }

     self.output = array(T, 0, 0)

     #  Stores updates
     self.updates = BinaryHierarchicalSoftmaxUpdates(Array(Int, 0), Array(Int, 0), array(T, 0))

     reset(self)

     return self
  end

end

# Resets
function reset{T<:_Real}(self::BinaryHierarchicalSoftmax{T}, stdv=0)
  if stdv != 0 then
    stdv = stdv * sqrt(3)
  else
    stdv = 1. / sqrt(size(self.weight, 2))
  end

  function r!(x)
    rand!(x)
    broadcast!(-, x, x, 0.5)
    broadcast!(*, x, x, stdv)
  end

  r!(self.weight)
  r!(self.bias)
end


function forward(self::BinaryHierarchicalSoftmax{CPUDouble}, inputs)
  input, targets = inputs

  T= CPUDouble

  @assert(ndims(input) == 1 || ndims(input) == 2, "input must be vector or matrix")
  @assert(size(input, 1) == size(targets, 1), "Batch size of inputs [$(size(inputs, 1))] and targets [$(size(targets, 1))] differ")

   nframe = size(input, 1)
   if size(self.output) != (nframe, 1)
     self.output = array(T, nframe, 1)
   end
   output = self.output

    #  Stores the heavy computation before computing gradInput or updating gradient
    tdepth::Int = 0
    for frame = 1:nframe
       local current::Int = self.parents[targets[frame]]
       local ix::Int = abs(current) - self.nleaves
       tdepth = tdepth + self.depth[ix] + 1
    end

    resize!(self.updates.offsets, nframe + 1)
    resize!(self.updates.nodes, tdepth)
    resize!(self.updates.values, tdepth)

    local offset::Int = 1

    for frame = 1:nframe
       self.updates.offsets[frame] = offset

       local current::Int = self.parents[targets[frame]]
       local logp::Float64 = 0
       while current != 0
          sign = 1.
          if current <= 0 then
             sign = -1.
             current = -current
          end

          local ix::Int = current - self.nleaves
          local current_p::Float64 = 1 + exp(sign * (dot(self.weight[ix], input[frame]) + self.bias[ix]))

          self.updates.nodes[offset] = current
          self.updates.values[offset] = sign * (1. / current_p - 1.)

          logp = logp - log(current_p)
          current = self.parents[current]
          offset = offset + 1

       end

       #  Set the output
       output[frame, 1] = logp
    end

    #  Marks the end
    self.updates.offsets[nframe+1] = offset
    return self.output
 end

 function updateGradInput{T<:Real}(self::BinaryHierarchicalSoftmax{T}, inputs, gradOutput)
    input, targets = inputs

    if self.gradInput != 0 then
       local gradInput0 = self.gradInput[1]
       if size(gradInput0) != size(input) then
         gradInput0 = self.gradInput[1] = zeros(T, size(input))
       end

       for frame = 1:(size(self.updates.offsets, 1) - 1)
          for j = self.updates.offsets[frame]:(self.updates.offsets[frame+1] - 1)
             local current = self.updates.nodes[j]
             local ix = current - self.nleaves
             gradInput0[frame] += gradOutput[frame] * self.updates.values[j], self.weight[ix]
          end
       end

       return self.gradInput
    end
 end


function accGradParameters{T<:Real}(self::BinaryHierarchicalSoftmax{T}, inputs, gradOutput, scale::Float64=1.)
  input, targets = inputs

  @assert(ndims(input) == 2, "Input must be a matrix")

  nframe = input:size(1)

  for frame = 1:(self.updates.offsets:size(1) - 1)
    for j = self.updates.offsets[frame]:(self.updates.offsets[frame+1] - 1)
      local current = self.updates.nodes[j]
      local ix = current - self.nleaves
      local c = scale * self.updates.values[j] * gradOutput[frame]

      self.gradWeight[ix] += c * input[frame]
      self.gradBias[ix] += c
    end
  end
 end





 ####### RANDOM TREE


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


nleaves = 100000
parents = Array(Int, 2 * nleaves - 1)
construct_tree(parents, 1, nleaves)

input_size = 100
hs = BinaryHierarchicalSoftmax{CPUDouble}(input_size, parents)

words = reshape([i for i=1:nleaves], nleaves, 1)
x = randn(1, input_size)
input = repmat(x, nleaves)
output = forward(hs, {input, words})
println(sum(exp(output)))

# -- Timeit

ninput = 10000
input = randn(ninput, input_size)
targets = rand(1:nleaves, ninput, 1)
println("Size of input = $(size(input)), # leaves = $nleaves")
for i=1:10
   t = @time forward(hs, {input, targets})
end
