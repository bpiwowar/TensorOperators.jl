#=

Binary Hierarchical SoftMax

A hierarchical softmax with a binary tree

B. Piwowarski
April 2015

=#

type BinaryHierarchicalSoftmaxUpdates
  size::Int
  offsets::Vector{Int}
  nodes::Vector{Int}
  values
end

# Holds the different structures
type BinaryHierarchicalSoftmax{D<:Device, F<:Float} <: Operator
  # List of parents, using the sign for denoting which branch
  parents::Vector{Int}

  # The number of leaves
  nleaves::Int

  # The depth of each leave
  depth::Vector{Int}

  weight::MatrixParameters{D,F}
  bias::VectorParameters{D,F}

  output::RealMatrix
  grad_input::RealVector

  updates::BinaryHierarchicalSoftmaxUpdates

  function BinaryHierarchicalSoftmax(inputSize::Int, parents::Vector{Int})
    self = new()


    self.parents = parents
    self.nleaves = div(size(parents, 1) + 1, 2)

    nbInnerNodes = size(parents, 1) - self.nleaves

    @assert(sum(parents) == 0, "Tree is not balanced")
    @assert(parents[size(parents,1)] == 0, "Root is not the last node")

    # Computes the depth at each inner node
    # (this is useful to store the updates)
    function compute_depth(ix::Int)
        if self.depth[ix] == -1 then
           local pix = abs(self.parents[ix + self.nleaves])
           if pix == 0 then
              self.depth[ix] = 0
           else
              self.depth[ix] = compute_depth(pix - self.nleaves) + 1
           end
        end
        return self.depth[ix]
     end

     self.depth = fill(-1, nbInnerNodes)
     for ix = 1:nbInnerNodes
        compute_depth(ix)
     end


     # Initialize the different fields
     self.weight = MatrixParameters{D,F}(nbInnerNodes, inputSize)
     self.bias = VectorParameters{D,F}(nbInnerNodes)

     #  No gradient for the targets
     self.grad_input = Any[ array(D, F), 0 ]

     self.output = array(D, F, 0, 0)

     #  Stores updates
     self.updates = BinaryHierarchicalSoftmaxUpdates(0, Array(Int, 0), Array(Int, 0), array(D, F, 0))

     reset(self)

     return self
  end

end


export BinaryHierarchicalSoftmax

# Resets
function reset{D<:Device, F<:Float}(self::BinaryHierarchicalSoftmax{D, F}, stdv=0)
  if stdv != 0 then
    stdv = stdv * sqrt(3)
  else
    stdv = 1. / sqrt(size(self.weight.values, 2))
  end

  function r!(x)
    rand!(x)
    broadcast!(-, x, x, 0.5)
    broadcast!(*, x, x, stdv)
  end

  r!(self.weight.values)
  r!(self.bias.values)
end


function forward!{F<:Float}(self::BinaryHierarchicalSoftmax{CPUDevice, F}, inputs::Array)
  const D = CPUDevice
  input::matrixOf(D, F), targets::Array{Int, 2} = inputs

  @assert(ndims(input) == 1 || ndims(input) == 2, "input must be vector or matrix")
  @assert(size(input, 1) == size(targets, 1), "Batch size of inputs [$(size(inputs, 1))] and targets [$(size(targets, 1))] differ")

   nframe::Int = size(input, 1)
   if size(self.output) != (nframe, 1)
     self.output = array(D, F, nframe, 1)
   end

   output::matrixOf(D,F) = self.output

    #  Stores the heavy computation before computing grad_input or updating gradient
    tdepth::Int = 0
    for frame = 1:nframe
       local current::Int = self.parents[targets[frame]]
       local ix::Int = abs(current) - self.nleaves
       tdepth = tdepth + self.depth[ix] + 1
    end

    self.updates.size = nframe
    if size(self.updates.offsets, 1) < nframe + 1 then
      resize!(self.updates.offsets, nframe + 1)
    end
    if size(self.updates.nodes, 1) < tdepth
      resize!(self.updates.nodes, tdepth)
      resize!(self.updates.values, tdepth)
    end

    # Create type stable values
    offset::Int = 1
    @stabilize values::vectorOf(D,F) = self.updates.values
    @stabilize weight::matrixOf(D,F) = self.weight.values
    @stabilize bias::vectorOf(D,F) = self.bias.values
    current_p::F = zero(F)

    # ---
    for frame = 1:nframe
       self.updates.offsets[frame] = offset

       local current::Int = self.parents[targets[frame, 1]]
       local logp::Float64 = 0
       while current != 0
          sign::Float64 = 1.
          if current <= 0 then
             sign = -1.
             current = -current
          end

          local ix::Int = current - self.nleaves
          current_p = 1. + exp(sign * (Base.dot(weight[ix], input[frame]) + bias[ix]))

          self.updates.nodes[offset] = current
          values[offset] = sign * (1. / current_p - 1.)

          logp = logp - log(current_p)
          current = self.parents[current]
          offset = offset + 1

       end

       #  Set the output
       output[frame, 1] = logp
    end

    #  Marks the end
    self.updates.offsets[nframe+1] = offset
    return output
 end

 function compute_inputgradient!{F<:Float}(self::BinaryHierarchicalSoftmax{CPUDevice, F}, inputs, gradOutput)
    input, targets = inputs

    if self.grad_input != 0 then
       local _grad_input0 = self.grad_input[1]
       if size(_grad_input0) != size(input) then
         _grad_input0 = self.grad_input[1] = zeros(D, F, size(input)...)
       end
       grad_input0::matrixOf(D,F) = _grad_input0

       values::vectorOf(D,F) = self.updates.values
       weight::matrixOf(D,F) = self.weight.values

       for frame = 1:self.updates.size
          for j = self.updates.offsets[frame]:(self.updates.offsets[frame+1] - 1)
             local current = self.updates.nodes[j]
             local ix = current - self.nleaves
             grad_input0[frame] += gradOutput[frame] * values[j] * weight[ix]
          end
       end

       return self.grad_input
    end
 end


function update_gradient!{D<:Device, F<:Float}(self::BinaryHierarchicalSoftmax{D, F}, inputs, gradOutput, scale::F=1.)
  input::matrixOf(D,F), targets = inputs

  @assert(ndims(input) == 2, "Input must be a matrix (ndims=$(ndims(input)))")

  values::vectorOf(D,F) = self.updates.values
  gradWeight::matrixOf(D,F) = self.weight.gradient
  gradBias::vectorOf(D,F) = self.bias.gradient

  for frame = 1:self.updates.size
    for j = self.updates.offsets[frame]:(self.updates.offsets[frame+1] - 1)
      local current = self.updates.nodes[j]
      local ix = current - self.nleaves
      local c = scale * values[j] * gradOutput[frame]

      gradWeight[ix] += c * input[frame]
      gradBias[ix] += c
    end
  end
 end



