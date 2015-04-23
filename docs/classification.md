# Softmax

## Hierarchical Softmax

### Binary Hierarchical Softmax

The `BinaryHierarchicalSoftmax` module implements a version of the hierarchical softmax where the tree is binary. For *n* classes,
It can be constructed with
```julia
BinaryHierarchicalSoftmax{D, F}(input_size::Int, parents::Vector{Int})
```
where:

- `input_size` is the size of the input vectors
- `parents` is an array of integers of the same of length of the number of nodes in the tree. Each corresponding index indicates which the index of the parent of the node. The sign of the integer gives an indication on the branch. **The parent node should be the last one in the array** and should have a value of 0 (no parent).

There is some basic checking of the `parents` array (sum of parent indices is 0, last value of the array is 0), but there is no further checking at the moment.