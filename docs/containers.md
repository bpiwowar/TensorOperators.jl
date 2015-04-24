# Sequence

A simple sequence of modules: in the forward phase, all the contained modules are called with the output of one being the input of the next; in the backward, the gradient with respect to the inputs is fed to the backward of the previous module.

# Parallel

The inputs are fed to each