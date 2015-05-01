# Tips for translating torch7 code to julia

## Tensor functions 

### select(dim, index) 

    slicedim(A, dim, index)

Returns a new Tensor which is a tensor slice at the given `index` in the dimension `dim`


### `torch.addmm([res,] [beta,] [v1,] M [v2,] mat1, mat2)`

res = res * beta + v1 * M + v2 * mat1*mat2

    gemm!('N', 'N', v2, mat1, mat2, beta, res)

