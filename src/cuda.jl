# Support for CUDA

if Pkg.installed("CUDArt") == nothing
    macro cuda(ex)
    end
else
    using CUDArt

    macro cuda(ex)
        ex
    end

    immutable CudaDevice <: Device end

    export CudaDevice



end