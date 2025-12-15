#pragma once

#include "Kernel.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK

#include <iostream>

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/native/cuda/Reduce.cuh

namespace at::native{

template<int nt, int output_vec_size, typename R>
__global__ void reduce_kernel(R reduction);

template<int nt, int output_vec_size, typename R>
class ReduceKernel: public Kernel {
public:
    ReduceKernel(
        const R& reduction,
        dim3 grid,
        dim3 block,
        int shared_memory,
        at::cuda::CUDAStream stream):
        reduction_(reduction),
        grid_(grid),
        block_(block),
        shared_memory_(shared_memory),
        stream_(stream)
    {
        // std::cout << "ReduceKernel: [已入队]" << std::endl;
    }

    void execute() override{
        // std::cout << "ReduceKernel: [已准备]" << std::endl;
        reduce_kernel<nt, output_vec_size, R><<<grid_, block_, shared_memory_, stream_>>>(reduction_);
        // std::cout << "ReduceKernel: [已发送]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    const R& reduction_;
    const dim3 grid_;
    const dim3 block_;
    const int shared_memory_;
    const at::cuda::CUDAStream stream_;
};

}