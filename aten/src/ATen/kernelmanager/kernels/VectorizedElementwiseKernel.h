#pragma once

#include "Kernel.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <iostream>

namespace at::native {

template <int vec_size, typename func_t, typename array_t>
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data);

template <int vec_size, typename func_t, typename array_t>
class VectorizedElementwiseKernel: public Kernel {
public:
    VectorizedElementwiseKernel(
        int N,
        const func_t& f,
        array_t data,
        int64_t grid,
        int64_t block,
        at::cuda::CUDAStream stream):
        N_(N),
        f_(f),
        data_(data),
        grid_(grid),
        block_(block),
        stream_(stream)
    {
        std::cout << "VectorizedElementwiseKernel: [已入队]" << std::endl;
    }

    void execute() override {
        std::cout << "VectorizedElementwiseKernel: [已准备]" << std::endl;
        vectorized_elementwise_kernel<vec_size, func_t, array_t><<<grid_, block_, 0, stream_>>>(N_, f_, data_);
        std::cout << "VectorizedElementwiseKernel: [已发送]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    int N_;
    const func_t& f_;
    array_t data_;
    int64_t grid_;
    int64_t block_;
    at::cuda::CUDAStream stream_;
};

}