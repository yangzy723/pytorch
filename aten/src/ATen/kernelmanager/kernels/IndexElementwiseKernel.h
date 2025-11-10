#pragma once

#include "Kernel.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <iostream>
#include <cxxabi.h>

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/native/cuda/IndexKernel.cu

namespace at::native {

template <int nt, int vt, typename func_t>
__global__ void index_elementwise_kernel(const int64_t N, const func_t f);

template <int nt, int vt, typename func_t>
class IndexElementwiseKernel: public Kernel {
public:
    IndexElementwiseKernel(
        const int64_t N,
        const func_t& f,
        const dim3 grid,
        const dim3 block,
        const at::cuda::CUDAStream stream):
        N_(N),
        f_(f),
        grid_(grid),
        block_(block),
        stream_(stream)
    {
        // std::cout << "IndexElementwiseKernel: [已入队]" << std::endl;
        // print_type_name<func_t>();
    }

    void execute() override {
        // std::cout << "IndexElementwiseKernel: [已准备]" << std::endl;
        index_elementwise_kernel<nt, vt, func_t><<<grid_, block_, 0, stream_>>>(N_, f_);
        // std::cout << "IndexElementwiseKernel: [已发送]" << std::endl;
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:

    const int64_t N_;
    const func_t& f_;
    const dim3 grid_;
    const dim3 block_;
    const at::cuda::CUDAStream stream_;

    template <typename T>
    void print_type_name() {
        int status;
        char* realname = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
        std::cout << "func_t = " << realname << std::endl;
        free(realname);
    }
};

}