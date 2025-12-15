#pragma once

#include "Kernel.h"

#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <iostream>

namespace at::native {

template <int nt, int vt, typename func_t>
__global__ void elementwise_kernel(int N, func_t f);

template <int nt, int vt, typename func_t>
class ElementwiseKernel: public Kernel {
public:
    ElementwiseKernel(
        int64_t N,
        const func_t& f,
        dim3 grid,
        dim3 block,
        at::cuda::CUDAStream stream):
        N_(N),
        f_(f),
        grid_(grid),
        block_(block),
        stream_(stream) 
    {
        // std::cout << "ElementwiseKernel: [已入队]" << std::endl;
    }

    void execute() override {
        // std::cout << "ElementwiseKernel: [已准备]" << std::endl;
        elementwise_kernel<nt, vt, func_t><<<grid_, block_, 0, stream_>>>(N_, f_);
        // std::cout << "ElementwiseKernel: [已发送]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    int64_t N_;
    const func_t& f_;
    dim3 grid_;
    dim3 block_;
    at::cuda::CUDAStream stream_;

    template <typename T>
    void print_type_name() {
        int status;
        char* realname = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
        std::cout << "func_t = " << realname << std::endl;
        free(realname);
    }
};

} // namespace at::native