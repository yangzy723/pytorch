#pragma once

#include "Kernel.h"

#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <iostream>
#include <cxxabi.h>

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/native/cuda/RangeFactories.cu

namespace at::native {

template<typename index_t, typename func_t>
__global__ void elementwise_kernel_with_index(index_t N, func_t f, typename function_traits<func_t>::result_type *data);

template<typename index_t, typename func_t>
class ElementwiseKernelWithIndex: public Kernel {
public:
    ElementwiseKernelWithIndex(
        index_t N,
        func_t f,
        typename function_traits<func_t>::result_type *data,
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
        // std::cout << "ElementwiseKernelWithIndex: [已入队]" << std::endl;
        // print_type_name<func_t>();
    }
    
    void execute() override {
        // std::cout << "ElementwiseKernelWithIndex: [已准备]" << std::endl;
        elementwise_kernel_with_index<index_t>
            <<<grid_, block_, 0, stream_>>>(N_, f_, data_);
        // std::cout << "ElementwiseKernelWithIndex: [已执行]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    index_t N_;
    func_t f_;
    typename function_traits<func_t>::result_type *data_;
    int64_t grid_;
    int64_t block_;
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