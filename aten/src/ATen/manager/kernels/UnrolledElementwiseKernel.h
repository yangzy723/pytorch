# pragma once

#include "Kernel.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <iostream>
#include <cxxabi.h>

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/native/cuda/CUDALoops.cuh

namespace at::native {

template <
    typename func_t,
    typename array_t,
    int elems_per_thread,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
__global__ void unrolled_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s);

template <
    typename func_t,
    typename array_t,
    int elems_per_thread,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
class UnrolledElementwiseKernel: public Kernel {
public:
    UnrolledElementwiseKernel(
        int N,
        func_t f,
        array_t data,
        inp_calc_t ic,
        out_calc_t oc,
        loader_t l,
        storer_t s,
        int64_t grid_unrolled,
        int64_t block_unrolled,
        at::cuda::CUDAStream stream):
        N_(N),
        f_(f),
        data_(data),
        ic_(ic),
        oc_(oc),
        l_(l),
        s_(s),
        grid_unrolled_(grid_unrolled),
        block_unrolled_(block_unrolled),
        stream_(stream)
    {
        // std::cout << "UnrolledElementwiseKernel: [已入队]" << std::endl;
        // print_type_name<func_t>();
    }

    void execute() override {
        // std::cout << "UnrolledElementwiseKernel: [已准备]" << std::endl;
        unrolled_elementwise_kernel<func_t, array_t, elems_per_thread, inp_calc_t, out_calc_t, loader_t, storer_t>
            <<<grid_unrolled_, block_unrolled_, 0, stream_>>>(N_, f_, data_, ic_, oc_, l_, s_);
        // std::cout << "UnrolledElementwiseKernel: [已发送]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    int N_;
    func_t f_;
    array_t data_;
    inp_calc_t ic_;
    out_calc_t oc_;
    loader_t l_;
    storer_t s_;
    int64_t grid_unrolled_;
    int64_t block_unrolled_;
    at::cuda::CUDAStream stream_;

    template <typename T>
    void print_type_name() {
        int status;
        char* realname = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
        std::cout << "func_t = " << realname << std::endl;
        free(realname);
    }
};

}
