#pragma once

#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <iostream>

// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...) do {                                       \
  size_t temp_storage_bytes = 0;                                          \
  AT_CUDA_CHECK(func(nullptr, temp_storage_bytes, __VA_ARGS__));          \
  auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get();    \
  auto temp_storage = caching_allocator.allocate(temp_storage_bytes);     \
  AT_CUDA_CHECK(func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__));\
} while (false)

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/cuda/cub.cuh

namespace at::cuda::cub {

template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT>
class InclusiveScan: public Kernel {
public:
    InclusiveScan(
        InputIteratorT input,
        OutputIteratorT output,
        ScanOpT scan_op,
        int size_cub,
        at::cuda::CUDAStream stream
    ):
        input_(input),
        output_(output),
        scan_op_(scan_op),
        size_cub_(size_cub),
        stream_(stream)
    {
        // std::cout << "KernelScan: [已入队]" << std::endl;
    }

    void execute() override {
        // std::cout << "KernelScan: [已准备]" << std::endl;
        CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveScan,
            input_,
            output_,
            scan_op_,
            size_cub_,
            stream_);
        // std::cout << "KernelScan: [已执行]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    InputIteratorT input_;
    OutputIteratorT output_;
    ScanOpT scan_op_;
    int size_cub_;
    at::cuda::CUDAStream stream_;
};

} // namespace at::cuda::cub