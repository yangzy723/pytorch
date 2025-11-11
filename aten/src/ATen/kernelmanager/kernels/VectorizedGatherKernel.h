#include "Kernel.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>        // at::cuda::CUDAStream
#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK

#include <iostream>

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/native/cuda/IndexKernelUtils.cu

namespace at::native {

template <int Alignment, typename index_t>
__global__ void vectorized_gather_kernel(char * out, char * inp, index_t * idx, int num_ind, int64_t slice_size, int64_t ind_dim_size, int64_t inp_stride, int64_t out_stride, bool allow_neg_indices);

template <int Alignment, typename index_t>
class VectorizedGatherKernel: public Kernel {
public:
    VectorizedGatherKernel(
        char *out,
        char *inp,
        index_t *idx,
        int num_ind,
        int64_t slice_size,
        int64_t ind_dim_size,
        int64_t inp_stride,
        int64_t out_stride,
        bool allow_neg_indices,
        dim3 grid,
        int64_t block,
        at::cuda::CUDAStream stream):
        out_(out),
        inp_(inp),
        idx_(idx),
        num_ind_(num_ind),
        slice_size_(slice_size),
        ind_dim_size_(ind_dim_size),
        inp_stride_(inp_stride),
        out_stride_(out_stride),
        allow_neg_indices_(allow_neg_indices),
        grid_(grid),
        block_(block),
        stream_(stream)
    {
        // std::cout << "VectorizedGatherKernel: [已入队]" << std::endl;
    }

    void execute() override{
        // std::cout << "VectorizedGatherKernel: [已准备]" << std::endl;
        
        vectorized_gather_kernel<Alignment, index_t>
            <<<grid_, block_, 0, stream_>>>
                (out_, inp_, idx_, num_ind_, slice_size_, ind_dim_size_, inp_stride_, out_stride_, allow_neg_indices_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // std::cout << "VectorizedGatherKernel: [已发送]" << std::endl;
    }

private:
    char *out_;
    char *inp_;
    index_t *idx_;
    int num_ind_;
    int64_t slice_size_;
    int64_t ind_dim_size_;
    int64_t inp_stride_;
    int64_t out_stride_;
    bool allow_neg_indices_;
    dim3 grid_;
    int64_t block_;
    at::cuda::CUDAStream stream_;
};

}