#pragma once

#include "Kernel.h"

#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <iostream>

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/native/cuda/Indexing.cu

namespace at::native {

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<const T, IndexType> src,
                                      cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize);

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
class IndexSelectSmallIndex: public Kernel {
public:
    IndexSelectSmallIndex(
        cuda::detail::TensorInfo<T, IndexType> dst,
        cuda::detail::TensorInfo<const T, IndexType> src,
        cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
        int dstSelectDim,
        int srcSelectDim,
        IndexType innerSize,
        int64_t srcSelectDimSize,
        dim3 smallIndexGrid,
        dim3 smallIndexBlock,
        cudaStream_t stream):
        dst_(dst),
        src_(src),
        indices_(indices),
        dstSelectDim_(dstSelectDim),
        srcSelectDim_(srcSelectDim),
        innerSize_(innerSize),
        srcSelectDimSize_(srcSelectDimSize),
        smallIndexGrid_(smallIndexGrid),
        smallIndexBlock_(smallIndexBlock),
        stream_(stream)
    {
        // std::cout << "IndexSelectSmallIndex: [已入队]" << std::endl;
    }

    void execute() override {
        // std::cout << "IndexSelectSmallIndex: [已准备]" << std::endl;
        indexSelectSmallIndex<T, IndicesType, IndexType, DstDim, SrcDim, IdxDim>
            <<<smallIndexGrid_, smallIndexBlock_, 0, stream_>>>( 
                dst_, src_, indices_, dstSelectDim_, srcSelectDim_, static_cast<IndexType>(innerSize_), srcSelectDimSize_);
        // std::cout << "IndexSelectSmallIndex: [已执行]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    cuda::detail::TensorInfo<T, IndexType> dst_;
    cuda::detail::TensorInfo<const T, IndexType> src_;
    cuda::detail::TensorInfo<const IndicesType, IndexType> indices_;
    int dstSelectDim_;
    int srcSelectDim_;
    IndexType innerSize_;
    int64_t srcSelectDimSize_;
    dim3 smallIndexGrid_;
    dim3 smallIndexBlock_;
    cudaStream_t stream_;
};

}