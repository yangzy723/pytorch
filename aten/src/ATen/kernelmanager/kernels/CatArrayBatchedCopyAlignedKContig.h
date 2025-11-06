#pragma once

#include "Kernel.h"

#include <c10/cuda/CUDAException.h>     // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cuda_runtime_api.h>
#include <iostream>

namespace at::native {

constexpr int CAT_ARRAY_BATCH_SIZE = 128;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;
constexpr int ALIGNED_VEC_LOAD_BYTES_16 = 16;
constexpr int ALIGNED_VEC_LOAD_BYTES_8 = 8;

template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline __device__ IndexType compute(
      const IndexType tensorSize[Dims],
      const IndexType tensorStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    // linearIndex is not really linear index, but instead the offset in
    // input tensor. If the input tensor is contiguous, then this offset
    // is the linear index, but if the input tensor is channels last, then
    // it is the linear index of the permuted contiguous tensor
    IndexType offset = 0;

    #pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : tensorSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * tensorStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * tensorStride[0];
  }
};

template<typename IndexType, unsigned int MaxDims>
struct TensorSizeStride {
  IndexType tensorSize[MaxDims];
  IndexType tensorStride[MaxDims];
};

template <typename T, typename IndexType, int n, int stride_size>
struct CatArrInputTensorMetadata {
  const T* input[n];
  IndexType offset[n];
  IndexType dimSize[n];
  IndexType nElements[n];
  bool isContiguous[n];
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> tensorStride[stride_size];
};

template <typename T, typename IndexType, int Dims, int batch_size, int stride_size, int aligned_vec_load_bytes>
__global__ void CatArrayBatchedCopy_alignedK_contig(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride);

template <typename T, typename IndexType, int Dims, int batch_size, int stride_size, int aligned_vec_load_bytes>
class CatArrayBatchedCopyAlignedKContig: public Kernel {
public:
    CatArrayBatchedCopyAlignedKContig(
        T* output,
        CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
        TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
        const int concatDim,
        IndexType dimStride,
        dim3 catGrid,
        dim3 applyBlock,
        cudaStream_t stream):
        output_(output),
        inputs_(inputs),
        os_(os),
        concatDim_(concatDim),
        dimStride_(dimStride),
        catGrid_(catGrid),
        applyBlock_(applyBlock),
        stream_(stream)
    {
        std::cout << "CatArrayBatchedCopyAlignedKContig: [已入队]" << std::endl;
    }

  void execute() override {
        std::cout << "CatArrayBatchedCopyAlignedKContig: [已准备]" << std::endl;
        CatArrayBatchedCopy_alignedK_contig<T, IndexType, Dims, batch_size, stride_size, aligned_vec_load_bytes>
            <<<catGrid_, applyBlock_, 0, stream_>>>(output_, inputs_, os_, concatDim_, dimStride_);
        std::cout << "CatArrayBatchedCopyAlignedKContig: [已执行]" << std::endl;

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    T* output_;
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs_;
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os_;
    int concatDim_;
    IndexType dimStride_;
    dim3 catGrid_;
    dim3 applyBlock_;
    cudaStream_t stream_;

};

}
