#pragma once

#include "Kernel.h"

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>   // TORCH_CUDABLAS_CHECK
#include <c10/util/BFloat16.h>
#include <c10/core/ScalarType.h>
#include <iostream>
#include <type_traits>              // std::is_same_v

// Rebuilt-PyTorch/pytorch-v2.8.0/aten/src/ATen/cuda/CUDABlas.cpp

// 确保 at::BFloat16 类型可用
using at::BFloat16;

/**
 * @brief GemmInternalCublasBF16Kernel
 *
 * 这是一个具体的 Kernel 实现，封装了对 cublasGemmEx 的一次特定调用。
 * 它的目的是捕获 `gemm_internal_cublas_bfloat16_helper` 函数中
 * 的所有必要参数，以便稍后在 `execute()` 方法中执行它。
 */
class GemmInternalCublas: public Kernel {
public:
    /**
     * @brief 构造函数，捕获 cublasGemmEx 调用所需的所有状态。
     * * @param handle cublas 句柄
     * @param opa 矩阵 A 的转置操作
     * @param opb 矩阵 B 的转置操作
     * @param m 矩阵 A 和 C 的行数
     * @param n 矩阵 B 和 C 的列数
     * @param k 矩阵 A 的列数 / B 的行数
     * @param falpha 缩放因子 alpha
     * @param Aarray 指向矩阵 A 的设备指针
     * @param lda 矩阵 A 的 leading dimension
     * @param fbeta 缩放因子 beta
     * @param b 指向矩阵 B 的设备指针
     * @param ldb 矩阵 B 的 leading dimension
     * @param Carray 指向矩阵 C 的设备指针 (输入/输出)
     * @param ldc 矩阵 C 的 leading dimension
     * @param compute_type cublas 的计算类型
     * @param cublas_flags cublas 的数学模式标志
     */
    GemmInternalCublas(
        cublasHandle_t handle,
        cublasMath_t cublas_flags,
        cublasOperation_t opa,
        cublasOperation_t opb,
        int64_t m, int64_t n, int64_t k,
        const void* falpha, const void* Aarray, cudaDataType_t Atype, int64_t lda,
        const void* Barray, cudaDataType_t Btype, int64_t ldb, const void* fbeta,
        void* Carray, cudaDataType_t Ctype, int64_t ldc,
        cudaDataType_t compute_type,
        cublasGemmAlgo_t algo
    ):
        handle_(handle), cublas_flags_(cublas_flags),
        opa_(opa), opb_(opb),
        m_(m), n_(n), k_(k),
        falpha_(falpha), Aarray_(Aarray), Atype_(Atype), lda_(lda),
        Barray_(Barray), Btype_(Btype), ldb_(ldb), fbeta_(fbeta),
        Carray_(Carray), Ctype_(Ctype), ldc_(ldc),
        compute_type_(compute_type),
        algo_(algo)
    {
        // std::cout << "GemmInternalCublas: [已入队] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;
    }

    void execute() override {
        // 设置数学模式 (从原函数移动而来)
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, cublas_flags_));

        // std::cout << "GemmInternalCublas: [已准备] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;
        TORCH_CUDABLAS_CHECK(cublasGemmEx(
            handle_,
            opa_, opb_,
            m_, n_, k_,
            falpha_, Aarray_, Atype_, lda_,
            Barray_, Btype_, ldb_, fbeta_,
            Carray_, Ctype_, ldc_,
            compute_type_, algo_
        ));
        // std::cout << "GemmInternalCublas: [已发送] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;

        // 恢复默认的数学模式 (从原函数移动而来)
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
    }

private:
    cublasHandle_t handle_;
    cublasMath_t cublas_flags_;
    cublasOperation_t opa_, opb_;
    int64_t m_, n_, k_;
    const void* falpha_;
    const void* Aarray_;
    cudaDataType_t Atype_;
    int64_t lda_;
    const void* Barray_;
    cudaDataType_t Btype_;
    int64_t ldb_;
    const void* fbeta_;
    void* Carray_;
    cudaDataType_t Ctype_;
    int64_t ldc_;
    cudaDataType_t compute_type_;
    cublasGemmAlgo_t algo_;
};