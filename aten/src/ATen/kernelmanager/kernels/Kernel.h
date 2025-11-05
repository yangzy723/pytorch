#pragma once

#include <memory> // 为了 std::unique_ptr

/**
 * @brief Kernel (内核) 的抽象基类。
 *
 * 定义了一个标准接口，所有可排队的操作 (无论是 CUDA Kernel 启动、
 * BLAS 库调用、还是其他任务) 都必须实现这个接口。
 * 任何希望被 KernelManager 管理和执行的计算单元都应从此类继承。
 */
class Kernel {
public:
    // 虚析构函数，确保派生类可以被正确地销毁
    virtual ~Kernel() = default;

    /**
     * @brief 纯虚函数，定义了 Kernel 的核心执行逻辑。
     *
     * 派生类将在这里实现具体的计算任务，
     * 例如调用 cublas, cudnn 或自定义的 CUDA kernel。
     * * 这个函数将由 KernelManager 的工作线程在稍后某个时间点调用。
     */
    virtual void execute() = 0;
};
