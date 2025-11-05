#pragma once

#include "kernels/Kernel.h"

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <atomic> // 用于 std::atomic<bool>

/**
 * @brief 内核管理器 (Singleton)。
 *
 * 负责管理一个线程安全的内核执行队列和一个专用的工作线程。
 * * 工作流程：
 * 1. 外部线程 (例如 PyTorch 的主线程) 通过 enqueue() 提交一个 Kernel 任务。
 * 2. 任务被放入一个先进先出 (FIFO) 的队列中。
 * 3. KernelManager 内部的工作线程被唤醒。
 * 4. 工作线程从队列中取出一个任务并调用其 execute() 方法。
 * * 这是一个单例类，确保整个应用程序中只有一个内核管理实例。
 */
class KernelManager {
public:
    /**
     * @brief 获取 KernelManager 的全局唯一实例。
     * @return KernelManager& 
     */
    static KernelManager& getInstance();

    /**
     * @brief 将一个内核任务添加到执行队列中。
     * * 这是一个线程安全的操作，可以从任何线程调用。
     *
     * @param kernel 一个指向 Kernel 派生类实例的智能指针 (std::unique_ptr)。
     * 所有权被转移到 KernelManager。
     */
    void enqueue(std::unique_ptr<Kernel> kernel);

    void launchKernels();

    /**
     * @brief 停止工作线程并清理资源。
     * * 会向工作线程发送停止信号，并等待其完成当前任务后退出。
     * 这在程序结束时很重要，以避免资源泄漏和确保所有排队的任务被处理。
     */
    void shutdown();

    // 禁用拷贝构造函数和拷贝赋值运算符，以维护单例模式
    KernelManager(const KernelManager&) = delete;
    KernelManager& operator=(const KernelManager&) = delete;

private:
    /**
     * @brief 私有构造函数 (单例模式)。
     * * 在构造时启动工作线程。
     */
    KernelManager();

    /**
     * @brief 析构函数。
     * * 自动调用 shutdown() 来确保工作线程被正确清理。
     */
    ~KernelManager();

    /**
     * @brief 工作线程的主循环函数。
     * * * 不断地从队列中拉取并执行内核，直到收到停止信号。
     */
    void workerLoop();

    // --- 成员变量 ---

    // 内核任务队列
    std::queue<std::unique_ptr<Kernel>> kernelQueue_;
    
    // 保护 kernelQueue_ 的互斥锁
    std::mutex queueMutex_;
    
    // 用于唤醒工作线程的条件变量
    std::condition_variable cv_;
    
    // 专用的工作线程
    std::thread workerThread_;
    
    // 停止标志，用于通知工作线程退出
    // 使用 std::atomic 以确保跨线程的可见性
    std::atomic<bool> stop_;
};
