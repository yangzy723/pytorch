#pragma once

#include "kernels/Kernel.h"

#include <memory>
#include <string>
#include <atomic>
#include <vector>

class KernelManager {
public:
    static KernelManager& getInstance();

    // 析构函数，用于关闭连接
    ~KernelManager();

    // 将内核提交给 Scheduler 审批
    // 此函数将通过持久连接发送请求并阻塞，直到收到响应
    void enqueue(std::unique_ptr<Kernel> kernel);

private:
    // 构造函数是 private 的，用于单例
    KernelManager();

    // 辅助函数，在构造时调用
    void connectToScheduler();

    // 辅助函数
    std::string generateRequestID();

    // 禁用拷贝和赋值
    KernelManager(const KernelManager&) = delete;
    KernelManager& operator=(const KernelManager&) = delete;

    // --- 成员变量 ---
    
    int sock_; // 持久化的套接字文件描述符
    std::atomic<uint64_t> requestIdCounter_;
};
