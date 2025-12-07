#pragma once

#include "kernels/Kernel.h"
#include "IPCProtocol.h"

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
    // 此函数将通过共享内存发送请求并阻塞，直到收到响应
    void enqueue(std::unique_ptr<Kernel> kernel);

    // 检查是否已连接到调度器
    bool isConnected() const { return channel_ != nullptr && connected_; }

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
    
    ClientChannel* channel_;             // 共享内存通道
    ClientRegistry* registry_;           // 注册表共享内存
    int registrySlot_;                   // 在注册表中的槽位
    std::string shmName_;                // 唯一的共享内存名称
    std::atomic<uint64_t> requestIdCounter_;
    bool connected_;                     // 连接状态
};
