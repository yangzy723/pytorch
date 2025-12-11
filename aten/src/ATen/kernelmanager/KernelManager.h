#pragma once

#include "kernels/Kernel.h"
#include "ipc/Interface.h"

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
    void enqueue(std::unique_ptr<Kernel> kernel);

    // 检查是否已连接到调度器
    bool isConnected() const { return channel_ != nullptr && connected_; }

    // 设置自定义的传输层工厂（用于测试或切换实现）
    static void setTransportFactory(std::unique_ptr<ipc::ITransportFactory> factory);

private:
    KernelManager();
    void connectToScheduler();
    std::string generateRequestID();

    // 禁用拷贝和赋值
    KernelManager(const KernelManager&) = delete;
    KernelManager& operator=(const KernelManager&) = delete;

    // --- 成员变量 ---
    static std::unique_ptr<ipc::ITransportFactory> transportFactory_;
    std::unique_ptr<ipc::IChannel> channel_;
    std::unique_ptr<ipc::IRegistry> registry_;
    
    int registrySlot_;
    std::string channelName_;
    std::atomic<uint64_t> requestIdCounter_;
    bool connected_;
};
