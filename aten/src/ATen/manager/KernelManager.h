#pragma once

#include "kernels/Kernel.h"
#include "shm_client.h"
#include <memory>
#include <string>
#include <atomic>

class KernelManager {
public:
    static KernelManager& getInstance();
    ~KernelManager();

    // 核心接口
    void enqueue(std::unique_ptr<Kernel> kernel);
    bool isConnected() const;

private:
    KernelManager();
    
    std::string generateRequestID();
    std::string getKernelType(const Kernel& kernel);
    std::string createRequestMessage(std::string reqId, const Kernel& kernel);

    // 禁止拷贝
    KernelManager(const KernelManager&) = delete;
    KernelManager& operator=(const KernelManager&) = delete;

private:
    std::unique_ptr<ShmClient> client_;
    std::string uniqueId_;
    std::atomic<uint64_t> requestIdCounter_{0};
};