#pragma once

#include "config.h"
#include <string>
#include <atomic>
#include <mutex>

/**
 * @brief 客户端核心通信类
 * 负责创建共享内存、注册到 Registry、并处理双向通信
 */
class ShmClient {
public:
    ShmClient(const std::string& type, const std::string& shm_uniqueId);
    ~ShmClient();

    // 初始化并连接到调度器
    bool connect();
    
    // 断开连接
    void disconnect();

    // 核心收发接口 (阻塞式 busy-wait)
    bool sendBlocking(const std::string& msg);
    bool recvBlocking(std::string& outMsg);
    
    bool isConnected() const;

private:
    // 内部辅助
    std::string generateShmName();
    bool initRegistry();
    bool createChannelShm();
    bool registerToServer();
    void waitForScheduler();

    // SPSC 队列操作
    bool spsc_try_push(const char* data, size_t len);
    bool spsc_try_pop(char* out_data, size_t max_len);

private:
    std::string clientType_;
    std::string uniqueId_;
    std::string myShmName_;
    int mySlot_ = -1;

    // 共享内存指针
    ClientRegistry* registry_ = nullptr;
    ClientChannelStruct* channel_ = nullptr;
    
    // 资源句柄
    int registryFd_ = -1;
    int channelFd_ = -1;
    
    std::atomic<bool> connected_{false};
};