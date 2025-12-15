#include "shm_client.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <chrono>

std::string get_user_suffix_client() {
    const char* u = std::getenv("USER");
    return (u && *u) ? std::string("_") + u : "_nouser";
}

ShmClient::ShmClient(const std::string& type, const std::string& shm_uniqueId)
    : clientType_(type), uniqueId_(shm_uniqueId) {
    myShmName_ = (type == "sglang" ? SHM_NAME_PREFIX_SGLANG : SHM_NAME_PREFIX_PYTORCH) 
                 + get_user_suffix_client() + "_" + shm_uniqueId;
}

ShmClient::~ShmClient() {
    disconnect();
}

void ShmClient::disconnect() {
    connected_ = false;

    // 1. 从 Registry 注销
    if (registry_ && mySlot_ >= 0) {
        registry_->entries[mySlot_].active.store(false, std::memory_order_release);
        registry_->version.fetch_add(1, std::memory_order_release);
    }

    // 2. 关闭通道
    if (channel_) {
        channel_->client_connected.store(false, std::memory_order_release);
        munmap(channel_, sizeof(ClientChannelStruct));
        shm_unlink(myShmName_.c_str()); // 客户端负责销毁自己的 SHM
        channel_ = nullptr;
    }

    // 3. 关闭 Registry 映射
    if (registry_) {
        munmap(registry_, sizeof(ClientRegistry));
        registry_ = nullptr;
    }
}

bool ShmClient::connect() {
    if (!initRegistry()) return false;
    if (!createChannelShm()) return false;
    if (!registerToServer()) return false;
    
    waitForScheduler();
    
    connected_ = true;
    return true;
}

bool ShmClient::initRegistry() {
    std::string regName = std::string(SHM_NAME_SCHEDULER) + get_user_suffix_client();
    
    // 客户端只读写 Registry，但不创建
    registryFd_ = shm_open(regName.c_str(), O_RDWR, 0666);
    if (registryFd_ == -1) {
        std::cerr << "[ShmClient] Failed to open registry: " << regName << std::endl;
        return false;
    }

    void* ptr = mmap(nullptr, sizeof(ClientRegistry), PROT_READ | PROT_WRITE, MAP_SHARED, registryFd_, 0);
    close(registryFd_);
    
    if (ptr == MAP_FAILED) return false;
    registry_ = static_cast<ClientRegistry*>(ptr);
    
    // 等待 Registry 初始化完成
    while (!registry_->scheduler_ready.load(std::memory_order_acquire)) {
        usleep(100000); // 100ms
    }
    return true;
}

bool ShmClient::createChannelShm() {
    // 客户端负责创建自己的通信通道
    channelFd_ = shm_open(myShmName_.c_str(), O_RDWR | O_CREAT, 0666);
    if (channelFd_ == -1) return false;

    if (ftruncate(channelFd_, sizeof(ClientChannelStruct)) == -1) {
        close(channelFd_);
        return false;
    }

    void* ptr = mmap(nullptr, sizeof(ClientChannelStruct), PROT_READ | PROT_WRITE, MAP_SHARED, channelFd_, 0);
    close(channelFd_);

    if (ptr == MAP_FAILED) return false;
    channel_ = static_cast<ClientChannelStruct*>(ptr);
    
    // 初始化队列
    channel_->request_queue.head.store(0);
    channel_->request_queue.tail.store(0);
    channel_->response_queue.head.store(0);
    channel_->response_queue.tail.store(0);
    channel_->client_connected.store(true, std::memory_order_release);
    channel_->scheduler_ready.store(false, std::memory_order_release); // 等待服务端置位

    return true;
}

bool ShmClient::registerToServer() {
    // 寻找空闲槽位
    for (int i = 0; i < MAX_REGISTERED_CLIENTS; ++i) {
        bool expected = false;
        // CAS 抢占槽位
        if (registry_->entries[i].active.compare_exchange_strong(expected, true)) {
            mySlot_ = i;
            auto& entry = registry_->entries[i];
            
            strncpy(entry.shm_name, myShmName_.c_str(), 63);
            strncpy(entry.client_type, clientType_.c_str(), 15);
            strncpy(entry.unique_id, uniqueId_.c_str(), 63);
            entry.client_pid.store(getpid(), std::memory_order_relaxed);
            
            // 通知 Scanner 更新
            registry_->version.fetch_add(1, std::memory_order_release);
            return true;
        }
    }
    std::cerr << "[ShmClient] Registry is full!" << std::endl;
    return false;
}

void ShmClient::waitForScheduler() {
    int attempts = 0;
    // 等待服务端发现我们并连接
    while (!channel_->scheduler_ready.load(std::memory_order_acquire)) {
        usleep(10000); // 10ms
        if (++attempts % 100 == 0) {
             std::cout << "[ShmClient] Waiting for scheduler..." << std::endl;
        }
    }
}

bool ShmClient::isConnected() const{
    return connected_ && channel_ && channel_->scheduler_ready.load(std::memory_order_acquire);
}

// ======================= SPSC Implementation =======================

bool ShmClient::spsc_try_push(const char* data, size_t len) {
    auto& q = channel_->request_queue;
    uint64_t tail = q.tail.load(std::memory_order_relaxed);
    uint64_t next_tail = (tail + 1) % SPSC_QUEUE_SIZE;
    
    if (next_tail == q.head.load(std::memory_order_acquire)) return false;

    size_t copy_len = (len < SPSC_MSG_SIZE - 1) ? len : (SPSC_MSG_SIZE - 1);
    memcpy(q.buffer[tail], data, copy_len);
    q.buffer[tail][copy_len] = '\0';

    q.tail.store(next_tail, std::memory_order_release);
    return true;
}

bool ShmClient::spsc_try_pop(char* out_data, size_t max_len) {
    auto& q = channel_->response_queue;
    uint64_t head = q.head.load(std::memory_order_relaxed);
    
    if (head == q.tail.load(std::memory_order_acquire)) return false;

    size_t copy_len = strlen(q.buffer[head]);
    if (copy_len >= max_len) copy_len = max_len - 1;
    memcpy(out_data, q.buffer[head], copy_len);
    out_data[copy_len] = '\0';

    q.head.store((head + 1) % SPSC_QUEUE_SIZE, std::memory_order_release);
    return true;
}

bool ShmClient::sendBlocking(const std::string& msg) {
    int attempts = 0;
    while (!spsc_try_push(msg.c_str(), msg.length())) {
        if (!isConnected()) return false;
        if (++attempts > 10000000) return false; // 简单超时保护
        __asm__ __volatile__("pause" ::: "memory");
    }
    return true;
}

bool ShmClient::recvBlocking(std::string& outMsg) {
    char buffer[SPSC_MSG_SIZE];
    while (!spsc_try_pop(buffer, SPSC_MSG_SIZE)) {
        if (!isConnected()) return false;
        __asm__ __volatile__("pause" ::: "memory");
    }
    outMsg = std::string(buffer);
    return true;
}