#pragma once

/**
 * @file ipc_common.h
 * @brief 共享的 IPC 数据结构和常量定义
 * 
 * 此文件定义了客户端和服务端共用的底层数据结构。
 * 这些结构直接映射到共享内存，必须是 POD 类型。
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <cstdlib>

namespace ipc {

// ============================================================
//  常量定义
// ============================================================

constexpr size_t SPSC_QUEUE_SIZE = 1024;        // 队列可存储的消息数量
constexpr size_t MAX_MSG_SIZE = 256;            // 每条消息的最大字节数
constexpr size_t CACHE_LINE_SIZE = 64;          // CPU 缓存行大小，用于避免伪共享
constexpr size_t MAX_CLIENTS = 64;              // 最多支持的客户端数量
constexpr int DEFAULT_TIMEOUT_MS = 5000;        // 默认超时时间(毫秒)

// 共享内存名称前缀
constexpr const char* SHM_PREFIX_PYTORCH = "/ks_pytorch_";
constexpr const char* SHM_PREFIX_SGLANG = "/ks_sglang_";

// 兼容性宏（用于旧代码）
#define SHM_NAME_PREFIX_PYTORCH "/ks_pytorch_"
#define SHM_NAME_PREFIX_SGLANG  "/ks_sglang_"

// ============================================================
//  辅助函数
// ============================================================

/**
 * 获取用户后缀，用于隔离不同用户的共享内存
 */
inline std::string getUserSuffix() {
    const char* u = std::getenv("USER");
    if (u && *u) {
        return std::string("_") + u;
    }
    return "_nouser";
}

/**
 * 获取注册表的共享内存名称
 */
inline std::string getRegistryName() {
    return std::string("/kernel_scheduler_registry") + getUserSuffix();
}

/**
 * 生成客户端通道名称
 * @param pid 进程 ID
 * @param uniqueId 可选的唯一标识
 * @param clientType 客户端类型 ("pytorch" 或 "sglang")
 */
inline std::string generateChannelName(pid_t pid, const std::string& uniqueId = "", 
                                        const std::string& clientType = "pytorch") {
    std::string suffix = getUserSuffix();
    if (!suffix.empty() && suffix[0] == '_') {
        suffix = suffix.substr(1);  // 去掉前导下划线
    }
    
    const char* prefix = (clientType == "sglang") ? SHM_PREFIX_SGLANG : SHM_PREFIX_PYTORCH;
    std::string id = uniqueId.empty() ? std::to_string(pid) : uniqueId;
    
    return std::string(prefix) + suffix + "_" + id;
}

// ============================================================
//  SPSC 无锁环形队列 (共享内存数据结构)
// ============================================================

/**
 * SPSCQueueData - SPSC 队列的共享内存布局
 * 
 * 这是一个单生产者单消费者的无锁环形队列。
 * 使用缓存行对齐避免伪共享问题。
 */
struct alignas(CACHE_LINE_SIZE) SPSCQueueData {
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> head;  // 消费者读取位置
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> tail;  // 生产者写入位置
    alignas(CACHE_LINE_SIZE) char buffer[SPSC_QUEUE_SIZE][MAX_MSG_SIZE];

    /**
     * 初始化队列
     */
    void init() {
        head.store(0, std::memory_order_relaxed);
        tail.store(0, std::memory_order_relaxed);
        std::memset(buffer, 0, sizeof(buffer));
    }

    /**
     * 尝试写入消息（非阻塞）
     * @return true 成功，false 队列已满
     */
    bool tryPush(const char* data, size_t len) {
        uint64_t current_tail = tail.load(std::memory_order_relaxed);
        uint64_t next_tail = (current_tail + 1) % SPSC_QUEUE_SIZE;
        
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false;  // 队列已满
        }

        size_t copy_len = (len < MAX_MSG_SIZE - 1) ? len : (MAX_MSG_SIZE - 1);
        std::memcpy(buffer[current_tail], data, copy_len);
        buffer[current_tail][copy_len] = '\0';

        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    bool tryPush(const std::string& msg) {
        return tryPush(msg.c_str(), msg.length());
    }

    /**
     * 尝试读取消息（非阻塞）
     * @return true 成功，false 队列为空
     */
    bool tryPop(char* out_data, size_t max_len) {
        uint64_t current_head = head.load(std::memory_order_relaxed);
        
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false;  // 队列为空
        }

        size_t copy_len = std::strlen(buffer[current_head]);
        if (copy_len >= max_len) copy_len = max_len - 1;
        std::memcpy(out_data, buffer[current_head], copy_len);
        out_data[copy_len] = '\0';

        head.store((current_head + 1) % SPSC_QUEUE_SIZE, std::memory_order_release);
        return true;
    }

    std::string tryPop() {
        char buf[MAX_MSG_SIZE];
        if (tryPop(buf, MAX_MSG_SIZE)) {
            return std::string(buf);
        }
        return "";
    }

    /**
     * 阻塞式写入（带超时）
     * @param timeout_ms 超时时间，负数表示无限等待
     */
    bool pushBlocking(const char* data, size_t len, int timeout_ms = -1) {
        if (timeout_ms < 0) {
            while (!tryPush(data, len)) {
                __asm__ __volatile__("pause" ::: "memory");
            }
            return true;
        }
        
        auto start = std::chrono::steady_clock::now();
        while (true) {
            if (tryPush(data, len)) return true;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms) return false;
            __asm__ __volatile__("pause" ::: "memory");
        }
    }

    bool pushBlocking(const std::string& msg, int timeout_ms = -1) {
        return pushBlocking(msg.c_str(), msg.length(), timeout_ms);
    }

    /**
     * 阻塞式读取（带超时）
     * @param timeout_ms 超时时间，负数表示无限等待
     */
    bool popBlocking(char* out_data, size_t max_len, int timeout_ms = -1) {
        if (timeout_ms < 0) {
            while (!tryPop(out_data, max_len)) {
                __asm__ __volatile__("pause" ::: "memory");
            }
            return true;
        }
        
        auto start = std::chrono::steady_clock::now();
        while (true) {
            if (tryPop(out_data, max_len)) return true;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms) return false;
            __asm__ __volatile__("pause" ::: "memory");
        }
    }

    bool empty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }

    size_t size() const {
        uint64_t h = head.load(std::memory_order_acquire);
        uint64_t t = tail.load(std::memory_order_acquire);
        return (t - h + SPSC_QUEUE_SIZE) % SPSC_QUEUE_SIZE;
    }
};

// ============================================================
//  通道共享内存数据结构
// ============================================================

/**
 * ChannelData - 双向通信通道的共享内存布局
 * 
 * 包含请求队列和响应队列，以及连接状态标志
 */
struct ChannelData {
    SPSCQueueData request_queue;    // 客户端 -> 服务端
    SPSCQueueData response_queue;   // 服务端 -> 客户端
    alignas(CACHE_LINE_SIZE) std::atomic<bool> client_connected;
    alignas(CACHE_LINE_SIZE) std::atomic<bool> server_ready;  // 又名 scheduler_ready

    void init() {
        request_queue.init();
        response_queue.init();
        client_connected.store(false, std::memory_order_relaxed);
        server_ready.store(false, std::memory_order_relaxed);
    }
};

// ============================================================
//  注册表共享内存数据结构
// ============================================================

/**
 * RegistryEntryData - 客户端注册条目
 */
struct RegistryEntryData {
    alignas(CACHE_LINE_SIZE) std::atomic<bool> active;
    char channel_name[64];     // 通道的共享内存名称
    char client_type[16];      // 客户端类型 (pytorch/sglang)
    char unique_id[64];        // 唯一标识
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> client_pid;
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> last_heartbeat;
    
    void init() {
        active.store(false, std::memory_order_relaxed);
        std::memset(channel_name, 0, sizeof(channel_name));
        std::memset(client_type, 0, sizeof(client_type));
        std::memset(unique_id, 0, sizeof(unique_id));
        client_pid.store(0, std::memory_order_relaxed);
        last_heartbeat.store(0, std::memory_order_relaxed);
    }
};

/**
 * RegistryData - 客户端注册表
 */
struct RegistryData {
    alignas(CACHE_LINE_SIZE) std::atomic<bool> server_ready;  // 调度器是否就绪
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> version;   // 版本号，每次变更递增
    RegistryEntryData entries[MAX_CLIENTS];

    void init() {
        server_ready.store(false, std::memory_order_relaxed);
        version.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < MAX_CLIENTS; i++) {
            entries[i].init();
        }
    }

    /**
     * 注册新客户端
     * @return 分配的槽位索引，失败返回 -1
     */
    int registerClient(const char* channelName, const char* clientType, 
                       const char* uniqueId, int64_t pid = 0) {
        for (size_t i = 0; i < MAX_CLIENTS; i++) {
            bool expected = false;
            if (entries[i].active.compare_exchange_strong(expected, true, 
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                std::strncpy(entries[i].channel_name, channelName, sizeof(entries[i].channel_name) - 1);
                std::strncpy(entries[i].client_type, clientType, sizeof(entries[i].client_type) - 1);
                std::strncpy(entries[i].unique_id, uniqueId, sizeof(entries[i].unique_id) - 1);
                entries[i].client_pid.store(pid, std::memory_order_release);
                entries[i].last_heartbeat.store(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count(), std::memory_order_release);
                version.fetch_add(1, std::memory_order_release);
                return static_cast<int>(i);
            }
        }
        return -1;
    }
    
    /**
     * 注销客户端
     */
    void unregisterClient(int slot) {
        if (slot >= 0 && slot < static_cast<int>(MAX_CLIENTS)) {
            entries[slot].active.store(false, std::memory_order_release);
            version.fetch_add(1, std::memory_order_release);
        }
    }
    
    /**
     * 更新心跳
     */
    void updateHeartbeat(int slot) {
        if (slot >= 0 && slot < static_cast<int>(MAX_CLIENTS)) {
            entries[slot].last_heartbeat.store(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count(), std::memory_order_release);
        }
    }
};

// ============================================================
//  协议消息构建函数
// ============================================================

/**
 * 创建请求消息
 * 格式: type|id|client_type[|unique_id]\n
 */
inline std::string createRequestMessage(const std::string& id, const std::string& type, 
                                         const std::string& uniqueId = "") {
    if (uniqueId.empty()) {
        return type + "|" + id + "|pytorch\n";
    }
    return type + "|" + id + "|pytorch|" + uniqueId + "\n";
}

/**
 * 创建响应消息
 * 格式: id|allowed|reason\n
 */
inline std::string createResponseMessage(const std::string& id, bool allowed, 
                                          const std::string& reason) {
    return id + "|" + (allowed ? "1" : "0") + "|" + reason + "\n";
}

// ============================================================
//  客户端信息结构（非共享内存，用于本地传递）
// ============================================================

struct ClientInfo {
    int slot;                   // 槽位索引
    std::string channelName;    // 通道名称
    std::string clientType;     // 客户端类型
    std::string uniqueId;       // 唯一标识
    int64_t pid;                // 进程 PID
    uint64_t lastHeartbeat;     // 最后心跳时间戳
    bool active;                // 是否活跃

    ClientInfo() : slot(-1), pid(0), lastHeartbeat(0), active(false) {}
};

// ============================================================
//  类型别名（向后兼容）
// ============================================================

// 客户端旧代码兼容
using SPSCQueue = SPSCQueueData;
using ClientChannel = ChannelData;
using ClientRegistryEntry = RegistryEntryData;
using ClientRegistry = RegistryData;

// 服务端旧代码兼容
using ClientChannelStruct = ChannelData;

// 常量别名
constexpr size_t SPSC_MSG_SIZE = MAX_MSG_SIZE;
constexpr size_t MAX_REGISTERED_CLIENTS = MAX_CLIENTS;

} // namespace ipc


