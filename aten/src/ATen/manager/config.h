#pragma once

#include <string>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>

// ============================================================
//  常量定义 (保持不变)
// ============================================================

#define SCHEDULER_PORT 9999
#define LOCALHOST "127.0.0.1"

constexpr size_t SPSC_QUEUE_SIZE = 1024;
constexpr size_t SPSC_MSG_SIZE = 256;
constexpr size_t CACHE_LINE_SIZE = 64;

#define SHM_NAME_SCHEDULER "/kernel_scheduler_registry"
#define SHM_NAME_PREFIX_PYTORCH "/ks_pytorch_"
#define SHM_NAME_PREFIX_SGLANG  "/ks_sglang_"
#define SHM_NAME_PYTORCH "/kernel_scheduler_pytorch"
#define SHM_NAME_SGLANG  "/kernel_scheduler_sglang"

constexpr size_t MAX_REGISTERED_CLIENTS = 64;

// ============================================================
//  数据结构 (POD, 用于共享内存布局)
// ============================================================

struct SPSCQueue {
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> head;
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> tail;
    alignas(CACHE_LINE_SIZE) char buffer[SPSC_QUEUE_SIZE][SPSC_MSG_SIZE];
};

struct ClientChannelStruct {
    SPSCQueue request_queue;
    SPSCQueue response_queue;
    alignas(CACHE_LINE_SIZE) std::atomic<bool> client_connected;
    alignas(CACHE_LINE_SIZE) std::atomic<bool> scheduler_ready;
};

struct ClientRegistryEntry {
    alignas(CACHE_LINE_SIZE) std::atomic<bool> active;
    char shm_name[64];
    char client_type[16];
    char unique_id[64];
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> client_pid;
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> last_heartbeat;
    
    void init() {
        active.store(false, std::memory_order_relaxed);
        std::memset(shm_name, 0, sizeof(shm_name));
        std::memset(client_type, 0, sizeof(client_type));
        std::memset(unique_id, 0, sizeof(unique_id));
        client_pid.store(0, std::memory_order_relaxed);
        last_heartbeat.store(0, std::memory_order_relaxed);
    }
};

struct ClientRegistry {
    alignas(CACHE_LINE_SIZE) std::atomic<bool> scheduler_ready;
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> version;
    ClientRegistryEntry entries[MAX_REGISTERED_CLIENTS];

    void init() {
        scheduler_ready.store(false, std::memory_order_relaxed);
        version.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < MAX_REGISTERED_CLIENTS; i++) {
            entries[i].init();
        }
    }
};