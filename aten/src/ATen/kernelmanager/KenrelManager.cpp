#include "KernelManager.h"
#include "IPCProtocol.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <unistd.h>

#include <typeinfo>       // 用于 typeid
#include <cxxabi.h>       // 用于 __cxa_demangle (GCC/Clang)
#include <memory>         // 用于 std::free
#include <cctype>

// --- 静态辅助函数  ---

const char* env_p = std::getenv("UNIQUE_ID");
std::string UNIQUE_ID = env_p;

namespace { // 使用匿名命名空间将它们限制在此文件

// 获取内核的 demangled 核心名称
std::string getClassName(const Kernel& kernel) {
    const char* mangledName = typeid(kernel).name();
    int status = 0;

    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(mangledName, nullptr, nullptr, &status),
        std::free
    };
    std::string demangledName = (status == 0 && res) ? res.get() : mangledName;

    size_t bracket_pos = demangledName.find_first_of("<(");
    std::string baseName = (bracket_pos != std::string::npos)
                             ? demangledName.substr(0, bracket_pos)
                             : demangledName;

    size_t ns_pos = baseName.rfind("::");
    if (ns_pos != std::string::npos) {
        return baseName.substr(ns_pos + 2); // 跳过两个字符 '::'
    }

    if (!baseName.empty() && !std::isalpha(static_cast<unsigned char>(baseName[0]))) {
        return baseName.substr(1);
    }
    return baseName;
}

std::vector<std::string> split_client(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

} // 匿名命名空间结束

// --- Singleton ---
KernelManager& KernelManager::getInstance() {
    static KernelManager instance;
    return instance;
}

// --- 构造函数 ---
// 在构造时建立共享内存连接
KernelManager::KernelManager() : channel_(nullptr), requestIdCounter_(0), connected_(false) {
    connectToScheduler();
<<<<<<< HEAD
=======
    std::cout << "[KernelManager] Connected to Scheduler (UNIQUE_ID: " << UNIQUE_ID << " )." << std::endl;
>>>>>>> 5063f7b15b49f24fb728cbedc209460fb2d74b3f
}

// --- 析构函数 ---
// 在析构时清理共享内存映射
KernelManager::~KernelManager() {
<<<<<<< HEAD
    if (channel_) {
        // 标记客户端已断开
        channel_->client_connected.store(false, std::memory_order_release);
        
        // 解除共享内存映射（不要 unlink，让调度器管理生命周期）
        SharedMemoryHelper::unmap(channel_);
        channel_ = nullptr;
        std::cout << "[KernelManager] 已关闭与调度器的共享内存连接。" << std::endl;
=======
    if (sock_ != -1) {
        close(sock_);
        std::cout << "[KernelManager] Connection closed (UNIQUE_ID: " << UNIQUE_ID << " )." << std::endl;
>>>>>>> 5063f7b15b49f24fb728cbedc209460fb2d74b3f
    }
}

// --- 私有辅助方法 ---

void KernelManager::connectToScheduler() {
    // 尝试打开共享内存（由调度器创建）
    channel_ = SharedMemoryHelper::create_or_open(SHM_NAME_PYTORCH, false);
    
    if (!channel_) {
        std::cerr << "[KernelManager] 无法打开共享内存，调度器可能未启动。" << std::endl;
        return;
    }

    // 等待调度器准备好（最多等待 5 秒）
    int waitCount = 0;
    const int maxWait = 50;  // 50 * 100ms = 5秒
    while (!channel_->scheduler_ready.load(std::memory_order_acquire)) {
        if (++waitCount > maxWait) {
            std::cerr << "[KernelManager] 等待调度器超时，共享内存已存在但调度器未就绪。" << std::endl;
            SharedMemoryHelper::unmap(channel_);
            channel_ = nullptr;
            return;
        }
        usleep(100000);  // 100ms
    }

    // 标记客户端已连接
    channel_->client_connected.store(true, std::memory_order_release);
    connected_ = true;

    std::cout << "[KernelManager] 已通过共享内存连接到调度器 (" << SHM_NAME_PYTORCH << ")" << std::endl;
}

std::string KernelManager::generateRequestID() {
    uint64_t id = ++requestIdCounter_;
    std::stringstream ss;
    ss << "req_" << id;
    return ss.str();
}

// --- Public 方法 ---

void KernelManager::enqueue(std::unique_ptr<Kernel> kernel) {
    if (!channel_ || !connected_) {
        std::cerr << "[KernelManager] 错误：未连接到调度器" << std::endl;
        // 在未连接时直接执行内核（降级模式）
        try {
            kernel->execute();
        } catch (const std::exception& e) {
            std::cerr << "[KernelManager] 降级执行时异常: " << e.what() << std::endl;
        }
        return;
    }

    std::string reqId = generateRequestID();
    std::string kernelType = getClassName(*kernel);
<<<<<<< HEAD
    std::string requestMessage = createRequestMessage(reqId, kernelType);
=======
    std::string requestMessage = createRequestMessage(reqId, kernelType, UNIQUE_ID);
    // std::cout << "[KernelManager] 准备提交 (ID: " << reqId << "): " << kernelType << std::endl;
>>>>>>> 5063f7b15b49f24fb728cbedc209460fb2d74b3f

    // 发送请求到请求队列
    if (!channel_->request_queue.push_blocking(requestMessage, 5000)) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 发送请求超时，队列可能已满" << std::endl;
        return;
    }

    // 等待响应
    char buffer[SPSC_MSG_SIZE];
    if (!channel_->response_queue.pop_blocking(buffer, SPSC_MSG_SIZE, 10000)) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 等待响应超时" << std::endl;
        return;
    }

    std::string responseMessage(buffer);
    // 去除尾部换行符
    while (!responseMessage.empty() && (responseMessage.back() == '\n' || responseMessage.back() == '\r')) {
        responseMessage.pop_back();
    }
    
    auto parts = split_client(responseMessage, '|');
    if (parts.size() != 3 || parts[0] != reqId) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 收到格式错误的响应: " << responseMessage << std::endl;
        return;
    }

    bool permissionGranted = (parts[1] == "1");
    std::string reason = parts[2];
    
    if (permissionGranted) {
        try {
            kernel->execute(); 
        } catch (const std::exception& e) {
            std::cerr << "[KernelManager] (ID: " << reqId << ") 执行时异常: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 内核被拒绝！原因: " << reason << std::endl;
    }
}