#include "KernelManager.h"
#include "ipc/Interface.h"
#include "ipc/ShmTransport.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <cctype>

// ============================================================
//  静态成员初始化
// ============================================================

std::unique_ptr<ipc::ITransportFactory> KernelManager::transportFactory_ = nullptr;

// ============================================================
//  静态辅助函数
// ============================================================

const char* env_p = std::getenv("UNIQUE_ID");
std::string UNIQUE_ID = env_p ? env_p : "";

// 生成唯一的通道名称
std::string generateChannelName() {
    std::string suffix;
    if (!UNIQUE_ID.empty()) {
        suffix = UNIQUE_ID;
    } else {
        suffix = std::to_string(getpid());
    }
    return ipc::generateChannelName(getpid(), suffix, "pytorch");
}

namespace {

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
        return baseName.substr(ns_pos + 2);
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

} // anonymous namespace

// ============================================================
//  KernelManager 实现
// ============================================================

void KernelManager::setTransportFactory(std::unique_ptr<ipc::ITransportFactory> factory) {
    transportFactory_ = std::move(factory);
}

KernelManager& KernelManager::getInstance() {
    static KernelManager instance;
    return instance;
}

KernelManager::KernelManager() 
    : registrySlot_(-1), requestIdCounter_(0), connected_(false) {
    
    // 如果没有设置自定义工厂，使用默认的共享内存工厂
    if (!transportFactory_) {
        transportFactory_ = std::make_unique<ipc::shm::ShmTransportFactory>();
    }
    
    channelName_ = generateChannelName();
    connectToScheduler();
    
    std::cout << "[KernelManager] Connected to Scheduler (UNIQUE_ID: " << UNIQUE_ID 
              << ", Channel: " << channelName_ << ", Transport: " 
              << transportFactory_->getName() << ")." << std::endl;
}

KernelManager::~KernelManager() {
    // 1. 从注册表注销
    if (registry_ && registrySlot_ >= 0) {
        registry_->unregisterClient(registrySlot_);
        registrySlot_ = -1;
    }

    // 2. 清理通道
    if (channel_) {
        channel_->setClientConnected(false);
        channel_.reset();
        transportFactory_->destroyChannel(channelName_);
    }

    // 3. 清理注册表
    registry_.reset();

    std::cout << "[KernelManager] 已关闭与调度器的连接 (UNIQUE_ID: " << UNIQUE_ID 
              << ", Channel: " << channelName_ << ")." << std::endl;
}

void KernelManager::connectToScheduler() {
    // 1. 首先尝试连接到注册表
    registry_ = transportFactory_->createRegistry(false);
    
    if (!registry_) {
        std::cerr << "[KernelManager] 无法打开注册表，调度器可能未启动。" << std::endl;
        return;
    }

    // 2. 等待调度器准备好
    int waitCount = 0;
    const int maxWait = 50;  // 50 * 100ms = 5秒
    while (!registry_->isServerReady()) {
        if (++waitCount > maxWait) {
            std::cerr << "[KernelManager] 等待调度器超时。" << std::endl;
            registry_.reset();
            return;
        }
        usleep(100000);
    }

    // 3. 创建通信通道
    channel_ = transportFactory_->createChannel(channelName_, true);
    if (!channel_) {
        std::cerr << "[KernelManager] 无法创建通道: " << channelName_ << std::endl;
        registry_.reset();
        return;
    }

    // 4. 注册
    std::string uniqueIdStr = UNIQUE_ID.empty() ? std::to_string(getpid()) : UNIQUE_ID;
    registrySlot_ = registry_->registerClient(channelName_, "pytorch", uniqueIdStr, static_cast<int64_t>(getpid()));
    if (registrySlot_ < 0) {
        std::cerr << "[KernelManager] 注册表已满。" << std::endl;
        transportFactory_->destroyChannel(channelName_);
        channel_.reset();
        registry_.reset();
        return;
    }

    // 5. 标记已连接
    channel_->setClientConnected(true);
    
    // 6. 等待服务端就绪
    waitCount = 0;
    const int maxWaitChannel = 100;
    while (!channel_->isServerReady()) {
        if (++waitCount > maxWaitChannel) {
            std::cerr << "[KernelManager] 等待服务端超时: " << channelName_ << std::endl;
            registry_->unregisterClient(registrySlot_);
            transportFactory_->destroyChannel(channelName_);
            channel_.reset();
            registry_.reset();
            registrySlot_ = -1;
            return;
        }
        usleep(100000);
    }

    connected_ = true;
    std::cout << "[KernelManager] 已通过 " << transportFactory_->getName() 
              << " 连接到调度器 (" << channelName_ << ")" << std::endl;
}

std::string KernelManager::generateRequestID() {
    uint64_t id = ++requestIdCounter_;
    std::stringstream ss;
    ss << "req_" << id;
    return ss.str();
}

void KernelManager::enqueue(std::unique_ptr<Kernel> kernel) {
    if (!channel_ || !connected_) {
        std::cerr << "[KernelManager] 错误：未连接到调度器" << std::endl;
        // 降级模式：直接执行
        try {
            kernel->execute();
        } catch (const std::exception& e) {
            std::cerr << "[KernelManager] 降级执行时异常: " << e.what() << std::endl;
        }
        return;
    }

    std::string reqId = generateRequestID();
    std::string kernelType = getClassName(*kernel);
    std::string requestMessage = ipc::createRequestMessage(reqId, kernelType, UNIQUE_ID);

    // 发送请求
    if (!channel_->getRequestQueue().sendBlocking(requestMessage, 5000)) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 发送请求超时" << std::endl;
        return;
    }

    // 等待响应
    std::string responseMessage;
    if (!channel_->getResponseQueue().receiveBlocking(responseMessage, 10000)) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 等待响应超时" << std::endl;
        return;
    }

    // 解析响应
    while (!responseMessage.empty() && (responseMessage.back() == '\n' || responseMessage.back() == '\r')) {
        responseMessage.pop_back();
    }
    
    auto parts = split_client(responseMessage, '|');
    if (parts.size() != 3 || parts[0] != reqId) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 响应格式错误: " << responseMessage << std::endl;
        return;
    }

    bool permissionGranted = (parts[1] == "1");
    std::string reason = parts[2];
    
    if (permissionGranted) {
        try {
            kernel->execute(); 
        } catch (const std::exception& e) {
            std::cerr << "[KernelManager] (ID: " << reqId << ") 执行异常: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 内核被拒绝: " << reason << std::endl;
    }
}


