#include "KernelManager.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <typeinfo>
#include <cxxabi.h>

std::string get_unique_id() {
    const char* env = std::getenv("UNIQUE_ID");
    if (env) 
        return std::string(env);
    return "";
}

KernelManager& KernelManager::getInstance() {
    static KernelManager instance;
    return instance;
}

KernelManager::KernelManager() {
    std::string pid = std::to_string(getpid());
    uniqueId_ = get_unique_id();
    client_ = std::make_unique<ShmClient>("pytorch", pid);
    std::cout << "[KernelManager] Initializing connection for pytorch: " << pid << "..." << std::endl;
    if (client_->connect()) {
        std::cout << "[KernelManager] Successfully connected to Scheduler." << std::endl;
    } else {
        std::cerr << "[KernelManager] Failed to connect to Scheduler!" << std::endl;
    }
}

KernelManager::~KernelManager() {
    if (client_) {
        client_->disconnect();
    }
}

bool KernelManager::isConnected() const {
    return client_ && client_->isConnected();
}

std::string KernelManager::generateRequestID() {
    return "req_" + std::to_string(++requestIdCounter_);
}

// 获取 Kernel 的类名 (Demangle)
std::string KernelManager::getKernelType(const Kernel& kernel) {
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

// 构建协议消息: KernelType|ReqId|ClientId[|UniqueId]
std::string KernelManager::createRequestMessage(std::string reqId, const Kernel& kernel) {
    std::string kType = getKernelType(kernel);
    std::stringstream ss;
    if (uniqueId_.empty()) {
        ss << kType << "|" << reqId << "|pytorch\n";
    } else {
        ss << kType << "|" << reqId << "|pytorch|" << uniqueId_ << "\n";
    }
    return ss.str();
}

// 简单的字符串分割
std::vector<std::string> split_resp(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void KernelManager::enqueue(std::unique_ptr<Kernel> kernel) {
    // 如果未连接，降级为直接执行
    if (!isConnected()) {
        std::cerr << "[KernelManager] Not connected. Executing directly." << std::endl;
        kernel->execute();
        return;
    }

    // 发送请求 (Busy Wait)
    std::string reqId = generateRequestID();
    std::string reqMsg = createRequestMessage(reqId, *kernel);
    if (!client_->sendBlocking(reqMsg)) {
        std::cerr << "[KernelManager] Send timeout. Skipping." << std::endl;
        return;
    }

    // 等待响应 (Busy Wait)
    std::string respMsg;
    if (!client_->recvBlocking(respMsg)) {
        std::cerr << "[KernelManager] Recv timeout." << std::endl;
        return;
    }

    // 解析响应: ReqId|1/0|Msg
    while (!respMsg.empty() && (respMsg.back() == '\n' || respMsg.back() == '\r')) {
        respMsg.pop_back();
    }

    auto parts = split_resp(respMsg, '|');
    if (parts.size() >= 2 && parts[0] == reqId) {
        bool allowed = (parts[1] == "1");
        if (allowed) {
            kernel->execute();
        } else {
            std::string reason = (parts.size() > 2) ? parts[2] : "Unknown";
            std::cerr << "[KernelManager] Kernel denied: " << reason << std::endl;
        }
    } else {
        std::cerr << "[KernelManager] Invalid response: " << respMsg << std::endl;
    }
}