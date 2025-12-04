#include "KernelManager.h"
#include "IPCProtocol.h" // 假设这个文件定义了 PORT 和 HOST

#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

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
// 在构造时建立连接
KernelManager::KernelManager() : sock_(-1), requestIdCounter_(0) {
    connectToScheduler();
    std::cout << "[KernelManager] Connected to Scheduler (UNIQUE_ID: " << UNIQUE_ID << " )." << std::endl;
}

// --- 析构函数 ---
// 在析构时关闭连接
KernelManager::~KernelManager() {
    if (sock_ != -1) {
        close(sock_);
        std::cout << "[KernelManager] Connection closed (UNIQUE_ID: " << UNIQUE_ID << " )." << std::endl;
    }
}

// --- 私有辅助方法 ---

void KernelManager::connectToScheduler() {
    struct sockaddr_in serv_addr;

    if ((sock_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "[KernelManager] Socket 创建失败" << std::endl;
        exit(EXIT_FAILURE);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SCHEDULER_PORT);

    if (inet_pton(AF_INET, LOCALHOST, &serv_addr.sin_addr) <= 0) {
        std::cerr << "[KernelManager] 无效的地址 / 不支持的地址" << std::endl;
        close(sock_);
        exit(EXIT_FAILURE);
    }

    if (connect(sock_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "[KernelManager] 连接失败！调度器进程是否已启动？" << std::endl;
        close(sock_);
        exit(EXIT_FAILURE);
    }

    std::cout << "[KernelManager] 成功连接到调度器。" << std::endl;
}

std::string KernelManager::generateRequestID() {
    uint64_t id = ++requestIdCounter_;
    std::stringstream ss;
    ss << "req_" << id;
    return ss.str();
}

// --- Public 方法 ---

void KernelManager::enqueue(std::unique_ptr<Kernel> kernel) {

    std::string reqId = generateRequestID();
    std::string kernelType = getClassName(*kernel);
    std::string requestMessage = createRequestMessage(reqId, kernelType, UNIQUE_ID);
    // std::cout << "[KernelManager] 准备提交 (ID: " << reqId << "): " << kernelType << std::endl;

    // std::cout << "[KernelManager] (ID: " << reqId << ") 发送请求..." << std::endl;
    if (send(sock_, requestMessage.c_str(), requestMessage.length(), 0) < 0) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 发送失败！连接可能已断开。" << std::endl;
        return;
    }

    // std::cout << "[KernelManager] (ID: " << reqId << ") 正在等待调度器批准..." << std::endl;
    char buffer[1024] = {0};
    ssize_t bytesRead = read(sock_, buffer, 1023);
   
    if (bytesRead <= 0) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 从服务器读取响应失败。连接已断开。" << std::endl;
        return;
    }

    std::string responseMessage(buffer, bytesRead);
    responseMessage.erase(responseMessage.find_last_not_of("\r\n") + 1);
    
    auto parts = split_client(responseMessage, '|');
    if (parts.size() != 3 || parts[0] != reqId) {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 收到格式错误的响应: " << responseMessage << std::endl;
        return;
    }

    bool permissionGranted = (parts[1] == "1");
    std::string reason = parts[2];
    if (permissionGranted) {
        // std::cout << "[KernelManager] (ID: " << reqId << ") 批准！正在执行内核..." << std::endl;
        try {
            kernel->execute(); 
            // std::cout << "[KernelManager] (ID: " << reqId << ") 内核执行完毕。" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[KernelManager] (ID: " << reqId << ") 执行时异常: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[KernelManager] (ID: " << reqId << ") 内核被拒绝！原因: " << reason << std::endl;
    }
}