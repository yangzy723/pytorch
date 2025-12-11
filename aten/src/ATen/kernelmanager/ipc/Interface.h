#pragma once

/**
 * @file ipc_interface.h
 * @brief 共享的 IPC 抽象接口定义
 * 
 * 此文件定义了客户端和服务端共用的抽象接口。
 * 具体实现可以是共享内存、TCP、Unix Socket 等。
 */

#include "Common.h"
#include <memory>
#include <functional>
#include <vector>

namespace ipc {

// ============================================================
//  消息队列接口
// ============================================================

/**
 * IMessageQueue - 消息队列抽象接口
 * 
 * 支持单向的消息传递，具体实现可以是：
 * - SPSC 无锁队列（共享内存）
 * - 管道
 * - TCP 连接
 * - Unix Domain Socket
 */
class IMessageQueue {
public:
    virtual ~IMessageQueue() = default;

    // 非阻塞发送
    virtual bool trySend(const std::string& message) = 0;
    virtual bool trySend(const char* data, size_t len) = 0;

    // 阻塞式发送（带超时，timeout_ms < 0 表示无限等待）
    virtual bool sendBlocking(const std::string& message, int timeout_ms = DEFAULT_TIMEOUT_MS) = 0;
    virtual bool sendBlocking(const char* data, size_t len, int timeout_ms = DEFAULT_TIMEOUT_MS) = 0;

    // 非阻塞接收
    virtual bool tryReceive(char* buffer, size_t buffer_size) = 0;
    virtual bool tryReceive(std::string& out_message) = 0;

    // 阻塞式接收（带超时，timeout_ms < 0 表示无限等待）
    virtual bool receiveBlocking(char* buffer, size_t buffer_size, int timeout_ms = DEFAULT_TIMEOUT_MS) = 0;
    virtual bool receiveBlocking(std::string& out_message, int timeout_ms = DEFAULT_TIMEOUT_MS) = 0;

    // 状态查询
    virtual bool isEmpty() const = 0;
    virtual size_t size() const = 0;
};

// ============================================================
//  通信通道接口
// ============================================================

/**
 * IChannel - 双向通信通道接口
 * 
 * 提供客户端与服务端之间的双向通信能力
 */
class IChannel {
public:
    virtual ~IChannel() = default;

    // 获取请求队列（客户端 -> 服务端）
    virtual IMessageQueue& getRequestQueue() = 0;

    // 获取响应队列（服务端 -> 客户端）
    virtual IMessageQueue& getResponseQueue() = 0;

    // 连接状态管理
    virtual bool isClientConnected() const = 0;
    virtual void setClientConnected(bool connected) = 0;

    virtual bool isServerReady() const = 0;
    virtual void setServerReady(bool ready) = 0;

    // 通道标识
    virtual std::string getName() const = 0;
    
    // 扩展：获取客户端信息（服务端使用）
    virtual std::string getClientType() const { return ""; }
    virtual std::string getUniqueId() const { return ""; }
    virtual pid_t getClientPid() const { return 0; }
};

// ============================================================
//  注册表接口
// ============================================================

/**
 * IRegistry - 客户端注册表接口
 * 
 * 用于客户端注册和发现
 */
class IRegistry {
public:
    virtual ~IRegistry() = default;

    // 服务端就绪状态
    virtual bool isServerReady() const = 0;
    virtual void setServerReady(bool ready) = 0;

    // 客户端注册（返回分配的槽位索引，失败返回 -1）
    virtual int registerClient(const std::string& channelName, 
                               const std::string& clientType,
                               const std::string& uniqueId,
                               int64_t pid = 0) = 0;

    // 客户端注销
    virtual void unregisterClient(int slot) = 0;

    // 更新心跳
    virtual void updateHeartbeat(int slot) = 0;

    // 获取客户端信息
    virtual bool getClientInfo(int slot, ClientInfo& outInfo) const = 0;

    // 获取所有活跃客户端
    virtual std::vector<ClientInfo> getActiveClients() const = 0;

    // 获取版本号
    virtual uint32_t getVersion() const = 0;
};

// ============================================================
//  工厂接口
// ============================================================

/**
 * ITransportFactory - IPC 传输层工厂接口
 */
class ITransportFactory {
public:
    virtual ~ITransportFactory() = default;

    // 创建或打开通道
    // isCreator: true 表示创建者（负责初始化），false 表示连接者
    virtual std::unique_ptr<IChannel> createChannel(const std::string& name, bool isCreator) = 0;

    // 创建或打开注册表
    // isCreator: true 表示创建者（服务端），false 表示连接者（客户端）
    virtual std::unique_ptr<IRegistry> createRegistry(bool isCreator) = 0;

    // 销毁通道
    virtual void destroyChannel(const std::string& name) = 0;

    // 销毁注册表
    virtual void destroyRegistry() = 0;

    // 获取工厂名称
    virtual std::string getName() const = 0;
};

// ============================================================
//  客户端连接管理接口
// ============================================================

/**
 * IClientConnection - 客户端连接管理接口
 * 
 * 封装了连接建立、消息收发、连接关闭等完整流程
 */
class IClientConnection {
public:
    virtual ~IClientConnection() = default;

    // 连接到服务端
    virtual bool connect(int timeout_ms = DEFAULT_TIMEOUT_MS) = 0;

    // 断开连接
    virtual void disconnect() = 0;

    // 检查连接状态
    virtual bool isConnected() const = 0;

    // 发送请求并等待响应
    virtual bool sendRequest(const std::string& request, 
                             std::string& response,
                             int timeout_ms = DEFAULT_TIMEOUT_MS) = 0;

    // 获取底层通道
    virtual IChannel* getChannel() = 0;
};

// ============================================================
//  服务端监听接口
// ============================================================

/**
 * IServerListener - 服务端监听接口
 * 
 * 用于服务端监听和处理客户端连接
 */
class IServerListener {
public:
    virtual ~IServerListener() = default;

    // 初始化
    virtual bool init() = 0;

    // 开始监听，当有新客户端连接时调用回调
    virtual void start(std::function<void(std::unique_ptr<IChannel>)> onNewClient) = 0;

    // 停止监听
    virtual void stop() = 0;

    // 是否正在运行
    virtual bool isRunning() const = 0;

    // 获取注册表
    virtual IRegistry* getRegistry() = 0;
};

} // namespace ipc


