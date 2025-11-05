#include "KernelManager.h"

#include <iostream> // 用于打印日志 (可选)

// --- Singleton ---
KernelManager& KernelManager::getInstance() {
    // C++11 保证了静态局部变量的初始化是线程安全的
    static KernelManager instance;
    return instance;
}

// --- 构造函数 ---
KernelManager::KernelManager() : stop_(false) {
    // 启动工作线程，workerLoop 将成为新线程的入口点
    // workerThread_ = std::thread(&KernelManager::workerLoop, this);
    // std::cout << "[KernelManager] 工作线程已启动。" << std::endl; // 可以取消注释以进行调试
}

// --- 析构函数 ---
KernelManager::~KernelManager() {
    // std::cout << "[KernelManager] 准备关闭..." << std::endl; // 可以取消注释以进行调试
    shutdown();
    // std::cout << "[KernelManager] 已关闭。" << std::endl; // 可以取消注释以进行调试
}

// --- Public 方法 ---
void KernelManager::enqueue(std::unique_ptr<Kernel> kernel) {
    {
        // 1. 获取互斥锁，保护队列
        std::lock_guard<std::mutex> lock(queueMutex_);
        
        // 2. 将内核推入队列
        kernelQueue_.push(std::move(kernel));
        
        // (调试信息)
        // std::cout << "[KernelManager] 内核已入队。队列大小: " << kernelQueue_.size() << std::endl;
    
    } // 互斥锁在此处自动释放

    // 3. 唤醒一个正在等待的线程
    // (即工作线程，如果它正在休眠)
    // cv_.notify_one();
}

void KernelManager::launchKernels() {
    while (!kernelQueue_.empty()) {
        std::unique_ptr<Kernel> kernelToExecute = std::move(kernelQueue_.front());
        kernelQueue_.pop();
        kernelToExecute->execute();
    }
}

void KernelManager::shutdown() {
    // 1. 设置停止标志
    // 使用 exchange 来确保我们只关闭一次
    if (stop_.exchange(true)) {
        // 如果已经停止了，就不要再执行了
        return;
    }

    // 2. 唤醒工作线程，以便它可以检查 stop_ 标志并退出
    cv_.notify_one();

    // 3. 等待工作线程执行完毕
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
}

// --- Private 方法 ---
void KernelManager::workerLoop() {
    // 线程循环，直到 stop_ 为 true 且队列为空
    while (true) {
        std::unique_ptr<Kernel> kernelToExecute;

        {
            // 1. 上锁，准备检查队列
            std::unique_lock<std::mutex> lock(queueMutex_);

            // 2. 等待
            // 线程将在此处休眠，直到...
            // 2a. 队列不为空 (kernelQueue_.empty() == false)
            // 2b. stop_ 为 true
            // ... 满足任一条件
            cv_.wait(lock, [this] {
                return !kernelQueue_.empty() || stop_.load();
            });

            // 3. 检查退出条件
            // 如果被唤醒是因为 'stop_' 且队列已空，则退出循环
            if (stop_.load() && kernelQueue_.empty()) {
                return; // 退出线程
            }

            // 4. 如果队列不为空，则取出任务
            if (!kernelQueue_.empty()) {
                kernelToExecute = std::move(kernelQueue_.front());
                kernelQueue_.pop();
            }
        
        } // 锁在此处释放

        // 5. 执行任务 (如果成功取出了任务)
        // *** 重要的是：在锁之外执行 ***
        // 这样可以避免在执行一个耗时任务时阻塞其他线程的入队操作。
        if (kernelToExecute) {
            try {
                // (调试信息)
                // std::cout << "[KernelManager] 正在执行内核..." << std::endl;
                kernelToExecute->execute();
                // std::cout << "[KernelManager] 内核执行完毕。" << std::endl;
            } catch (const std::exception& e) {
                // 捕获并报告异常，防止工作线程崩溃
                std::cerr << "[KernelManager] 执行内核时捕获到异常: " << e.what() << std::endl;
            } catch (...) {
                // 捕获所有其他类型的异常
                std::cerr << "[KernelManager] 执行内核时捕获到未知异常。" << std::endl;
            }
        }
    }
}
