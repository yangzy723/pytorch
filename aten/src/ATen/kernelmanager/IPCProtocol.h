#pragma once

#include <string>

#define SCHEDULER_PORT 9999
#define LOCALHOST "127.0.0.1"

static std::string createRequestMessage(const std::string& id, const std::string& type) {
    return id + "|" + type + "\n";
}

static std::string createResponseMessage(const std::string& id, bool allowed, const std::string& reason) {
    return id + "|" + (allowed ? "1" : "0") + "|" + reason + "\n";
}