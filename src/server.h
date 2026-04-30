#ifndef EDGE_LLAMA_SERVER_H
#define EDGE_LLAMA_SERVER_H

#include "model_qwen2.h"
#include <string>

namespace edge_llama {

class Server {
public:
    Server(Qwen2Model& model);
    ~Server();

    bool start(const std::string& socket_path);
    bool start_tcp(int port);
    void stop();
    bool serve();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Qwen2Model& model_;
};

} // namespace edge_llama

#endif // EDGE_LLAMA_SERVER_H
