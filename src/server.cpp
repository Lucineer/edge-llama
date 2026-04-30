#include "server.h"
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sstream>

namespace edge_llama {

struct Server::Impl {
    int server_fd = -1;
    int client_fd = -1;
    std::string socket_path;
    bool is_tcp = false;
    bool running = false;
};

Server::Server(Qwen2Model& model)
    : model_(model)
    , impl_(std::make_unique<Impl>()) {}

Server::~Server() {
    stop();
}

bool Server::start(const std::string& socket_path) {
    stop();
    unlink(socket_path.c_str());

    impl_->server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (impl_->server_fd < 0) return false;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(impl_->server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(impl_->server_fd);
        impl_->server_fd = -1;
        return false;
    }

    chmod(socket_path.c_str(), 0666);
    listen(impl_->server_fd, 5);
    fcntl(impl_->server_fd, F_SETFL, O_NONBLOCK);

    impl_->socket_path = socket_path;
    impl_->is_tcp = false;
    impl_->running = true;
    std::cerr << "edge-llama: Unix socket at " << socket_path << std::endl;
    return true;
}

bool Server::start_tcp(int port) {
    stop();

    impl_->server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (impl_->server_fd < 0) return false;

    int opt = 1;
    setsockopt(impl_->server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (bind(impl_->server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(impl_->server_fd);
        impl_->server_fd = -1;
        return false;
    }

    listen(impl_->server_fd, 5);
    fcntl(impl_->server_fd, F_SETFL, O_NONBLOCK);
    impl_->is_tcp = true;
    impl_->running = true;
    std::cerr << "edge-llama: TCP on port " << port << std::endl;
    return true;
}

void Server::stop() {
    impl_->running = false;
    if (impl_->client_fd >= 0) { close(impl_->client_fd); impl_->client_fd = -1; }
    if (impl_->server_fd >= 0) { close(impl_->server_fd); impl_->server_fd = -1; }
    if (!impl_->socket_path.empty() && !impl_->is_tcp) unlink(impl_->socket_path.c_str());
}

static std::string read_line(int fd) {
    std::string result;
    char c;
    while (read(fd, &c, 1) == 1) {
        if (c == '\n') break;
        result += c;
    }
    return result;
}

bool Server::serve() {
    if (!impl_->running) return false;

    struct pollfd pfd;
    pfd.fd = impl_->server_fd;
    pfd.events = POLLIN;
    pfd.revents = 0;

    int ret = ::poll(&pfd, 1, 0);
    if (ret <= 0) return false;

    if (pfd.revents & POLLIN) {
        int client = accept(impl_->server_fd, nullptr, nullptr);
        if (client < 0) return false;

        std::string prompt = read_line(client);
        if (!prompt.empty()) {
            std::string response = model_.generate(prompt, 128);
            std::string output = response + "\n";
            (void)write(client, output.c_str(), output.length());
        }
        close(client);
    }
    return true;
}

} // namespace edge_llama
