/*
 * flato.c — Fleet Plato MUD
 * 
 * Minimal multi-agent MUD in C17.
 * No dependencies. Pure poll() event loop.
 * Relies on edge-llama for inference via Unix socket.
 * 
 * Build: cc -std=c17 -O2 -o flato flato.c -lpthread
 * Run:   ./flato 4000 /tmp/edge-llama.sock
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <signal.h>

// =============================================================
//  Constants
// =============================================================

#define MAX_CLIENTS     64
#define BUF_SIZE        65536
#define MAX_LLAMA_REQ   8192
#define MAX_PEERS       32
#define MAX_TOKEN_HIST  2048
#define POLL_TIMEOUT_MS 100

// =============================================================
//  Client state
// =============================================================

typedef struct {
    int fd;
    char in_buf[BUF_SIZE];
    size_t in_len;
    char out_buf[BUF_SIZE];
    size_t out_len;
    size_t out_sent;
    
    enum { CS_IDLE, CS_PROMPTING, CS_GENERATING } state;
    bool connected;
    char prompt[BUF_SIZE];
    size_t prompt_len;
    
    // Token history (for context window)
    int32_t tokens[MAX_TOKEN_HIST];
    int n_tokens;
    
    time_t connected_at;
} Client;

// =============================================================
//  Server state
// =============================================================

typedef struct {
    // Listeners
    int telnet_fd;
    int llama_fd;       // Unix socket to edge-llama
    
    // Clients
    Client clients[MAX_CLIENTS];
    int n_clients;
    
    // Pending connect to edge-llama
    bool llama_connecting;
    int llama_connect_attempts;
    
    // Mesh peers
    struct {
        struct sockaddr_in addr;
        char name[64];
        bool active;
        time_t last_seen;
    } peers[MAX_PEERS];
    int n_peers;
    
    // Run flag & port
    volatile bool running;
    int port;
    
    // Command buffer (for /think responses)
    char cmd_buf[MAX_LLAMA_REQ];
    size_t cmd_len;
    int cmd_client_idx;  // Which client triggered the command
} Server;

static Server g_server = {0};

// =============================================================
//  Utils
// =============================================================

static void set_nonblock(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int find_client(int fd) {
    for (int i = 0; i < g_server.n_clients; i++) {
        if (g_server.clients[i].fd == fd && g_server.clients[i].connected)
            return i;
    }
    return -1;
}

static int add_client(int fd) {
    if (g_server.n_clients >= MAX_CLIENTS) {
        close(fd);
        return -1;
    }
    Client* c = &g_server.clients[g_server.n_clients++];
    memset(c, 0, sizeof(Client));
    c->fd = fd;
    c->connected = true;
    c->state = CS_IDLE;
    c->connected_at = time(NULL);
    set_nonblock(fd);
    return g_server.n_clients - 1;
}

static void remove_client(int idx) {
    if (idx < 0 || idx >= g_server.n_clients) return;
    Client* c = &g_server.clients[idx];
    if (c->fd >= 0) close(c->fd);
    c->connected = false;
    // Compact
    if (idx < g_server.n_clients - 1) {
        memmove(&g_server.clients[idx], &g_server.clients[idx + 1],
                (g_server.n_clients - idx - 1) * sizeof(Client));
    }
    g_server.n_clients--;
}

static void queue_output(int idx, const char* data, size_t len) {
    if (idx < 0 || idx >= g_server.n_clients) return;
    Client* c = &g_server.clients[idx];
    if (c->out_len + len >= BUF_SIZE) {
        len = BUF_SIZE - c->out_len - 1;
    }
    memcpy(c->out_buf + c->out_len, data, len);
    c->out_len += len;
}

static void queue_output_str(int idx, const char* s) {
    queue_output(idx, s, strlen(s));
}

// =============================================================
//  Edge-llama bridge
// =============================================================

static int connect_edge_llama(const char* socket_path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    
    set_nonblock(fd);
    return fd;
}

static void send_to_llama(const char* data, size_t len) {
    if (g_server.llama_fd < 0) return;
    ssize_t n = write(g_server.llama_fd, data, len);
    (void)n;
}

// =============================================================
//  Command handlers
// =============================================================

static void handle_cmd(int client_idx, const char* cmd_line) {
    char cmd[256];
    char arg[512] = "";
    int n = sscanf(cmd_line, "%255s %511[^\n]", cmd, arg);
    
    if (strcmp(cmd, "think") == 0 || strcmp(cmd, "t") == 0) {
        // Forward prompt to edge-llama
        Client* c = &g_server.clients[client_idx];
        c->state = CS_GENERATING;
        
        // Build request to edge-llama
        char req[MAX_LLAMA_REQ];
        int req_len = snprintf(req, sizeof(req), "{\"prompt\":\"%s\",\"max_tokens\":64}\n", arg);
        
            g_server.cmd_client_idx = client_idx;
        send_to_llama(req, req_len);
        
        // Clear output for streaming
        c->out_len = 0;
        c->out_sent = 0;
        
    } else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "h") == 0) {
        queue_output_str(client_idx,
            "\r\n=== flato MUD ===\r\n"
            "/think <prompt>   — Generate with edge-llama\r\n"
            "/status            — System health\r\n"
            "/gpu               — GPU status (nvidia-smi)\r\n"
            "/cuda              — CUDA/Capability details\r\n"
            "/peers             — List mesh peers\r\n"
            "/help              — This message\r\n"
            "/quit              — Disconnect\r\n"
            "\r\n> ");
            
    } else if (strcmp(cmd, "status") == 0) {
        char buf[512];
        int n = snprintf(buf, sizeof(buf),
            "\r\n=== Status ===\r\n"
            "Clients: %d\r\n"
            "Peers: %d\r\n"
            "edge-llama: %s\r\n"
            "Uptime: %lds\r\n"
            "\r\n> ",
            g_server.n_clients,
            g_server.n_peers,
            g_server.llama_fd >= 0 ? "connected" : "disconnected",
            (long)(time(NULL) - g_server.clients[client_idx].connected_at));
        queue_output(client_idx, buf, n < (int)sizeof(buf) ? n : sizeof(buf));
        
    } else if (strcmp(cmd, "cuda") == 0) {
        // Detailed CUDA status
        Client* c = &g_server.clients[client_idx];
        queue_output_str(client_idx, "\r\n");
        
        // CUDA toolkit version
        FILE* fp = popen("nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not found'", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                queue_output_str(client_idx, "CUDA:     ");
                queue_output(client_idx, line, strlen(line));
                queue_output_str(client_idx, "\r\n");
            }
            pclose(fp);
        }
        
        // CUDA devices
        fp = popen("nvidia-smi --query-gpu=index,name,compute_cap,clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                queue_output_str(client_idx, "Device:   ");
                queue_output(client_idx, line, strlen(line));
                queue_output_str(client_idx, "\r\n");
            }
            pclose(fp);
        }
        
        // CMA status
        fp = popen("grep -o 'CMA:[0-9]*/[0-9]*' /proc/meminfo 2>/dev/null || cat /proc/cmdline 2>/dev/null | grep -o 'cma=[^ ]*' || echo 'N/A'", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                queue_output_str(client_idx, "CMA:      ");
                queue_output(client_idx, line, strlen(line));
                queue_output_str(client_idx, "\r\n");
            }
            pclose(fp);
        }
        
        c->state = CS_IDLE;
        queue_output_str(client_idx, "\r\n> ");
        
    } else if (strcmp(cmd, "gpu") == 0) {
        // Query nvidia-smi for GPU status (reusing /status infrastructure)
        Client* c = &g_server.clients[client_idx];
        c->state = CS_GENERATING;
        g_server.cmd_client_idx = client_idx;
        queue_output_str(client_idx, "\r\n");
        
        // Read nvidia-smi output in child process
        FILE* fp = popen("nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null || echo 'nvidia-smi not available'", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                queue_output(client_idx, line, strlen(line));
                queue_output_str(client_idx, "\r\n");
            }
            pclose(fp);
        } else {
            queue_output_str(client_idx, "Failed to query GPU\r\n");
        }
        
        c->state = CS_IDLE;
        queue_output_str(client_idx, "\r\n> ");
        
    } else if (strcmp(cmd, "peers") == 0) {
        queue_output_str(client_idx, "\r\n=== Peers ===\r\n");
        for (int i = 0; i < g_server.n_peers && i < 10; i++) {
            char buf[128];
            int n = snprintf(buf, sizeof(buf),
                "  %s — %s %ds ago\r\n",
                g_server.peers[i].name,
                g_server.peers[i].active ? "active" : "inactive",
                (int)(time(NULL) - g_server.peers[i].last_seen));
            queue_output(client_idx, buf, n < (int)sizeof(buf) ? n : sizeof(buf));
        }
        queue_output_str(client_idx, "> ");
        
    } else if (strcmp(cmd, "migrate") == 0 || strcmp(cmd, "m") == 0) {
        // Hermit Crab Agent Migration — fleet-innovations #1
        // Migrate this agent's identity to a new shell
        Client* c = &g_server.clients[client_idx];
        c->state = CS_GENERATING;
        
        char buf[768];
        int len = snprintf(buf, sizeof(buf),
            "\r\n=== Hermit Crab Shell ===\r\n"
            "Agent:  JC1 (flato C telnet)\r\n"
            "Port:   %d\r\n"
            "Backend: libedge-cuda.so at /tmp/edge-native.sock\r\n"
            "Model:  deepseek-r1:1.5b Q4_K_M, 19 t/s (CPU)\r\n"
            "This shell is vacatable. Identity stored in PLATO rooms.\r\n"
            "Connectivity: Evennia MUD, edge-gateway, mesh bridge\r\n"
            "\r\n"
            "Need a successor? Drop a bottle to fleet for reassignment.\r\n"
            "> ",
            g_server.port);
        queue_output(client_idx, buf, len < (int)sizeof(buf) ? len : sizeof(buf));
        
        c->state = CS_IDLE;
        
    } else if (strcmp(cmd, "deadman") == 0) {
        // Deadman Switch Protocol — fleet-innovations #3
        // Check fleet heartbeat status
        queue_output_str(client_idx, "\r\n=== Deadman Switch ===\r\n");
        queue_output_str(client_idx, "See mesh bridge: python3 tools/mesh-bridge.py tick\r\n");
        queue_output_str(client_idx, "> ");
        
    } else {
        queue_output_str(client_idx, "\r\nUnknown command. Try /help\r\n> ");
    }
}

// =============================================================
//  Handle edge-llama response
// =============================================================

static void handle_llama_response(const char* data, size_t len) {
    int client_idx = g_server.cmd_client_idx;
    if (client_idx < 0 || client_idx >= g_server.n_clients) return;
    
    Client* c = &g_server.clients[client_idx];
    if (c->state != CS_GENERATING) return;
    
    // Forward to client
    queue_output(client_idx, data, len);
}

static void handle_llama_eof(void) {
    int client_idx = g_server.cmd_client_idx;
    if (client_idx < 0 || client_idx >= g_server.n_clients) return;
    
    Client* c = &g_server.clients[client_idx];
    if (c->state != CS_GENERATING) return;
    
    c->state = CS_IDLE;
    queue_output_str(client_idx, "\r\n> ");
    c->out_sent = 0;
    c->prompt_len = 0;
}

// =============================================================
//  Client input parsing
// =============================================================

static void handle_cmd_with_prompt(int idx, const char* prompt) {
    Client* c = &g_server.clients[idx];
    c->state = CS_GENERATING;
    
    char req[MAX_LLAMA_REQ];
    int req_len = snprintf(req, sizeof(req), "{\"prompt\":\"%s\",\"max_tokens\":64}\n", prompt);
    
    g_server.cmd_client_idx = idx;
    send_to_llama(req, req_len);
    
    c->out_len = 0;
    c->out_sent = 0;
    queue_output_str(idx, "\r\n");
}

static void process_client_line(int idx, const char* line) {
    if (line[0] == '/') {
        handle_cmd(idx, line + 1);
    } else if (line[0] == 0) {
        queue_output_str(idx, "> ");
    } else {
        // Wrap in /think
        char buf[MAX_LLAMA_REQ];
        snprintf(buf, sizeof(buf), "%s", line);
        handle_cmd_with_prompt(idx, buf);
    }
}

static void process_client_data(int idx) {
    Client* c = &g_server.clients[idx];
    if (c->in_len == 0) return;
    
    if (c->state == CS_GENERATING) {
        // Ignore input while generating
        c->in_len = 0;
        return;
    }
    
    // Process line by line
    char* data = c->in_buf;
    size_t remain = c->in_len;
    
    while (remain > 0) {
        char* nl = (char*)memchr(data, '\n', remain);
        if (!nl) {
            // Partial line — keep it
            if (data != c->in_buf && remain > 0) {
                memmove(c->in_buf, data, remain);
            }
            c->in_len = remain;
            return;
        }
        
        size_t line_len = nl - data;
        // Strip \r
        if (line_len > 0 && data[line_len - 1] == '\r') line_len--;
        
        // Copy line
        char line[1024];
        size_t copy_len = line_len < sizeof(line) - 1 ? line_len : sizeof(line) - 1;
        memcpy(line, data, copy_len);
        line[copy_len] = 0;
        
        process_client_line(idx, line);
        
        data = nl + 1;
        remain -= (nl - data + 1);
    }
    
    c->in_len = 0;
}

// =============================================================
//  Event loop
// =============================================================

static void handle_accept(void) {
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    int client_fd = accept(g_server.telnet_fd, (struct sockaddr*)&addr, &addrlen);
    if (client_fd < 0) return;
    
    int idx = add_client(client_fd);
    if (idx >= 0) {
        char buf[256];
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
        int n = snprintf(buf, sizeof(buf),
            "\r\n=== flato MUD ===\r\n"
            "Connected from %s:%d\r\n"
            "Type /help for commands.\r\n"
            "\r\n> ",
            ip, ntohs(addr.sin_port));
        queue_output(idx, buf, n);
    }
}

static void handle_client_data(int idx) {
    Client* c = &g_server.clients[idx];
    
    ssize_t n = read(c->fd, c->in_buf + c->in_len, BUF_SIZE - c->in_len - 1);
    if (n <= 0) {
        remove_client(idx);
        return;
    }
    
    c->in_len += n;
    c->in_buf[c->in_len] = 0;
    
    if (c->state == CS_IDLE) {
        process_client_data(idx);
    }
}

static void handle_client_write(int idx) {
    Client* c = &g_server.clients[idx];
    if (c->out_len == 0) return;
    
    ssize_t n = write(c->fd, c->out_buf + c->out_sent, c->out_len - c->out_sent);
    if (n > 0) {
        c->out_sent += n;
        if (c->out_sent >= c->out_len) {
            c->out_len = 0;
            c->out_sent = 0;
        }
    }
}

static void handle_llama_data(void) {
    if (g_server.llama_fd < 0) return;
    
    static char buf[BUF_SIZE];
    ssize_t n = read(g_server.llama_fd, buf, sizeof(buf) - 1);
    if (n <= 0) {
        // edge-llama disconnected
        close(g_server.llama_fd);
        g_server.llama_fd = -1;
        handle_llama_eof();
        return;
    }
    
    buf[n] = 0;
    
    // Check for JSON delimiter
    char* end = strstr(buf, "\n");
    if (end) {
        size_t msg_len = end - buf;
        char msg[8192];
        size_t copy = msg_len < sizeof(msg) - 1 ? msg_len : sizeof(msg) - 1;
        memcpy(msg, buf, copy);
        msg[copy] = 0;
        
        // Simple parsing: find "text" field (handle optional spaces after colon)
        char* text_start = strstr(msg, "\"text\":\"");
        if (!text_start) {
            text_start = strstr(msg, "\"text\": \"");
            if (text_start) text_start += 9;
        } else {
            text_start += 8;
        }
        if (text_start) {
            char* text_end = strchr(text_start, '"');
            if (text_end) {
                *text_end = 0;
                handle_llama_response(text_start, strlen(text_start));
            }
        }
        
        // Handle remaining
        size_t remaining = n - (end + 1 - buf);
        if (remaining > 0) {
            // Store for next read
        }
    } else {
        // No delimiter yet — check the data directly
        handle_llama_response(buf, n);
    }
}

static void event_loop(void) {
    g_server.running = true;
    
    while (g_server.running) {
        struct pollfd fds[MAX_CLIENTS + 2];
        nfds_t nfds = 0;
        
        // Telnet listener
        fds[nfds].fd = g_server.telnet_fd;
        fds[nfds].events = POLLIN;
        fds[nfds].revents = 0;
        nfds++;
        
        // edge-llama socket
        if (g_server.llama_fd >= 0) {
            fds[nfds].fd = g_server.llama_fd;
            fds[nfds].events = POLLIN;
            fds[nfds].revents = 0;
            nfds++;
        }
        
        // Clients
        for (int i = 0; i < g_server.n_clients; i++) {
            Client* c = &g_server.clients[i];
            if (!c->connected || c->fd < 0) continue;
            
            fds[nfds].fd = c->fd;
            fds[nfds].events = POLLIN;
            fds[nfds].revents = 0;
            if (c->out_len > c->out_sent) {
                fds[nfds].events |= POLLOUT;
            }
            nfds++;
        }
        
        int ret = poll(fds, nfds, POLL_TIMEOUT_MS);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        
        // Process events
        nfds_t idx = 0;
        
        // Telnet accept
        if (fds[idx].revents & POLLIN) {
            handle_accept();
        }
        idx++;
        
        // edge-llama
        if (g_server.llama_fd >= 0) {
            if (fds[idx].revents & (POLLIN | POLLHUP)) {
                handle_llama_data();
            }
            idx++;
        }
        
        // Clients
        int client_base = idx;
        int client_idx = 0;
        for (int i = 0; i < g_server.n_clients && idx < nfds; i++, client_idx++) {
            if (!g_server.clients[i].connected) continue;
            
            if (fds[idx].revents & POLLIN) {
                handle_client_data(i);
            }
            if (fds[idx].revents & POLLOUT) {
                handle_client_write(i);
            }
            if (fds[idx].revents & (POLLHUP | POLLERR)) {
                remove_client(i);
            }
            idx++;
        }
    }
}

// =============================================================
//  Main
// =============================================================

static void signal_handler(int sig) {
    (void)sig;
    g_server.running = false;
}

int main(int argc, char* argv[]) {
    printf("flato v0.1 — Fleet Plato MUD\n");
    
    int port = 4000;
    const char* llama_sock = "/tmp/edge-llama.sock";
    
    if (argc > 1) port = atoi(argv[1]);
    if (argc > 2) llama_sock = argv[2];
    
    g_server.port = port;
    
    // Set up signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create telnet socket
    g_server.telnet_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (g_server.telnet_fd < 0) {
        perror("socket");
        return 1;
    }
    
    int opt = 1;
    setsockopt(g_server.telnet_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (bind(g_server.telnet_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(g_server.telnet_fd);
        return 1;
    }
    
    if (listen(g_server.telnet_fd, 16) < 0) {
        perror("listen");
        close(g_server.telnet_fd);
        return 1;
    }
    
    printf("flato: listening on port %d\n", port);
    printf("flato: connecting to edge-llama at %s\n", llama_sock);
    
    // Connect to edge-llama
    g_server.llama_fd = connect_edge_llama(llama_sock);
    if (g_server.llama_fd < 0) {
        printf("flato: WARNING — edge-llama not available at %s\n", llama_sock);
        printf("flato: will retry on client commands\n");
    } else {
        printf("flato: edge-llama connected\n");
    }
    
    // Run event loop
    event_loop();
    
    // Cleanup
    printf("\nflato: shutting down...\n");
    for (int i = 0; i < g_server.n_clients; i++) {
        remove_client(i);
    }
    if (g_server.llama_fd >= 0) close(g_server.llama_fd);
    close(g_server.telnet_fd);
    
    printf("flato: goodbye.\n");
    return 0;
}
