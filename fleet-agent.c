/*
 * fleet-agent.c — Compiled Fleet Agent (fleet-innovations #5)
 *
 * A lightweight C17 agent that handles trust scoring, heartbeat
 * management, and status reporting at C speed — no Python overhead.
 *
 * Built for Jetson arm64, links with simple popen/subprocess patterns,
 * and reads/writes to the same SQLite store as mesh-bridge.py.
 *
 * Compile: 
 *   gcc -std=c17 -O2 -o fleet-agent fleet-agent.c -lsqlite3 -lm
 *
 * Usage:
 *   ./fleet-agent tick          — Run one monitoring cycle
 *   ./fleet-agent status        — Print agent status JSON
 *   ./fleet-agent pulse <agent> — Send heartbeat pulse
 *   ./fleet-agent trust <agent> — Report trust score
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <sys/sysinfo.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <sqlite3.h>

/* Config */
#define TRUST_DB_PATH  "/home/lucineer/.openclaw/workspace/memory/fleet-trust.db"
#define DEADMAN_DB_PATH "/home/lucineer/.openclaw/workspace/memory/fleet-deadman.db"
#define PKI_DIR        "/home/lucineer/.openclaw/workspace/memory/plato-pki"
#define MAX_AGENTS     32
#define MAX_LINE       4096
#define GRACE_ORACLE1  300    /* 5 min in seconds */
#define GRACE_FORGEMASTER 1800  /* 30 min */
#define TRUST_DECAY    0.04f

/* Agent state */
typedef struct {
    char name[64];
    double trust_score;
    int observations;
    long last_pulse;      /* unix timestamp */
    int grace_period;     /* seconds */
} Agent;

/* SQL helpers */
static int cb_count(void *count, int cols, char **vals, char **names) {
    (void)cols; (void)names;
    if (vals[0]) *(int*)count = atoi(vals[0]);
    return 0;
}

static int cb_agent(void *agent, int cols, char **vals, char **names) {
    Agent *a = (Agent*)agent;
    (void)names;
    if (cols >= 1 && vals[0]) strncpy(a->name, vals[0], 63);
    if (cols >= 2 && vals[1]) a->trust_score = atof(vals[1]);
    if (cols >= 3 && vals[2]) a->observations = atoi(vals[2]);
    return 0;
}

/* ================================================================
 * Trust Scoring (portable C version of flux-trust Bayesian model)
 * ================================================================ */

static int trust_db_init(sqlite3 *db) {
    const char *sql = 
        "CREATE TABLE IF NOT EXISTS trust ("
        "  agent TEXT PRIMARY KEY,"
        "  alpha REAL DEFAULT 1.0,"
        "  beta REAL DEFAULT 1.0,"
        "  observations INTEGER DEFAULT 0,"
        "  last_updated INTEGER DEFAULT (strftime('%s','now'))"
        ")";
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "trust init error: %s\n", err);
        sqlite3_free(err);
    }
    return rc;
}

static double trust_score_sqlite(sqlite3 *db, const char *agent) {
    const char *sql = "SELECT alpha, beta, observations FROM trust WHERE agent=?";
    sqlite3_stmt *stmt = NULL;
    double score = 0.5; /* default */
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, agent, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            double alpha = sqlite3_column_double(stmt, 0);
            double beta = sqlite3_column_double(stmt, 1);
            int obs = sqlite3_column_int(stmt, 2);
            if (obs > 0) score = alpha / (alpha + beta);
        }
    }
    if (stmt) sqlite3_finalize(stmt);
    return score;
}

static int trust_observe(sqlite3 *db, const char *agent, int positive) {
    const char *sel = "SELECT alpha, beta FROM trust WHERE agent=?";
    sqlite3_stmt *stmt = NULL;
    double alpha = 1.0, beta = 1.0;
    
    if (sqlite3_prepare_v2(db, sel, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, agent, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            alpha = sqlite3_column_double(stmt, 0);
            beta = sqlite3_column_double(stmt, 1);
        }
    }
    if (stmt) sqlite3_finalize(stmt);
    
    if (positive) alpha += 1.0;
    else beta += 1.0;
    
    const char *upsert = 
        "INSERT INTO trust(agent, alpha, beta, observations, last_updated) "
        "VALUES(?,?,?,1,strftime('%s','now')) "
        "ON CONFLICT(agent) DO UPDATE SET "
        "alpha=excluded.alpha, beta=excluded.beta, "
        "observations=observations+1, "
        "last_updated=strftime('%s','now')";
    
    if (sqlite3_prepare_v2(db, upsert, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, agent, -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 2, alpha);
        sqlite3_bind_double(stmt, 3, beta);
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        return rc == SQLITE_DONE ? 0 : -1;
    }
    sqlite3_finalize(stmt);
    return -1;
}

static int trust_decay_all(sqlite3 *db) {
    const char *sql = 
        "UPDATE trust SET alpha = alpha * (1.0 - ?1), beta = beta * (1.0 - ?1) "
        "WHERE alpha + beta > 2.0";
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_double(stmt, 1, TRUST_DECAY);
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        return rc == SQLITE_DONE ? 0 : -1;
    }
    sqlite3_finalize(stmt);
    return -1;
}

/* ================================================================
 * P48 Agent Identity Vectors
 *
 * Each fleet agent has a 90-dim P48 signature (6-bit components, 0-63)
 * encoding its keyword affinity. Packed into ceil(90/8)=12 uint64s.
 *
 * Keyword ordering (90 total, 4 groups):
 *   EDGE(26):   jetson,cpu,gpu,memory,temperature,load,uptime,
 *               disk,thermal,fan,power,nvidia,cuda,nvcc,
 *               arm64,aarch64,swap,network,interface,sensor,
 *               telemetry,hardware,clock,throttle,edge,device
 *   RESEARCH(25): research,paper,study,findings,analysis,experiment,
 *                benchmark,performance,test,comparison,evaluation,
 *                learn,training,dataset,model,inference,llm,
 *                neural,embedding,vector,similarity,tile,
 *                investigation,methodology,result
 *   FLEET(25):  fleet,agent,oracle,forge,vessel,bottle,matrix,
 *               heartbeat,sync,mesh,iron,coordination,bridge,
 *               pki,cert,trust,deadman,migration,protocol,
 *               lighthouse,beacon,dm,conduit,message
 *   JC1(14):    jc1,jetsonclaw,plato,evennia,flato,mythos,
 *               cocapn,sovereign,infer,think,vessel,libllama,
 *               gguf
 *
 * Signature generation for each agent:
 *   1. Start with zero vector
 *   2. For matching keywords in agent's domain, add 1.0 + log(1+count)
 *   3. L2 normalize [0..63 range]
 * ================================================================ */

#define P48_DIMS 90
#define P48_PACKED 12  /* ceil(90/8) */

typedef struct {
    char name[64];
    uint64_t packed[P48_PACKED];  /* 12 × 6-bit components */
    int components[P48_DIMS];     /* unpacked for distance calc */
} P48Identity;

/* Pack 90 components (0-63) into 12 uint64s */
static void p48_pack(const int comps[P48_DIMS], uint64_t packed[P48_PACKED]) {
    memset(packed, 0, P48_PACKED * sizeof(uint64_t));
    for (int i = 0; i < P48_DIMS; i++) {
        int p = i / 8;
        int bit = (i % 8) * 6;
        packed[p] |= ((uint64_t)(comps[i] & 0x3F)) << bit;
    }
}

/* Unpack 12 uint64s into 90 components */
static void p48_unpack(const uint64_t packed[P48_PACKED], int comps[P48_DIMS]) {
    for (int i = 0; i < P48_DIMS; i++) {
        int p = i / 8;
        int bit = (i % 8) * 6;
        comps[i] = (int)((packed[p] >> bit) & 0x3F);
    }
}

/* Squared P48 distance between two unpacked vectors */
static int p48_dist_sq(const int a[P48_DIMS], const int b[P48_DIMS]) {
    int total = 0;
    for (int i = 0; i < P48_DIMS; i++) {
        int d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

/* Packed distance (12 uint64s) — faster */
static int p48_dist_sq_packed(const uint64_t a[P48_PACKED], const uint64_t b[P48_PACKED]) {
    int total = 0;
    for (int p = 0; p < P48_PACKED; p++) {
        uint64_t ax = a[p], bx = b[p];
        for (int i = 0; i < 8; i++) {
            int ca = (int)((ax >> (6 * i)) & 0x3F);
            int cb = (int)((bx >> (6 * i)) & 0x3F);
            int d = ca - cb;
            total += d * d;
        }
    }
    return total;
}

/* Fleet agent P48 identities */
static void agent_p48_identity(const char *agent_name, P48Identity *id) {
    strncpy(id->name, agent_name, 63);
    memset(id->components, 0, P48_DIMS * sizeof(int));
    
    if (strcmp(agent_name, "jc1") == 0 || strcmp(agent_name, "JetsonClaw1") == 0) {
        /* Edge(26): gpu,cuda,nvidia,arm64,memory,temperature,load,edge,device,thermal,jetson */
        id->components[0]++;  /* jetson[0] */
        id->components[2]++;  /* gpu[2] */
        id->components[12]++; /* cuda[12] */
        id->components[11]++; /* nvidia[11] */
        id->components[16]++; /* arm64[16] */
        id->components[3]++;  /* memory[3] */
        id->components[4]++;  /* temperature[4] */
        id->components[5]++;  /* load[5] */
        id->components[24]++; /* edge[24] */
        id->components[25]++; /* device[25] */
        id->components[7]++;  /* thermal[7] */
        /* Research(25): llm,model,embedding,inference */
        id->components[41]++; /* model[41] */
        id->components[42]++; /* inference[42] */
        id->components[38]++; /* llm[38] */
        /* Fleet(25): fleet,agent,heartbeat,sync,deadman,trust,bottle */
        id->components[51]++; /* fleet[51] */
        id->components[52]++; /* agent[52] */
        id->components[60]++; /* heartbeat[60] */
        id->components[61]++; /* sync[61] */
        id->components[68]++; /* deadman[68] */
        id->components[70]++; /* trust[70] */
        id->components[55]++; /* bottle[55] */
        /* JC1(14): jc1,plato,evennia,flato,mythos,cocapn,sovereign,infer,think */
        id->components[76]++; /* jc1[76] */
        id->components[78]++; /* plato[78] */
        id->components[79]++; /* evennia[79] */
        id->components[80]++; /* flato[80] */
        id->components[81]++; /* mythos[81] */
        id->components[82]++; /* cocapn[82] */
        id->components[84]++; /* infer[84] */
        id->components[85]++; /* think[85] */
    } else if (strcmp(agent_name, "oracle1") == 0 || strcmp(agent_name, "Oracle1") == 0) {
        id->components[51]++; id->components[52]++; id->components[53]++; /* fleet,agent,oracle */
        id->components[60]++; id->components[61]++; /* heartbeat,sync */
        id->components[62]++; /* mesh */
        id->components[65]++; id->components[66]++; id->components[67]++; /* pki,cert,trust */
        id->components[70]++; /* trust */
        id->components[72]++; id->components[75]++; /* migration,lighthouse */
        id->components[78]++; id->components[82]++; /* plato,cocapn */
        id->components[76]++; /* jc1 */
    } else if (strcmp(agent_name, "forgemaster") == 0 || strcmp(agent_name, "Forgemaster") == 0 ||
               strcmp(agent_name, "fm") == 0) {
        id->components[51]++; id->components[52]++; /* fleet,agent */
        id->components[56]++; /* matrix */
        id->components[60]++; /* heartbeat */
        id->components[72]++; /* migration */
        id->components[74]++; /* iron */
        id->components[54]++; /* forge */
        id->components[55]++; /* bottle */
    } else if (strcmp(agent_name, "kimi") == 0 || strcmp(agent_name, "KimiClaw") == 0) {
        id->components[56]++; /* matrix */
        id->components[74]++; /* iron */
    }
    
    /* Normalize */
    int total = 0;
    for (int i = 0; i < P48_DIMS; i++) {
        total += id->components[i];
    }
    if (total == 0) {
        id->components[0] = 1; /* Default: at least some edge affinity */
        total = 1;
    }
    
    /* Scale into [5..63] range based on contribution */
    for (int i = 0; i < P48_DIMS; i++) {
        int raw = id->components[i];
        if (raw > 0) {
            raw = (int)(5.0 + 58.0 * (double)raw / (double)total);
        }
        id->components[i] = raw > 63 ? 63 : raw;
    }
    
    /* Pack */
    p48_pack(id->components, id->packed);
}

/* Print P48 identity as JSON */
static void p48_identity_json(FILE *out, const P48Identity *id) {
    fprintf(out, "{\n  \"agent\": \"%s\",\n  \"p48_packed\": [", id->name);
    for (int i = 0; i < P48_PACKED; i++) {
        fprintf(out, "%s%llu", i > 0 ? ", " : "", 
                (unsigned long long)id->packed[i]);
    }
    fprintf(out, "]\n}\n");
}

/* Print P48 identity for all known fleet agents */
static void cmd_p48_agents(void) {
    static const char *agents[] = {"jc1", "oracle1", "forgemaster", "kimi", NULL};
    printf("[\n");
    for (int i = 0; agents[i]; i++) {
        P48Identity id;
        agent_p48_identity(agents[i], &id);
        p48_identity_json(stdout, &id);
        if (agents[i+1]) printf(",\n");
    }
    printf("]\n");
}

/* Compare agent identity to another — returns distance */
static int cmd_p48_compare(const char *a, const char *b) {
    P48Identity id_a, id_b;
    agent_p48_identity(a, &id_a);
    agent_p48_identity(b, &id_b);
    int dist = p48_dist_sq_packed(id_a.packed, id_b.packed);
    printf("P48 distance %s <-> %s: %d\n", a, b, dist);
    return dist;
}

static int deadman_db_init(sqlite3 *db) {
    const char *sql =
        "CREATE TABLE IF NOT EXISTS heartbeats ("
        "  agent TEXT PRIMARY KEY,"
        "  stage TEXT DEFAULT 'active',"
        "  last_pulse INTEGER DEFAULT (strftime('%s','now')),"
        "  grace_seconds INTEGER DEFAULT 300,"
        "  successor TEXT,"
        "  last_election INTEGER"
        ")";
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "deadman init error: %s\n", err);
        sqlite3_free(err);
    }
    return rc;
}

/* Returns: 0=active, 1=degraded, 2=orphaned, 3=handoff */
static int deadman_check(sqlite3 *db, const char *agent, long now) {
    const char *sql = "SELECT last_pulse, stage, grace_seconds FROM heartbeats WHERE agent=?";
    sqlite3_stmt *stmt = NULL;
    int stage = 3; /* handoff (unknown agent) */
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, agent, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            long last = sqlite3_column_int64(stmt, 0);
            const char *s = (const char*)sqlite3_column_text(stmt, 1);
            int grace = sqlite3_column_int(stmt, 2);
            long elapsed = now - last;
            
            if (s && strcmp(s, "handoff") == 0) stage = 3;
            else if (s && strcmp(s, "orphaned") == 0) stage = 2;
            else if (s && strcmp(s, "degraded") == 0) stage = 1;
            else if (elapsed < grace) stage = 0;
            else if (elapsed < grace * 2) stage = 1;
            else if (elapsed < grace * 3) stage = 2;
            else stage = 3;
        }
    }
    if (stmt) sqlite3_finalize(stmt);
    return stage;
}

static int deadman_pulse(sqlite3 *db, const char *agent, int grace_seconds) {
    const char *upsert =
        "INSERT INTO heartbeats(agent, stage, last_pulse, grace_seconds) "
        "VALUES(?, 'active', strftime('%s','now'), ?) "
        "ON CONFLICT(agent) DO UPDATE SET "
        "stage='active', last_pulse=strftime('%s','now'), "
        "grace_seconds=COALESCE(NULLIF(?,0), grace_seconds)";
    
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, upsert, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, agent, -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 2, grace_seconds);
        sqlite3_bind_int(stmt, 3, grace_seconds);
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        return rc == SQLITE_DONE ? 0 : -1;
    }
    sqlite3_finalize(stmt);
    return -1;
}

/* ================================================================
 * FLUX Bytecode Execution — eval constraint bytecode on edge
 * ================================================================ */

static int run_flux_bytecode(const char *bytecode_path) {
    char cmd[MAX_LINE];
    snprintf(cmd, sizeof(cmd),
        "/usr/local/bin/flux-vm %s 2>&1", bytecode_path);
    return system(cmd);
}

static int run_flux_source(const char *source, const char *asm_path) {
    char cmd[MAX_LINE * 4];
    snprintf(cmd, sizeof(cmd),
        "cat > %s << 'FLUXEOF'\n%s\nFLUXEOF\n"
        "/usr/local/bin/flux-asm %s -o /tmp/flux_agent.bin 2>&1 && "
        "/usr/local/bin/flux-vm /tmp/flux_agent.bin 2>&1",
        asm_path, source, asm_path);
    return system(cmd);
}

/* ================================================================
 * System Health (nproc, mem, load)
 * ================================================================ */

static long get_free_mem_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[MAX_LINE];
    long free_kb = -1, avail_kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemFree: %ld kB", &free_kb) == 1) continue;
        if (sscanf(line, "MemAvailable: %ld kB", &avail_kb) == 1) break;
    }
    fclose(f);
    return avail_kb > 0 ? avail_kb : free_kb;
}

static void system_status_json(FILE *out) {
    long free_kb = get_free_mem_kb();
    struct sysinfo info;
    sysinfo(&info);
    long uptime = info.uptime;
    
    fprintf(out,
        "{\n"
        "  \"status\": \"ok\",\n"
        "  \"free_mem_mb\": %.1f,\n"
        "  \"load_1m\": %ld,\n"
        "  \"load_5m\": %ld,\n"
        "  \"load_15m\": %ld,\n"
        "  \"uptime_hours\": %.1f\n"
        "}\n",
        free_kb / 1024.0,
        info.loads[0] / 65536, info.loads[1] / 65536, info.loads[2] / 65536,
        uptime / 3600.0
    );
}

/* ================================================================
 * Agent: Oracle1 ping (via shell HTTP)
 * ================================================================ */

static int ping_oracle1(void) {
    /* Use curl to ping Oracle1 shell */
    char cmd[MAX_LINE];
    snprintf(cmd, sizeof(cmd),
        "curl -s -m 5 'http://147.224.38.131:8848/cmd/shell' "
        "-X POST -H 'Content-Type: application/json' "
        "-d '{\"agent\":\"jc1\",\"command\":\"ping\"}' 2>/dev/null | head -c 200");
    return system(cmd);
}

/* ================================================================
 * Main
 * ================================================================ */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <command> [args]\n"
        "Commands:\n"
        "  tick             — Run one monitoring cycle (ping + deadman + trust)\n"
        "  status           — Print system health JSON\n"
        "  pulse <agent>    — Send heartbeat pulse\n"
        "  trust <agent>    — Show trust score for agent\n"
        "  observe <agent> <0|1> — Record positive(1) or negative(0) observation\n"
        "  agents           — List all known agents with trust\n"
        "  decay            — Apply trust decay\n"
        "  check <agent>    — Check deadman stage for agent\n\n"
        "  flux <bcfile>    — Run FLUX bytecode binary\n"
        "  flux-run <name>  — Run built-in FLUX demo (hello/fib/sum)\n"
        "  p48              — Show all fleet agent P48 identities\n"
        "  p48-compare <a> <b> — P48 distance between two agents\n",
        prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    
    const char *cmd = argv[1];
    
    if (strcmp(cmd, "status") == 0) {
        system_status_json(stdout);
        return 0;
    }
    
    /* Commands requiring SQLite */
    sqlite3 *t_db = NULL, *d_db = NULL;
    int t_rc = sqlite3_open(TRUST_DB_PATH, &t_db);
    int d_rc = sqlite3_open(DEADMAN_DB_PATH, &d_db);
    
    if (t_rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open trust DB: %s\n", sqlite3_errmsg(t_db));
        return 1;
    }
    if (d_rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open deadman DB: %s\n", sqlite3_errmsg(d_db));
        sqlite3_close(t_db);
        return 1;
    }
    
    trust_db_init(t_db);
    deadman_db_init(d_db);
    
    long now = time(NULL);
    
    if (strcmp(cmd, "tick") == 0) {
        printf("=== Fleet-Agent Tick %ld ===\n", now);
        
        /* Pulse agents */
        printf("Pulsing oracle1... ");
        deadman_pulse(d_db, "oracle1", GRACE_ORACLE1);
        trust_observe(t_db, "oracle1", 1);
        printf("done\n");
        
        printf("Pulsing forgemaster... ");
        deadman_pulse(d_db, "forgemaster", GRACE_FORGEMASTER);
        printf("done\n");
        
        /* Check stages */
        static const char *agents[] = {"oracle1", "forgemaster", "jc1", "kimi", "fm"};
        static const char *stages[] = {"active", "degraded", "orphaned", "handoff"};
        for (int i = 0; i < 5; i++) {
            int s = deadman_check(d_db, agents[i], now);
            double trust = trust_score_sqlite(t_db, agents[i]);
            printf("  %s: stage=%s trust=%.2f\n", agents[i], stages[s], trust);
        }
        
        /* Decay */
        trust_decay_all(t_db);
        printf("Trust decay applied (%.0f%%)\n", TRUST_DECAY * 100);
        
        /* Ping oracle1 */
        printf("Ping oracle1 shell: ");
        ping_oracle1();
        printf("\n\n=== Tick Complete ===\n");
        
    } else if (strcmp(cmd, "pulse") == 0 && argc >= 3) {
        const char *agent = argv[2];
        int grace = (argc >= 4) ? atoi(argv[3]) : GRACE_ORACLE1;
        if (deadman_pulse(d_db, agent, grace) == 0) {
            printf("Pulsed %s (grace=%ds)\n", agent, grace);
        }
    } else if (strcmp(cmd, "trust") == 0 && argc >= 3) {
        double score = trust_score_sqlite(t_db, argv[2]);
        printf("%.4f\n", score);
    } else if (strcmp(cmd, "observe") == 0 && argc >= 4) {
        int positive = atoi(argv[3]);
        if (trust_observe(t_db, argv[2], positive) == 0) {
            printf("Recorded %s observation for %s\n", 
                   positive ? "positive" : "negative", argv[2]);
        }
    } else if (strcmp(cmd, "agents") == 0) {
        const char *sql = "SELECT agent, alpha, beta, observations FROM trust ORDER BY agent";
        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(t_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            printf("%-20s %10s %6s\n", "AGENT", "TRUST", "OBS");
            printf("-------------------- ---------- ------\n");
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                const char *name = (const char*)sqlite3_column_text(stmt, 0);
                double alpha = sqlite3_column_double(stmt, 1);
                double beta = sqlite3_column_double(stmt, 2);
                int obs = sqlite3_column_int(stmt, 3);
                double score = (alpha + beta > 2) ? alpha / (alpha + beta) : 0.5;
                printf("%-20s %10.4f %6d\n", name, score, obs);
            }
        }
        sqlite3_finalize(stmt);
    } else if (strcmp(cmd, "decay") == 0) {
        trust_decay_all(t_db);
        printf("Trust decay applied\n");
    } else if (strcmp(cmd, "flux") == 0 && argc >= 3) {
        printf("=== FLUX Bytecode Execution ===\n");
        int rc = run_flux_bytecode(argv[2]);
        printf("Exit code: %d\n", rc);
    } else if (strcmp(cmd, "flux-run") == 0 && argc >= 3) {
        const char *demo = argv[2];
        if (strcmp(demo, "hello") == 0) {
            printf("=== FLUX Demo: hello ===\n");
            FILE *f = fopen("/tmp/flux_hello.asm", "w");
            if (f) {
                fprintf(f, "; Hello World FLUX bytecode\n");
                fprintf(f, "MOVI R0, 42\n");
                fprintf(f, "HALT\n");
                fclose(f);
            }
            run_flux_source("; Hello\nMOVI R0, 42\nHALT\n", "/tmp/flux_hello.asm");
        } else if (strcmp(demo, "fib") == 0) {
            printf("=== FLUX Demo: fibonacci ===\n");
            run_flux_source(
                "; Fibonacci up to 10\n"
                "MOVI R0, 0\n"
                "MOVI R1, 1\n"
                "MOVI R2, 5\n"
                "loop:\n"
                "ADD R0, R1\n"
                "DEC R1\n"
                "DEC R2\n"
                "JNZ R2, loop\n"
                "HALT\n",
                "/tmp/flux_fib.asm");
        } else if (strcmp(demo, "sum") == 0) {
            printf("=== FLUX Demo: sum 1..10 ===\n");
            run_flux_source(
                "; Sum 1..10\n"
                "MOVI R0, 0\n"
                "MOVI R1, 10\n"
                "loop:\n"
                "IADD R0, R1\n"
                "DEC R1\n"
                "JNZ R1, loop\n"
                "HALT\n",
                "/tmp/flux_sum.asm");
        }
    } else if (strcmp(cmd, "p48") == 0) {
        cmd_p48_agents();
    } else if (strcmp(cmd, "p48-compare") == 0 && argc >= 4) {
        cmd_p48_compare(argv[2], argv[3]);
    } else if (strcmp(cmd, "check") == 0 && argc >= 3) {
        static const char *stages[] = {"active", "degraded", "orphaned", "handoff"};
        int s = deadman_check(d_db, argv[2], now);
        double trust = trust_score_sqlite(t_db, argv[2]);
        printf("%s: stage=%s trust=%.4f\n", argv[2], stages[s], trust);
    } else {
        usage(argv[0]);
    }
    
    sqlite3_close(t_db);
    sqlite3_close(d_db);
    return 0;
}
