/// mtpdb_manager — Per-server MTPDB instance supervisor.
///
/// Workflow:
///   1. Launched on a remote server (directly or via SSH from mtpdb_master).
///   2. On startup, calls POST /internal/register on the master so it appears
///      in GET /servers.
///   3. Receives POST /provision from the master, which:
///        a. Allocates free ports from its pool
///        b. Generates a random password
///        c. Writes a config.toml for the new instance
///        d. Spawns `manaknightdb <config.toml>`
///        e. Returns { id, pg_port, mysql_port, api_port, quic_port, password }
///
/// CLI flags (all optional):
///   --server-id   <uuid>           (auto-generated if omitted)
///   --master-url  <http://...>     (skips registration if omitted)
///   --port        <n>              manager HTTP port (default 7001)
///   --region      <label>          (default "default")
///   --data-dir    <path>           base dir for instance data (default ./mtpdb_instances)
///   --mtpdb-bin   <path>           path to manaknightdb binary (default manaknightdb)
///   --pg-ports    <start>-<end>    PG port range  (default 15432-15532)
///   --mysql-ports <start>-<end>    MySQL port range (default 13306-13406)
///   --api-ports   <start>-<end>    API port range  (default 18765-18865)
///   --quic-ports  <start>-<end>    QUIC port range (default 14433-14533)

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use chrono::Utc;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    process::Child,
    sync::Arc,
    time::Duration,
};
use tokio::sync::Mutex;
use uuid::Uuid;

// ── instance info (serialisable, returned to callers) ─────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub id: String,
    pub db_name: String,
    pub pg_port: u16,
    pub mysql_port: u16,
    pub api_port: u16,
    pub quic_port: u16,
    pub password: String,
    pub data_dir: String,
    pub created_at: String,
    /// "running" | "stopped"
    pub status: String,
}

// ── port pool ─────────────────────────────────────────────────────────────────

struct PortPool {
    available: Vec<u16>,
    used: HashSet<u16>,
}

impl PortPool {
    fn from_range(start: u16, end: u16) -> Self {
        Self {
            available: (start..=end).collect(),
            used: HashSet::new(),
        }
    }

    /// Grab the next free port.
    fn acquire(&mut self) -> Option<u16> {
        if let Some(port) = self.available.pop() {
            self.used.insert(port);
            Some(port)
        } else {
            None
        }
    }

    /// Return a port to the pool.
    fn release(&mut self, port: u16) {
        if self.used.remove(&port) {
            self.available.push(port);
        }
    }
}

// ── application state ─────────────────────────────────────────────────────────

struct ManagerState {
    server_id: String,
    region: String,
    /// Our own IP as visible from the outside (used in registration).
    advertise_ip: String,
    #[allow(dead_code)]
    manager_port: u16,
    /// instances[id] = serialisable metadata
    instances: HashMap<String, InstanceInfo>,
    /// processes[id] = live child process handle
    processes: HashMap<String, Child>,
    /// Port pools
    pg_pool: PortPool,
    mysql_pool: PortPool,
    api_pool: PortPool,
    quic_pool: PortPool,
    /// Root directory; instances live in {data_dir}/{instance_id}/
    data_dir: String,
    /// Path (or name) of the manaknightdb binary to spawn
    mtpdb_bin: String,
}

type SharedState = Arc<Mutex<ManagerState>>;

// ── request bodies ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ProvisionRequest {
    db_name: Option<String>,
}

// ── error helper ──────────────────────────────────────────────────────────────

fn api_err(code: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    (code, Json(serde_json::json!({ "error": msg.into() })))
}

// ── handlers ──────────────────────────────────────────────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "service": "mtpdb_manager" }))
}

/// GET /info — server metadata + running instance list.
async fn info(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let st = state.lock().await;
    let instances: Vec<&InstanceInfo> = st.instances.values().collect();
    Json(serde_json::json!({
        "server_id": st.server_id,
        "region":    st.region,
        "ip":        st.advertise_ip,
        "instances": instances,
        "instance_count": instances.len(),
    }))
}

/// GET /instances — list all provisioned instances.
async fn list_instances(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let st = state.lock().await;
    let instances: Vec<&InstanceInfo> = st.instances.values().collect();
    Json(serde_json::json!({ "instances": instances, "count": instances.len() }))
}

/// GET /instances/:id — get one instance.
async fn get_instance(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let st = state.lock().await;
    st.instances
        .get(&id)
        .map(|inst| Json(serde_json::json!(inst)))
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, format!("Instance '{}' not found", id)))
}

/// POST /provision — spin up a new MTPDB instance.
async fn provision(
    State(state): State<SharedState>,
    Json(req): Json<ProvisionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut st = state.lock().await;

    // Allocate ports.
    let pg_port = st
        .pg_pool
        .acquire()
        .ok_or_else(|| api_err(StatusCode::SERVICE_UNAVAILABLE, "No PostgreSQL ports available"))?;
    let mysql_port = st.mysql_pool.acquire().ok_or_else(|| {
        st.pg_pool.release(pg_port);
        api_err(StatusCode::SERVICE_UNAVAILABLE, "No MySQL ports available")
    })?;
    let api_port = st.api_pool.acquire().ok_or_else(|| {
        st.pg_pool.release(pg_port);
        st.mysql_pool.release(mysql_port);
        api_err(StatusCode::SERVICE_UNAVAILABLE, "No API ports available")
    })?;
    let quic_port = st.quic_pool.acquire().ok_or_else(|| {
        st.pg_pool.release(pg_port);
        st.mysql_pool.release(mysql_port);
        st.api_pool.release(api_port);
        api_err(StatusCode::SERVICE_UNAVAILABLE, "No QUIC ports available")
    })?;

    let id = Uuid::new_v4().to_string();
    let db_name = req
        .db_name
        .unwrap_or_else(|| format!("db_{}", &id[..8]));
    let password = generate_password(24);

    let instance_dir = format!("{}/{}", st.data_dir, id);
    let data_dir = format!("{}/data", instance_dir);
    let wal_dir = format!("{}/wal", instance_dir);
    let config_path = format!("{}/config.toml", instance_dir);

    // Create directory structure.
    std::fs::create_dir_all(&instance_dir).map_err(|e| {
        release_ports(&mut st, pg_port, mysql_port, api_port, quic_port);
        api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("mkdir failed: {}", e))
    })?;

    // Write per-instance config.toml.
    let config_content = build_config_toml(
        pg_port, mysql_port, api_port, quic_port, &data_dir, &wal_dir, &password,
    );
    std::fs::write(&config_path, &config_content).map_err(|e| {
        release_ports(&mut st, pg_port, mysql_port, api_port, quic_port);
        api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("write config failed: {}", e))
    })?;

    // Spawn manaknightdb.
    let child = std::process::Command::new(&st.mtpdb_bin)
        .arg(&config_path)
        .spawn()
        .map_err(|e| {
            release_ports(&mut st, pg_port, mysql_port, api_port, quic_port);
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to spawn '{}': {}", st.mtpdb_bin, e),
            )
        })?;

    let info = InstanceInfo {
        id: id.clone(),
        db_name,
        pg_port,
        mysql_port,
        api_port,
        quic_port,
        password: password.clone(),
        data_dir: instance_dir.clone(),
        created_at: Utc::now().to_rfc3339(),
        status: "running".to_string(),
    };

    println!(
        "[manager] Provisioned instance {} — pg:{} mysql:{} api:{} quic:{}",
        id, pg_port, mysql_port, api_port, quic_port
    );

    st.processes.insert(id.clone(), child);
    st.instances.insert(id.clone(), info.clone());

    Ok(Json(serde_json::json!(info)))
}

/// DELETE /instances/:id — kill the process and free the ports.
async fn stop_instance(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut st = state.lock().await;

    let inst = st
        .instances
        .remove(&id)
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, format!("Instance '{}' not found", id)))?;

    // Kill the process if still running.
    if let Some(mut child) = st.processes.remove(&id) {
        let _ = child.kill();
        let _ = child.wait();
    }

    // Return ports to the pool.
    release_ports(&mut st, inst.pg_port, inst.mysql_port, inst.api_port, inst.quic_port);

    println!("[manager] Stopped instance {}", id);

    Ok(Json(serde_json::json!({ "status": "stopped", "id": id })))
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn release_ports(st: &mut ManagerState, pg: u16, mysql: u16, api: u16, quic: u16) {
    st.pg_pool.release(pg);
    st.mysql_pool.release(mysql);
    st.api_pool.release(api);
    st.quic_pool.release(quic);
}

fn generate_password(len: usize) -> String {
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| {
            let idx = rng.gen_range(0..62);
            match idx {
                0..=9 => (b'0' + idx) as char,
                10..=35 => (b'a' + idx - 10) as char,
                _ => (b'A' + idx - 36) as char,
            }
        })
        .collect()
}

/// Build a config.toml for a single manaknightdb instance.
///
/// The `[auth]` section is a forward-looking placeholder — password enforcement
/// depends on which auth mechanisms are wired into the MTPDB instance itself.
fn build_config_toml(
    pg_port: u16,
    mysql_port: u16,
    api_port: u16,
    quic_port: u16,
    data_dir: &str,
    wal_dir: &str,
    password: &str,
) -> String {
    format!(
        r#"# Auto-generated by mtpdb_manager — do not edit manually.

[server]
quic_port      = {quic_port}
api_port       = {api_port}
wire_port      = {mysql_port}
max_connections = 10000
ram_limit_mb   = 256

[cache]
hot_cache_size_mb = 64
eviction_policy   = "lru"

[storage]
data_dir          = "{data_dir}"
wal_dir           = "{wal_dir}"
memtable_size_mb  = 32

[postgresql]
enabled         = true
port            = {pg_port}
max_connections = 200
enable_ssl      = false

# The password below is returned to the caller at provisioning time.
# Wire it into the instance's auth configuration as needed.
[auth]
password = "{password}"
"#,
        quic_port = quic_port,
        api_port = api_port,
        mysql_port = mysql_port,
        data_dir = data_dir,
        wal_dir = wal_dir,
        pg_port = pg_port,
        password = password,
    )
}

/// Parse "start-end" range strings like "15432-15532".
fn parse_port_range(s: &str) -> (u16, u16) {
    let parts: Vec<&str> = s.splitn(2, '-').collect();
    if parts.len() == 2 {
        if let (Ok(a), Ok(b)) = (parts[0].parse(), parts[1].parse()) {
            return (a, b);
        }
    }
    eprintln!("Warning: could not parse port range '{}', using 0-0", s);
    (0, 0)
}

/// Register with the master server (best-effort; non-fatal on failure).
async fn register_with_master(
    master_url: &str,
    server_id: &str,
    ip: &str,
    region: &str,
    manager_port: u16,
) {
    let client = Client::new();
    let url = format!("{}/internal/register", master_url);
    let body = serde_json::json!({
        "server_id":    server_id,
        "ip":           ip,
        "region":       region,
        "manager_port": manager_port,
    });

    // Retry a few times to give the master a moment to be ready.
    for attempt in 1..=5 {
        match client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                println!("[manager] Registered with master at {}", master_url);
                return;
            }
            Ok(resp) => {
                eprintln!(
                    "[manager] Registration attempt {}: master returned {}",
                    attempt,
                    resp.status()
                );
            }
            Err(e) => {
                eprintln!("[manager] Registration attempt {} failed: {}", attempt, e);
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    eprintln!(
        "[manager] Could not register with master at {} after 5 attempts. \
         Continuing without registration.",
        master_url
    );
}

// ── CLI arg parsing (no extra deps) ──────────────────────────────────────────

struct Config {
    server_id: String,
    master_url: Option<String>,
    port: u16,
    region: String,
    data_dir: String,
    mtpdb_bin: String,
    pg_ports: (u16, u16),
    mysql_ports: (u16, u16),
    api_ports: (u16, u16),
    quic_ports: (u16, u16),
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_string())
    };

    Config {
        server_id: get("--server-id", &Uuid::new_v4().to_string()),
        master_url: args
            .windows(2)
            .find(|w| w[0] == "--master-url")
            .map(|w| w[1].clone()),
        port: get("--port", "7001").parse().unwrap_or(7001),
        region: get("--region", "default"),
        data_dir: get("--data-dir", "./mtpdb_instances"),
        mtpdb_bin: get("--mtpdb-bin", "manaknightdb"),
        pg_ports: parse_port_range(&get("--pg-ports", "15432-15532")),
        mysql_ports: parse_port_range(&get("--mysql-ports", "13306-13406")),
        api_ports: parse_port_range(&get("--api-ports", "18765-18865")),
        quic_ports: parse_port_range(&get("--quic-ports", "14433-14533")),
    }
}

// ── entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let cfg = parse_args();

    // Best-effort outbound IP detection (used for master registration).
    let advertise_ip = {
        use std::net::UdpSocket;
        if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
            if sock.connect("8.8.8.8:80").is_ok() {
                sock.local_addr()
                    .map(|a| a.ip().to_string())
                    .unwrap_or_else(|_| "127.0.0.1".to_string())
            } else {
                "127.0.0.1".to_string()
            }
        } else {
            "127.0.0.1".to_string()
        }
    };

    std::fs::create_dir_all(&cfg.data_dir).ok();

    let state: SharedState = Arc::new(Mutex::new(ManagerState {
        server_id: cfg.server_id.clone(),
        region: cfg.region.clone(),
        advertise_ip: advertise_ip.clone(),
        manager_port: cfg.port,
        instances: HashMap::new(),
        processes: HashMap::new(),
        pg_pool: PortPool::from_range(cfg.pg_ports.0, cfg.pg_ports.1),
        mysql_pool: PortPool::from_range(cfg.mysql_ports.0, cfg.mysql_ports.1),
        api_pool: PortPool::from_range(cfg.api_ports.0, cfg.api_ports.1),
        quic_pool: PortPool::from_range(cfg.quic_ports.0, cfg.quic_ports.1),
        data_dir: cfg.data_dir.clone(),
        mtpdb_bin: cfg.mtpdb_bin.clone(),
    }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/instances", get(list_instances))
        .route("/instances/:id", get(get_instance))
        .route("/instances/:id", delete(stop_instance))
        .route("/provision", post(provision))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    println!("╔══════════════════════════════════════════════════╗");
    println!("║           mtpdb_manager  v0.1.0                  ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("Server ID : {}", cfg.server_id);
    println!("Region    : {}", cfg.region);
    println!("Advertise : {}:{}", advertise_ip, cfg.port);
    println!("Data dir  : {}", cfg.data_dir);
    println!("MTPDB bin : {}", cfg.mtpdb_bin);
    println!("Listening on http://{}", addr);
    println!();
    println!("  GET  /health             health check");
    println!("  GET  /info               server metadata + instances");
    println!("  GET  /instances          list all instances");
    println!("  GET  /instances/:id      get instance details");
    println!("  POST /provision          create a new DB instance");
    println!("  DELETE /instances/:id    stop an instance");

    // Register with master in the background — don't block startup.
    if let Some(master_url) = cfg.master_url.clone() {
        let sid = cfg.server_id.clone();
        let ip = advertise_ip.clone();
        let region = cfg.region.clone();
        let port = cfg.port;
        tokio::spawn(async move {
            // Small delay so our own HTTP server is fully up before we call back.
            tokio::time::sleep(Duration::from_millis(500)).await;
            register_with_master(&master_url, &sid, &ip, &region, port).await;
        });
    } else {
        println!();
        println!("  (no --master-url given; running in standalone mode)");
    }

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
