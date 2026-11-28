// SPDX-License-Identifier: GPL-3.0-or-later
//! First-run setup from a source checkout. The repository ships no game code: every step runs one of the repository's own scripts
//! on the player's PC, against the player's own dump, and writes only inside the checkout (`game/`, `work/`, build folders) and the
//! run folder. The dump itself is only read.
//!
//! Steps, in order (each is skipped when its output is already there and current, so Set up after a pause continues):
//!   python   tools/.venv with recomp/requirements.txt (uv)
//!   exefs    game/dump/exefs: the dump's NSOs, `main` written uncompressed (tools/nso/prepare_exefs.py)
//!   lift     work/build/fefw/fefw_*.lib: lift and compile the six modules (recomp/scripts/build_game.cmd fefw)
//!   textures tools/textures/third_party: astc-encoder and bc7enc sources for the ASTC to BC7 transcoder (fetch_third_party.ps1)
//!   exe      game-build/fefw/build/release/fefw.exe (game-build/scripts/build_fefw.ps1)
//!   harness  the shader translator (tools/shaders: Ryujinx sources staged, dotnet build)
//!   shaders  work/shaders/spirv: shaders found in the romfs and main, translated to SPIR-V
//!   ini      fefw.ini in the run folder (settings.rs)
//! Pause kills the running step's process tree; the build tools keep finished work (ninja, the harness's results.tsv).

use serde::Serialize;
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const STEPS: [(&str, &str); 7] = [
    ("doctor", "Check this machine"),
    ("sources", "Fetch the build sources"),
    ("runtime", "Build the runtime"),
    ("ingest", "Read the game files"),
    ("lift", "Recompile the game code"),
    ("link", "Link the game"),
    ("done", "Finish"),
];

/// The steps a release package runs: the engine has nothing to fetch and no runtime to build there, because the
/// runtime and the compilers travel with it. Must match what xc2_setup.py reports in package mode.
pub const PACKAGE_STEP_IDS: [&str; 6] = ["doctor", "ingest", "lift", "link", "shaders", "done"];

fn step_title(id: &str) -> &'static str {
    match id {
        "doctor" => "Check this machine",
        "sources" => "Fetch the build sources",
        "runtime" => "Build the runtime",
        "ingest" => "Read the game files",
        "lift" => "Recompile the game code",
        "link" => "Link the game",
        "shaders" => "Translate the shaders",
        "done" => "Finish",
        _ => "Working",
    }
}

/// Every path the setup and Play use, derived from the checkout (or the release package), the dump and the run folder.
#[derive(Clone)]
pub struct Paths {
    /// the checkout; in the release package, the install folder
    pub repo: PathBuf,
    pub exefs: PathBuf,
    pub romfs: PathBuf,
    /// the dump's dlc/ folder (add-on content the player dumped), when it holds any
    pub dlc: Option<PathBuf>,
    pub game_dir: PathBuf,
    /// the release package's tools/ folder (tools/build/fefw_build.py next to the launcher); None in a checkout
    pub package: Option<PathBuf>,
}

impl Paths {
    fn in_game(&self, rel: &[&str]) -> PathBuf {
        rel.iter().fold(self.game_dir.clone(), |p, r| p.join(r))
    }
    pub fn python(&self) -> PathBuf {
        // A package carries its own interpreter and that is the one a player runs, so XC2_PYTHON must not override it:
        // on the maintainer's machine it did, and the package test ran on a different Python (whose path also ended
        // up in the log).
        if let Some(t) = &self.package {
            return package_python(t);
        }
        if let Some(p) = std::env::var_os("XC2_PYTHON") {
            return PathBuf::from(p);
        }
        match &self.package {
            Some(t) => package_python(t),
            None => {
                let venv = self.repo.join("tools").join(".venv").join("Scripts").join("python.exe");
                if venv.is_file() {
                    venv
                } else {
                    python_path().unwrap_or_else(|| PathBuf::from("python"))
                }
            }
        }
    }
    pub fn plain_exefs(&self) -> PathBuf {
        match &self.package {
            Some(_) => self.in_game(&["exefs"]),
            None => self.repo.join("game").join("dump").join("exefs"),
        }
    }
    /// RomFS mods ([fs] mods): <run folder>/mods/<name>/romfs
    pub fn mods(&self) -> PathBuf {
        self.in_game(&["mods"])
    }
    pub fn libs(&self) -> PathBuf {
        self.repo.join("work").join("build").join("fefw")
    }
    pub fn exe(&self) -> PathBuf {
        match &self.package {
            Some(_) => self.in_game(&["xc2_boot.exe"]),
            None => exe_path(&self.repo),
        }
    }
    pub fn harness(&self) -> PathBuf {
        match &self.package {
            Some(t) => t.join("spirv_harness").join(if cfg!(windows) { "spirv_harness.exe" } else { "spirv_harness" }),
            None => self.repo.join("tools").join("shaders").join("build").join("bin").join("Release").join("net10.0").join("spirv_harness.dll"),
        }
    }
    pub fn extractor(&self) -> PathBuf {
        match &self.package {
            Some(t) => t.join("shaders").join("extract_corpus.py"),
            None => self.repo.join("tools").join("shaders").join("ampere").join("extract_corpus.py"),
        }
    }
    pub fn corpus(&self) -> PathBuf {
        match &self.package {
            Some(_) => self.in_game(&["shader_cache", "corpus"]),
            None => self.repo.join("work").join("shaders").join("corpus"),
        }
    }
    pub fn spirv(&self) -> PathBuf {
        match &self.package {
            Some(_) => self.in_game(&["shader_cache", "spirv"]),
            None => self.repo.join("work").join("shaders").join("spirv"),
        }
    }
    pub fn bc7(&self) -> PathBuf {
        match &self.package {
            Some(_) => self.in_game(&["shader_cache", "bc7"]),
            None => self.repo.join("work").join("bc7_cache"),
        }
    }
    fn shader_stamp(&self) -> PathBuf {
        match &self.package {
            Some(_) => self.in_game(&["shader_cache", "setup.json"]),
            None => self.repo.join("work").join("shaders").join("setup.json"),
        }
    }
    pub fn log(&self) -> PathBuf {
        match &self.package {
            Some(_) => self.in_game(&["setup.log"]),
            None => self.repo.join("work").join("setup.log"),
        }
    }
}

/// The last `fefw_build.py --check`: (done.json mtime and size, fefw.exe mtime) -> current. The tools do not change while the
/// launcher runs and a finished build rewrites done.json, so one Python run per finished build is enough.
type CheckKey = (Option<std::time::SystemTime>, u64, Option<std::time::SystemTime>);
static BUILD_CHECK: Mutex<Option<(CheckKey, bool)>> = Mutex::new(None);

fn package_build_ok(p: &Paths) -> bool {
    let Some(tools) = &p.package else { return false };
    let done = p.in_game(&["build", "done.json"]);
    let (Ok(dm), true) = (fs::metadata(&done), p.exe().is_file()) else { return false };
    let key: CheckKey = (dm.modified().ok(), dm.len(), mtime(&p.exe()));
    if let Some((k, b)) = BUILD_CHECK.lock().unwrap().as_ref() {
        if *k == key {
            return *b;
        }
    }
    let mut cmd = Command::new(p.python());
    cmd.arg("-I").arg(tools.join("build").join("fefw_build.py")).arg("--check").arg("--game").arg(&p.game_dir);
    // the build for this processor (cpu.rs): a fefw.exe built for the other one is not current, so Set up rebuilds it
    if let Some(v) = crate::cpu::build_variant() {
        cmd.args(["--cpu", v]);
    }
    quiet(&mut cmd);
    let ok = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
    *BUILD_CHECK.lock().unwrap() = Some((key, ok));
    ok
}

/// The release package's Python: tools/python/python.exe (the embeddable package) on Windows. On Linux tools/python/bin/python3
/// (a relocatable build) when the package has one, else the system's python3, which then needs the lifter's packages (zbic, lz4,
/// capstone, zstandard) installed.
pub(crate) fn package_python(tools: &Path) -> PathBuf {
    if cfg!(windows) {
        return tools.join("python").join("python.exe");
    }
    let bundled = tools.join("python").join("bin").join("python3");
    if bundled.is_file() {
        return bundled;
    }
    which("python3").unwrap_or(bundled)
}

pub fn exe_path(repo: &Path) -> PathBuf {
    repo.join("xc2").join("build").join("boot-301").join("xc2_boot.exe")
}

pub const MODULES: [&str; 6] = ["main", "sdk", "subsdk0", "subsdk1", "subsdk2", "rtld"];

#[derive(Serialize, Clone, Default)]
pub struct StepState {
    pub id: String,
    pub title: String,
    /// "done", "todo", "running", "error"
    pub state: String,
    pub detail: String,
}

#[derive(Serialize, Clone, Default)]
pub struct Progress {
    /// "idle", "running", "done", "cancelled", "error"
    pub state: String,
    pub step: String,
    pub done: u64,
    pub total: u64,
    pub message: String,
}

/// The running job's state, shared with the page.
#[derive(Default)]
pub struct Shared {
    pub progress: Mutex<Progress>,
    pub log: Mutex<VecDeque<String>>,
    pub cancel: AtomicBool,
    pub pid: AtomicU32,
}

const LOG_LINES: usize = 600;

fn mtime(p: &Path) -> Option<std::time::SystemTime> {
    fs::metadata(p).ok().and_then(|m| m.modified().ok())
}

/// NSO flags (0x0C) of a file, None when it is not a readable NSO.
fn nso_flags(p: &Path) -> Option<u32> {
    let mut h = [0u8; 16];
    fs::File::open(p).ok()?.read_exact(&mut h).ok()?;
    (&h[..4] == b"NSO0").then(|| u32::from_le_bytes([h[12], h[13], h[14], h[15]]))
}

fn python_ok(p: &Paths) -> bool {
    if !p.python().is_file() {
        return false;
    }
    let mut cmd = Command::new(p.python());
    cmd.args(["-I", "-c", "import zbic, lz4, capstone, unicorn, elftools, zstandard"]);
    quiet(&mut cmd);
    cmd.stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

fn exefs_ok(p: &Paths) -> bool {
    let dir = p.plain_exefs();
    let stamp = fs::read_to_string(dir.join("source.txt")).unwrap_or_default();
    MODULES.iter().all(|m| dir.join(m).is_file())
        && nso_flags(&dir.join("main")).map_or(false, |f| f & 0x87 == 0)
        && stamp.lines().next().map(|l| l.trim()) == Some(crate::dump::fwd(&p.exefs).as_str())
}

fn libs_ok(p: &Paths) -> bool {
    let main = mtime(&p.plain_exefs().join("main"));
    MODULES.iter().all(|m| {
        let lib = p.libs().join(format!("fefw_{m}.lib"));
        lib.is_file() && mtime(&lib) >= main
    })
}

fn exe_ok(p: &Paths) -> bool {
    let exe = mtime(&p.exe());
    if p.package.is_some() {
        // A package has no per-module libraries to compare against: the engine's own record is the source of truth,
        // and the file being there is what the Play button needs.
        return exe.is_some();
    }
    exe.is_some() && MODULES.iter().all(|m| mtime(&p.libs().join(format!("fefw_{m}.lib"))) <= exe)
}

fn textures_ok(p: &Paths) -> bool {
    p.repo.join("tools").join("textures").join("third_party").join("astc-encoder").join("Source").join("astcenc.h").is_file()
}

const SHADER_TRANSLATOR_VERSION: u32 = 3; // 3: real derivatives for the NVN2 shuffle pattern (#29 eyes); 2: portable 32-lane warps (game-build/release/fefw_build.py has the same number)

fn shaders_ok(p: &Paths) -> bool {
    let stamp = fs::read_to_string(p.shader_stamp()).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&stamp).unwrap_or_default();
    p.spirv().join("results.tsv").is_file()
        && v["complete"].as_bool() == Some(true)
        && v["translator_version"].as_u64() == Some(SHADER_TRANSLATOR_VERSION as u64)
        && v["romfs"].as_str() == Some(crate::dump::fwd(&p.romfs).as_str())
}

/// The engine's own record of the run: <run folder>/setup/status.json.
fn engine_status(p: &Paths) -> (std::collections::HashMap<String, String>, String, String) {
    let text = fs::read_to_string(p.game_dir.join("setup").join("status.json")).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    let mut map = std::collections::HashMap::new();
    if let Some(arr) = v["steps"].as_array() {
        for s in arr {
            if let (Some(n), Some(st)) = (s["name"].as_str(), s["state"].as_str()) {
                map.insert(n.to_string(), st.to_string());
            }
        }
    }
    (map, v["state"].as_str().unwrap_or("").to_string(), v["step"].as_str().unwrap_or("").to_string())
}

/// Outcome of checking `<game_dir>/build/install-manifest.json` and synced `runtime/*.dll` against the current package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestGate {
    pub ok: bool,
    pub upgrade_needed: bool,
    pub stale_steps: Vec<String>,
    pub reason: String,
}

impl Default for ManifestGate {
    fn default() -> Self {
        Self { ok: true, upgrade_needed: false, stale_steps: Vec::new(), reason: String::new() }
    }
}

type FileStamp = (Option<std::time::SystemTime>, u64);
type ManifestCacheKey = (PathBuf, PathBuf, FileStamp, FileStamp, FileStamp, FileStamp, FileStamp, FileStamp);
static MANIFEST_GATE_CACHE: Mutex<Option<(ManifestCacheKey, ManifestGate)>> = Mutex::new(None);

pub fn invalidate_manifest_cache() {
    *MANIFEST_GATE_CACHE.lock().unwrap() = None;
}

fn quick_stamp(p: &Path) -> FileStamp {
    fs::metadata(p).map(|m| (m.modified().ok(), m.len())).unwrap_or((None, 0))
}

fn manifest_cache_key(repo: &Path, game_dir: &Path) -> ManifestCacheKey {
    (
        repo.to_path_buf(),
        game_dir.to_path_buf(),
        quick_stamp(&game_dir.join("build").join("install-manifest.json")),
        quick_stamp(&game_dir.join("setup").join("status.json")),
        quick_stamp(&game_dir.join("xc2_boot.exe")),
        quick_stamp(&repo.join("runtime").join("xc2_rt.dll")),
        quick_stamp(&game_dir.join("xc2_rt.dll")),
        quick_stamp(&repo.join("tools").join("spirv_harness")),
    )
}

fn mcpu_for_setup() -> Option<&'static str> {
    match crate::cpu::build_variant() {
        Some("fma") => Some("x86_64+cx16+fma"),
        Some("compat") => Some("x86_64+cx16+sse4_2+popcnt"),
        _ => None,
    }
}

fn engine_script_for_repo(repo: &Path) -> PathBuf {
    let in_repo = repo.join("engine").join("xc2_setup.py");
    if in_repo.is_file() {
        return in_repo;
    }
    engine_script(repo)
}

pub(crate) fn parse_manifest_gate_json(text: &str) -> Option<ManifestGate> {
    let v: serde_json::Value = serde_json::from_str(text.trim()).ok()?;
    let ok = v.get("ok")?.as_bool()?;
    let upgrade_needed = v.get("upgrade_needed").and_then(|x| x.as_bool()).unwrap_or(!ok);
    let stale_steps = v
        .get("stale_steps")
        .and_then(|x| x.as_array())
        .map(|arr| arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let reason = v.get("reason").and_then(|x| x.as_str()).unwrap_or_default().to_string();
    Some(ManifestGate { ok, upgrade_needed, stale_steps, reason })
}

fn run_manifest_check(py: &Path, engine: &Path, repo: &Path, game_dir: &Path, seed_legacy: bool) -> Option<ManifestGate> {
    if !py.is_file() || !engine.is_file() {
        return None;
    }
    let mut cmd = Command::new(py);
    cmd.arg("-u")
        .arg(engine)
        .arg("--check-manifest")
        .arg("--repo")
        .arg(repo)
        .arg("--install")
        .arg(game_dir);
    if let Some(mcpu) = mcpu_for_setup() {
        cmd.arg("--cpu").arg(mcpu);
    }
    if seed_legacy {
        cmd.arg("--seed-legacy");
    }
    quiet(&mut cmd);
    let out = cmd.stdin(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_manifest_gate_json(&String::from_utf8_lossy(&out.stdout))
}

/// Read-only package manifest check for `update::rebuild_needed`. Returns `None` when the package tools are absent.
pub fn check_package_manifest_readonly(install: &Path, game_dir: &Path) -> Option<ManifestGate> {
    let tools = install.join("tools");
    let py = package_python(&tools);
    let engine = engine_script_for_repo(install);
    if !py.is_file() || !engine.is_file() {
        return None;
    }
    run_manifest_check(&py, &engine, install, game_dir, false)
}

/// Package manifest gate: checks whether an already-built package install matches the current package's lift, link
/// (ABI + stub + synced runtime DLLs), and shader/prefill inputs. Checkout builds and not-yet-built installs pass
/// through unchanged. Cached across repeated `setup_status` polls and invalidated after setup or update runs.
pub fn manifest_gate(p: &Paths) -> ManifestGate {
    let Some(_) = &p.package else {
        return ManifestGate::default();
    };
    let status_done = engine_status(p).1 == "done";
    if !status_done {
        return ManifestGate::default();
    }
    if !p.exe().is_file() {
        return ManifestGate {
            ok: false,
            upgrade_needed: false,
            stale_steps: vec!["link".into()],
            reason: "xc2_boot.exe is not built yet".into(),
        };
    }
    let key = manifest_cache_key(&p.repo, &p.game_dir);
    if let Some((k, cached)) = MANIFEST_GATE_CACHE.lock().unwrap().as_ref() {
        if *k == key {
            return cached.clone();
        }
    }
    let engine = engine_script_for_repo(&p.repo);
    let gate = run_manifest_check(&p.python(), &engine, &p.repo, &p.game_dir, true).unwrap_or_else(|| ManifestGate {
        ok: false,
        upgrade_needed: true,
        stale_steps: vec!["link".into()],
        reason: "could not verify install manifest".into(),
    });
    // Re-snapshot key after optional legacy seeding so the written install-manifest.json stays a cache hit.
    let final_key = manifest_cache_key(&p.repo, &p.game_dir);
    *MANIFEST_GATE_CACHE.lock().unwrap() = Some((final_key, gate.clone()));
    gate
}

/// What the engine says is done already, step by step (the page's checklist).
pub fn steps(p: &Paths, running: Option<&str>) -> Vec<StepState> {
    let (states, _, current) = engine_status(p);
    let gate = if running.is_none() { manifest_gate(p) } else { ManifestGate::default() };
    // A package runs five steps, a checkout seven: showing the checkout's list in a package leaves "Fetch the build
    // sources" and "Build the runtime"永远 to do, which is what made a finished setup look unfinished.
    let ids: Vec<&str> = if p.package.is_some() {
        PACKAGE_STEP_IDS.to_vec()
    } else {
        STEPS.iter().map(|(id, _)| *id).collect()
    };
    ids.iter()
        .map(|id| {
            let stale_by_gate = !gate.ok && (gate.stale_steps.iter().any(|s| s == id) || *id == "done");
            let state = match states.get(*id).map(|s| s.as_str()) {
                Some("done") if stale_by_gate => "todo",
                Some("done") => "done",
                Some("failed") => "error",
                _ if current == *id || running == Some(*id) => "running",
                _ => "todo",
            };
            StepState { id: (*id).to_string(), title: step_title(id).to_string(), state: state.into(), detail: String::new() }
        })
        .collect()
}

pub fn complete(p: &Paths) -> bool {
    engine_status(p).1 == "done" && manifest_gate(p).ok
}

/// A program started from the AppImage gets the host's environment, not the AppImage's: its AppRun points PYTHONHOME, GTK_*,
/// GIO/GST/QT module paths and more into its temporary mount, which breaks Python programs (umu-run, Proton's script) and anything
/// that outlives the launcher (Steam started again by Add to Steam). Variables naming the mount lose those entries (and go when
/// nothing is left); the AppImage's own variables go. Nothing changes outside an AppImage.
pub(crate) fn host_env(cmd: &mut Command) {
    if !cfg!(target_os = "linux") {
        return;
    }
    let vars: Vec<(String, String)> = std::env::vars_os().filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?))).collect();
    for (k, v) in appimage_fixes(&vars) {
        match v {
            Some(v) => cmd.env(k, v),
            None => cmd.env_remove(k),
        };
    }
}

/// What host_env changes in `vars` (an environment with an AppImage's APPDIR in it): (name, new value), None removes it.
pub(crate) fn appimage_fixes(vars: &[(String, String)]) -> Vec<(String, Option<String>)> {
    // only an AppImage's AppRun: on Windows (';' lists) or with an APPDIR some other program set, nothing is touched
    if !cfg!(target_os = "linux") || !vars.iter().any(|(k, v)| k == "APPIMAGE" && !v.is_empty()) {
        return vec![];
    }
    let Some(appdir) = vars.iter().find(|(k, _)| k == "APPDIR").map(|(_, d)| d.trim_end_matches('/').to_string()).filter(|d| d.len() > 1) else {
        return vec![];
    };
    let mut out = vec![];
    for (k, v) in vars {
        if matches!(k.as_str(), "APPDIR" | "APPIMAGE" | "ARGV0" | "OWD" | "PYTHONHOME" | "PYTHONPATH" | "PYTHONDONTWRITEBYTECODE" | "PERLLIB") {
            out.push((k.clone(), None));
        } else if v.contains(&appdir) {
            let kept: Vec<&str> = v.split(':').filter(|p| !p.is_empty() && !p.contains(&appdir)).collect();
            out.push((k.clone(), (!kept.is_empty()).then(|| kept.join(":"))));
        }
    }
    out
}

pub(crate) fn quiet(cmd: &mut Command) {
    host_env(cmd);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS (the compilers it starts inherit it)
        cmd.creation_flags(0x0800_0000 | 0x0000_4000);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // its own process group, so kill_tree ends everything it starts (the lifter's workers, the compilers)
        cmd.process_group(0);
    }
}

/// Ends a process and everything it started (cmake, ninja, the compilers, the lifter's workers).
pub fn kill_tree(pid: u32) {
    if pid == 0 {
        return;
    }
    #[cfg(windows)]
    {
        let mut cmd = Command::new("taskkill");
        cmd.args(["/T", "/F", "/PID", &pid.to_string()]);
        quiet(&mut cmd);
        let _ = cmd.stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
    #[cfg(unix)]
    {
        // quiet() made the process a group leader: a negative pid signals the whole group
        let _ = Command::new("kill").args(["-KILL", "--", &format!("-{pid}")]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
}

/// ninja's "[12/345] ..." prefix.
fn ninja_progress(line: &str) -> Option<(u64, u64)> {
    let rest = line.strip_prefix('[')?;
    let (nums, _) = rest.split_once(']')?;
    let (a, b) = nums.split_once('/')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// "scan: 12 345" (extract_corpus.py --progress) or the harness's "done N/M" style lines.
fn tool_progress(line: &str) -> Option<(u64, u64)> {
    if let Some(rest) = line.strip_prefix("@progress ") {
        let mut it = rest.split_whitespace().skip(1);
        return Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?));
    }
    if let Some(rest) = line.strip_prefix("scan: ") {
        let mut it = rest.split_whitespace();
        return Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?));
    }
    ninja_progress(line)
}

pub struct Runner<'a> {
    pub shared: &'a Arc<Shared>,
    pub emit: &'a (dyn Fn(Progress) + Sync),
    pub emit_log: &'a (dyn Fn(Vec<String>) + Sync),
    pub log_file: Mutex<Option<fs::File>>,
    pub log_path: PathBuf,
}

impl Runner<'_> {
    fn log(&self, lines: Vec<String>) {
        if lines.is_empty() {
            return;
        }
        {
            let mut q = self.shared.log.lock().unwrap();
            for l in &lines {
                q.push_back(l.clone());
                while q.len() > LOG_LINES {
                    q.pop_front();
                }
            }
        }
        if let Some(f) = self.log_file.lock().unwrap().as_mut() {
            use std::io::Write;
            for l in &lines {
                let _ = writeln!(f, "{l}");
            }
        }
        (self.emit_log)(lines);
    }

    fn progress(&self, step: &str, done: u64, total: u64, message: &str) {
        let p = Progress { state: "running".into(), step: step.into(), done, total, message: message.into() };
        *self.shared.progress.lock().unwrap() = p.clone();
        (self.emit)(p);
    }

    /// Runs one command to the end, streaming its output into the log. Ok(true) = success, Ok(false) = paused.
    fn run(&self, step: &str, message: &str, mut cmd: Command, cwd: &Path) -> Result<bool, String> {
        if self.shared.cancel.load(Ordering::SeqCst) {
            return Ok(false);
        }
        self.progress(step, 0, 0, message);
        self.log(vec![format!("> {}", describe(&cmd))]);
        cmd.current_dir(cwd);
        cmd.env("PYTHONUNBUFFERED", "1").env("NINJA_STATUS", "[%f/%t] ");
        quiet(&mut cmd);
        let mut child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not start {:?}: {}", cmd.get_program(), e))?;
        self.shared.pid.store(child.id(), Ordering::SeqCst);
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let tx2 = tx.clone();
        let out = child.stdout.take().unwrap();
        let err = child.stderr.take().unwrap();
        let read = |r: Box<dyn Read + Send>, tx: std::sync::mpsc::Sender<String>| {
            std::thread::spawn(move || {
                for l in BufReader::new(r).split(b'\n').map_while(Result::ok) {
                    let s = String::from_utf8_lossy(&l).trim_end_matches('\r').to_string();
                    if tx.send(s).is_err() {
                        break;
                    }
                }
            })
        };
        let t1 = read(Box::new(out), tx);
        let t2 = read(Box::new(err), tx2);
        let mut batch = Vec::new();
        let mut script_error: Option<String> = None;
        let mut last = std::time::Instant::now();
        loop {
            if self.shared.cancel.load(Ordering::SeqCst) {
                kill_tree(child.id());
            }
            match rx.recv_timeout(Duration::from_millis(150)) {
                Ok(l) => {
                    if let Some((d, t)) = tool_progress(&l) {
                        let text = l.strip_prefix("@progress ").and_then(|r| r.splitn(4, ' ').nth(3)).unwrap_or(message).to_string();
                        self.progress(step, d, t, &text);
                    }
                    if let Some(e) = l.strip_prefix("@error ") {
                        script_error = Some(e.replace("\\n", "\n"));
                    }
                    // "@progress" carries numbers for the bar; it is not text for the log window
                    if !l.starts_with("@progress ") {
                        batch.push(l);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if !batch.is_empty() && (last.elapsed() > Duration::from_millis(250) || batch.len() > 200) {
                self.log(std::mem::take(&mut batch));
                last = std::time::Instant::now();
            }
        }
        self.log(batch);
        let _ = t1.join();
        let _ = t2.join();
        let st = child.wait().map_err(|e| e.to_string())?;
        self.shared.pid.store(0, Ordering::SeqCst);
        if self.shared.cancel.load(Ordering::SeqCst) {
            return Ok(false);
        }
        if st.success() {
            Ok(true)
        } else if st.code() == Some(2) && script_error.is_none() && step == "build" {
            Ok(false)
        } else if let Some(e) = script_error {
            Err(e)
        } else {
            Err(format!("{} failed ({}). The last lines of the log say why; the full log is {}.", message, st, crate::dump::fwd(&self.log_path)))
        }
    }
}

fn which(exe: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe)).find(|p| p.is_file())
}

/// The toolchain the project's own environment script puts on PATH (XC2_ENV), resolved once: the build scripts dot-source
/// it, so a machine that builds has its cmake/ninja/dotnet there and not necessarily on the launcher's PATH.
fn env_tools() -> &'static std::collections::HashMap<String, String> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<std::collections::HashMap<String, String>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut map = std::collections::HashMap::new();
        let Some(env_script) = std::env::var_os("XC2_ENV") else {
            return map;
        };
        let quoted = env_script.to_string_lossy().replace('\'', "''");
        let script = format!(
            ". '{}' | Out-Null; foreach ($t in 'cmake','ninja','dotnet','clang-cl','git','uv','pwsh') {{ $c = Get-Command $t -ErrorAction SilentlyContinue; if ($c) {{ 'TOOL=' + $t + '|' + $c.Source }} }}",
            quoted
        );
        // not find_tool(): that would call back into env_tools while the OnceLock is still being initialized
        let ps = shell_for_env();
        let out = Command::new(&ps)
            .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &script])
            .stdin(Stdio::null())
            .output();
        match out {
            Ok(o) => {
                for line in String::from_utf8_lossy(&o.stdout).lines() {
                    if let Some(rest) = line.trim().strip_prefix("TOOL=") {
                        if let Some((k, v)) = rest.split_once('|') {
                            map.insert(k.to_lowercase(), v.trim().to_string());
                        }
                    }
                }
                if map.is_empty() {
                    // Saying nothing here is how this fails: the page then reports every tool as missing while the
                    // machine has them all. The reason goes on the page instead.
                    let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
                    let tail: String = err.lines().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" / ");
                    *ENV_ERROR.lock().unwrap() = Some(if tail.is_empty() {
                        format!("{} read {} but found no tool on its PATH", ps.display(), env_script.to_string_lossy())
                    } else {
                        format!("{} could not read {}: {}", ps.display(), env_script.to_string_lossy(), tail)
                    });
                }
            }
            Err(e) => {
                *ENV_ERROR.lock().unwrap() = Some(format!("could not run {}: {e}", ps.display()));
            }
        }
        map
    })
}

/// Why the XC2_ENV toolchain could not be read, for the page. None when it was read (or was not asked for).
static ENV_ERROR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub fn env_error() -> Option<String> {
    let _ = env_tools(); // resolve it now if nothing has yet
    ENV_ERROR.lock().unwrap().clone()
}

/// The PowerShell to read the environment script with. Absolute paths first: the launcher's PATH is the user's, and a
/// shell is not on every one of them.
fn shell_for_env() -> PathBuf {
    for p in [
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    ] {
        let p = PathBuf::from(p);
        if p.is_file() {
            return p;
        }
    }
    which("pwsh.exe").or_else(|| which("powershell.exe")).unwrap_or_else(|| PathBuf::from("powershell.exe"))
}

/// A tool from XC2_ENV, else on PATH, else in one of its usual install folders.
fn find_tool(names: &[&str], extra: &[PathBuf]) -> Option<PathBuf> {
    let env = env_tools();
    for n in names {
        let key = n.trim_end_matches(".exe").to_lowercase();
        if let Some(p) = env.get(&key) {
            if Path::new(p).is_file() {
                return Some(PathBuf::from(p));
            }
        }
    }
    names.iter().find_map(|n| which(n)).or_else(|| extra.iter().find(|p| p.is_file()).cloned())
}

/// A command as one log line. The interpreter path carries the account name and the arguments carry the dump's, and
/// logs get pasted into issues: the program is logged by file name and the home folder becomes %USERPROFILE%.
fn describe(cmd: &Command) -> String {
    let prog = std::path::Path::new(cmd.get_program())
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| cmd.get_program().to_string_lossy().to_string());
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    let mut out = prog;
    for a in cmd.get_args() {
        let s = if home.is_empty() { a.to_string_lossy().to_string() } else { a.to_string_lossy().replace(&home, "%USERPROFILE%") };
        out.push(' ');
        out.push_str(&s);
    }
    out
}

/// The interpreter the setup engine runs: XC2_PYTHON, else python.exe on PATH.
fn python_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("XC2_PYTHON") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    find_tool(&["python.exe"], &[])
}

fn local_appdata(rel: &str) -> PathBuf {
    PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default()).join(rel)
}

pub fn powershell() -> PathBuf {
    find_tool(&["pwsh.exe"], &[PathBuf::from(r"C:\Program Files\PowerShell\7\pwsh.exe")]).unwrap_or_else(|| PathBuf::from("powershell.exe"))
}

/// One build prerequisite and whether it was found (the page lists them before Set up).
#[derive(Serialize, Clone)]
pub struct Tool {
    pub name: String,
    pub ok: bool,
    pub found: String,
    pub hint: String,
}

pub fn check_tools(package: bool) -> Vec<Tool> {
    if package {
        return vec![]; // everything the release package needs is in it
    }
    let pf = |rel: &str| PathBuf::from(std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".into())).join(rel);
    let pf86 = |rel: &str| PathBuf::from(std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".into())).join(rel);
    let llvm = std::env::var("NX_LLVM").ok().map(|d| PathBuf::from(d).join("clang-cl.exe"));
    let tc_llvm = std::env::var("NX_TOOLCHAIN").ok().map(|d| PathBuf::from(d).join("llvm").join("bin").join("clang-cl.exe"));
    let items: Vec<(&str, Option<PathBuf>, &str)> = vec![
        ("Visual Studio C++ tools", find_tool(&[], &[pf86(r"Microsoft Visual Studio\Installer\vswhere.exe")]),
         "Visual Studio 2022 Build Tools or newer with \"Desktop development with C++\"."),
        ("LLVM (clang-cl)", find_tool(&["clang-cl.exe"], &[llvm.unwrap_or_default(), tc_llvm.unwrap_or_default(), pf(r"LLVM\bin\clang-cl.exe")]),
         "LLVM 23 or newer from github.com/llvm/llvm-project/releases (or set NX_LLVM to its bin folder, or NX_TOOLCHAIN to a folder with llvm\\bin)."),
        ("CMake", find_tool(&["cmake.exe"], &[pf(r"CMake\bin\cmake.exe")]), "CMake 3.25 or newer (winget install Kitware.CMake)."),
        ("Ninja", find_tool(&["ninja.exe"], &[local_appdata(r"Microsoft\WinGet\Links\ninja.exe")]), "winget install Ninja-build.Ninja"),
        ("uv (Python)", find_tool(&["uv.exe"], &[local_appdata(r"Microsoft\WinGet\Links\uv.exe")]), "winget install astral-sh.uv"),
        ("Python 3.12 (zstandard)", python_path(), "Python 3.12 with zstandard (pip install zstandard), or set XC2_PYTHON to it. The Store alias on PATH is not one."),
        (".NET SDK", find_tool(&["dotnet.exe"], &[pf(r"dotnet\dotnet.exe")]), ".NET 10 SDK (winget install Microsoft.DotNet.SDK.10)."),
        ("Git", find_tool(&["git.exe"], &[pf(r"Git\cmd\git.exe")]), "winget install Git.Git (fetches the shader translator's sources)."),
    ];
    let mut out: Vec<Tool> = items
        .into_iter()
        .map(|(name, p, hint)| Tool { name: name.into(), ok: p.is_some(), found: p.map(|p| crate::dump::fwd(&p)).unwrap_or_default(), hint: hint.into() })
        .collect();
    // The toolchain script is itself a prerequisite, and the one whose failure used to be invisible.
    if std::env::var_os("XC2_ENV").is_some() {
        let err = env_error();
        out.insert(
            0,
            Tool {
                name: "XC2_ENV toolchain".into(),
                ok: err.is_none(),
                found: std::env::var("XC2_ENV").unwrap_or_default(),
                hint: err.unwrap_or_else(|| "The build tools below come from this script.".into()),
            },
        );
    }
    out
}

/// The Ryujinx source tree the shader translator is built from (MIT). `RYUBING_DIR`, else `<repo>/work/refs/ryubing`, cloned on
/// first use from the project's public repository.
const RYUBING_GIT: &str = "https://git.ryujinx.app/ryubing/ryujinx.git";
/// The revision tools/shaders' patches were made against (tools/shaders/scripts/setup_ryujinx.ps1).
const RYUBING_COMMIT: &str = "a82350bb774f70fcbd41c9987bf67a3775409963";

fn run_package_steps(r: &Runner, p: &Paths, tools: &Path, jobs: usize) -> Result<bool, String> {
    let game = &p.game_dir;
    fs::create_dir_all(game).map_err(|e| format!("cannot create {}: {}", game.display(), e))?;
    if !p.python().is_file() {
        return Err(format!("The bundled Python is missing ({}). Extract the release zip again so tools/ is complete.", crate::dump::fwd(&p.python())));
    }
    if !package_build_ok(p) {
        let mut c = Command::new(p.python());
        c.arg("-I").arg(tools.join("build").join("fefw_build.py")).arg("--exefs").arg(&p.exefs).arg("--game").arg(game);
        c.args(["--jobs", &jobs.to_string()]);
        let variant = crate::cpu::build_variant();
        if let Some(v) = variant {
            c.args(["--cpu", v]);
        }
        let what = if variant == Some("compat") {
            "Recompiling the game code from your dump (the compatible build for processors without FMA3)"
        } else {
            "Recompiling the game code from your dump"
        };
        if !r.run("build", what, c, game)? {
            return Ok(false);
        }
        *BUILD_CHECK.lock().unwrap() = None;
    }
    shader_steps(r, p, jobs, game)?;
    if r.shared.cancel.load(Ordering::SeqCst) {
        return Ok(false);
    }
    r.progress("ini", 0, 0, "Writing fefw.ini");
    crate::settings::ensure_play_paths(&p.game_dir, p)?;
    Ok(true)
}

/// Size and modification time of a file ("" when missing): enough to tell that it was replaced.
fn file_stamp(path: &Path) -> String {
    fs::metadata(path)
        .map(|m| {
            let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
            format!("{}:{}", m.len(), t)
        })
        .unwrap_or_default()
}

/// What the shader corpus was extracted from; a corpus whose marker differs is extracted again.
fn corpus_marker(p: &Paths) -> serde_json::Value {
    serde_json::json!({
        "translator_version": SHADER_TRANSLATOR_VERSION,
        "romfs": crate::dump::fwd(&p.romfs),
        "exefs": crate::dump::fwd(&p.plain_exefs()),
        "main": file_stamp(&p.plain_exefs().join("main")),
        "extractor": file_stamp(&p.extractor()),
    })
}

/// Blocks the harness set aside because they crashed the translator (status "crash" in results.tsv).
fn crashed_shaders(p: &Paths) -> usize {
    fs::read_to_string(p.spirv().join("results.tsv"))
        .unwrap_or_default()
        .lines()
        .skip(1)
        .filter(|l| l.split('\t').nth(4) == Some("crash"))
        .count()
}

/// Finds the shaders in the romfs and main and translates them to SPIR-V (both modes).
fn shader_steps(r: &Runner, p: &Paths, jobs: usize, cwd: &Path) -> Result<bool, String> {
    if shaders_ok(p) {
        return Ok(true);
    }
    if !p.harness().is_file() {
        return Err(format!("The shader translator is missing ({}).", crate::dump::fwd(&p.harness())));
    }
    let _ = fs::remove_file(p.shader_stamp());
    // The corpus is deleted once the shaders are translated, so one that is still there is from a Set up that stopped in
    // the translation. Its marker (written only after the extractor finished) says which dump and extractor made it.
    let marker = p.corpus().join("setup-corpus.json");
    let want = corpus_marker(p);
    let have: serde_json::Value = serde_json::from_str(&fs::read_to_string(&marker).unwrap_or_default()).unwrap_or_default();
    if have == want && p.corpus().join("unique.tsv").is_file() {
        r.log(vec![format!("Reusing the shaders found earlier in {}", crate::dump::fwd(&p.corpus()))]);
    } else {
        let _ = fs::remove_file(&marker);
        let mut c = Command::new(p.python());
        c.arg("-I").arg(p.extractor()).arg("--romfs").arg(&p.romfs).arg("--main").arg(p.plain_exefs().join("main"));
        c.arg("--exefs").arg(p.plain_exefs()).arg("--out").arg(p.corpus()).args(["--jobs", &jobs.min(8).to_string(), "--progress"]);
        if !r.run("shaders", "Finding the shaders in your romfs", c, cwd)? {
            return Ok(false);
        }
        fs::write(&marker, want.to_string()).map_err(|e| e.to_string())?;
    }
    let harness = |jobs: usize| {
        let mut c = if p.package.is_some() {
            Command::new(p.harness())
        } else {
            let mut c = Command::new("dotnet");
            c.arg(p.harness());
            c
        };
        c.arg("--batch").arg(p.corpus()).arg("--out").arg(p.spirv()).args(["--jobs", &jobs.to_string()]);
        // SPIR-V validation needs spirv-val from the Vulkan SDK, which players do not have (the release package never validates)
        if p.package.is_some() || (std::env::var_os("VULKAN_SDK").is_none() && which("spirv-val.exe").is_none()) {
            c.arg("--no-val");
        }
        c
    };
    // The harness survives translator crashes itself (it runs the translation in a child process and sets a crashing
    // shader aside, GitHub issue #46). If it still fails, once more with one job: it resumes from its journal.tsv.
    match r.run("shaders", "Translating the shaders to SPIR-V", harness(jobs), cwd) {
        Ok(true) => {}
        Ok(false) => return Ok(false),
        Err(e) => {
            r.log(vec![format!("The shader translation failed ({}); trying again with one job, from where it stopped.", e.lines().next().unwrap_or(""))]);
            if !r.run("shaders", "Translating the shaders to SPIR-V (again, one at a time)", harness(1), cwd)? {
                return Ok(false);
            }
        }
    }
    let crashed = crashed_shaders(p);
    if crashed > 0 {
        r.log(vec![format!("{crashed} shaders crashed the translator and were left out; the game draws without them (status \"crash\" in {}).",
                           crate::dump::fwd(&p.spirv().join("results.tsv")))]);
    }
    let stamp = serde_json::json!({ "complete": true, "translator_version": SHADER_TRANSLATOR_VERSION, "romfs": crate::dump::fwd(&p.romfs), "crashed": crashed });
    fs::write(p.shader_stamp(), stamp.to_string()).map_err(|e| e.to_string())?;
    // the corpus is only the extractor's output for the translator: several GB the game never reads
    let _ = fs::remove_dir_all(p.corpus());
    Ok(true)
}

fn engine_script(repo: &Path) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("engine").join("xc2_setup.py");
            if p.is_file() {
                return p;
            }
        }
    }
    repo.join("tools").join("setup").join("xc2_setup.py")
}

/// Runs the setup engine (engine/xc2_setup.py). The launcher is only the view: the engine writes setup/status.json and the
/// checklist above reads it back.
fn run_steps(r: &Runner, p: &Paths, _jobs: usize) -> Result<bool, String> {
    let engine = engine_script(&p.repo);
    if !engine.is_file() {
        return Err(format!("the setup engine was not found at {}", engine.display()));
    }
    if python_path().is_none() && !p.python().is_file() {
        return Err("Python was not found. Install Python 3.12 with zstandard, or set XC2_PYTHON to its python.exe.".into());
    }
    let mut c = Command::new(p.python());
    c.arg("-u")
        .arg(&engine)
        .arg("--repo")
        .arg(&p.repo)
        .arg("--install")
        .arg(&p.game_dir)
        .arg("--exefs")
        .arg(&p.exefs)
        .arg("--romfs")
        .arg(&p.romfs)
        // the page's own tool card is this check; doctor.ps1 does not consult XC2_ENV yet, so on a machine that builds
        // it still reports cmake/ninja missing (measured 2026-11-17: exit 1 with the toolchain active)
        // A package's doctor step only says "the package carries everything" (0.4 s), and skipping it left the page's
        // first circle unticked for the whole run. A checkout still skips it: doctor.ps1 does not consult XC2_ENV yet
        // and reports the toolchain missing on a machine that builds.
        .arg("--skip")
        .arg(if p.package.is_some() { "" } else { "doctor" });
    if let Some(dlc) = &p.dlc {
        c.env("XC2_AOC_DIR", dlc);
    }
    if !r.run("setup", "Setting up...", c, &p.repo)? {
        invalidate_manifest_cache();
        return Ok(false);
    }
    invalidate_manifest_cache();
    Ok(complete(p))
}

#[allow(dead_code)]
fn run_steps_legacy(r: &Runner, p: &Paths, jobs: usize) -> Result<bool, String> {
    if let Some(tools) = &p.package {
        return run_package_steps(r, p, tools, jobs);
    }
    let repo = &p.repo;
    let py = p.python();
    if !python_ok(p) {
        let uv = find_tool(&["uv.exe"], &[local_appdata(r"Microsoft\WinGet\Links\uv.exe")]).ok_or("uv was not found. Install it (winget install astral-sh.uv) and press Set up again.")?;
        if !py.is_file() {
            let mut c = Command::new(&uv);
            c.args(["venv", "tools/.venv", "--python", "3.13"]);
            if !r.run("python", "Creating the Python environment (tools/.venv)", c, repo)? {
                return Ok(false);
            }
        }
        let mut c = Command::new(&uv);
        c.args(["pip", "install", "--python", "tools/.venv", "-r", "recomp/requirements.txt"]);
        if !r.run("python", "Installing the Python packages", c, repo)? {
            return Ok(false);
        }
    }
    if !exefs_ok(p) {
        let mut c = Command::new(&py);
        c.arg("-I").arg("tools/nso/prepare_exefs.py").arg("--exefs").arg(&p.exefs).arg("--out").arg(p.plain_exefs());
        if !r.run("exefs", "Reading the game code from your dump", c, repo)? {
            return Ok(false);
        }
    }
    if !libs_ok(p) {
        let mut c = Command::new("cmd.exe");
        c.args(["/d", "/c", "recomp\\scripts\\build_game.cmd", "fefw"]).env("NX_JOBS", jobs.to_string());
        if !r.run("lift", "Recompiling the game code (an hour or more the first time)", c, repo)? {
            return Ok(false);
        }
    }
    if !textures_ok(p) {
        let mut c = Command::new(powershell());
        c.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "tools/textures/scripts/fetch_third_party.ps1"]);
        if !r.run("textures", "Fetching the texture transcoder sources (astc-encoder, bc7enc)", c, repo)? {
            return Ok(false);
        }
    }
    if !exe_ok(p) {
        let mut c = Command::new(powershell());
        c.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "game-build\\scripts\\build_fefw.ps1", "-Jobs", &jobs.to_string()]);
        if !r.run("exe", "Building fefw.exe", c, repo)? {
            return Ok(false);
        }
    }
    if !p.harness().is_file() || !shaders_ok(p) {
        let staged = repo.join("tools").join("shaders").join("third_party").join("ryubing");
        if !staged.is_dir() {
            let src = std::env::var("RYUBING_DIR").map(PathBuf::from).unwrap_or_else(|_| repo.join("work").join("refs").join("ryubing"));
            if !src.is_dir() {
                let git = find_tool(&["git.exe"], &[]).ok_or("Git was not found; it is needed once to fetch the shader translator's sources.")?;
                let mut c = Command::new(&git);
                c.args(["clone", "--filter=blob:none", "--no-checkout", RYUBING_GIT]).arg(&src);
                if !r.run("harness", "Fetching the Ryujinx shader translator sources (MIT)", c, repo)? {
                    return Ok(false);
                }
                let mut c = Command::new(&git);
                c.arg("-C").arg(&src).args(["checkout", RYUBING_COMMIT]);
                if !r.run("harness", "Checking out the translator's tested revision", c, repo)? {
                    return Ok(false);
                }
            }
            let mut c = Command::new(powershell());
            c.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "tools\\shaders\\scripts\\setup_ryujinx.ps1", "-Ryubing"]).arg(&src);
            if !r.run("harness", "Staging the translator sources", c, repo)? {
                return Ok(false);
            }
        }
        let mut c = Command::new("dotnet");
        c.args(["build", "tools\\shaders\\harness\\SpirvHarness.csproj", "-c", "Release"]);
        if !r.run("harness", "Building the shader translator", c, repo)? {
            return Ok(false);
        }
    }
    if !shader_steps(r, p, jobs, repo)? {
        return Ok(false);
    }
    r.progress("ini", 0, 0, "Writing fefw.ini");
    crate::settings::ensure_play_paths(&p.game_dir, p)?;
    r.log(vec![format!("fefw.ini is in {}", crate::dump::fwd(&p.game_dir))]);
    Ok(true)
}

/// Runs the whole setup on this thread. Returns the final progress (done, cancelled or error).
pub fn run(shared: &Arc<Shared>, p: &Paths, jobs: usize, emit: &(dyn Fn(Progress) + Sync), emit_log: &(dyn Fn(Vec<String>) + Sync)) -> Progress {
    let _ = fs::create_dir_all(p.repo.join("work"));
    let file = fs::OpenOptions::new().create(true).append(true).open(p.log()).ok();
    let r = Runner { shared, emit, emit_log, log_file: Mutex::new(file), log_path: p.log() };
    r.log(vec![format!("--- setup: dump exefs {}, romfs {}", crate::dump::fwd(&p.exefs), crate::dump::fwd(&p.romfs))]);
    let end = match run_steps(&r, p, jobs) {
        Ok(true) => Progress { state: "done".into(), message: "Setup finished. Press Play.".into(), ..Default::default() },
        Ok(false) => Progress { state: "cancelled".into(), message: "Paused. Set up continues where it stopped.".into(), ..Default::default() },
        Err(e) => {
            r.log(vec![format!("error: {e}")]);
            Progress { state: "error".into(), message: e, ..Default::default() }
        }
    };
    *shared.progress.lock().unwrap() = end.clone();
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_lines() {
        assert_eq!(ninja_progress("[12/345] Building CXX object x.obj"), Some((12, 345)));
        assert_eq!(ninja_progress("[ 3/ 9] x"), Some((3, 9)));
        assert_eq!(ninja_progress("no"), None);
        assert_eq!(tool_progress("scan: 5 10"), Some((5, 10)));
    }

    #[test]
    fn manifest_gate_json_parsing_and_missing_exe_gate() {
        let ok = parse_manifest_gate_json(r#"{"ok":true,"upgrade_needed":false,"stale_steps":[],"reason":""}"#).unwrap();
        assert!(ok.ok && !ok.upgrade_needed && ok.stale_steps.is_empty());

        let stale = parse_manifest_gate_json(
            r#"{"ok":false,"upgrade_needed":true,"stale_steps":["link","shaders"],"reason":"runtime DLLs out of sync (xc2_rt.dll)"}"#,
        )
        .unwrap();
        assert!(!stale.ok && stale.upgrade_needed);
        assert_eq!(stale.stale_steps, vec!["link".to_string(), "shaders".to_string()]);
        assert!(stale.reason.contains("xc2_rt.dll"));
        assert!(parse_manifest_gate_json("not json").is_none());

        // Package install whose status.json says "done" but xc2_boot.exe is missing must fail complete() and mark link+done todo.
        let root = std::env::temp_dir().join(format!("xc2-gate-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let setup_dir = root.join("setup");
        fs::create_dir_all(&setup_dir).unwrap();
        fs::write(
            setup_dir.join("status.json"),
            r#"{"state":"done","step":"","steps":[{"name":"doctor","state":"done"},{"name":"ingest","state":"done"},{"name":"lift","state":"done"},{"name":"link","state":"done"},{"name":"shaders","state":"done"},{"name":"done","state":"done"}]}"#,
        )
        .unwrap();
        let p = Paths {
            repo: root.clone(),
            exefs: root.join("exefs"),
            romfs: root.join("romfs"),
            dlc: None,
            game_dir: root.clone(),
            package: Some(root.join("tools")),
        };
        invalidate_manifest_cache();
        assert!(!complete(&p), "missing xc2_boot.exe must fail complete()");
        let st = steps(&p, None);
        assert_eq!(st.iter().find(|s| s.id == "link").unwrap().state, "todo");
        assert_eq!(st.iter().find(|s| s.id == "done").unwrap().state, "todo");
        assert_eq!(st.iter().find(|s| s.id == "shaders").unwrap().state, "done");
        let _ = fs::remove_dir_all(&root);
    }
}
