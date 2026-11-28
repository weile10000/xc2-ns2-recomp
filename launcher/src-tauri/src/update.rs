// SPDX-License-Identifier: GPL-3.0-or-later
//! Updates of the release package from inside the launcher: asks GitHub for the latest release, downloads its package for this
//! platform (XC2-package-dist-*.zip) with curl (Windows' own curl.exe), checks it against the SHA-256 digest GitHub lists for
//! the asset, unpacks it with tar (Windows' own tar.exe) and moves the new files over the install folder. Everything the old
//! package had is moved to update/old first and moved back if any step fails.
//! The player's files (xc2_boot.exe, nxrt.ini, controls.ini, save/, shader_cache/, logs) are not in the zip and stay as they are.
//! Afterwards `rebuild_needed` checks whether xc2_boot.exe is missing or the hook tables / stub sources changed.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// The GitHub repository whose releases the launcher follows. FEFW_UPDATE_REPO at build time: a fork's launcher follows the fork.
pub const REPO: &str = match option_env!("FEFW_UPDATE_REPO") {
    Some(r) => r,
    None => "weile10000/xc2-ns2-recomp",
};
/// FEFW_BUILD_ID at build time: the release tag this launcher was built for (a fork tagging its releases by commit). Any other
/// latest release then counts as newer, as tags like that have no order.
pub const BUILD_ID: Option<&str> = option_env!("FEFW_BUILD_ID");
/// The release asset for this platform, and files a downloaded package must have.
#[cfg(windows)]
const PACKAGE: (&str, &str, [&str; 4]) = (".zip", "Windows", ["xc2-launcher.exe", "runtime/xc2_rt.dll", "engine/xc2_setup.py", "tools/python/python.exe"]);
#[cfg(not(windows))]
const PACKAGE: (&str, &str, [&str; 4]) =
    ("-linux-x64.tar.gz", "Linux", ["xc2-launcher.AppImage", "runtime/xc2_rt.dll", "engine/xc2_setup.py", "tools/python/bin/python3"]);
const AGENT: &str = "xc2-launcher";
const CURL: &str = if cfg!(windows) { "curl.exe" } else { "curl" };
const TAR: &str = if cfg!(windows) { "tar.exe" } else { "tar" };
/// The launcher's working folder inside the install folder: downloads, the unpacked package, the previous version's files.
const WORK: &str = "update";

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Release {
    pub version: String,
    pub name: String,
    pub notes: String,
    pub page: String,
    pub asset_name: String,
    pub asset_url: String,
    pub size: u64,
    /// lowercase hex; empty when GitHub lists no digest (the update then refuses to install)
    pub sha256: String,
}

#[derive(Serialize, Clone, Default)]
pub struct Check {
    pub current: String,
    pub latest: Option<Release>,
    pub newer: bool,
    /// false when the launcher runs in a checkout of the repository (updates come from git there)
    pub package: bool,
}

#[derive(Serialize, Clone, Default)]
pub struct Status {
    /// idle, running, done, error
    pub state: String,
    /// download, verify, unpack, install, check
    pub phase: String,
    pub done: u64,
    pub total: u64,
    pub message: String,
    pub version: String,
    /// after an install: whether Set up has to build fefw.exe again (None: could not tell)
    pub rebuild: Option<bool>,
    pub rebuild_reason: String,
}

#[derive(Default)]
pub struct Shared {
    pub status: Mutex<Status>,
    pub cancel: AtomicBool,
}

/// "v1.2.3" or "1.2.3-x" -> (1, 2, 3)
pub fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim().trim_start_matches(['v', 'V']);
    let mut it = s.split('.').map(|p| p.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<u64>().ok());
    Some((it.next()??, it.next().flatten().unwrap_or(0), it.next().flatten().unwrap_or(0)))
}

fn system_tool(name: &str) -> PathBuf {
    if !cfg!(windows) {
        return PathBuf::from(name.trim_end_matches(".exe")); // curl and tar from PATH
    }
    let root = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let p = root.join("System32").join(name);
    if p.is_file() { p } else { PathBuf::from(name) }
}

fn curl() -> Command {
    let mut c = Command::new(system_tool(CURL));
    c.args(["--fail", "--location", "--silent", "--show-error", "--user-agent", AGENT]);
    crate::setup::quiet(&mut c);
    c
}

fn parse_release(v: &serde_json::Value) -> Release {
    let tag = v["tag_name"].as_str().unwrap_or_default();
    let version = tag.trim_start_matches(['v', 'V']).to_string();
    let assets = v["assets"].as_array().cloned().unwrap_or_default();
    let none = serde_json::Value::Null;
    let zip = assets
        .iter()
        .find(|a| a["name"].as_str().is_some_and(|n| n.starts_with("XC2-package-dist-") && n.ends_with(PACKAGE.0)))
        .unwrap_or(&none);
    let digest = zip["digest"].as_str().unwrap_or_default();
    Release {
        name: v["name"].as_str().filter(|s| !s.is_empty()).unwrap_or(tag).to_string(),
        notes: v["body"].as_str().unwrap_or_default().replace("\r\n", "\n"),
        page: v["html_url"].as_str().unwrap_or_default().to_string(),
        asset_name: zip["name"].as_str().unwrap_or_default().to_string(),
        asset_url: zip["browser_download_url"].as_str().unwrap_or_default().to_string(),
        size: zip["size"].as_u64().unwrap_or(0),
        sha256: digest.strip_prefix("sha256:").unwrap_or("").to_ascii_lowercase(),
        version,
    }
}

/// Picks the newest published release from GitHub's `/releases` list. `/releases/latest` skips pre-releases
/// (`prerelease = true`), whereas all `0.x` releases are marked pre-release, so listing + sorting by semver finds them.
fn select_latest_release(v: &serde_json::Value) -> Result<Release, String> {
    if let Some(arr) = v.as_array() {
        let mut best: Option<(Option<(u64, u64, u64)>, Release)> = None;
        for item in arr {
            if item["draft"].as_bool().unwrap_or(false) {
                continue;
            }
            let rel = parse_release(item);
            if rel.version.is_empty() {
                continue;
            }
            let ver = parse_version(&rel.version);
            let replace = match &best {
                None => true,
                Some((bv, _)) => match (ver, *bv) {
                    (Some(a), Some(b)) => a > b,
                    (Some(_), None) => true,
                    _ => false,
                },
            };
            if replace {
                best = Some((ver, rel));
            }
        }
        return best.map(|(_, r)| r).ok_or_else(|| "GitHub has no published releases yet.".to_string());
    }
    if v["tag_name"].is_string() {
        return Ok(parse_release(v));
    }
    Err("GitHub sent an unexpected release list.".to_string())
}

/// The latest published release of REPO and its package for this platform. A release without one (Linux, while only the Windows
/// zip is published) comes back with an empty asset: it can be shown, not installed.
pub fn latest() -> Result<Release, String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases?per_page=20");
    let out = curl()
        .args(["--retry", "2", "--retry-all-errors", "--max-time", "30", "--header", "Accept: application/vnd.github+json", &url])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run {CURL}: {e}"))?;
    if !out.status.success() {
        return Err(format!("Could not reach GitHub: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("GitHub sent an unexpected answer: {e}"))?;
    select_latest_release(&v)
}

pub fn check(current: &str, package: bool) -> Result<Check, String> {
    let rel = latest()?;
    let newer = match BUILD_ID {
        Some(_) => !rel.version.is_empty() && rel.version != current.trim_start_matches(['v', 'V']),
        None => matches!((parse_version(&rel.version), parse_version(current)), (Some(a), Some(b)) if a > b),
    };
    Ok(Check { current: current.to_string(), latest: Some(rel), newer, package })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut f = fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Runs one update: download, verify, unpack, install, check. `emit` gets every status change.
pub struct Run<'a> {
    pub shared: &'a Shared,
    pub install: PathBuf,
    pub game_dir: PathBuf,
    pub emit: &'a dyn Fn(Status),
}

impl Run<'_> {
    fn set(&self, phase: &str, done: u64, total: u64, message: &str) {
        let s = {
            let mut st = self.shared.status.lock().unwrap();
            st.state = "running".into();
            st.phase = phase.into();
            st.done = done;
            st.total = total;
            st.message = message.into();
            st.clone()
        };
        (self.emit)(s);
    }

    fn cancelled(&self) -> Result<(), String> {
        if self.shared.cancel.load(Ordering::SeqCst) { Err("Update cancelled.".into()) } else { Ok(()) }
    }

    pub fn run(&self, rel: &Release) -> Result<Status, String> {
        if rel.sha256.len() != 64 {
            return Err("GitHub lists no checksum for this download, so it cannot be verified. Download the package from the release page instead.".into());
        }
        if rel.asset_name.contains(['/', '\\']) || rel.asset_name.contains("..") {
            return Err("The release's file name is not usable.".into());
        }
        let work = self.install.join(WORK);
        fs::create_dir_all(&work).map_err(|e| format!("cannot create {}: {e}", work.display()))?;
        let zip = self.download(rel, &work)?;
        self.cancelled()?;
        let top = self.unpack(&zip, &work.join("stage"))?;
        self.cancelled()?;
        self.set("install", 0, 0, "Installing");
        install_with_game_dir(&self.install, &self.game_dir, &top, &work.join("old"))?;
        crate::setup::invalidate_manifest_cache();
        let _ = fs::remove_file(&zip);
        let _ = fs::remove_dir_all(work.join("stage"));
        self.set("check", 0, 0, "Checking whether the game has to be built again");
        let (rebuild, reason) = rebuild_needed(&self.install, &self.game_dir);
        let st = Status {
            state: "done".into(),
            phase: "done".into(),
            message: format!("Version {} is installed. Restart the launcher to use it.", rel.version),
            version: rel.version.clone(),
            rebuild,
            rebuild_reason: reason,
            ..Default::default()
        };
        Ok(st)
    }

    fn download(&self, rel: &Release, work: &Path) -> Result<PathBuf, String> {
        let zip = work.join(&rel.asset_name);
        // a finished download of the same file from an earlier attempt
        if zip.is_file() && fs::metadata(&zip).map(|m| m.len()).unwrap_or(0) == rel.size && sha256_file(&zip)? == rel.sha256 {
            return Ok(zip);
        }
        let part = work.join(format!("{}.part", rel.asset_name));
        let _ = fs::remove_file(&zip);
        let _ = fs::remove_file(&part);
        let mut child = curl()
            .args(["--retry", "3", "--speed-limit", "1024", "--speed-time", "60", "--output"])  // a stalled connection fails after a minute instead of hanging
            .arg(&part)
            .arg(&rel.asset_url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot run {CURL}: {e}"))?;
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        let status = loop {
            if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
                break s;
            }
            if self.shared.cancel.load(Ordering::SeqCst) {
                let _ = child.kill();
                let _ = child.wait();
                let _ = fs::remove_file(&part);
                return Err("Update cancelled.".into());
            }
            let got = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
            self.set("download", got, rel.size, &format!("Downloading {} ({:.0} of {:.0} MiB)", rel.asset_name, mb(got), mb(rel.size)));
            std::thread::sleep(Duration::from_millis(250));
        };
        if !status.success() {
            let mut err = String::new();
            if let Some(mut e) = child.stderr.take() {
                let _ = e.read_to_string(&mut err);
            }
            let _ = fs::remove_file(&part);
            return Err(format!("The download failed: {}", err.trim()));
        }
        self.set("verify", 0, 0, "Checking the download");
        let got = sha256_file(&part)?;
        if got != rel.sha256 {
            let _ = fs::remove_file(&part);
            return Err(format!("The download is damaged (SHA-256 {got}, expected {}). Try again.", rel.sha256));
        }
        fs::rename(&part, &zip).map_err(|e| format!("cannot rename {}: {e}", part.display()))?;
        Ok(zip)
    }

    /// Unpacks the zip into `stage` and returns its package root (supporting both a flat zip root and a single top folder).
    fn unpack(&self, zip: &Path, stage: &Path) -> Result<PathBuf, String> {
        self.set("unpack", 0, 0, "Unpacking");
        if stage.exists() {
            fs::remove_dir_all(stage).map_err(|e| format!("cannot clear {}: {e}", stage.display()))?;
        }
        fs::create_dir_all(stage).map_err(|e| format!("cannot create {}: {e}", stage.display()))?;
        let mut c = Command::new(system_tool(TAR));
        // -m: do not restore the file times. The files only need to be there; a zip with a time the file system cannot hold
        // ("Can't restore time: Invalid argument", seen on some Windows setups) must not fail the whole update.
        c.arg("-xmf").arg(zip).arg("-C").arg(stage);
        crate::setup::quiet(&mut c);
        let out = c.stdin(Stdio::null()).output().map_err(|e| format!("cannot run {TAR}: {e}"))?;
        if !out.status.success() && !only_time_errors(&String::from_utf8_lossy(&out.stderr)) {
            return Err(format!("Unpacking failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        let top = if stage.join(PACKAGE.2[0]).is_file() {
            stage.to_path_buf()
        } else {
            let dirs: Vec<PathBuf> = fs::read_dir(stage).map_err(|e| e.to_string())?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
            match dirs.as_slice() {
                [one] => one.clone(),
                _ => return Err("The package does not have the expected layout.".into()),
            }
        };
        for need in PACKAGE.2 {
            if !top.join(need).is_file() {
                return Err(format!("The package is missing {need}."));
            }
        }
        Ok(top)
    }
}

/// True when every line tar printed is a failure to set a file time (the contents were written). The package check after
/// unpacking still catches a missing file.
fn only_time_errors(stderr: &str) -> bool {
    let mut any = false;
    for l in stderr.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let l = l.to_ascii_lowercase();
        if l.contains("error exit delayed") {   // bsdtar's closing line after any non-fatal error
            continue;
        }
        if !l.contains("can't restore time") && !l.contains("cannot utime") {
            return false;
        }
        any = true;
    }
    any
}

/// Files of a tree, as paths relative to `root` (sorted).
fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![PathBuf::new()];
    while let Some(rel) = stack.pop() {
        let Ok(rd) = fs::read_dir(root.join(&rel)) else { continue };
        for e in rd.flatten() {
            let r = rel.join(e.file_name());
            match e.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(r),
                Ok(_) => out.push(r),
                Err(_) => {}
            }
        }
    }
    out.sort();
    out
}

/// fs::rename with a few retries: antivirus scanners and the search indexer hold new files open for a moment.
fn rename_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut wait = 100;
    loop {
        match fs::rename(from, to) {
            Err(e) if wait <= 1600 && e.kind() == std::io::ErrorKind::PermissionDenied => {
                std::thread::sleep(Duration::from_millis(wait));
                wait *= 2;
            }
            r => return r,
        }
    }
}

/// Top-level `*.dll` files in `dir`, sorted by file name.
fn top_level_dlls(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        if !p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dll")) {
            continue;
        }
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            out.push((name.to_string(), p));
        }
    }
    out.sort_by(|a, b| a.0.to_ascii_lowercase().cmp(&b.0.to_ascii_lowercase()));
    out
}

/// Syncs `install/runtime/*.dll` next to an existing `game_dir/xc2_boot.exe`, backing up previous copies inside
/// `game_dir/.update-old-dlls` (same volume as `game_dir` so renames never fail across drives) and rolling back
/// cleanly if any copy or rename fails. Player-added DLLs in `game_dir` are left untouched; only DLLs that previously
/// shipped in `old/runtime/` and disappeared from `install/runtime/` are removed.
fn sync_runtime_dlls(install: &Path, game_dir: &Path, old: &Path) -> Result<(), String> {
    if !game_dir.join("xc2_boot.exe").is_file() {
        return Ok(());
    }
    let runtime_dir = install.join("runtime");
    if !runtime_dir.is_dir() {
        return Ok(());
    }
    let new_dlls = top_level_dlls(&runtime_dir);
    let new_names: std::collections::HashSet<String> = new_dlls.iter().map(|(n, _)| n.to_ascii_lowercase()).collect();
    let stale_dlls: Vec<String> = top_level_dlls(&old.join("runtime"))
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| !new_names.contains(&n.to_ascii_lowercase()))
        .collect();

    let backup_dir = game_dir.join(".update-old-dlls");
    if backup_dir.exists() {
        let _ = fs::remove_dir_all(&backup_dir);
    }
    // (filename, old file moved to backup_dir, new file placed at game_dir/<name>)
    let mut done: Vec<(String, bool, bool)> = Vec::new();
    let mut tmp_paths: Vec<PathBuf> = Vec::new();
    let mut step = || -> Result<(), String> {
        let ensure_backup = || -> Result<(), String> {
            fs::create_dir_all(&backup_dir).map_err(|e| format!("cannot create {}: {e}", backup_dir.display()))
        };
        for name in &stale_dlls {
            let target = game_dir.join(name);
            if fs::symlink_metadata(&target).is_ok() {
                ensure_backup()?;
                rename_retry(&target, &backup_dir.join(name)).map_err(|e| {
                    format!("cannot move {} aside (is the game or another program using it?): {e}", target.display())
                })?;
                done.push((name.clone(), true, false));
            }
        }
        for (name, src) in &new_dlls {
            let target = game_dir.join(name);
            if target.is_file() && files_equal(src, &target).unwrap_or(false) {
                continue;
            }
            let staged = game_dir.join(format!(".{name}.upd-new"));
            let _ = fs::remove_file(&staged);
            fs::copy(src, &staged).map_err(|e| format!("cannot copy {} to {}: {e}", src.display(), staged.display()))?;
            tmp_paths.push(staged.clone());
            let had = fs::symlink_metadata(&target).is_ok();
            if had {
                ensure_backup()?;
                rename_retry(&target, &backup_dir.join(name)).map_err(|e| {
                    format!("cannot move {} aside (is the game or another program using it?): {e}", target.display())
                })?;
            }
            done.push((name.clone(), had, false));
            rename_retry(&staged, &target).map_err(|e| format!("cannot put {} in place: {e}", target.display()))?;
            done.last_mut().unwrap().2 = true;
        }
        Ok(())
    };
    if let Err(e) = step() {
        for p in &tmp_paths {
            let _ = fs::remove_file(p);
        }
        for (name, had, placed) in done.iter().rev() {
            let target = game_dir.join(name);
            if *placed {
                let _ = fs::remove_file(&target);
            }
            if *had {
                let _ = rename_retry(&backup_dir.join(name), &target);
            }
        }
        let _ = fs::remove_dir_all(&backup_dir);
        return Err(e);
    }
    let _ = fs::remove_dir_all(&backup_dir);
    Ok(())
}

/// Puts the new package `top` in place over `install`, file by file (a folder cannot be renamed while anything holds a handle
/// under it, a file can). Each old file goes to `old` first: the files the new package replaces, and in the package's own folders
/// (those the new package has, such as tools/) the files it no longer has. Anything else in the install folder (xc2_boot.exe,
/// nxrt.ini, save/, shader_cache/, build/) is left alone. When `xc2_boot.exe` already exists in `game_dir`, `runtime/*.dll` are
/// also synced beside it. Any failure moves back what was already moved. The running launcher can be moved too (Windows allows
/// renaming a running exe; on Linux the running AppImage keeps its old file until it exits).
pub fn install(install: &Path, top: &Path, old: &Path) -> Result<(), String> {
    install_with_game_dir(install, install, top, old)
}

pub fn install_with_game_dir(install: &Path, game_dir: &Path, top: &Path, old: &Path) -> Result<(), String> {
    if old.exists() {
        fs::remove_dir_all(old).map_err(|e| format!("cannot clear {} (close any program using it): {e}", old.display()))?;
    }
    fs::create_dir_all(old).map_err(|e| format!("cannot create {}: {e}", old.display()))?;
    let new_files = walk(top);
    let new_set: std::collections::HashSet<&PathBuf> = new_files.iter().collect();
    // the package's own folders: their files that the new package does not have are stale
    let mut stale = Vec::new();
    for e in fs::read_dir(top).map_err(|e| e.to_string())?.flatten() {
        if e.path().is_dir() {
            let name = PathBuf::from(e.file_name());
            // locales/ holds the shipped language tables, but the player may have put their own translation there too:
            // a table the package does not ship is not stale, it is theirs.
            if name.to_string_lossy().eq_ignore_ascii_case("locales") {
                continue;
            }
            stale.extend(walk(&install.join(&name)).into_iter().map(|r| name.join(r)).filter(|r| !new_set.contains(r)));
        }
    }
    // (relative path, the old file was moved to old/, a new file was moved in)
    let mut done: Vec<(PathBuf, bool, bool)> = Vec::new();
    let mut step = || -> Result<(), String> {
        let aside = |rel: &Path| -> Result<(), String> {
            let target = install.join(rel);
            let dest = old.join(rel);
            if let Some(d) = dest.parent() {
                fs::create_dir_all(d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
            }
            rename_retry(&target, &dest).map_err(|e| format!("cannot move {} aside (is the game or another program using it?): {e}", target.display()))
        };
        for rel in &stale {
            aside(rel)?;
            done.push((rel.clone(), true, false));
        }
        for rel in &new_files {
            let target = install.join(rel);
            // symlink_metadata: a symlink (python-build-standalone has them) whose target was moved aside still counts
            let had = fs::symlink_metadata(&target).is_ok();
            // a language table is added, never replaced: the copy in the install folder may be the player's own.
            let shared = rel.components().next().is_some_and(|c| c.as_os_str().to_string_lossy().eq_ignore_ascii_case("locales"));
            if had && shared {
                continue;
            }
            if had {
                aside(rel)?;
            }
            done.push((rel.clone(), had, false));
            if let Some(d) = target.parent() {
                fs::create_dir_all(d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
            }
            rename_retry(&top.join(rel), &target).map_err(|e| format!("cannot put {} in place: {e}", target.display()))?;
            done.last_mut().unwrap().2 = true;
        }
        Ok(())
    };
    let rollback_pkg = |done: &[(PathBuf, bool, bool)]| {
        for (rel, had, placed) in done.iter().rev() {
            let target = install.join(rel);
            if *placed {
                let _ = fs::remove_file(&target);
            }
            if *had {
                let _ = rename_retry(&old.join(rel), &target);
            }
        }
    };
    if let Err(e) = step() {
        rollback_pkg(&done);
        return Err(format!("{e} Nothing was changed."));
    }
    if let Err(e) = sync_runtime_dlls(install, game_dir, old) {
        rollback_pkg(&done);
        return Err(format!("{e} Nothing was changed."));
    }
    Ok(())
}

/// Compares two files without loading whole multi-hundred-MB DLLs into memory: checks length first, then streams
/// in chunks (returning `Err` if either file cannot be opened/read so callers fail closed instead of silently skipping).
fn files_equal(a: &Path, b: &Path) -> std::io::Result<bool> {
    if fs::metadata(a)?.len() != fs::metadata(b)?.len() {
        return Ok(false);
    }
    let mut fa = fs::File::open(a)?;
    let mut fb = fs::File::open(b)?;
    let mut ba = vec![0u8; 256 * 1024];
    let mut bb = vec![0u8; 256 * 1024];
    loop {
        let na = fa.read(&mut ba)?;
        if na == 0 {
            return Ok(fb.read(&mut bb[..1])? == 0);
        }
        let mut got = 0;
        while got < na {
            let nb = fb.read(&mut bb[got..na])?;
            if nb == 0 {
                return Ok(false);
            }
            got += nb;
        }
        if ba[..na] != bb[..na] {
            return Ok(false);
        }
    }
}

/// Compares a directory under `install` against the backup saved aside under `update/old` (because `tar -m` gives all
/// unpacked files the current timestamp). Returns `Some(reason)` when files were added, removed, modified, or unreadable.
fn dir_changed(install: &Path, old_dir: &Path, rel_dir: &str, missing_or_list_reason: &str) -> Option<String> {
    let new_d = install.join(rel_dir);
    let old_d = old_dir.join(rel_dir);
    if !new_d.is_dir() && !old_d.is_dir() {
        return None;
    }
    let keep = |p: &PathBuf| !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("pdb"));
    let mut new_files: Vec<PathBuf> = walk(&new_d).into_iter().filter(keep).collect();
    let mut old_files: Vec<PathBuf> = walk(&old_d).into_iter().filter(keep).collect();
    new_files.sort();
    old_files.sort();
    if new_files != old_files {
        return Some(missing_or_list_reason.into());
    }
    for rel in &new_files {
        match files_equal(&new_d.join(rel), &old_d.join(rel)) {
            Ok(true) => {}
            Ok(false) => return Some(format!("{}/{} changed", rel_dir, crate::dump::fwd(rel))),
            // Fail closed on read/lock errors (e.g. AV scanner holding a file) rather than treating as unchanged.
            Err(_) => return Some(format!("{}/{} could not be verified", rel_dir, crate::dump::fwd(rel))),
        }
    }
    None
}

/// Whether the updated package requires running Set up again. When the package's manifest check is available,
/// consults it directly so routine ABI-compatible `runtime/*.dll` updates already synced by `install()` do not
/// trigger an unnecessary Set up. Otherwise falls back to comparing `xc2/`, `runtime/`, and the shader toolchain
/// against `update/old/` saved aside during `install()`.
pub fn rebuild_needed(install: &Path, game_dir: &Path) -> (Option<bool>, String) {
    if !game_dir.join("xc2_boot.exe").is_file() {
        return (Some(true), "xc2_boot.exe is not built yet".into());
    }
    if let Some(gate) = crate::setup::check_package_manifest_readonly(install, game_dir) {
        return if gate.ok {
            (Some(false), String::new())
        } else {
            (Some(true), gate.reason)
        };
    }
    let old_dir = install.join(WORK).join("old");
    if !old_dir.is_dir() {
        return (None, String::new());
    }
    for (rel_dir, list_reason) in [
        ("xc2", "hook tables or stub sources changed"),
        ("runtime", "runtime libraries changed"),
        ("tools/spirv_harness", "shader translator changed"),
        ("tools/shader-cache", "shader cache scripts changed"),
        ("tools/shader", "shader patch scripts changed"),
    ] {
        if let Some(reason) = dir_changed(install, &old_dir, rel_dir, list_reason) {
            return (Some(true), reason);
        }
    }
    (Some(false), String::new())
}

/// At start: removes what an earlier update left behind (the previous version, downloads, the unpacked package).
pub fn cleanup(install: &Path) {
    let work = install.join(WORK);
    if work.is_dir() {
        let _ = fs::remove_dir_all(work.join("old"));
        let _ = fs::remove_dir_all(work.join("stage"));
        if let Ok(rd) = fs::read_dir(&work) {
            for e in rd.flatten() {
                if e.path().extension().is_some_and(|x| x == "part") {
                    let _ = fs::remove_file(e.path());
                }
            }
        }
        let _ = fs::remove_dir(&work); // only when empty: a finished download is kept for a retry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacking_tolerates_only_time_errors() {
        // #129: tar could not set the file times (some Windows file systems); the files themselves were written
        assert!(only_time_errors("fefw-recomp-0.1.7/fefw.exe: Can't restore time: Invalid argument\nfefw-recomp-0.1.7/LEGAL.txt: Can't restore time: Invalid argument\n"));
        assert!(only_time_errors("a: Can't restore time: Invalid argument\ntar.exe: Error exit delayed from previous errors.\n"));
        assert!(!only_time_errors("a: Can't restore time: Invalid argument\ntar: b: Cannot open: Permission denied\n"));
        assert!(!only_time_errors("tar.exe: Error exit delayed from previous errors.\n"));
        assert!(!only_time_errors("fefw-recomp-0.1.7/a: Cannot open: No space left on device"));
        assert!(!only_time_errors(""));
    }

    #[test]
    fn versions() {
        assert_eq!(parse_version("v0.1.2"), Some((0, 1, 2)));
        assert_eq!(parse_version("1.10"), Some((1, 10, 0)));
        assert_eq!(parse_version("0.2.0-rc1"), Some((0, 2, 0)));
        assert!(parse_version("0.1.10") > parse_version("0.1.9"));
        assert_eq!(parse_version("dev"), None);
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fefw-upd-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }
    fn put(p: PathBuf, s: &str) {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, s).unwrap();
    }
    fn get(p: PathBuf) -> String {
        fs::read_to_string(p).unwrap_or_else(|_| "<none>".into())
    }

    #[test]
    fn install_replaces_package_files_only() {
        let (inst, top, old) = (tmp("i1"), tmp("t1"), tmp("o1").join("old"));
        put(inst.join("runtime/xc2_rt.dll"), "old dll");
        put(inst.join("tools/a.py"), "old a");
        put(inst.join("tools/gone.py"), "stale");
        put(inst.join("nxrt.ini"), "player ini");
        put(inst.join("save/x"), "player save");
        put(top.join("runtime/xc2_rt.dll"), "new dll");
        put(top.join("tools/a.py"), "new a");
        put(top.join("tools/sub/b.py"), "new b");
        install(&inst, &top, &old).unwrap();
        assert_eq!(get(inst.join("runtime/xc2_rt.dll")), "new dll");
        assert_eq!(get(inst.join("tools/a.py")), "new a");
        assert_eq!(get(inst.join("tools/sub/b.py")), "new b");
        assert!(!inst.join("tools/gone.py").exists());
        assert_eq!(get(inst.join("nxrt.ini")), "player ini");
        assert_eq!(get(inst.join("save/x")), "player save");
        assert_eq!(get(old.join("runtime/xc2_rt.dll")), "old dll");
        assert_eq!(get(old.join("tools/gone.py")), "stale");
    }

    #[test]
    fn install_adds_language_tables_without_replacing_the_players_own() {
        let (inst, top, old) = (tmp("i3"), tmp("t3"), tmp("o3").join("old"));
        put(inst.join("locales/zh.ini"), "the player's own zh");
        put(inst.join("locales/mine.ini"), "a table the player wrote");
        put(top.join("locales/zh.ini"), "the shipped zh");
        put(top.join("runtime/xc2_rt.dll"), "new dll");
        install(&inst, &top, &old).unwrap();
        // the shipped table does not overwrite the copy already there
        assert_eq!(get(inst.join("locales/zh.ini")), "the player's own zh");
        // and a table only the player has is not treated as stale
        assert_eq!(get(inst.join("locales/mine.ini")), "a table the player wrote");
        assert_eq!(get(inst.join("runtime/xc2_rt.dll")), "new dll");

        // with no copy there yet, the shipped table is put in place
        let (inst2, top2, old2) = (tmp("i4"), tmp("t4"), tmp("o4").join("old"));
        put(top2.join("locales/zh.ini"), "the shipped zh");
        put(top2.join("runtime/xc2_rt.dll"), "new dll");
        install(&inst2, &top2, &old2).unwrap();
        assert_eq!(get(inst2.join("locales/zh.ini")), "the shipped zh");
    }

    #[test]
    fn install_rolls_back_on_failure() {
        let (inst, top, old) = (tmp("i2"), tmp("t2"), tmp("o2").join("old"));
        put(inst.join("runtime/xc2_rt.dll"), "old dll");
        put(inst.join("tools/a.py"), "old a");
        put(inst.join("tools/gone.py"), "stale");
        put(inst.join("zz"), "a file where the new package has a folder");
        put(top.join("runtime/xc2_rt.dll"), "new dll");
        put(top.join("tools/a.py"), "new a");
        put(top.join("zz/inner"), "cannot go under a file");
        // zz is a file in the install but a folder in the package: zz/inner cannot be created, after the other files moved
        assert!(install(&inst, &top, &old).is_err());
        assert_eq!(get(inst.join("runtime/xc2_rt.dll")), "old dll");
        assert_eq!(get(inst.join("tools/a.py")), "old a");
        assert_eq!(get(inst.join("tools/gone.py")), "stale");
        assert_eq!(get(inst.join("zz")), "a file where the new package has a folder");
    }

    #[test]
    fn install_syncs_runtime_dlls_beside_boot_exe_and_preserves_player_dlls() {
        // 1) Separate game_dir with existing xc2_boot.exe: runtime/*.dll sync beside it, stale runtime DLL removed,
        //    player-added DLL kept, and temp backup folder cleaned up.
        let (inst, game, top, old) = (tmp("sync-i1"), tmp("sync-g1"), tmp("sync-t1"), tmp("sync-o1").join("old"));
        put(inst.join("runtime/xc2_rt.dll"), "rt-v1");
        put(inst.join("runtime/avcodec-62.dll"), "ff-old");
        put(game.join("xc2_boot.exe"), "exe");
        put(game.join("xc2_rt.dll"), "rt-v1");
        put(game.join("avcodec-62.dll"), "ff-old");
        put(game.join("amd_fidelityfx_upscaler_vk.dll"), "player-fsr");
        put(top.join("runtime/xc2_rt.dll"), "rt-v2");
        put(top.join("runtime/avcodec-63.dll"), "ff-new");

        install_with_game_dir(&inst, &game, &top, &old).unwrap();
        assert_eq!(get(inst.join("runtime/xc2_rt.dll")), "rt-v2");
        assert_eq!(get(game.join("xc2_rt.dll")), "rt-v2");
        assert_eq!(get(game.join("avcodec-63.dll")), "ff-new");
        assert!(!game.join("avcodec-62.dll").exists(), "obsolete shipped runtime DLL must be removed beside exe");
        assert_eq!(get(game.join("amd_fidelityfx_upscaler_vk.dll")), "player-fsr", "player DLL must stay untouched");
        assert!(!game.join(".update-old-dlls").exists());

        // 2) Fresh install without xc2_boot.exe: runtime/*.dll must NOT be copied to root yet.
        let (inst2, top2, old2) = (tmp("sync-i2"), tmp("sync-t2"), tmp("sync-o2").join("old"));
        put(inst2.join("runtime/xc2_rt.dll"), "rt-v1");
        put(top2.join("runtime/xc2_rt.dll"), "rt-v2");
        install(&inst2, &top2, &old2).unwrap();
        assert_eq!(get(inst2.join("runtime/xc2_rt.dll")), "rt-v2");
        assert!(!inst2.join("xc2_rt.dll").exists());
    }

    #[test]
    fn install_rolls_back_runtime_dlls_and_package_on_dll_sync_failure() {
        let (inst, game, top, old) = (tmp("rb-dll-i"), tmp("rb-dll-g"), tmp("rb-dll-t"), tmp("rb-dll-o").join("old"));
        put(inst.join("runtime/a_ok.dll"), "a-v1");
        put(inst.join("runtime/z_fail.dll"), "z-v1");
        put(inst.join("tools/a.py"), "tool-v1");
        put(game.join("xc2_boot.exe"), "exe");
        put(game.join("a_ok.dll"), "a-v1");
        put(game.join("z_fail.dll"), "z-v1");
        // Block z_fail.dll staging AFTER a_ok.dll has already been replaced, so we prove mid-sync rollback works.
        put(game.join(".z_fail.dll.upd-new/blocker"), "dir-blocker");
        put(top.join("runtime/a_ok.dll"), "a-v2");
        put(top.join("runtime/z_fail.dll"), "z-v2");
        put(top.join("tools/a.py"), "tool-v2");

        assert!(install_with_game_dir(&inst, &game, &top, &old).is_err());
        assert_eq!(get(inst.join("runtime/a_ok.dll")), "a-v1", "package runtime DLL must roll back");
        assert_eq!(get(inst.join("runtime/z_fail.dll")), "z-v1", "package runtime DLL must roll back");
        assert_eq!(get(inst.join("tools/a.py")), "tool-v1", "package tool must roll back");
        assert_eq!(get(game.join("a_ok.dll")), "a-v1", "already-replaced game_dir DLL must roll back to v1");
        assert_eq!(get(game.join("z_fail.dll")), "z-v1");
        assert!(!game.join(".update-old-dlls").exists());
    }

    #[test]
    fn release_selection_includes_prereleases_and_picks_dist_zip() {
        let list = serde_json::json!([
            {
                "tag_name": "v0.2.3-draft",
                "draft": true,
                "prerelease": true,
                "assets": [{"name": "XC2-package-dist-0.2.3.zip", "browser_download_url": "https://x/0.2.3.zip", "size": 10, "digest": "sha256:aa"}]
            },
            {
                "tag_name": "v0.2.1",
                "draft": false,
                "prerelease": true,
                "assets": [{"name": "XC2-package-dist-0.2.1.zip", "browser_download_url": "https://x/0.2.1.zip", "size": 20, "digest": "sha256:bb"}]
            },
            {
                "tag_name": "v0.2.2",
                "draft": false,
                "prerelease": true,
                "assets": [
                    {"name": "xc2-launcher-0.2.2-src.zip", "browser_download_url": "https://x/src.zip", "size": 1, "digest": "sha256:cc"},
                    {"name": "XC2-package-dist-0.2.2.zip", "browser_download_url": "https://x/0.2.2.zip", "size": 30, "digest": "sha256:dd"}
                ]
            }
        ]);
        let rel = select_latest_release(&list).unwrap();
        assert_eq!(rel.version, "0.2.2");
        assert_eq!(rel.asset_name, "XC2-package-dist-0.2.2.zip");
        assert_eq!(rel.sha256, "dd");
    }

    #[test]
    fn rebuild_needed_tracks_boot_exe_xc2_runtime_and_shader_harness() {
        let inst = tmp("rb1");
        let old = inst.join(WORK).join("old");
        put(inst.join("xc2/sites/main.txt"), "same");
        put(old.join("xc2/sites/main.txt"), "same");
        put(inst.join("runtime/xc2_rt.dll"), "dll-v1");
        put(old.join("runtime/xc2_rt.dll"), "dll-v1");
        put(inst.join("tools/spirv_harness/Ryujinx.Graphics.Shader.dll"), "tr-v1");
        put(old.join("tools/spirv_harness/Ryujinx.Graphics.Shader.dll"), "tr-v1");
        assert_eq!(rebuild_needed(&inst, &inst).0, Some(true));
        put(inst.join("xc2_boot.exe"), "exe");
        assert_eq!(rebuild_needed(&inst, &inst), (Some(false), String::new()));

        // Runtime DLL change must trigger Set up so the new DLL is synced / manifest-checked.
        put(inst.join("runtime/xc2_rt.dll"), "dll-v2");
        let (need, why) = rebuild_needed(&inst, &inst);
        assert_eq!(need, Some(true));
        assert!(why.contains("runtime/xc2_rt.dll"), "unexpected reason: {why}");
        put(inst.join("runtime/xc2_rt.dll"), "dll-v1");

        // Shader translator change (including Spv.Generator.dll / AmpereDecoder.dll) must trigger Set up;
        // debug-symbol .pdb changes must not.
        put(inst.join("tools/spirv_harness/spirv_harness.pdb"), "pdb-v2");
        put(old.join("tools/spirv_harness/spirv_harness.pdb"), "pdb-v1");
        assert_eq!(rebuild_needed(&inst, &inst), (Some(false), String::new()));

        put(inst.join("tools/spirv_harness/Spv.Generator.dll"), "spv-v2");
        put(old.join("tools/spirv_harness/Spv.Generator.dll"), "spv-v1");
        let (need, why) = rebuild_needed(&inst, &inst);
        assert_eq!(need, Some(true));
        assert!(why.contains("tools/spirv_harness/Spv.Generator.dll"), "unexpected reason: {why}");
        put(inst.join("tools/spirv_harness/Spv.Generator.dll"), "spv-v1");

        // Same-length byte change must be caught by streaming comparison too.
        put(inst.join("tools/spirv_harness/Ryujinx.Graphics.Shader.dll"), "tr-v2");
        let (need, why) = rebuild_needed(&inst, &inst);
        assert_eq!(need, Some(true));
        assert!(why.contains("tools/spirv_harness/"), "unexpected reason: {why}");
        put(inst.join("tools/spirv_harness/Ryujinx.Graphics.Shader.dll"), "tr-v1");

        // Hook table change still triggers Set up.
        put(inst.join("xc2/sites/main.txt"), "changed");
        let (need, why) = rebuild_needed(&inst, &inst);
        assert_eq!(need, Some(true));
        assert!(why.contains("xc2/sites/main.txt"), "unexpected reason: {why}");
    }

    /// End to end against GitHub: FEFW_UPDATE_TEST_DIR = an unpacked older package.
    /// cargo test --release -- --ignored update_end_to_end --nocapture
    #[test]
    #[ignore]
    fn update_end_to_end() {
        let install = PathBuf::from(std::env::var("FEFW_UPDATE_TEST_DIR").expect("FEFW_UPDATE_TEST_DIR"));
        let rel = latest().expect("latest");
        println!("latest {} {} {} sha256 {}", rel.version, rel.asset_name, rel.size, rel.sha256);
        let shared = Shared::default();
        let emit = |s: Status| {
            if s.phase != "download" || s.done % (16 << 20) < 200_000 {
                println!("{} {} {}/{} {}", s.state, s.phase, s.done, s.total, s.message)
            }
        };
        let run = Run { shared: &shared, install: install.clone(), game_dir: install.clone(), emit: &emit };
        let st = run.run(&rel).expect("update");
        println!("done: {} rebuild {:?} ({})", st.message, st.rebuild, st.rebuild_reason);
        assert!(install.join("runtime/xc2_rt.dll").is_file() && install.join("engine/xc2_setup.py").is_file());
        assert!(install.join("update/old/runtime/xc2_rt.dll").is_file());
        cleanup(&install);
        assert!(!install.join("update/old").exists());
    }
}
