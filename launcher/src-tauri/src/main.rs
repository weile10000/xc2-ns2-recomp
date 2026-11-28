// SPDX-License-Identifier: GPL-3.0-or-later
//! Launcher for the fefw recomp (fefw.exe). It works on a checkout of this repository: it reads the player's own dump, runs the
//! repository's own lift and build scripts on this PC, writes fefw.ini in a run folder and starts the game. It ships no game files.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod arh;
mod cpu;
mod dump;
mod fsr;
mod ini;
mod locale;
mod mod_import;
mod mods;
mod nup;
mod play;
mod settings;
mod setup;
mod steam;
mod streamline;
mod update;

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::Emitter;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

/// The launcher's own options, in the user's app config folder.
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(default)]
struct Config {
    /// The repository checkout. Empty: found from the launcher's own location.
    repo_dir: String,
    /// The dump folder the player picked (exefs/ and romfs/, or base/ and updates/).
    dump_dir: String,
    /// Where fefw.ini, controls.ini, the log and the saves live. Empty: <repo>/work/play.
    game_dir: String,
    /// Build processes. 0: one fewer than the CPU threads.
    jobs: u32,
    minimize_on_play: bool,
    /// do not ask GitHub for a newer release when the launcher starts
    skip_update_check: bool,
    /// Linux: the command that starts fefw.exe (Wine or Proton, e.g. `umu-run` or `wine`). Empty: Steam's own Proton if it is
    /// installed, else umu-run, else wine. Unused on Windows.
    runner: String,
    /// The launcher's own interface language ("en", "zh", or the code of a language pack in locales/). Empty: English.
    lang: String,
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {}", dir.display(), e))?;
    Ok(dir.join("launcher.json"))
}

fn load_config(app: &AppHandle) -> Config {
    config_path(app).ok().and_then(|p| fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn exe_dir() -> PathBuf {
    // an AppImage runs from a temporary mount: the package is the folder the .AppImage file is in
    #[cfg(target_os = "linux")]
    if let Some(d) = std::env::var_os("APPIMAGE").map(PathBuf::from).and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        return d;
    }
    std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| PathBuf::from("."))
}

/// The release package's folder, when the launcher runs in one: the tools it carries are what makes the package
/// portable. A checkout has no tools/python next to the exe, so this stays None there and everything below takes the
/// build-it-here path instead.
fn package_tools() -> Option<PathBuf> {
    let tools = exe_dir().join("tools");
    tools.join("python").join("python.exe").is_file().then_some(tools)
}

fn is_repo(dir: &Path) -> bool {
    dir.join("xc2").join("recomp").join("xc2ns2").join("config.py").is_file()
        && dir.join("third_party").join("fefw-recomp").join("runtime").join("CMakeLists.txt").is_file()
}

/// The checkout the launcher runs in: the configured one, else the first parent of the launcher's exe that is one.
fn repo_of(cfg: &Config) -> Option<PathBuf> {
    if package_tools().is_some() {
        // the release package: the launcher's folder holds tools/, runtime/xc2_rt.lib and xc2/sites even when the
        // player installs game data into a separate folder (cfg.game_dir); passing cfg.game_dir as --repo would
        // send xc2_setup.py into checkout mode instead of package mode (#58).
        return Some(exe_dir());
    }
    if !cfg.repo_dir.trim().is_empty() {
        let p = PathBuf::from(cfg.repo_dir.trim());
        return is_repo(&p).then_some(p);
    }
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().skip(1).take(8).find(|d| is_repo(d)).map(|d| d.to_path_buf())
}

fn game_dir_of(cfg: &Config, repo: &Path) -> PathBuf {
    if cfg.game_dir.trim().is_empty() {
        if package_tools().is_some() {
            return exe_dir();
        }
        repo.join("work").join("play")
    } else {
        PathBuf::from(cfg.game_dir.trim())
    }
}

fn need_repo(cfg: &Config) -> Result<PathBuf, String> {
    repo_of(cfg).ok_or_else(|| "The repository was not found. Build the launcher inside a checkout of this project, or set the repository folder in Settings.".to_string())
}

/// Setup's paths from the configuration, when the dump is usable.
fn paths(cfg: &Config) -> Result<setup::Paths, String> {
    let repo = need_repo(cfg)?;
    let report = dump::scan(&cfg.dump_dir, &builds_of(&repo));
    let chosen = report.chosen.ok_or_else(|| {
        if cfg.dump_dir.trim().is_empty() { "Choose your game files first.".to_string() } else { format!("The game files at {} are not usable: {}", cfg.dump_dir, report.errors.join(" ")) }
    })?;
    let game_dir = game_dir_of(cfg, &repo);
    Ok(setup::Paths {
        repo,
        exefs: PathBuf::from(chosen.exefs),
        romfs: PathBuf::from(chosen.romfs),
        dlc: report.dlc_dir.map(PathBuf::from),
        game_dir,
        package: package_tools(),
    })
}

/// The supported build of main: the checkout's xc2/recomp/xc2ns2/config.py.
fn builds_of(repo: &Path) -> Vec<(String, String)> {
    dump::supported_builds(repo)
}

fn jobs_of(cfg: &Config) -> usize {
    if cfg.jobs > 0 {
        return cfg.jobs as usize;
    }
    std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(2).max(1)
}

/// The account folder the runtime serves `save:` from: `<game_dir>\save\account\<uid>\`. The uid is the one the runtime's
/// title profile uses, so a save copied here is the save the game opens.
const SAVE_ACCOUNT_UID: &str = "0100000000000000635232534e324358";

/// The save-file copy behind `import_saves`: every .sav under `src` (four folders deep, which covers a port's save tree)
/// into the account folder the game opens. Returns (copied, seen).
fn copy_saves(account: &std::path::Path, src: &std::path::Path) -> Result<(usize, usize), String> {
    let dest = fs::read_dir(account)
        .ok()
        .and_then(|rd| rd.flatten().map(|e| e.path()).find(|q| q.is_dir()))
        .unwrap_or_else(|| account.join(SAVE_ACCOUNT_UID));
    fs::create_dir_all(&dest).map_err(|e| format!("cannot create {}: {}", dest.display(), e))?;
    let mut copied = 0usize;
    let mut seen = 0usize;
    let mut stack = vec![(src.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                if depth < 4 {
                    stack.push((path, depth + 1));
                }
                continue;
            }
            let is_save = path.extension().and_then(|x| x.to_str()).is_some_and(|x| x.eq_ignore_ascii_case("sav"));
            if !is_save {
                continue;
            }
            seen += 1;
            let Some(name) = path.file_name() else { continue };
            if fs::copy(&path, dest.join(name)).is_ok() {
                copied += 1;
            }
        }
    }
    Ok((copied, seen))
}

/// Copy the player's own save files from another install (or from the original XC2 Recomp) into this one. The files are
/// the game's own (bf2savefile.sav, bf2option.sav, irasavefile.sav): nothing is converted and nothing else is read.
#[tauri::command]
fn import_saves(app: AppHandle, path: String) -> Result<String, String> {
    let cfg = load_config(&app);
    let p = paths(&cfg)?;
    let src = PathBuf::from(path.trim());
    if !src.is_dir() {
        return Err(format!("{} is not a folder", src.display()));
    }
    let account = p.game_dir.join("save").join("account");
    let (copied, seen) = copy_saves(&account, &src)?;
    if seen == 0 {
        return Err(format!("no .sav files under {}", src.display()));
    }
    Ok(format!(
        "{} of {} save file(s) copied into {} (restart the game to see them)",
        copied,
        seen,
        crate::dump::fwd(&account)
    ))
}

#[derive(Serialize)]
struct Overview {
    repo_dir: String,
    repo_ok: bool,
    game_dir: String,
    exe: String,
    exe_exists: bool,
}

#[tauri::command]
fn get_config(app: AppHandle) -> Config {
    load_config(&app)
}

#[tauri::command]
fn save_config(app: AppHandle, config: Config) -> Result<(), String> {
    let text = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    settings::write_atomic(&config_path(&app)?, &text).map_err(|e| format!("cannot save launcher settings: {e}"))
}

#[tauri::command]
fn get_overview(app: AppHandle) -> Overview {
    let cfg = load_config(&app);
    match repo_of(&cfg) {
        Some(repo) => {
            let exe = if package_tools().is_some() { game_dir_of(&cfg, &repo).join("xc2_boot.exe") } else { setup::exe_path(&repo) };
            Overview {
                repo_dir: dump::fwd(&repo),
                repo_ok: true,
                game_dir: dump::fwd(&game_dir_of(&cfg, &repo)),
                exe_exists: exe.is_file(),
                exe: dump::fwd(&exe),
            }
        }
        None => Overview { repo_dir: cfg.repo_dir.clone(), repo_ok: false, game_dir: String::new(), exe: String::new(), exe_exists: false },
    }
}

#[tauri::command]
async fn pick_folder(app: AppHandle, title: String, start: Option<String>) -> Option<String> {
    let mut dlg = app.dialog().file().set_title(title);
    if let Some(s) = start.filter(|s| PathBuf::from(s).is_dir()) {
        dlg = dlg.set_directory(s);
    }
    dlg.blocking_pick_folder().and_then(|p| p.into_path().ok()).map(|p| dump::fwd(&p))
}

#[tauri::command]
async fn pick_nup(app: AppHandle, start: Option<String>) -> Option<String> {
    let mut dlg = app.dialog().file().set_title("Choose your .nup package").add_filter("NUP package", &["nup"]);
    if let Some(s) = start.filter(|s| PathBuf::from(s).is_dir()) {
        dlg = dlg.set_directory(s);
    }
    dlg.blocking_pick_file().and_then(|p| p.into_path().ok()).map(|p| dump::fwd(&p))
}

#[derive(Serialize, Clone)]
struct NupProgress {
    done: u64,
    total: u64,
    file: String,
}

/// Unpacks a .nup (https://github.com/a55uka/nup) into `<repo>/game/nup_dump` (base/ and updates/<version>/, the layout of a dump
/// folder) and returns that folder. Emits "nup-progress". A finished unpack of the same package is reused.
#[tauri::command]
async fn import_nup(app: AppHandle, state: State<'_, Arc<setup::Shared>>, path: String) -> Result<String, String> {
    if state.progress.lock().unwrap().state == "running" {
        return Err("Setup is running. Pause it first.".into());
    }
    let cfg = load_config(&app);
    let repo = need_repo(&cfg)?;
    let out = if package_tools().is_some() { game_dir_of(&cfg, &repo).join("nup_dump") } else { repo.join("game").join("nup_dump") };
    let src = PathBuf::from(path.trim());
    let emitter = app.clone();
    let res = tauri::async_runtime::spawn_blocking(move || {
        let mut last = 0u64;
        nup::extract(
            &src,
            &out,
            &mut |done, total, file| {
                if total == 0 || done == total || done - last >= total / 100 {
                    last = done;
                    let _ = emitter.emit("nup-progress", NupProgress { done, total, file: file.to_string() });
                }
            },
            &|| false,
        )
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(dump::fwd(&res))
}

#[tauri::command]
fn validate_dump(app: AppHandle, path: String) -> Result<dump::DumpReport, String> {
    Ok(dump::scan(&path, &builds_of(&need_repo(&load_config(&app))?)))
}

#[tauri::command]
fn save_dump(app: AppHandle, path: String) -> Result<dump::DumpReport, String> {
    let mut cfg = load_config(&app);
    let report = dump::scan(&path, &builds_of(&need_repo(&cfg)?));
    if report.chosen.is_none() {
        return Err(report.errors.join(" "));
    }
    cfg.dump_dir = report.path.clone();
    save_config(app, cfg)?;
    Ok(report)
}

#[tauri::command]
fn read_settings(app: AppHandle) -> Result<settings::SettingsDump, String> {
    let cfg = load_config(&app);
    Ok(settings::read(&game_dir_of(&cfg, &need_repo(&cfg)?)))
}

#[tauri::command]
fn write_settings(app: AppHandle, edits: Vec<settings::Edit>) -> Result<(), String> {
    let cfg = load_config(&app);
    settings::apply(&game_dir_of(&cfg, &need_repo(&cfg)?), &edits)
}

/// What one upscaler / frame generation feature finds (Settings): `found` only when every file it needs is in one folder.
#[derive(Serialize)]
struct FeatureFiles {
    found: bool,
    /// the folder the game will use (found), or the files missing and where they were looked for
    detail: String,
}

#[derive(Serialize)]
struct UpscalerFiles {
    fsr_upscaling: FeatureFiles,
    fsr_framegen: FeatureFiles,
    dlss: FeatureFiles,
    dlss_framegen: FeatureFiles,
}

/// The folders the game searches (temporal_ffx.cpp, temporal_dlss.cpp, framegen_ffx.cpp): the setting's folder (a file in it counts), then
/// `upscalers/<sub>` next to the game, next to the launcher and in the run folder.
fn upscaler_dirs(setting: &str, sub: &str, game: &Path, exe: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    let s = setting.trim();
    if !s.is_empty() {
        let p = PathBuf::from(s);
        v.push(if p.is_file() { p.parent().map(|d| d.to_path_buf()).unwrap_or(p) } else { p });
    }
    for base in [exe.parent().map(|d| d.to_path_buf()).unwrap_or_default(), exe_dir(), game.to_path_buf()] {
        let d = base.join("upscalers").join(sub);
        if !v.contains(&d) {
            v.push(d);
        }
    }
    v
}

fn files_in(dirs: &[PathBuf], sets: &[&[&str]], what: &str) -> FeatureFiles {
    for d in dirs {
        for set in sets {
            if set.iter().all(|f| d.join(f).is_file()) {
                return FeatureFiles { found: true, detail: dump::fwd(d) };
            }
        }
    }
    // the first folder holding any of the files says which ones it lacks
    for d in dirs {
        let set = sets[0];
        let have: Vec<&str> = set.iter().copied().filter(|f| d.join(f).is_file()).collect();
        if !have.is_empty() {
            let lack: Vec<&str> = set.iter().copied().filter(|f| !d.join(f).is_file()).collect();
            return FeatureFiles { found: false, detail: format!("{} lacks {}", dump::fwd(d), lack.join(", ")) };
        }
    }
    FeatureFiles { found: false, detail: format!("{} not found in {}", what, dirs.iter().map(|d| dump::fwd(d)).collect::<Vec<_>>().join(", ")) }
}

#[tauri::command]
fn upscaler_files(app: AppHandle, fsr_dir: String, dlss_dir: String) -> Result<UpscalerFiles, String> {
    let cfg = load_config(&app);
    let repo = need_repo(&cfg)?;
    let game = game_dir_of(&cfg, &repo);
    let exe = if package_tools().is_some() { game.join("xc2_boot.exe") } else { setup::exe_path(&repo) };
    let mut fsr = upscaler_dirs(&fsr_dir, "fsr3", &game, &exe);
    for d in upscaler_dirs(&fsr_dir, "fsr", &game, &exe) {
        if !fsr.contains(&d) {
            fsr.push(d);
        }
    }
    // Streamline: the game takes the setting's folder, else the first default folder holding sl.interposer.dll, and needs every file there
    let sl = upscaler_dirs(&dlss_dir, "streamline", &game, &exe);
    let sl_pick: Vec<PathBuf> = if !dlss_dir.trim().is_empty() {
        sl[..1].to_vec()
    } else {
        sl.iter().find(|d| d.join("sl.interposer.dll").is_file()).cloned().map(|d| vec![d]).unwrap_or(sl.clone())
    };
    const DLSS: [&str; 4] = ["sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "nvngx_dlss.dll"];
    const DLSS_G: [&str; 9] =
        ["sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "nvngx_dlss.dll", "sl.dlss_g.dll", "sl.reflex.dll", "sl.pcl.dll", "nvngx_dlssg.dll", "NvLowLatencyVk.dll"];
    Ok(UpscalerFiles {
        fsr_upscaling: files_in(&fsr, &[&["amd_fidelityfx_upscaler_vk.dll"], &["amd_fidelityfx_vk.dll"]], "amd_fidelityfx_upscaler_vk.dll"),
        fsr_framegen: files_in(&fsr, &[&["amd_fidelityfx_vk.dll"]], "amd_fidelityfx_vk.dll (FidelityFX SDK v1.1.4)"),
        dlss: files_in(&sl_pick, &[&DLSS], "the Streamline files"),
        dlss_framegen: files_in(&sl_pick, &[&DLSS_G], "the Streamline frame generation files"),
    })
}

#[derive(Serialize)]
struct SetupStatus {
    progress: setup::Progress,
    steps: Vec<setup::StepState>,
    complete: bool,
    /// true when an existing build is out of sync with the updated package's manifest inputs or runtime DLLs
    upgrade_needed: bool,
    upgrade_reason: String,
    tools: Vec<setup::Tool>,
    log: Vec<String>,
    jobs: usize,
    /// running from the release package (no build tools needed)
    package: bool,
    /// why this PC's processor cannot run the compiled game (missing CMPXCHG16B / SSE4.2 / POPCNT), if it cannot
    cpu_problem: Option<String>,
    /// set when this processor gets the compatible build of the game code (no FMA3: cpu.rs)
    cpu_note: Option<String>,
}

#[tauri::command]
fn setup_status(app: AppHandle, state: State<Arc<setup::Shared>>) -> SetupStatus {
    let cfg = load_config(&app);
    let mut progress = state.progress.lock().unwrap().clone();
    if progress.state.is_empty() {
        progress.state = "idle".into();
    }
    let running = (progress.state == "running").then(|| progress.step.clone());
    let resolved = paths(&cfg).ok();
    let gate = match &resolved {
        Some(p) if running.is_none() => setup::manifest_gate(p),
        _ => setup::ManifestGate::default(),
    };
    let steps = resolved.as_ref().map(|p| setup::steps(p, running.as_deref())).unwrap_or_default();
    SetupStatus {
        complete: !steps.is_empty() && steps.iter().all(|s| s.state == "done") && gate.ok,
        upgrade_needed: gate.upgrade_needed,
        upgrade_reason: gate.reason,
        steps,
        progress,
        tools: setup::check_tools(package_tools().is_some()),
        package: package_tools().is_some(),
        cpu_problem: cpu::check(),
        cpu_note: cpu::note(),
        log: state.log.lock().unwrap().iter().cloned().collect(),
        jobs: jobs_of(&cfg),
    }
}

#[tauri::command]
fn start_setup(app: AppHandle, state: State<Arc<setup::Shared>>) -> Result<(), String> {
    if let Some(why) = cpu::check() {
        return Err(why);
    }
    let cfg = load_config(&app);
    let p = paths(&cfg)?;
    {
        let mut pr = state.progress.lock().unwrap();
        if pr.state == "running" {
            return Err("Setup is already running.".into());
        }
        *pr = setup::Progress { state: "running".into(), step: "python".into(), message: "Starting...".into(), ..Default::default() };
    }
    let shared = state.inner().clone();
    shared.cancel.store(false, Ordering::SeqCst);
    let jobs = jobs_of(&cfg);
    std::thread::spawn(move || {
        let a1 = app.clone();
        let a2 = app.clone();
        let emit = move |p: setup::Progress| {
            let _ = a1.emit("setup-status", p);
        };
        let emit_log = move |lines: Vec<String>| {
            let _ = a2.emit("setup-log", lines);
        };
        let end = setup::run(&shared, &p, jobs, &emit, &emit_log);
        let _ = app.emit("setup-status", end);
    });
    Ok(())
}

#[tauri::command]
fn cancel_setup(state: State<Arc<setup::Shared>>) {
    state.cancel.store(true, Ordering::SeqCst);
    setup::kill_tree(state.pid.load(Ordering::SeqCst));
}

#[tauri::command]
fn play(app: AppHandle, state: State<Arc<setup::Shared>>, safe_mode: Option<bool>) -> Result<(), String> {
    if state.progress.lock().unwrap().state == "running" {
        return Err("Setup is running. Wait for it or pause it first.".into());
    }
    let cfg = load_config(&app);
    let p = paths(&cfg)?;
    if !setup::complete(&p) {
        let gate = setup::manifest_gate(&p);
        if gate.upgrade_needed {
            return Err(if gate.reason.is_empty() {
                "Upgrade is not finished: open Setup and press Finish upgrade.".into()
            } else {
                format!("Upgrade is not finished ({}): open Setup and press Finish upgrade.", gate.reason)
            });
        }
        return Err("Setup is not finished: open Setup and press Set up.".into());
    }
    settings::ensure_play_paths(&p.game_dir, &p)?;
    play::start(&app, &p, &exe_dir(), cfg.minimize_on_play, &cfg.runner, safe_mode.unwrap_or(false))
}

/// Linux: whether Steam is installed and the shortcut to fefw.exe exists.
#[tauri::command]
async fn steam_status(app: AppHandle) -> steam::State {
    // off the main thread: it reads the dump folder and walks /proc
    let none = steam::State { available: false, running: false, added: false };
    if !cfg!(target_os = "linux") {
        return none;
    }
    tauri::async_runtime::spawn_blocking(move || match paths(&load_config(&app)) {
        Ok(p) => steam::state(&p.exe()),
        Err(_) => steam::State { available: false, running: false, added: false },
    })
    .await
    .unwrap_or(none)
}

/// Linux: adds fefw.exe to Steam as a non-Steam game. Steam has to be closed for that: when it runs, the player is asked first.
#[tauri::command]
async fn add_to_steam(app: AppHandle) -> Result<String, String> {
    use tauri_plugin_dialog::{MessageDialogButtons, MessageDialogKind};
    let cfg = load_config(&app);
    let p = paths(&cfg)?;
    let exe = p.exe();
    if !exe.is_file() {
        return Err("fefw.exe is not built yet. Run Set up first.".into());
    }
    let close = if steam::state(&exe).running {
        app.dialog()
            .message("Steam saves its game list when it closes, so it has to close while the launcher adds the game. It is started again right after.")
            .title("Add to Steam")
            .kind(MessageDialogKind::Info)
            .buttons(MessageDialogButtons::OkCancelCustom("Close Steam and add".into(), "Cancel".into()))
            .blocking_show()
    } else {
        false
    };
    if steam::state(&exe).running && !close {
        return Err("Not added: Steam is still running.".into());
    }
    // the upscaler DLLs next to the launcher: fefw.exe finds them by itself when the run folder is the install folder
    let launcher_dir = (p.game_dir != exe_dir()).then(|| dump::game_path(&exe_dir()));
    let game_dir = p.game_dir.clone();
    // what fefw.ini points the game at, for Flatpak Steam's sandbox
    let mut reads = vec![p.game_dir.clone(), p.romfs.clone(), p.plain_exefs(), exe_dir()];
    reads.extend(p.dlc.clone());
    tauri::async_runtime::spawn_blocking(move || steam::add(&exe, &game_dir, launcher_dir.as_deref(), &reads, close)).await.map_err(|e| e.to_string())?
}

#[tauri::command]
fn game_status(state: State<play::PlayState>) -> play::Status {
    state.0.lock().unwrap().clone()
}

#[tauri::command]
fn open_game_file(app: AppHandle, name: String) -> Result<(), String> {
    if !play::REPORT_FILES.contains(&name.as_str()) {
        return Err("That file cannot be opened from here.".into());
    }
    let cfg = load_config(&app);
    let path = game_dir_of(&cfg, &need_repo(&cfg)?).join(&name);
    if !path.is_file() {
        return Err(format!("{name} does not exist."));
    }
    play::open_with_shell(&path)
}

#[tauri::command]
fn open_game_dir(app: AppHandle) -> Result<(), String> {
    let cfg = load_config(&app);
    let dir = game_dir_of(&cfg, &need_repo(&cfg)?);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    play::open_with_shell(&dir)
}

/// Opens <run folder>/mods ([fs] mods), creating it: RomFS mods go in mods/<name>/romfs.
#[tauri::command]
fn open_mods_dir(app: AppHandle) -> Result<(), String> {
    let cfg = load_config(&app);
    let dir = game_dir_of(&cfg, &need_repo(&cfg)?).join("mods");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    play::open_with_shell(&dir)
}

fn mods_context(app: &AppHandle, state: &State<play::PlayState>) -> Result<(PathBuf, mods::ExefsTargetInfo, Option<PathBuf>, bool), String> {
    let cfg = load_config(app);
    let repo = need_repo(&cfg)?;
    let game_dir = game_dir_of(&cfg, &repo);
    let plain = if package_tools().is_some() { game_dir.join("exefs") } else { repo.join("game").join("dump").join("exefs") };
    let exefs_dir = if plain.join("main").is_file() { Some(plain) } else { paths(&cfg).ok().map(|p| p.exefs) };
    let target = mods::exefs_target_info(exefs_dir.as_deref(), Some(&repo));
    let romfs_cand = game_dir.join("game").join("romfs-301");
    let vanilla_romfs = if romfs_cand.is_dir() {
        Some(romfs_cand)
    } else {
        paths(&cfg).ok().map(|p| p.romfs).filter(|p| p.is_dir())
    };
    let running = state.0.lock().map(|s| s.state == "running").unwrap_or(false);
    Ok((game_dir, target, vanilla_romfs, running))
}

#[tauri::command]
fn list_mods(app: AppHandle, state: State<play::PlayState>) -> Result<mods::ModsSnapshot, String> {
    let (game_dir, target, vanilla_romfs, running) = mods_context(&app, &state)?;
    Ok(mods::compute_snapshot(&game_dir, &target, vanilla_romfs.as_deref(), running))
}

#[tauri::command]
fn save_mods_state(app: AppHandle, state: State<play::PlayState>, input: mods::SaveModsStateInput) -> Result<mods::ModsSnapshot, String> {
    let (game_dir, target, vanilla_romfs, running) = mods_context(&app, &state)?;
    mods::save_state(&game_dir, &target, vanilla_romfs.as_deref(), input, running)
}

#[tauri::command]
fn open_mod_folder(app: AppHandle, name: String) -> Result<(), String> {
    if name.trim().is_empty() || name.starts_with('.') || name.contains(['/', '\\', ':']) || name == ".." {
        return Err("Invalid mod folder name.".into());
    }
    let cfg = load_config(&app);
    let dir = game_dir_of(&cfg, &need_repo(&cfg)?).join("mods").join(&name);
    if !dir.is_dir() {
        return Err(format!("Mod folder '{name}' does not exist."));
    }
    play::open_with_shell(&dir)
}

#[tauri::command]
async fn pick_mod_file(app: AppHandle, title: String) -> Option<String> {
    app.dialog()
        .file()
        .set_title(title)
        .add_filter("XC2 Mods & Cheats", &["zip", "7z", "txt", "json", "ips", "pchtxt", "bdat", "wismt"])
        .blocking_pick_file()
        .and_then(|p| p.into_path().ok())
        .map(|p| dump::fwd(&p))
}

#[tauri::command]
fn import_mod(app: AppHandle, state: State<play::PlayState>, source_path: String) -> Result<mods::ImportModResult, String> {
    let (game_dir, target, vanilla_romfs, running) = mods_context(&app, &state)?;
    let arh = arh::cached_arh_index(vanilla_romfs.as_deref());
    let summary = mod_import::import_mod_source(
        &game_dir.join("mods"),
        Path::new(source_path.trim()),
        &target,
        vanilla_romfs.as_deref(),
        arh.as_deref(),
    )?;
    let snapshot = mods::compute_snapshot(&game_dir, &target, vanilla_romfs.as_deref(), running);
    Ok(mods::ImportModResult {
        folder: summary.folder,
        warnings: summary.warnings,
        has_exefs: summary.has_exefs,
        snapshot,
    })
}

#[tauri::command]
fn install_mod_folder(app: AppHandle, state: State<play::PlayState>, source_path: String) -> Result<mods::ModsSnapshot, String> {
    let res = import_mod(app, state, source_path)?;
    Ok(res.snapshot)
}

#[tauri::command]
fn delete_mod(app: AppHandle, state: State<play::PlayState>, name: String) -> Result<mods::DeleteModResult, String> {
    let (game_dir, target, vanilla_romfs, running) = mods_context(&app, &state)?;
    mods::delete_mod(&game_dir, &target, vanilla_romfs.as_deref(), &name, running)
}

#[derive(Serialize, Clone)]
struct VoiceSubtitlesStatus {
    dir: String,
    packs: Vec<String>,
}

#[derive(Serialize, Clone)]
struct VoiceSubtitlesImportResult {
    installed: Vec<String>,
    status: VoiceSubtitlesStatus,
}

fn list_voice_subtitles(dir: &Path) -> VoiceSubtitlesStatus {
    let mut packs: Vec<String> = fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if !p.is_file() {
                return None;
            }
            let ext = p.extension()?.to_str()?;
            if !ext.eq_ignore_ascii_case("json") {
                return None;
            }
            let stem = p.file_stem()?.to_str()?.trim();
            if stem.is_empty() {
                return None;
            }
            Some(stem.to_string())
        })
        .collect();
    packs.sort();
    packs.dedup();
    VoiceSubtitlesStatus { dir: dump::fwd(dir), packs }
}

fn sanitize_voice_lang_stem(raw: &str) -> Option<String> {
    let s: String = raw
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    (!s.is_empty() && s.len() <= 32).then_some(s)
}

fn validate_and_copy_voice_json(src: &Path, dst_dir: &Path) -> Result<Option<String>, String> {
    let Some(ext) = src.extension().and_then(|s| s.to_str()) else {
        return Ok(None);
    };
    if !ext.eq_ignore_ascii_case("json") {
        return Ok(None);
    }
    let Some(raw_stem) = src.file_stem().and_then(|s| s.to_str()) else {
        return Ok(None);
    };
    let Some(stem) = sanitize_voice_lang_stem(raw_stem) else {
        return Ok(None);
    };
    let bytes = fs::read(src).map_err(|e| format!("cannot read {}: {e}", src.display()))?;
    let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Ok(None);
    };
    let valid_cues = val
        .as_object()
        .and_then(|o| o.get("cues"))
        .and_then(|c| c.as_object())
        .map(|m| !m.is_empty())
        .unwrap_or(false);
    if !valid_cues {
        return Ok(None);
    }
    fs::create_dir_all(dst_dir).map_err(|e| format!("cannot create {}: {e}", dst_dir.display()))?;
    let dest = dst_dir.join(format!("{stem}.json"));
    fs::write(&dest, &bytes).map_err(|e| format!("cannot write {}: {e}", dest.display()))?;
    Ok(Some(stem))
}

fn import_voice_subtitles_from(src: &Path, dst_dir: &Path) -> Result<Vec<String>, String> {
    if !src.is_file() {
        return Err(format!("File not found: {}", src.display()));
    }
    let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    let mut installed = Vec::new();
    if ext == "json" {
        if let Some(stem) = validate_and_copy_voice_json(src, dst_dir)? {
            installed.push(stem);
        }
    } else if ext == "zip" {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = std::env::temp_dir().join(format!("xc2-voice-import-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&tmp).map_err(|e| format!("cannot create temp dir: {e}"))?;
        let extract_res = mod_import::extract_zip(src, &tmp);
        if let Err(e) = extract_res {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e);
        }
        let mut stack = vec![(tmp.clone(), 0usize)];
        while let Some((dir, depth)) = stack.pop() {
            if depth > 4 {
                continue;
            }
            if let Ok(rd) = fs::read_dir(&dir) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.is_dir() {
                        stack.push((p, depth + 1));
                    } else if p.is_file() {
                        if let Ok(Some(stem)) = validate_and_copy_voice_json(&p, dst_dir) {
                            installed.push(stem);
                        }
                    }
                }
            }
        }
        let _ = fs::remove_dir_all(&tmp);
    } else {
        return Err("Unsupported file type. Select a .zip or .json voice subtitle pack.".into());
    }
    installed.sort();
    installed.dedup();
    if installed.is_empty() {
        return Err("No valid voice subtitle JSON files (with a non-empty 'cues' table) were found.".into());
    }
    Ok(installed)
}

#[tauri::command]
fn voice_subtitles_status(app: AppHandle) -> Result<VoiceSubtitlesStatus, String> {
    let cfg = load_config(&app);
    let dir = game_dir_of(&cfg, &need_repo(&cfg)?).join("voice_subtitles");
    Ok(list_voice_subtitles(&dir))
}

#[tauri::command]
async fn pick_voice_subtitle_file(app: AppHandle, title: String) -> Option<String> {
    app.dialog()
        .file()
        .set_title(title)
        .add_filter("Voice Subtitle Pack (.zip, .json)", &["zip", "json"])
        .blocking_pick_file()
        .and_then(|p| p.into_path().ok())
        .map(|p| dump::fwd(&p))
}

#[tauri::command]
fn import_voice_subtitles(app: AppHandle, path: String) -> Result<VoiceSubtitlesImportResult, String> {
    let cfg = load_config(&app);
    let dir = game_dir_of(&cfg, &need_repo(&cfg)?).join("voice_subtitles");
    let installed = import_voice_subtitles_from(Path::new(path.trim()), &dir)?;
    let status = list_voice_subtitles(&dir);
    Ok(VoiceSubtitlesImportResult { installed, status })
}

#[tauri::command]
fn open_voice_subtitles_dir(app: AppHandle) -> Result<(), String> {
    let cfg = load_config(&app);
    let dir = game_dir_of(&cfg, &need_repo(&cfg)?).join("voice_subtitles");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    play::open_with_shell(&dir)
}

#[tauri::command]
fn read_mod_log(app: AppHandle) -> Result<mods::ModLogView, String> {
    let cfg = load_config(&app);
    let game_dir = game_dir_of(&cfg, &need_repo(&cfg)?);
    mods::read_last_mod_log(&game_dir)
}

#[tauri::command]
fn open_setup_log(app: AppHandle) -> Result<(), String> {
    let cfg = load_config(&app);
    let repo = need_repo(&cfg)?;
    let log = if package_tools().is_some() { game_dir_of(&cfg, &repo).join("setup.log") } else { repo.join("work").join("setup.log") };
    if !log.is_file() {
        return Err("There is no setup log yet.".into());
    }
    play::open_with_shell(&log)
}

/// Starts tools/report.py, which plays the game once with the diagnostics on and packs work/reports/report-<stamp>.zip when
/// the window closes. The player does not have to find a log or copy lines out of it: the zip is the whole report.
#[tauri::command]
fn start_report(app: AppHandle) -> Result<String, String> {
    let cfg = load_config(&app);
    let repo = need_repo(&cfg)?;
    let dir = game_dir_of(&cfg, &repo);
    let (py, script) = if let Some(tools) = package_tools() {
        (tools.join("python").join("python.exe"), tools.join("report.py"))
    } else {
        let py = std::env::var_os("XC2_PYTHON").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("python"));
        (py, repo.join("tools").join("report.py"))
    };
    if !script.is_file() {
        return Err("tools/report.py is not in this folder.".into());
    }
    let reports = dir.join("work").join("reports");
    fs::create_dir_all(&reports).map_err(|e| e.to_string())?;
    let mut c = std::process::Command::new(&py);
    c.arg(&script).current_dir(&dir);
    setup::host_env(&mut c);
    c.spawn().map_err(|e| format!("cannot start {}: {e}", py.display()))?;
    Ok(reports.to_string_lossy().to_string())
}

/// Opens work/reports, where report.py puts the zip.
#[tauri::command]
fn open_reports_dir(app: AppHandle) -> Result<(), String> {
    let cfg = load_config(&app);
    let dir = game_dir_of(&cfg, &need_repo(&cfg)?).join("work").join("reports");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    play::open_with_shell(&dir)
}

#[tauri::command]
async fn check_update(app: AppHandle) -> Result<update::Check, String> {
    // FEFW_LAUNCHER_PRETEND_VERSION: test the update path against the published release from an older-looking launcher
    let current = std::env::var("FEFW_LAUNCHER_PRETEND_VERSION").ok().filter(|v| update::parse_version(v).is_some()).unwrap_or_else(|| update::BUILD_ID.map(String::from).unwrap_or_else(|| app.package_info().version.to_string()));
    let package = package_tools().is_some();
    tauri::async_runtime::spawn_blocking(move || update::check(&current, package)).await.map_err(|e| e.to_string())?
}

#[tauri::command]
fn update_status(state: State<Arc<update::Shared>>) -> update::Status {
    let mut s = state.status.lock().unwrap().clone();
    if s.state.is_empty() {
        s.state = "idle".into();
    }
    s
}

#[tauri::command]
fn start_update(
    app: AppHandle,
    state: State<Arc<update::Shared>>,
    setup_state: State<Arc<setup::Shared>>,
    play_state: State<play::PlayState>,
    release: update::Release,
) -> Result<(), String> {
    if package_tools().is_none() {
        return Err("Updates are for the release package. In a checkout, use git.".into());
    }
    if release.asset_url.is_empty() {
        return Err(format!("Release {} has no package for this system yet.", release.version));
    }
    if setup_state.progress.lock().unwrap().state == "running" {
        return Err("Setup is running. Wait for it or pause it first.".into());
    }
    if play_state.0.lock().unwrap().state == "running" {
        return Err("The game is running. Close it first.".into());
    }
    {
        let mut s = state.status.lock().unwrap();
        if s.state == "running" {
            return Err("An update is already running.".into());
        }
        *s = update::Status { state: "running".into(), phase: "download".into(), message: "Starting".into(), version: release.version.clone(), ..Default::default() };
    }
    let cfg = load_config(&app);
    let install = exe_dir();
    let game_dir = repo_of(&cfg).map(|r| game_dir_of(&cfg, &r)).unwrap_or_else(|| install.clone());
    let shared = state.inner().clone();
    shared.cancel.store(false, Ordering::SeqCst);
    std::thread::spawn(move || {
        let a = app.clone();
        let emit = move |s: update::Status| {
            let _ = a.emit("update-status", s);
        };
        let run = update::Run { shared: &shared, install, game_dir, emit: &emit };
        let end = match run.run(&release) {
            Ok(s) => s,
            Err(e) => update::Status { state: "error".into(), message: e, version: release.version.clone(), ..Default::default() },
        };
        *shared.status.lock().unwrap() = end.clone();
        let _ = app.emit("update-status", end);
    });
    Ok(())
}

#[tauri::command]
fn cancel_update(state: State<Arc<update::Shared>>) {
    state.cancel.store(true, Ordering::SeqCst);
}

/// The folder the game is started from: the DLSS files go to upscalers/streamline there (streamline.rs).
fn game_run_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let cfg = load_config(app);
    let repo = need_repo(&cfg)?;
    let game = game_dir_of(&cfg, &repo);
    let exe = if package_tools().is_some() { game.join("xc2_boot.exe") } else { setup::exe_path(&repo) };
    Ok(exe.parent().map(|d| d.to_path_buf()).unwrap_or(game))
}

#[derive(Serialize)]
struct UpscalerDownloadInfo {
    #[serde(flatten)]
    manifest: streamline::DownloadManifest,
    dest: String,
}

#[tauri::command]
fn upscaler_download_manifest(app: AppHandle, kind: String) -> Result<UpscalerDownloadInfo, String> {
    let game_dir = game_run_dir(&app)?;
    match kind.as_str() {
        "dlss" => Ok(UpscalerDownloadInfo {
            manifest: streamline::manifest(),
            dest: dump::fwd(&streamline::dest(&game_dir)),
        }),
        "fsr" => Ok(UpscalerDownloadInfo {
            manifest: fsr::manifest(),
            dest: dump::fwd(&fsr::dest(&game_dir)),
        }),
        _ => Err(format!("unknown upscaler download kind: {kind}")),
    }
}

#[tauri::command]
fn dlss_download_status(state: State<Arc<streamline::Shared>>) -> streamline::Status {
    let mut s = state.status.lock().unwrap().clone();
    if s.state.is_empty() {
        s.state = "idle".into();
    }
    s
}

/// Downloads NVIDIA's Streamline SDK release and installs the DLSS files (streamline.rs); progress through dlss_download_status.
#[tauri::command]
fn start_dlss_download(app: AppHandle, state: State<Arc<streamline::Shared>>, play_state: State<play::PlayState>, confirmed: bool) -> Result<(), String> {
    if !confirmed {
        return Err("Download requires explicit user confirmation.".into());
    }
    if play_state.0.lock().unwrap().state == "running" {
        return Err("The game is running. Close it first: it keeps the DLSS files open.".into());
    }
    let game_dir = game_run_dir(&app)?;
    {
        let mut s = state.status.lock().unwrap();
        if s.state == "running" {
            return Err("The DLSS download is already running.".into());
        }
        *s = streamline::Status { state: "running".into(), phase: "download".into(), message: "Starting".into(), ..Default::default() };
    }
    let shared = state.inner().clone();
    shared.cancel.store(false, Ordering::SeqCst);
    std::thread::spawn(move || {
        let run = streamline::Run { shared: &shared, game_dir };
        let end = match run.run() {
            Ok(s) => s,
            Err(e) => streamline::Status { state: "error".into(), message: e, ..Default::default() },
        };
        *shared.status.lock().unwrap() = end;
    });
    Ok(())
}

#[tauri::command]
fn cancel_dlss_download(state: State<Arc<streamline::Shared>>) {
    state.cancel.store(true, Ordering::SeqCst);
}

#[tauri::command]
fn fsr_download_status(state: State<'_, Arc<fsr::Shared>>) -> streamline::Status {
    let mut s = state.status.lock().unwrap().clone();
    if s.state.is_empty() {
        s.state = "idle".into();
    }
    s
}

/// Downloads AMD's signed FidelityFX SDK DLLs (v2.4.0 + v1.1.4) directly from GPUOpen's official GitHub tags.
#[tauri::command]
fn start_fsr_download(app: AppHandle, state: State<'_, Arc<fsr::Shared>>, play_state: State<play::PlayState>, confirmed: bool) -> Result<(), String> {
    if !confirmed {
        return Err("Download requires explicit user confirmation.".into());
    }
    if play_state.0.lock().unwrap().state == "running" {
        return Err("The game is running. Close it first: it keeps the FSR files open.".into());
    }
    let game_dir = game_run_dir(&app)?;
    {
        let mut s = state.status.lock().unwrap();
        if s.state == "running" {
            return Err("The FSR download is already running.".into());
        }
        *s = streamline::Status { state: "running".into(), phase: "download".into(), message: "Starting".into(), ..Default::default() };
    }
    let shared = state.inner().clone();
    shared.cancel.store(false, Ordering::SeqCst);
    std::thread::spawn(move || {
        let run = fsr::Run { shared: &shared, game_dir };
        let end = match run.run() {
            Ok(s) => s,
            Err(e) => streamline::Status { state: "error".into(), message: e, ..Default::default() },
        };
        *shared.status.lock().unwrap() = end;
    });
    Ok(())
}

#[tauri::command]
fn cancel_fsr_download(state: State<'_, Arc<fsr::Shared>>) {
    state.cancel.store(true, Ordering::SeqCst);
}

/// Starts the (new) launcher from the install folder and closes this one.
#[tauri::command]
fn restart_launcher(app: AppHandle) -> Result<(), String> {
    let exe = match std::env::var_os("APPIMAGE") {
        Some(a) if !cfg!(windows) => PathBuf::from(a),
        _ => exe_dir().join(if cfg!(windows) { "xc2-launcher.exe" } else { "xc2-launcher" }),
    };
    let mut c = std::process::Command::new(&exe);
    setup::host_env(&mut c); // not this AppImage's environment: its mount goes away when this launcher exits
    c.current_dir(exe_dir()).spawn().map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
    app.exit(0);
    Ok(())
}

fn main() {
    if package_tools().is_some() {
        update::cleanup(&exe_dir());
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(play::PlayState::default())
        .manage(Arc::new(setup::Shared::default()))
        .manage(Arc::new(update::Shared::default()))
        .manage(Arc::new(streamline::Shared::default()))
        .manage(Arc::new(fsr::Shared::default()))
        .on_window_event(|window, event| {
            // closing the window pauses a running setup rather than leaving its compilers behind
            if let tauri::WindowEvent::Destroyed = event {
                let st = window.state::<Arc<setup::Shared>>();
                st.cancel.store(true, Ordering::SeqCst);
                setup::kill_tree(st.pid.load(Ordering::SeqCst));
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            get_overview,
            pick_folder,
            pick_nup,
            import_nup,
            validate_dump,
            save_dump,
            read_settings,
            write_settings,
            upscaler_files,
            setup_status,
            start_setup,
            cancel_setup,
            play,
            game_status,
    import_saves,
            steam_status,
            add_to_steam,
            open_game_file,
            open_game_dir,
            open_mods_dir,
            list_mods,
            save_mods_state,
            open_mod_folder,
            pick_mod_file,
            import_mod,
            install_mod_folder,
            delete_mod,
            read_mod_log,
            voice_subtitles_status,
            pick_voice_subtitle_file,
            import_voice_subtitles,
            open_voice_subtitles_dir,
            open_setup_log,
            start_report,
            open_reports_dir,
            check_update,
            update_status,
            start_update,
            cancel_update,
            upscaler_download_manifest,
            dlss_download_status,
            start_dlss_download,
            cancel_dlss_download,
            fsr_download_status,
            start_fsr_download,
            cancel_fsr_download,
            restart_launcher,
            locale::list_locales,
            locale::read_locale,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the launcher");
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_saves_keeps_names_and_uses_the_runtime_account() {
        let root = std::env::temp_dir().join(format!("xc2-saves-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let src = root.join("ported");
        fs::create_dir_all(src.join("slot1")).unwrap();
        fs::write(src.join("bf2savefile.sav"), b"main").unwrap();
        fs::write(src.join("slot1").join("irasavefile.sav"), b"torna").unwrap();
        fs::write(src.join("notes.txt"), b"not a save").unwrap();
        let account = root.join("save").join("account");

        let (copied, seen) = copy_saves(&account, &src).unwrap();
        assert_eq!((copied, seen), (2, 2), "only .sav files, both depths");
        let dest = account.join(SAVE_ACCOUNT_UID);
        assert_eq!(fs::read(dest.join("bf2savefile.sav")).unwrap(), b"main");
        assert_eq!(fs::read(dest.join("irasavefile.sav")).unwrap(), b"torna");
        assert!(!dest.join("notes.txt").exists());

        // a second import into the same install lands in the folder that is already there
        let (again, _) = copy_saves(&account, &src).unwrap();
        assert_eq!(again, 2);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn import_voice_subtitles_validates_and_copies_json() {
        let root = std::env::temp_dir().join(format!("xc2-voice-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let good = root.join("en.json");
        fs::write(&good, br#"{"speakers":{},"cues":{"voice_01010101_Play":{"text_full":"Hi"}}}"#).unwrap();
        let bad = root.join("not_voice.json");
        fs::write(&bad, br#"{"hello":"world"}"#).unwrap();
        let dst = root.join("voice_subtitles");

        let ok = import_voice_subtitles_from(&good, &dst).unwrap();
        assert_eq!(ok, vec!["en".to_string()]);
        assert!(dst.join("en.json").is_file());
        assert!(import_voice_subtitles_from(&bad, &dst).is_err());

        let st = list_voice_subtitles(&dst);
        assert_eq!(st.packs, vec!["en".to_string()]);
        let _ = fs::remove_dir_all(&root);
    }
}
