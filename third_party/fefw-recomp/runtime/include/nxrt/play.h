// SPDX-License-Identifier: GPL-3.0-or-later
// Play mode (S8): the PC front end of the recompiled game. One window, the SDL event pump on the game's main thread, the frame
// limiter on the presenting thread, real controller / keyboard / mouse input into the hid service, raw mouse aim, and the F1 settings page.
//
// Start-up order in a launcher (game-build/src/mpr_main.cpp):
//   nxrt::init();  play::load();                     // reads controls.ini (or [play] controls) and the [play] keys of nxrt.ini
//   window + gfx::Device + gfx::Renderer             // window size and present mode come from play::display()
//   nvn::install_backend({... limiter = play::limiter(), keep_aspect = play::display().keep_aspect})
//   play::start({...});                              // input source, hooks, overlay page; the game may run from here
// What is about one game (the frame-boundary site, raw mouse aim, camera look, lock-on, display hooks, menu widgets, context detection, F1
// entries) comes from the registered game module's play table (nxrt/game.h, src/gfx/play/play_game.h; docs/GAME_BOUNDARY.md). The game must
// have been lifted with entry hooks at those sites (Metroid Prime Remastered: recomp/games/mpr/main.sites). Without them the window still
// opens but nothing is pumped (a warning says so after a few seconds).
//
// Configuration (nxrt.ini section [play], environment NXRT_PLAY_<KEY>):
//   controls = controls.ini     the settings file (play_types.h documents the format); written with the factory settings when missing
//   capture_mouse = 1           capture the pointer (relative mode) while the game is being played
//   pitch_limit_deg = 0         > 0: pitch limit of the mouse aim in degrees instead of the game's tweak value
//   quit_on_close = 1           closing the window ends the process (exit code 0)
//   game_base = 0x7100000000    base of the game's main module
//   log_status = 0              1: one status line per second in the log (context, aim state, input counts)
#pragma once
#include <cstdint>
#include <functional>
#include <string>

#include "nxrt/frame_limiter.h"
#include "nxrt/gfx.h"
#include "nxrt/overlay.h"

namespace nxrt::play {

// Loads the settings file and creates the frame limiter. Idempotent. Returns false (and uses the factory settings) when the file is invalid.
bool load();
// What the launcher needs from the settings before the window exists.
struct DisplayConfig {
    int width = 1280, height = 720;         // windowed client size
    std::string window_mode = "windowed";   // "windowed" "borderless" "exclusive"
    bool vsync = false;
    int fps_cap = 60;                       // 0 = unlimited
    bool keep_aspect = true;
    gfx::PresentMode present_mode = gfx::PresentMode::Immediate;   // what the vsync setting asks for (the renderer falls back to what the surface supports)
};
DisplayConfig display();
FrameLimiter* limiter();                    // owned by play mode; pass to nvn::BackendDesc::limiter
std::string settings_path();

struct StartOptions {
    gfx::Device* device = nullptr;          // needed for the overlay
    gfx::Renderer* renderer = nullptr;      // needed for the overlay and for present-mode changes; null = no graphics (unit tests)
    bool window = false;                    // a platform window exists: pump its events, capture the pointer, fullscreen, quit on close
    bool overlay = true;                    // create the F1 overlay (needs device and renderer)
};
bool start(const StartOptions& opt);        // after nxrt::init, the renderer and the NVN back end; installs the hid input source and the hooks
void stop();                                // saves changed settings, removes the hooks and the input source
void set_hud_callback(std::function<void(const gfx::HudCanvas&)> fn);   // optional foreground HUD drawn over the game frame
uint64_t create_hud_texture_bc3(uint32_t w, uint32_t h, const void* bc3_linear, size_t bytes);  // upload BC3 texture for HudCanvas::image
uint64_t create_hud_texture_bc7(uint32_t w, uint32_t h, const void* bc7_linear, size_t bytes);  // upload BC7 texture (menu face atlas)
void set_picker(gfx::PickerCallbacks cb);                                // optional searchable destination picker window (e.g. F4)
void toggle_picker();                                                    // toggle destination picker window
bool overlay_blocks_game();                 // true while the F1 overlay (or on-screen keyboard) wants keyboard/mouse/pad input

// The binding engine's contribution at `now_ns`: Switch buttons plus stick deflection in -1..1. Play mode installs its
// own input source (PlaySource) that already merges this, but a game whose own source replaces it (xc2_boot's LiveInput
// does, on purpose) gets nothing from the keyboard unless it asks for this. False when play mode is not running.
bool binding_output(uint64_t now_ns, uint64_t* buttons, float* lx, float* ly, float* rx, float* ry);

// Hands the pointer to an external input source (xc2_boot's LiveInput turns the counts into right-stick camera
// movement). While this is on, play mode stops capturing and releasing the mouse itself, and stops reading the
// counts for its own aim, so the two never fight over the same deltas.
void set_external_mouse_capture(bool on);

// Live gamepad tuning read by external input sources (e.g. xc2_boot's LiveInput). Lock-free snapshot.
struct LivePadConfig {
    bool label_abxy = false;
    float stick_deadzone = 0.10f;
    float right_deadzone = 0.10f;
    float stick_max = 1.0f;                 // left stick outer-range scale (0.5..1.0; issue #76)
    float look_sens = 1.0f;
    float mouse_stick_scale = 8.0f;         // keyboard/mouse camera: mouse counts for a full right-stick deflection
    bool mouse_camera_capture = true;       // live toggle from F1 Mouse group (issue #78)
    int ls_center_x = 0, ls_center_y = 0;   // left-stick centre from the F1 calibration, subtracted before the dead zone
    float raw_ls_x = 0, raw_ls_y = 0;       // the left stick as the pad reports it, for the calibration readout
    int rs_center_x = 0, rs_center_y = 0;
    float raw_rs_x = 0, raw_rs_y = 0;
    bool invert_y = false;
    bool rumble = true;
    float rumble_scale = 1.0f;
};
LivePadConfig live_pad_config();
// Publishes the external input source's mapped Switch state (buttons + sticks -32767..32767) so the F1 menu's
// gamepad navigation (- + + chord, A/B/D-pad) and Live Switch Pad Monitor work live. Returns the Switch button mask
// currently held back from the game by the overlay (e.g. buttons still held right after closing the menu).
uint64_t note_external_pad(uint64_t buttons, int lx, int ly, int rx, int ry);
// The left stick as the pad reports it (-1..1, y up), published by the external input source so the F1 calibration can
// show it and the player can see how far the rest position has drifted (issue #76).
void publish_raw_left_stick(float x, float y);
void publish_raw_right_stick(float x, float y);

// Counters for the tests and the log.
struct Stats {
    uint64_t frames = 0;                    // Update hook calls
    uint64_t process_input_calls = 0;
    uint64_t aim_frames = 0;                // ProcessInput calls where the mouse aim was applied
    uint64_t events = 0;                    // input events drained
    uint64_t mouse_counts_applied = 0;      // sum of |dx| + |dy| taken by the aim
    bool pump_thread_mismatch = false;      // the Update hook ran on another thread than start() was called on
};
Stats stats();

}  // namespace nxrt::play
