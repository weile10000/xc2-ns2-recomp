// SPDX-License-Identifier: GPL-3.0-or-later
// Play mode: the glue. See include/nxrt/play.h for the start-up order and docs/S8.md for the design.
//
// Threads:
//   main (the guest main thread, which created the window)  the Update hook (frame boundary: event pump, input, settings changes) and the
//                                                            ProcessInput hook (raw mouse aim) run here
//   presenting thread (the game's render thread)             FrameLimiter::begin_frame (the pacing wait, and the pre-frame hook that changes the
//                                                            present mode), the ImGui overlay and the settings page
//   hid sampling thread                                      PlaySource::poll (200 Hz): reads the pad snapshot and asks the binding engine for the
//                                                            keyboard and mouse contribution at that instant
#include "nxrt/diag.h"
#include "nxrt/extras.h"
#include "nxrt/lowlat.h"
#include "nxrt/play.h"

#include <windows.h>

#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_timer.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "nxrt/audio_sink.h"
#include "nxrt/cheats.h"
#include "nxrt/ftrace.h"
#include "nxrt/game.h"
#include "play_display.h"
#include "nxrt/kernel.h"
#include "nxrt/svc.h"
#include "nxrt/config.h"
#include "nxrt/jc2_mouse.h"
#include "nxrt/dispatch.h"
#include "nxrt/hooks.h"
#include "nxrt/input.h"
#include "nxrt/services_api.h"
#include "nxrt/input_source.h"
#include "nxrt/log.h"
#include "nxrt/nvn_backend.h"
#include "nxrt/overlay.h"
#include "nxrt/platform.h"
#include "nxrt/runtime.h"
#include "nxrt/timestep.h"
#include "nxrt/tt.h"
#include "extras/temporal.h"
#include "menu_mouse.h"
#include "play_engine.h"
#include "play_game.h"
#include "play_selftest.h"
#include "play_types.h"
#include "play_vpad.h"
#include "play_ui.h"
#include "prompts.h"
#include "pad_map.h"
#include "pad_orient.h"

namespace nxrt::play {
namespace {
// Keep-alive pump: the window's events are normally pumped at the frame boundary. While the game boots or loads it can go ten seconds and
// more without one, and Windows then calls the window "Not Responding". busy_pump runs before every SVC (svc_set_busy_hook); on the
// thread that owns the window it pumps the events when the last pump is 100 ms old. Elsewhere it costs a thread-id compare.
std::atomic<DWORD> g_pump_tid{0};
// Set by an external input source that drives the pointer itself (xc2_boot's LiveInput, which turns the counts into
// right-stick camera movement). While on, play mode neither captures nor releases the mouse and its aim path leaves
// the deltas alone - otherwise the two would toggle the capture against each other every frame.
std::atomic<bool> g_external_mouse{false};
std::atomic<uint64_t> g_last_pump_ms{0};
void keyboard_service();
void busy_pump() {
    if (GetCurrentThreadId() != g_pump_tid.load(std::memory_order_relaxed)) return;
    keyboard_service();
    const uint64_t t = SDL_GetTicks();
    if (t - g_last_pump_ms.load(std::memory_order_relaxed) < 100) return;
    g_last_pump_ms.store(t, std::memory_order_relaxed);
    platform::pump_events();
}
}  // namespace

namespace {

constexpr const char* kTag = "play";
constexpr uint64_t kMs = 1000000ull;

// SDL mouse button (1 left, 2 middle, 3 right, 4 X1, 5 X2) to nn::hid::MouseButton (Left 0, Right 1, Middle 2, Forward 3, Back 4).
uint32_t jc2_button_bit(uint8_t b) {
    switch (b) {
        case 1: return 1u << 0;
        case 3: return 1u << 1;
        case 2: return 1u << 2;
        case 5: return 1u << 3;
        case 4: return 1u << 4;
        default: return 0;
    }
}

struct PadFrame {
    uint64_t buttons = 0;
    hid::StickValue left, right;
    hid::SixAxisSample six;
};

struct Touch {
    bool down = false;
    uint32_t x = 0, y = 0;     // on the 1280 x 720 touch screen
};

struct G;

// ---- the hid input source ------------------------------------------------------------------------------------------------------------
class PlaySource : public hid::InputSource {
public:
    explicit PlaySource(Engine* e) : engine_(e) {}
    void poll(hid::InputSnapshot& out) override {
        const EngineOut kb = engine_->output(FrameLimiter::now_ns());
        PadFrame pf;
        Touch t;
        {
            std::lock_guard<std::mutex> lk(mu_);
            pf = pad_;
            t = touch_;
        }
        hid::NpadInput& n = out.npad[0];
        n.connected = true;
        n.style = hid::NpadStyle::FullKey;
        n.buttons = pf.buttons | kb.buttons;
        n.left = combine(pf.left, kb.lx, kb.ly);
        n.right = combine(pf.right, kb.rx, kb.ry);
        {   // [nvn] pad_dbg: the hid analog range is ±32767; anything past it makes XC2's ml::DevPadNx::Impl::updateTpad assert
            // (`mi<=ma` on the scaled value), so show which input produced it: the pad frame, the keyboard's unit contribution, the result.
            static const bool dbg = nxrt::config_get_bool("nvn", "pad_dbg", true);
            if (dbg && (std::abs(n.left.x) > 32767 || std::abs(n.left.y) > 32767 || std::abs(n.right.x) > 32767 || std::abs(n.right.y) > 32767))
                NXLOG_W("hid", "play input over range: pad l %d %d r %d %d | kb l %g %g r %g %g | out l %d %d r %d %d", pf.left.x, pf.left.y, pf.right.x,
                        pf.right.y, static_cast<double>(kb.lx), static_cast<double>(kb.ly), static_cast<double>(kb.rx), static_cast<double>(kb.ry), n.left.x,
                        n.left.y, n.right.x, n.right.y);
        }
        n.six_axis = pf.six;
        {   // the pad's orientation (the game's pointer scheme reads it): integrated from the gyro, held level by the accelerometer
            const uint64_t t = FrameLimiter::now_ns();
            const double dt = last_orient_ns_ ? static_cast<double>(t - last_orient_ns_) / 1e9 : 0.0;
            last_orient_ns_ = t;
            orient_.step(pf.six.gyro, pf.six.accel, dt);
            std::memcpy(n.six_axis.direction, orient_.direction, sizeof orient_.direction);
            std::memcpy(n.six_axis.angle, orient_.angle, sizeof orient_.angle);
        }
        right_y_.store(n.right.y, std::memory_order_relaxed);
        right_x_f_.store(static_cast<float>(n.right.x) / 32767.f, std::memory_order_relaxed);
        right_y_f_.store(static_cast<float>(n.right.y) / 32767.f, std::memory_order_relaxed);
        gyro_pitch_.store(std::fabs(pf.six.gyro[0]), std::memory_order_relaxed);
        n.six_axis.gyro[0] *= look_scale_.load(std::memory_order_relaxed);      // the game integrates the pitch rate per call: see play_lookrate.h
        out.touch.count = 0;
        if (t.down) {
            out.touch.count = 1;
            hid::TouchFinger& f = out.touch.fingers[0];
            f = hid::TouchFinger{};
            f.id = 0;
            f.x = t.x;
            f.y = t.y;
        }
    }
    void set_pad(const PadFrame& p) {
        std::lock_guard<std::mutex> lk(mu_);
        pad_ = p;
    }
    // Is a pad looking up or down right now (the right stick's vertical axis off centre, or the gyro turning about the pad's pitch axis)? Read by the aim
    // on the main thread; the values are the ones the game gets.
    bool pad_look(bool gyro_enabled) const {
        return std::abs(right_y_.load(std::memory_order_relaxed)) > 650 || (gyro_enabled && gyro_pitch_.load(std::memory_order_relaxed) > 0.02f);
    }
    // The right stick as the game gets it (dead zone, invert and sensitivity applied), -1..1, + = right and up: the alternate body's camera (mpr: the morph ball camera) orbits with it.
    void right_stick(float* x, float* y) const {
        *x = right_x_f_.load(std::memory_order_relaxed);
        *y = right_y_f_.load(std::memory_order_relaxed);
    }
    void set_look_scale(float k) { look_scale_.store(k, std::memory_order_relaxed); }      // the gyro's pitch rate times k (step * 60), for the game's per-call pitch
    void set_touch(const Touch& t) {
        std::lock_guard<std::mutex> lk(mu_);
        touch_ = t;
    }

private:
    static hid::StickValue combine(hid::StickValue pad, float kx, float ky) {
        double x = pad.x + static_cast<double>(kx) * 32767.0, y = pad.y + static_cast<double>(ky) * 32767.0;
        const double len = std::sqrt(x * x + y * y);
        if (len > 32767.0) { x *= 32767.0 / len; y *= 32767.0 / len; }
        hid::StickValue r;
        // The integer clamp is what finally holds the hid range: the pull-back above leaves a rounded value and a value of 32768 (= the
        // negation of the SDL axis minimum) reached the game's pad state, where XC2 asserts on it (mm_math.h:819, mi<=ma).
        r.x = std::clamp<int32_t>(static_cast<int32_t>(std::lround(x)), -32767, 32767);
        r.y = std::clamp<int32_t>(static_cast<int32_t>(std::lround(y)), -32767, 32767);
        return r;
    }
    Engine* engine_;
    PadOrient orient_;
    uint64_t last_orient_ns_ = 0;
    std::atomic<int32_t> right_y_{0};
    std::atomic<float> right_x_f_{0.f}, right_y_f_{0.f};
    std::atomic<float> look_scale_{1.f};
    std::atomic<float> gyro_pitch_{0.f};
    std::mutex mu_;
    PadFrame pad_;
    Touch touch_;
};

// ---- the state ------------------------------------------------------------------------------------------------------------------------
struct G {
    // settings
    std::mutex mu;                       // guards settings (written on the main thread, copied by the page on the presenting thread) and pending
    Settings settings;
    Settings pending;
    bool have_pending = false;
    bool dirty = false;                  // settings changed in the game (Alt+Enter): written at stop
    std::string path;
    bool loaded = false, started = false, load_ok = true;

    FrameLimiter limiter;
    std::unique_ptr<Engine> engine;
    std::unique_ptr<PlaySource> source_owner;
    std::shared_ptr<hid::InputSource> source;
    std::unique_ptr<gfx::Overlay> overlay;
    gfx::Renderer* renderer = nullptr;
    gfx::Device* device = nullptr;
    bool window = false;

    std::thread::id pump_thread;
    uint64_t base = 0;                   // set by load: [play] game_base, default the game's Info::main_base
    uint32_t hook_update = 0;            // the frame boundary (the game's GamePlay::frame_update)
    bool capture_mouse = true, quit_on_close = true, log_status = false;
    bool pointer_draw = false;         // [play] pointer_draw: the menu pointer is also drawn into the frame (screenshots show it); a test aid
    bool late_sample = true;           // pump the window's events again right before the mouse counts are taken (S11: the counts are as new as ProcessInput)
    double pitch_limit_deg = 0;

    // frame state (main thread)
    uint64_t update_seq = 0, aim_seq = 0;
    std::atomic<uint64_t> last_pi_ns{0};
    std::atomic<uint64_t> last_pi_frame{0};       // value of `frames` at the last ProcessInput (the game is playing while it keeps being called)
    uint64_t map_acts_seen = 0;
    bool map_hint = false;
    bool overlay_was_visible = false;
    bool was_in_game = false;
    float mouse_x = 0, mouse_y = 0;
    bool mouse1 = false;
    uint64_t last_mouse_activity_ns = 0;
    float last_cursor_x = -1.f, last_cursor_y = -1.f;
    bool have_cursor_pos = false;
    bool was_cursor_autohide_eligible = false;
    bool cursor_auto_hidden = false;
    uint32_t jc2_buttons = 0;            // host mouse buttons as nn::hid::MouseButton bits, for the Joy-Con 2 mouse (hid jc2_mouse.h)
    float jc2_wheel_x = 0, jc2_wheel_y = 0;   // wheel notches since the last feed
    uint64_t last_status_ns = 0;
    bool warned_thread = false;

    // button prompts: the device the player used last (Auto style) and a counter that changes with the bindings
    LastDevice last_dev = LastDevice::None;
    int last_pad_kind = 0;
    uint64_t settings_epoch = 1;

    // pads
    uint64_t last_rumble_ns = 0;
    uint16_t rumble_lo = 0, rumble_hi = 0;

    // present mode requested by the page, applied on the presenting thread
    std::atomic<int> want_present{-1};

    // vibration from the game (hid thread) -> pads (main thread)
    std::atomic<float> vib[2][2] = {};   // [hand][low, high] amplitude

    // external input source bridge (xc2_boot's LiveInput): lock-free pad config & pad state -> F1 menu/monitor
    std::atomic<bool> pad_cfg_label_abxy{false};
    std::atomic<float> pad_cfg_ls_dz{0.10f};
    std::atomic<float> pad_cfg_rs_dz{0.10f};
    std::atomic<float> pad_cfg_stick_max{1.0f};
    std::atomic<float> pad_cfg_look_sens{1.0f};
    std::atomic<float> mouse_cfg_stick_scale{8.0f};
    std::atomic<bool> mouse_cfg_camera_capture{true};
    std::atomic<int> pad_cfg_ls_cx{0}, pad_cfg_ls_cy{0}, pad_cfg_rs_cx{0}, pad_cfg_rs_cy{0};
    std::atomic<float> pad_raw_ls_x{0.0f}, pad_raw_ls_y{0.0f}, pad_raw_rs_x{0.0f}, pad_raw_rs_y{0.0f};
    std::atomic<bool> pad_cfg_invert_y{false};
    std::atomic<bool> pad_cfg_rumble{true};
    std::atomic<float> pad_cfg_rumble_scale{1.0f};

    std::atomic<bool> ext_pad_active{false};
    std::atomic<uint64_t> ext_pad_buttons{0};
    std::atomic<int> ext_pad_lx{0}, ext_pad_ly{0}, ext_pad_rx{0}, ext_pad_ry{0};
    std::atomic<uint64_t> ext_menu_mask{0};

    // rebinding
    std::atomic<int> capture_state{0};   // 0 idle, 1 waiting for an input
    std::mutex cap_mu;
    bool cap_have = false;
    InputId cap_id;

    // live status
    mutable std::mutex st_mu;
    LiveStatus live;
    uint64_t fps_last_frames = 0, fps_last_ns = 0;
    double fps = 0;

    SelfTest selftest;

    // counters
    std::atomic<uint64_t> frames{0}, pi_calls{0}, aim_frames{0}, events{0}, counts{0};
    std::atomic<bool> thread_mismatch{false};
};

SelfTestHost* selftest_host();      // defined after the settings page host

G& g() {
    static G inst;
    return inst;
}

uint64_t now_ns() { return FrameLimiter::now_ns(); }

// [gfx] present_mode (nxrt.ini) is the mode the settings page writes and the only one a player can reach; [display] vsync
// (controls.ini) is what this function knew before that key existed. The explicit key wins. Without it the renderer used
// the key once, when it was created, and play::start() overwrote it a tenth of a second later: picking "FIFO" in the
// launcher still presented in Immediate (issue 12, two swapchains 115 ms apart, measured 2026-11-20).
// No key at all: vsync off is immediate (tearing, lowest latency; mailbox where immediate is missing). vsync on is FIFO,
// except with [display] fps_limit = 0 (uncapped): mailbox, so the game renders as fast as it can and the display still shows whole frames.
gfx::PresentMode present_mode_for(const DisplaySettings& d) {
    const std::string cfg = config_get_str("gfx", "present_mode", "");
    gfx::PresentMode pm = gfx::PresentMode::Fifo;
    if (!cfg.empty() && gfx::parse_present_mode(cfg.c_str(), &pm)) return pm;
    if (!d.vsync) return gfx::PresentMode::Immediate;
    return d.fps_cap == 0 ? gfx::PresentMode::Mailbox : gfx::PresentMode::Fifo;
}

const char* window_mode_name(platform::FullscreenMode m) {
    switch (m) {
        case platform::FullscreenMode::Windowed: return "windowed";
        case platform::FullscreenMode::Borderless: return "borderless";
        case platform::FullscreenMode::Exclusive: return "exclusive";
    }
    return "?";
}

// Master volume (live). The mute while the window has no focus is set every frame (update_hook) and on focus changes.
void apply_audio(const AudioSettings& a) { audio::set_master_volume(static_cast<float>(a.volume)); }

void publish_live_pad_config(const PadSettings& p) {
    G& st = g();
    st.pad_cfg_label_abxy.store(p.label_abxy, std::memory_order_relaxed);
    st.pad_cfg_ls_dz.store(static_cast<float>(std::clamp(p.stick_deadzone, 0.0, 0.5)), std::memory_order_relaxed);
    st.pad_cfg_rs_dz.store(static_cast<float>(std::clamp(p.right_deadzone, 0.0, 0.5)), std::memory_order_relaxed);
    st.pad_cfg_stick_max.store(static_cast<float>(std::clamp(p.stick_max, 0.5, 1.0)), std::memory_order_relaxed);
    st.pad_cfg_look_sens.store(static_cast<float>(std::clamp(p.look_sens, 0.25, 3.0)), std::memory_order_relaxed);
    st.pad_cfg_invert_y.store(p.invert_y, std::memory_order_relaxed);
    st.pad_cfg_rumble.store(p.rumble, std::memory_order_relaxed);
    st.pad_cfg_rumble_scale.store(static_cast<float>(std::clamp(p.rumble_scale, 0.0, 1.0)), std::memory_order_relaxed);
    st.pad_cfg_ls_cx.store(std::clamp(p.ls_center_x, -32767, 32767), std::memory_order_relaxed);
    st.pad_cfg_ls_cy.store(std::clamp(p.ls_center_y, -32767, 32767), std::memory_order_relaxed);
    st.pad_cfg_rs_cx.store(std::clamp(p.rs_center_x, -32767, 32767), std::memory_order_relaxed);
    st.pad_cfg_rs_cy.store(std::clamp(p.rs_center_y, -32767, 32767), std::memory_order_relaxed);
}

void publish_live_mouse_config(const MouseSettings& m) {
    G& st = g();
    st.mouse_cfg_stick_scale.store(static_cast<float>(std::clamp(m.stick_scale, 1.0, 200.0)), std::memory_order_relaxed);
    st.mouse_cfg_camera_capture.store(m.camera_capture, std::memory_order_relaxed);
}

// The F1 render quality: the upscaler's input as a share of the output height (table in nxrt/render_quality.h).
}  // namespace

// The output height the quality is a share of: [am] resolution (the console mode the game renders for, the window's size), else 1080.
int render_output_height() {
    const std::string am = config_get_str("am", "resolution", "");
    const size_t x = am.find_first_of("xX");
    const int h = x == std::string::npos ? 0 : std::atoi(am.c_str() + x + 1);
    return h >= 720 && h <= 4320 ? h : 1080;
}

namespace {

Game3dSettings hydrate_game3d_from_config() {
    Game3dSettings r;
    const std::string q = config_get_str("upscale", "render_quality", "");
    r.res_preset = 0;   // native (100%) default when [upscale] render_quality is unset
    for (int i = 0; i < kRenderQualityCount; ++i)
        if (q == kRenderQuality[i].key) r.res_preset = i;
    if (q.empty() && config_get_bool("upscale", "lock_render_size", false)) {   // set by hand: the nearest quality to the ini's scales
        const double scene = config_get_double("gfx", "render_scale", 1.0);
        const double out = config_get_double("gfx", "output_scale", scene);
        const double share = out > 0.0 ? (scene * 720.0) / (out * 1080.0) : 0.0;
        double best = 1e9;
        for (int i = 0; i + 1 < kRenderQualityCount; ++i) {
            const double d = std::fabs(std::log(share / kRenderQuality[i].share));
            if (share > 0.0 && d < best) { best = d; r.res_preset = i; }
        }
    }

    std::string up = config_get_str("dlss", "upscaler", "taau");
    for (char& c : up) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (up == "auto") up = extras::temporal_dlss_started() ? "dlss" : "fsr3";
    if (up != "dlss" && up != "fsr3" && up != "taau" && up != "bilinear") up = up == "fsr" ? "fsr3" : "taau";
    r.upscaler = up;
    r.dlss_preset = extras::dlss_preset_current();

    std::string fv = config_get_str("upscale", "fsr_version", "3.1");
    if (fv.rfind("4", 0) == 0) fv = "4.1";
    else if (fv.rfind("2", 0) == 0) fv = "2.3";
    else fv = "3.1";
    r.fsr_version = fv;
    r.sharpness = std::clamp(config_get_double("dlss", "sharpness", 0.5), 0.0, 1.0);
    r.out_height = render_output_height();
    std::string cap = config_get_str("nvn", "mirror_capture", "auto");
    for (char& c : cap) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    r.capture = cap == "auto" ? "auto" : (cap == "1" || cap == "on" || cap == "true" || cap == "yes") ? "on" : "off";
    return r;
}

void apply_game3d_settings(const Game3dSettings& old_g, const Game3dSettings& g3d) {
    const bool up_changed = (g3d.upscaler != old_g.upscaler || g3d.fsr_version != old_g.fsr_version ||
                             std::fabs(g3d.sharpness - old_g.sharpness) > 1e-4);
    if (up_changed) {
        nvn::set_dlss_live(g3d.upscaler.c_str(), g3d.fsr_version.c_str(), static_cast<float>(g3d.sharpness));
        std::string err;
        char sbuf[32];
        std::snprintf(sbuf, sizeof sbuf, "%.2f", std::clamp(g3d.sharpness, 0.0, 1.0));
        const bool ok1 = config_write("dlss", "upscaler", g3d.upscaler, &err);
        const bool ok2 = config_write("dlss", "sharpness", sbuf, &err);
        const bool ok3 = config_write("upscale", "fsr_version", g3d.fsr_version, &err);
        const bool ok4 = config_write("upscale", "fsr_follow_size", (!g3d.fsr_version.empty() && g3d.fsr_version[0] == '4') ? "1" : "0", &err);
        if (!ok1 || !ok2 || !ok3 || !ok4) NXLOG_W(kTag, "cannot persist 3D upscaler settings to nxrt.ini: %s", err.c_str());
    }
    if (g3d.dlss_preset != old_g.dlss_preset) {   // the DLSS model: live, and kept for the next start
        extras::set_dlss_preset_live(g3d.dlss_preset.c_str());
        std::string err;
        if (!config_write("dlss", "preset", g3d.dlss_preset, &err)) NXLOG_W(kTag, "cannot persist [dlss] preset to nxrt.ini: %s", err.c_str());
    }
    if (g3d.capture != old_g.capture) {   // [nvn] mirror_capture: decided at start-up (the copies, the frame lead), so it waits for a restart
        std::string err;
        if (!config_write("nvn", "mirror_capture", g3d.capture, &err)) NXLOG_W(kTag, "cannot persist [nvn] mirror_capture to nxrt.ini: %s", err.c_str());
        else NXLOG_I(kTag, "mirror_capture -> %s saved to nxrt.ini (takes effect on restart)", g3d.capture.c_str());
    }
    if (g3d.out_height != old_g.out_height && g3d.out_height >= 720) {   // the output resolution: what the game is told its TV is
        char res[32];
        std::snprintf(res, sizeof res, "%dx%d", g3d.out_height * 16 / 9, g3d.out_height);
        std::string err;
        if (!config_write("am", "resolution", res, &err)) NXLOG_W(kTag, "cannot persist [am] resolution to nxrt.ini: %s", err.c_str());
        else NXLOG_I(kTag, "output resolution -> %s saved to nxrt.ini (takes effect on restart)", res);
    }
    // The mode is stored as intent: the runtime derives the scales and the lock from it and [am] resolution at start-up
    // (nxrt/render_quality.h), so changing the output resolution here or in the launcher cannot leave the scales of another output.
    if (g3d.res_preset != old_g.res_preset) {
        const int idx = std::clamp(g3d.res_preset, 0, kRenderQualityCount - 1);
        const RenderQuality& rq = kRenderQuality[idx];
        std::string err;
        // Preserve an explicit [xc2] native_res > 0 for fixed render quality presets: xc2_boot scales native_res=2
        // from [upscale] render_quality and disables NVN attachment scaling via game_handles_render_quality, so the
        // two never multiply (issue #94). Only "console" (share <= 0, dynamic resolution) requires native_res = 0.
        bool ok = config_write("upscale", "render_quality", rq.key, &err);
        if (ok && rq.share <= 0.0 && config_get_int("xc2", "native_res", 0) > 0) {
            ok = config_write("xc2", "native_res", "0", &err) && config_write("xc2", "res_class", "0", &err);
        }
        if (!ok) NXLOG_W(kTag, "cannot persist the 3D render quality to nxrt.ini: %s", err.c_str());
        else NXLOG_I(kTag, "3D render quality -> %s saved to nxrt.ini (takes effect on restart)", rq.key);
    }
}

void apply_unfocused_mute() {
    G& st = g();
    audio::set_muted(st.window && st.settings.audio.mute_unfocused && !platform::focused());
}

void apply_frame_cap(const Settings& s) {
    G& st = g();
    double cap = s.display.fps_cap;
    if (cap < 0) {
        // fps_limit = auto: the display's refresh rate (no limit while it is unknown, e.g. headless runs)
        const float hz = st.window ? platform::display_refresh_hz() : 0.f;
        cap = hz > 0 ? static_cast<double>(hz) : 0;
    }
    if (s.display.vsync && s.display.fps_cap == 0) {
        cap = 0;     // uncapped with vsync: mailbox presents, no limiter
    } else if (s.display.vsync) {
        // The display paces; a cap only matters when it is below the refresh rate. With no cap below the refresh rate, the frame limiter still keeps just
        // under it (S21, [latency] vsync_cap): the game otherwise renders as fast as the swapchain takes frames, the queue behind the display
        // fills (three images) and every picture is shown one to two frames after it was simulated; just under the refresh rate the queue stays empty.
        const float hz = st.window ? platform::display_refresh_hz() : 0.f;
        static const bool vsync_cap = config_get_bool("latency", "vsync_cap", true);
        if (cap <= 0 || (hz > 0 && cap >= hz)) cap = (vsync_cap && hz > 0) ? static_cast<double>(hz) * 0.9995 : 0;
    }
    st.limiter.set_target_fps(cap);
    extras::lowlat_note_cap(cap);     // Reflex / Anti-Lag pace to the same cap (remembered for a renderer that is created later)
}

// ---- hooks ----------------------------------------------------------------------------------------------------------------------------
void pre_frame(void*) {
    // On the presenting thread, before the frame: the one place a present-mode change cannot collide with a frame in flight.
    G& st = g();
    const int want = st.want_present.exchange(-1);
    if (st.renderer && want >= 0) st.renderer->set_present_mode(static_cast<gfx::PresentMode>(want));
    if (st.overlay) st.overlay->apply_present_pending();
}

bool key_sink(const char* name, bool down) {
    InputId id;
    static float s_mx = 0, s_my = 0;     // the last scripted pointer position: a scripted click lands there, as a real one would
    // Scripted pointer: `keydown mpos <x> <y>` puts the pointer at window pixel (x, y) the way a real motion event does (S28 tests).
    if (name && std::strncmp(name, "mpos ", 5) == 0) {
        if (!down) return true;
        float px = 0, py = 0;
        if (std::sscanf(name + 5, "%f %f", &px, &py) != 2) return false;
        s_mx = px;
        s_my = py;
        input::Event m;
        m.type = input::EventType::MouseMotion;
        m.u.motion.x = px;
        m.u.motion.y = py;
        input::queue_push(m);
        return true;
    }
    // Scripted virtual gamepad (S35 tests): `keydown vpad attach xbox`, `keydown vpad axis ry 0.8`, `keydown vpad button south 1`, `keydown vpad gyro 0 1 0` ...
    if (name && std::strncmp(name, "vpad ", 5) == 0) return !down || vpad_command(name + 5);
    if (!name || !parse_input(name, &id)) return false;
    input::Event e;
    e.timestamp_ns = 0;
    switch (id.kind) {
        case InKind::Key:
            e.type = down ? input::EventType::KeyDown : input::EventType::KeyUp;
            e.u.key.scancode = id.code;
            e.u.key.keycode = static_cast<int32_t>(SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(id.code), SDL_KMOD_NONE, false));
            break;
        case InKind::Mouse:
            e.type = down ? input::EventType::MouseButtonDown : input::EventType::MouseButtonUp;
            e.u.button.button = static_cast<uint8_t>(id.code);
            e.u.button.clicks = 1;
            e.u.button.x = s_mx;
            e.u.button.y = s_my;
            break;
        case InKind::Wheel:
            if (!down) return true;      // a notch has no release
            e.type = input::EventType::MouseWheel;
            e.u.wheel.dy = static_cast<float>(id.code);
            break;
        default: return false;
    }
    input::queue_push(e);
    return true;
}

// @shot in a script: the renderer writes the next presented frame as a PNG; the call returns when the file exists (or after 5 s).
bool screenshot_provider(const char* path, void*) {
    G& st = g();
    if (!st.renderer || !path) return false;
    DeleteFileA(path);
    if (std::strstr(path, "nvntrace_")) nvn::backend_trace_next_frame();   // `@shot nvntrace_<name>`: also trace the frame that is captured
    // The caller is the game's thread at a frame boundary, and the capture happens at the render thread's next end of frame, which
    // needs the game's next frame: waiting here for the file held every shot for the 5 s timeout. Wait only for an earlier request to
    // be taken (one capture path at a time), then return; the file is written on a worker and the test runner waits for it at exit.
    for (int i = 0; i < 200 && st.renderer->capture_pending(); ++i) Sleep(5);
    st.renderer->request_screenshot(path);
    return true;
}

void vib_sink(uint32_t slot, int hand, const hid::VibrationValue& v, void*) {
    if (slot != 0 || hand < 0 || hand > 1) return;
    G& st = g();
    st.vib[hand][0].store(v.amp_low);
    st.vib[hand][1].store(v.amp_high);
}

}  // namespace

[[noreturn]] void exit_process(int code, const char* verdict, const std::string& message) {
    G& st = g();
    NXLOG_I(kTag, "ending the process (%s): %s", verdict, message.c_str());
    diag_set_exiting();   // the watchdog stops looking for a stall: the frames are meant to stop now
    if (st.dirty) {
        std::string err;
        std::lock_guard<std::mutex> lk(st.mu);
        if (!save_settings(st.path, st.settings, &err)) NXLOG_W(kTag, "cannot save %s: %s", st.path.c_str(), err.c_str());
        st.dirty = false;
    }
    nvn::backend_flush_caches();                    // the pipeline cache; this path does not shut the back end down
    nxrt::log_flush();
    if (tt::enabled() && !tt::finished()) tt::end_run(code, verdict, message);     // writes result.json, then terminates
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
    for (;;) Sleep(1000);
}

namespace {

[[noreturn]] void quit_now(const char* why) { exit_process(0, "pass", why); }

void set_window_mode(const std::string& mode) {
    G& st = g();
    if (!st.window) return;
    const platform::FullscreenMode want = mode == "borderless" ? platform::FullscreenMode::Borderless
                                         : mode == "exclusive" ? platform::FullscreenMode::Exclusive
                                                               : platform::FullscreenMode::Windowed;
    if (platform::fullscreen_mode() == want) return;
    if (!platform::set_fullscreen(want)) {
        NXLOG_W(kTag, "window mode '%s' was refused", mode.c_str());
        // Exclusive fullscreen needs the display to accept a mode change (it does not on a remote or locked session, or with some drivers).
        // The next best thing is borderless at the desktop mode, which always works; the setting stays as the player chose it.
        if (want == platform::FullscreenMode::Exclusive && platform::fullscreen_mode() == platform::FullscreenMode::Windowed) {
            if (platform::set_fullscreen(platform::FullscreenMode::Borderless)) NXLOG_W(kTag, "using borderless fullscreen instead");
            else NXLOG_W(kTag, "borderless fullscreen was refused too; staying windowed");
        }
    } else if (want == platform::FullscreenMode::Windowed &&
               st.settings.display.width >= 320 && st.settings.display.height >= 240) {
        // Restore the player's configured windowed client size (e.g. 2560x1440 on a 4K display) and centre it,
        // since fullscreen startup creates the window at desktop dimensions to avoid a white-box flash.
        platform::set_window_size(st.settings.display.width, st.settings.display.height);
        platform::center_window();
    }
    st.limiter.reset_schedule();
}

void toggle_fullscreen() {
    G& st = g();
    if (!st.window) return;
    std::string target;
    if (platform::fullscreen_mode() == platform::FullscreenMode::Windowed)
        target = st.settings.display.window_mode == "exclusive" ? "exclusive" : "borderless";
    else
        target = "windowed";
    set_window_mode(target);
    std::lock_guard<std::mutex> lk(st.mu);
    st.settings.display.window_mode = window_mode_name(platform::fullscreen_mode());
    st.dirty = true;
}

void toggle_fps_counter();       // defined after the settings page host

// Tells the overlay how the frame rate counter should look (the overlay draws it; nothing happens while it is off).
void apply_fps_counter(const FpsSettings& f) {
    G& st = g();
    if (!st.overlay) return;
    gfx::Overlay::CounterConfig c;
    c.show = f.show;
    c.corner = static_cast<int>(f.corner);
    c.detail = f.detail;
    c.in_screenshots = f.in_screenshots;
    c.lang_zh = (f.menu_lang != "en");
    st.overlay->set_counter(c);
}

// The pointer is over a slider's track in the game's menu (its menu hooks are active this frame): the wheel steps the value.
bool game_menu_over_value() {
    const GamePlay* gp = game_play();
    return gp && gp->menu_active && gp->menu_over_value && gp->menu_active() && gp->menu_over_value();
}

// The Joy-Con 2 mouse is on its surface while the game is in play (menus and the map keep their bindings and the menu pointer).
// [input] jc2_clicks: "both" (default) also hands clicks to the bindings (a real Joy-Con 2 sends its buttons as npad R/ZR),
// "device" sends them to the Joy-Con mouse only.
bool jc2_owns_clicks() {
    G& st = g();
    static const bool device_only = config_get_str("input", "jc2_clicks", "both") == "device";
    return device_only && st.engine && ctx_in_play(st.engine->context()) && hid::jc2_mouse_active();
}

constexpr uint64_t kCursorIdleHideNs = 3000ull * kMs;

void note_mouse_activity(uint64_t now) {
    G& st = g();
    st.last_mouse_activity_ns = now;
    if (st.cursor_auto_hidden && st.window && !platform::relative_mouse()) {
        st.cursor_auto_hidden = false;
        platform::set_cursor_visible(true);
    }
}

bool should_autohide_fullscreen_cursor(uint64_t now, bool overlay_blocks) {
    G& st = g();
    if (!st.window || platform::relative_mouse()) {
        st.was_cursor_autohide_eligible = false;
        st.cursor_auto_hidden = false;
        return false;
    }
    const bool full = platform::fullscreen_mode() != platform::FullscreenMode::Windowed;
    const bool eligible = full && platform::focused() && !platform::minimized() && !overlay_blocks;
    if (!eligible) {
        st.was_cursor_autohide_eligible = false;
        st.cursor_auto_hidden = false;
        return false;
    }
    if (!st.was_cursor_autohide_eligible || st.last_mouse_activity_ns == 0) {
        st.was_cursor_autohide_eligible = true;
        st.last_mouse_activity_ns = now;
        st.cursor_auto_hidden = false;
        return false;
    }
    const bool hide = (now >= st.last_mouse_activity_ns) && (now - st.last_mouse_activity_ns >= kCursorIdleHideNs);
    st.cursor_auto_hidden = hide;
    return hide;
}

void handle_event(const input::Event& e, uint64_t now) {
    G& st = g();
    using input::EventType;
    st.events++;
    if (e.type == EventType::MouseMotion && !(e.flags & input::kFlagRelative)) {
        const float mx = e.u.motion.x;
        const float my = e.u.motion.y;
        if (!st.have_cursor_pos) {
            st.last_cursor_x = mx;
            st.last_cursor_y = my;
            st.have_cursor_pos = true;
        } else if (std::fabs(mx - st.last_cursor_x) >= 1.f || std::fabs(my - st.last_cursor_y) >= 1.f) {
            st.last_cursor_x = mx;
            st.last_cursor_y = my;
            note_mouse_activity(now);
        }
    } else if (e.type == EventType::MouseButtonDown || e.type == EventType::MouseButtonUp || e.type == EventType::MouseWheel) {
        note_mouse_activity(now);
    }
    // rebinding capture comes first: the page asked for the next input
    if (st.capture_state.load() == 1) {
        InputId id;
        bool got = false, cancel = false;
        if (e.type == EventType::KeyDown && !(e.flags & input::kFlagRepeat)) {
            if (e.u.key.scancode == SDL_SCANCODE_ESCAPE) cancel = true;
            else { id = {InKind::Key, e.u.key.scancode}; got = true; }
        } else if (e.type == EventType::MouseButtonDown) {
            id = {InKind::Mouse, e.u.button.button}; got = true;
        } else if (e.type == EventType::MouseWheel && e.u.wheel.dy != 0) {
            id = {InKind::Wheel, e.u.wheel.dy > 0 ? 1 : -1}; got = true;
        }
        if (got) {
            std::lock_guard<std::mutex> lk(st.cap_mu);
            st.cap_id = id;
            st.cap_have = true;
            st.capture_state.store(0);
            return;
        }
        if (cancel) { st.capture_state.store(0); return; }
        if (e.type == EventType::KeyDown || e.type == EventType::KeyUp) return;    // nothing reaches the game while capturing
    }
    // The frame rate counter hotkey: flips the counter and never reaches the game (press, repeat and release).
    if (st.overlay && (e.type == EventType::KeyDown || e.type == EventType::KeyUp) && st.settings.fps.key.kind == InKind::Key &&
        e.u.key.scancode == st.settings.fps.key.code) {
        if (e.type == EventType::KeyDown && !(e.flags & input::kFlagRepeat)) toggle_fps_counter();
        return;
    }
    if (st.overlay && st.overlay->handle_event(e)) return;
    const bool menu_blocked = st.overlay && st.overlay->blocks_game();
    if ((e.type == EventType::KeyDown && !(e.flags & input::kFlagRepeat)) || e.type == EventType::MouseButtonDown || e.type == EventType::MouseWheel)
        st.last_dev = LastDevice::KeyboardMouse;
    switch (e.type) {
        case EventType::Quit:
            if (st.quit_on_close) quit_now("window closed");
            break;
        case EventType::KeyDown:
            if ((e.u.key.scancode == SDL_SCANCODE_F11 && !(e.mods & (SDL_KMOD_CTRL | SDL_KMOD_ALT | SDL_KMOD_SHIFT))) ||
                ((e.u.key.scancode == SDL_SCANCODE_RETURN || e.u.key.scancode == SDL_SCANCODE_KP_ENTER) && (e.mods & SDL_KMOD_ALT))) {
                if (!(e.flags & input::kFlagRepeat)) toggle_fullscreen();
                return;
            }
            if (!(e.flags & input::kFlagRepeat)) mm::pointer().on_other_input();
            if (!(e.flags & input::kFlagRepeat) && !menu_blocked) st.engine->key(e.u.key.scancode, true, now);
            break;
        case EventType::KeyUp:
            st.engine->key(e.u.key.scancode, false, now);
            break;
        case EventType::MouseButtonDown:
            if (e.u.button.button == 1) st.mouse1 = true;
            st.jc2_buttons |= jc2_button_bit(e.u.button.button);
            if (!menu_blocked) mm::pointer().on_button(e.u.button.button, true);
            // While the Joy-Con 2 mouse owns the mouse in play, the left and right buttons belong to the device alone (the middle button stays
            // the mouse-mode key). A release is always passed on: it only matters when the press was seen before the device took over.
            if (!menu_blocked && !(jc2_owns_clicks() && (e.u.button.button == 1 || e.u.button.button == 3))) st.engine->mouse_button(e.u.button.button, true, now);
            break;
        case EventType::MouseButtonUp:
            if (e.u.button.button == 1) st.mouse1 = false;
            st.jc2_buttons &= ~jc2_button_bit(e.u.button.button);
            mm::pointer().on_button(e.u.button.button, false);
            st.engine->mouse_button(e.u.button.button, false, now);
            break;
        case EventType::MouseWheel: {
            if (menu_blocked) break;
            st.jc2_wheel_x += e.u.wheel.dx;
            st.jc2_wheel_y += e.u.wheel.dy;
            if (jc2_owns_clicks()) break;           // the wheel goes to the Joy-Con 2 mouse only
            const float dy = e.u.wheel.dy;
            // the map: the wheel zooms (the L and R buttons), a longer pulse than a button tap so one notch is a visible step
            if (dy != 0 && st.settings.mouse.menu_cursor && st.engine->context() == Ctx::Map && !(st.overlay && st.overlay->visible())) {
                st.engine->pulse(dy > 0 ? Act::BtnR : Act::BtnL, now, 160);
                break;
            }
            // over a slider's track in a menu the wheel steps the value (right is more) instead of moving the selection
            if (dy != 0 && st.settings.mouse.menu_cursor && mm::pointer().enabled() && game_menu_over_value()) {
                st.engine->pulse(dy > 0 ? Act::DpadRight : Act::DpadLeft, now);
                break;
            }
            const int n = std::max(1, static_cast<int>(std::lround(std::fabs(dy))));
            if (dy != 0) for (int i = 0; i < n; ++i) st.engine->wheel(dy > 0 ? 1 : -1, now);
            break;
        }
        case EventType::MouseMotion:
            if (!(e.flags & input::kFlagRelative)) {
                st.mouse_x = e.u.motion.x;
                st.mouse_y = e.u.motion.y;
                if (!menu_blocked) mm::pointer().on_motion(e.u.motion.x, e.u.motion.y);
                // moving the pointer over a menu counts as using the mouse (the prompts switch to keyboard icons); the raw motion of the aim in play does not
                if (!menu_blocked && st.settings.mouse.menu_cursor && mm::pointer().has_moved()) st.last_dev = LastDevice::KeyboardMouse;
            }
            break;
        case EventType::WindowFocusLost:
            audio::set_muted(st.window && st.settings.audio.mute_unfocused);
            st.engine->release_all(now);
            st.mouse1 = false;
            st.jc2_buttons = 0;
            mm::pointer().on_focus_lost();
            break;
        case EventType::WindowFocusGained:
            audio::set_muted(false);
            st.limiter.reset_schedule();
            break;
        case EventType::WindowRestored:
        case EventType::WindowMinimized:
            st.limiter.reset_schedule();
            break;
        case EventType::GamepadAdded:
            NXLOG_I(kTag, "controller %u connected", static_cast<unsigned>(e.u.gamepad.id));
            // a pad that replaces the one in use: the prompts follow its kind at once (without it they change at the first press)
            if (st.last_dev == LastDevice::Pad) st.last_pad_kind = static_cast<int>(input::gamepad_kind_of(e.u.gamepad.id));
            break;
        case EventType::GamepadRemoved:
            NXLOG_I(kTag, "controller %u disconnected", static_cast<unsigned>(e.u.gamepad.id));
            break;
        default: break;
    }
}

// Reads every open pad, merges them (buttons OR, the stick with the larger deflection wins) and publishes the result to the source.
void poll_pads(uint64_t now) {
    G& st = g();
    PadFrame pf;
    input::GamepadInfo infos[8];
    const size_t n = input::gamepad_list(infos, 8);
    bool have_six = false;
    uint16_t lo = 0, hi = 0;
    {
        const float l = std::max(st.vib[0][0].load(), st.vib[1][0].load()), h = std::max(st.vib[0][1].load(), st.vib[1][1].load());
        map_rumble(l, h, st.settings.pad, &lo, &hi);
    }
    const Ctx cx = st.engine ? st.engine->context() : Ctx::Menu;
    const bool look_ctx = cx == Ctx::Play || cx == Ctx::Alt || cx == Ctx::Inspect;      // the right stick is a look only in play
    const bool rumble_changed = lo != st.rumble_lo || hi != st.rumble_hi;
    const bool rumble_refresh = (lo || hi) && now - st.last_rumble_ns > 100 * kMs;
    for (size_t i = 0; i < n; ++i) {
        platform::GamepadRaw raw;
        if (!platform::gamepad_poll(infos[i].id, &raw)) continue;
        hid::NpadInput tmp;
        map_pad(raw, st.settings.pad, &tmp);
        {   // a pressed button or a stick well off centre makes this the pad the prompts follow
            auto off_centre = [](hid::StickValue v) { return static_cast<int64_t>(v.x) * v.x + static_cast<int64_t>(v.y) * v.y > 20000ll * 20000ll; };
            if (tmp.buttons != 0 || off_centre(tmp.left) || off_centre(tmp.right)) {
                st.last_dev = LastDevice::Pad;
                st.last_pad_kind = static_cast<int>(infos[i].kind);
                mm::pointer().on_other_input();
            }
        }
        if (look_ctx) tmp.right = adjust_look_stick(tmp.right, st.settings.pad);
        pf.buttons |= tmp.buttons;
        auto big = [](hid::StickValue a, hid::StickValue b) {
            return static_cast<int64_t>(a.x) * a.x + static_cast<int64_t>(a.y) * a.y >= static_cast<int64_t>(b.x) * b.x + static_cast<int64_t>(b.y) * b.y ? a : b;
        };
        pf.left = big(pf.left, tmp.left);
        pf.right = big(pf.right, tmp.right);
        if (!have_six) {
            float a[3], gy[3];
            if (platform::gamepad_sensors(infos[i].id, a, gy)) {
                map_sensors(a, gy, st.settings.pad, &pf.six);
                have_six = true;
            }
        }
        if (rumble_changed || rumble_refresh) platform::gamepad_rumble(infos[i].id, lo, hi, 250);
    }
    if (rumble_changed || rumble_refresh) {
        st.rumble_lo = lo;
        st.rumble_hi = hi;
        st.last_rumble_ns = now;
    }
    // The on-screen keyboard reads the merged pad before it is held back; what it returns stays away from the game (everything while it is
    // open, then the button that closed it until it is released).
    const uint64_t raw_buttons = pf.buttons;
    if (st.overlay) {
        st.overlay->set_keyboard_abxy(st.settings.pad.label_abxy);
        pf.buttons &= ~st.overlay->set_keyboard_pad(pf.buttons, pf.left.x, pf.left.y);
        // the menu's pad chord and its gamepad navigation; the press that opens or closes it stays away from the game until released
        pf.buttons &= ~st.overlay->set_menu_pad(pf.buttons, pf.left.x, pf.left.y);
    }
    // The settings menu or the on-screen keyboard is open: the pad does not reach the game (the keyboard and mouse are held back the same way in
    // handle_event).
    if (st.overlay && st.overlay->blocks_game()) {
        static uint64_t s_held_logged = 0;   // test runs check this line: a new press while the menu is open is logged once
        if (const uint64_t fresh = raw_buttons & ~s_held_logged; fresh && tt::enabled())
            NXLOG_I(kTag, "menu open: pad buttons 0x%llx held back from the game (sent 0x%llx)", static_cast<unsigned long long>(raw_buttons),
                    static_cast<unsigned long long>(pf.buttons));
        s_held_logged = raw_buttons;
        pf.buttons = 0;
        pf.left = hid::StickValue{};
        pf.right = hid::StickValue{};
    }
    st.source_owner->set_pad(pf);
}

// set while update_hook runs: keyboard_service stays out of it
thread_local bool t_in_update = false;

void apply_pending_settings(uint64_t now) {
    G& st = g();
    std::unique_lock<std::mutex> lk(st.mu);
    if (!st.have_pending) return;
    Settings s = st.pending;
    st.have_pending = false;
    const std::string old_mode = st.settings.display.window_mode;
    const int old_monitor = st.settings.display.monitor;
    const int old_w = st.settings.display.width, old_h = st.settings.display.height;
    const Game3dSettings old_g3d = st.settings.game3d;
    st.settings = s;
    st.settings_epoch++;
    lk.unlock();
    if (st.engine) st.engine->set_settings(s, now);
    publish_live_pad_config(s.pad);
    publish_live_mouse_config(s.mouse);
    apply_frame_cap(s);
    display_tuning(s.display);
    nvn::set_keep_aspect(s.display.keep_aspect);
    apply_game3d_settings(old_g3d, s.game3d);
    if (const GamePlay* gp = game_play(); gp && gp->settings_changed) gp->settings_changed(s);     // the game's display options, lock-on
    extras::set_settings(s.extras);
    extras::hdr_refresh_levels();
    apply_fps_counter(s.fps);
    st.want_present.store(static_cast<int>(present_mode_for(s.display)));
    apply_audio(s.audio);
    if (st.window && s.display.monitor != old_monitor && s.display.monitor > 0) {
        if (platform::move_to_display(s.display.monitor)) st.limiter.reset_schedule();
        apply_frame_cap(s);                        // another display can mean another refresh rate
    }
    if (s.display.window_mode != old_mode) {
        set_window_mode(s.display.window_mode);
        apply_frame_cap(s);                        // another window mode can mean another refresh rate (vsync cap)
    }
    // The windowed size is live while the window is windowed; in fullscreen it is the size the window goes back to (the next start).
    if (st.window && (s.display.width != old_w || s.display.height != old_h) && s.display.width >= 320 && s.display.height >= 240 &&
        platform::fullscreen_mode() == platform::FullscreenMode::Windowed)
        platform::set_window_size(s.display.width, s.display.height);
    std::string err;
    if (!save_settings(st.path, s, &err)) NXLOG_W(kTag, "cannot save %s: %s", st.path.c_str(), err.c_str());
    else NXLOG_I(kTag, "settings applied and saved to %s", st.path.c_str());
}

int update_hook(::Ctx*, uint64_t, void*) {
    G& st = g();
    ft::mark("main.update");
    struct InUpdate {
        InUpdate() { t_in_update = true; }
        ~InUpdate() { t_in_update = false; }
    } in_update;
    // frame pacing (S18): the limiter waits here, before the input is sampled, so the picture shown is the one simulated one render time
    // earlier. The wait for frames still in flight comes first, then the limiter's schedule.
    // All of it runs once per presented frame: the game's main loop runs more iterations than it presents frames (Pokopia ~79 against ~50 a
    // second), and per iteration the limiter took several slots per frame while the in-flight list (one id per sim_begin, one popped per
    // present) filled up until wait_in_flight held the main thread for presents that only come after more main-loop work.
    const bool frame_start = extras::pace_boundary_due();
    if (frame_start) {
        extras::LowLatency* ll = extras::lowlat();
        if (extras::pace_in_main()) {
            if (ll) ll->wait_in_flight();
            // With Reflex on and a frame cap, the Reflex sleep that follows is the limiter (minimum interval between simulation starts)
            if (!(ll && ll->paces_frames())) { NXFT("limiter.main"); st.limiter.begin_frame_nohook(); }
            extras::pace_boundary_done();
        }
        // low latency (S14): the sleep that lets the frame start as late as the GPU allows, then the input sample marker; the events are pumped right after
        if (ll) {
            ll->sim_begin();
            ll->input_sample();
        }
    }
    const uint64_t now = now_ns();
    st.frames++;
    st.update_seq++;

    const bool on_pump_thread = std::this_thread::get_id() == st.pump_thread;
    if (on_pump_thread) g_last_pump_ms.store(SDL_GetTicks(), std::memory_order_relaxed);
    if (!on_pump_thread && !st.warned_thread) {
        st.warned_thread = true;
        st.thread_mismatch.store(true);
        NXLOG_W(kTag, "the frame-boundary hook runs on another thread than the one that started play mode: SDL events are not pumped");
    }
    if (st.window && on_pump_thread) platform::pump_events();

    // settings queued by the page
    apply_pending_settings(now);

    apply_unfocused_mute();

    // events
    input::Event buf[128];
    for (;;) {
        const size_t got = input::queue_drain(buf, 128);
        if (!got) break;
        for (size_t i = 0; i < got; ++i) handle_event(buf[i], now);
    }
    poll_pads(now);
    const GamePlay* gp = game_play();
    if (gp && gp->frame_begin) gp->frame_begin();
    if (st.engine) st.engine->frame_tick();

    // what the game is doing: the context, the cycle hint, the pointer
    const uint64_t pi = st.last_pi_ns.load();
    // "In game" = the game called ProcessInput during one of the last few frame boundaries. Counted in frames, not host time: a slow frame (the game
    // can take 100 ms or more per frame under load) must not read as a menu.
    // Also within 100 ms of host time: a catch-up burst of frame boundaries with no input call (1-2 frames about once a second at ~50 fps)
    // dropped the context to the menu and released the mouse for a moment.
    const bool in_game = pi != 0 && (st.frames.load() - st.last_pi_frame.load() <= 4 || pi + 100000000ull >= now);
    // The game's part (play_game.h GamePlay::frame): the gyro's pitch-rate scale for this step, the cycle hint, the context while in game. This
    // step's length is known here, before the game reads the sample (the timestep hook on the same frame update runs first: hooks of one address run
    // in registration order).
    // One of the game's menus over play (bag, shop, choice box: GamePlay::menu_in_game): the frame is a menu frame for the context, the
    // pointer and the mouse capture (with the menu pointer off, play stays play).
    const bool game_menu = gp && gp->menu_in_game && gp->menu_in_game(in_game && st.settings.mouse.menu_cursor) && in_game;
    const bool playing = in_game && !game_menu;
    GameFrame gf;
    if (gp && gp->frame) {
        const double step = timestep::active() ? static_cast<double>(timestep::current_dt()) : static_cast<double>(timestep::kGameStep);
        gp->frame(playing, step, &gf);
    }
    if (st.source_owner) st.source_owner->set_look_scale(gf.look_scale);
    if (gf.hints_valid) st.engine->set_cycle_hint(gf.cycle_index, gf.cycle_mask, now);
    const uint64_t map_acts = st.engine->activations(Act::Map);
    if (in_game && map_acts != st.map_acts_seen) st.map_hint = true;
    st.map_acts_seen = map_acts;
    if (in_game && !st.was_in_game) st.map_hint = false;
    st.was_in_game = in_game;
    Ctx ctx;
    if (!in_game) ctx = st.map_hint ? Ctx::Map : Ctx::Menu;
    else if (game_menu) ctx = Ctx::Menu;
    else ctx = gf.play_ctx;
    st.engine->set_context(ctx, now);

    {   // button prompts: the style that shows, and the icon textures behind it
        PromptInputs pin;
        pin.requested = st.settings.prompts.style;
        pin.label_abxy = st.settings.pad.label_abxy;
        pin.ctx = ctx;
        pin.settings_epoch = st.settings_epoch;
        pin.last = st.last_dev;
        pin.last_pad_kind = st.last_pad_kind;
        input::GamepadInfo pads[1];
        pin.connected_pad_kind = input::gamepad_list(pads, 1) > 0 ? static_cast<int>(pads[0].kind) : -1;
        prompts_update(st.settings, pin);
    }

    const bool overlay_visible = st.overlay && st.overlay->blocks_game();
    if (overlay_visible && !st.overlay_was_visible) st.engine->release_all(now);
    st.overlay_was_visible = overlay_visible;

    if (st.window) {
        platform::set_text_input(st.overlay && st.overlay->keyboard_open());   // typed text for the on-screen keyboard only
        const bool want = st.capture_mouse && platform::focused() && !platform::minimized() && !overlay_visible && playing &&
                          st.capture_state.load() == 0 && !hid::jc2_mouse_wants_pointer() &&
                          !g_external_mouse.load(std::memory_order_relaxed);
        if (want != platform::relative_mouse()) {
            platform::set_relative_mouse(want);
            if (want) input::mouse_delta_take();      // nothing that piled up before the capture
        }
    }
    {   // the map: dragging with the left button rotates it (the left stick), with the middle button moves it (the right stick); the speed is the pointer's speed
        static float px = 0, py = 0;
        static uint64_t last_ns = 0;
        static bool have = false;
        float lx = 0, ly = 0, rx = 0, ry = 0;
        mm::Pointer& pp = mm::pointer();
        const bool on_map = st.window && ctx == Ctx::Map && st.settings.mouse.menu_cursor && pp.enabled() && (platform::focused() || tt::enabled()) && !overlay_visible;
        if (on_map && have && (pp.held() || pp.mid_held())) {
            int ww = 1, wh = 1;
            platform::pixel_size(&ww, &wh);
            // the stick follows the pointer's speed: a full stick is a drag of 0.8 window heights per second, whatever the frame rate
            const float dt = std::max(1e-3f, static_cast<float>(now - last_ns) / 1e9f);
            const float k = 1.f / (0.8f * static_cast<float>(std::max(1, wh)) * dt);
            const float dx = std::max(-1.f, std::min(1.f, (pp.x() - px) * k)), dy = std::max(-1.f, std::min(1.f, (pp.y() - py) * k));
            if (pp.held()) { lx = dx; ly = -dy; }
            else { rx = dx; ry = -dy; }
        }
        px = pp.x();
        py = pp.y();
        last_ns = now;
        have = true;
        st.engine->set_aux_sticks(lx, ly, rx, ry);
    }
    {   // the mouse as a pointer in the menus (S28): the cursor, and the snapshot the widget hooks read this frame
        mm::Pointer& ptr = mm::pointer();
        ptr.set_enabled(st.settings.mouse.menu_cursor);
        // (a scripted run has no focus: the test tooling drives the pointer with synthetic events)
        const bool menu_up = st.window && !playing && ctx == Ctx::Menu && (platform::focused() || tt::enabled()) && !platform::minimized() && !overlay_visible;
        if (tt::enabled()) {   // test runs: why the menu pointer is or is not up, on each change
            static int last = -1;
            const int sig = (st.window ? 1 : 0) | (playing ? 2 : 0) | (static_cast<int>(ctx) << 2) | (overlay_visible ? 256 : 0);
            if (sig != last && (last = sig, true))
                NXLOG_I("play", "menu pointer gate: window %d playing %d ctx %d (menu %d) minimized %d overlay %d -> %d", st.window ? 1 : 0, playing ? 1 : 0,
                        static_cast<int>(ctx), static_cast<int>(Ctx::Menu), platform::minimized() ? 1 : 0, overlay_visible ? 1 : 0, menu_up ? 1 : 0);
        }
        mm::Geometry geo;
        platform::pixel_size(&geo.win_w, &geo.win_h);
        const DisplayLive dl = gp && gp->display_live ? gp->display_live() : DisplayLive{};
        geo.render_w = dl.render_w;
        geo.render_h = dl.render_h;
        geo.display_aspect = dl.aspect > 0 ? dl.aspect : 0;
        geo.keep_aspect = st.settings.display.keep_aspect;
        if (const Act a = gp && gp->menu_frame ? gp->menu_frame(menu_up, geo) : Act::None; a != Act::None) st.engine->pulse(a, now);
        if (st.overlay && st.pointer_draw) st.overlay->set_pointer(menu_up && ptr.enabled() && ptr.cursor_visible(menu_up), ptr.x(), ptr.y(), true);
        // In play with the Joy-Con 2 mouse on its surface the game draws its own cursor: the OS cursor is hidden (unless the overlay is up).
        const bool game_cursor = playing && !overlay_visible && hid::jc2_mouse_active();
        const bool auto_hide = should_autohide_fullscreen_cursor(now, overlay_visible);
        if (st.window && !platform::relative_mouse())
            platform::set_cursor_visible(!auto_hide && !game_cursor && (overlay_visible || !(ptr.enabled() && menu_up) || ptr.cursor_visible(menu_up)));
        if (st.window) {   // the Joy-Con 2 mouse (hid): the pointer on the picture, the buttons and the wheel
            hid::HostMouse hm;
            bool inside = false;
            mm::window_to_picture(geo, st.mouse_x, st.mouse_y, &hm.u, &hm.v, &inside);
            hm.buttons = overlay_visible ? 0u : st.jc2_buttons;
            hm.wheel_x = st.jc2_wheel_x;
            hm.wheel_y = st.jc2_wheel_y;
            hm.focused = (platform::focused() || tt::enabled()) && !platform::minimized() && !overlay_visible;
            hm.captured = platform::relative_mouse();
            hid::jc2_mouse_feed(hm);
            st.jc2_wheel_x = st.jc2_wheel_y = 0;
        }
    }
    if (!playing) input::mouse_delta_take();         // aim only counts what arrives while the game is playing (not under one of its menus)

    if (st.log_status && now - st.last_status_ns > 1000 * kMs) {
        st.last_status_ns = now;
        GameStatus gs;
        if (gp && gp->status) gp->status(&gs);
        char game_part[512] = "";
        if (gp && gp->status_text) gp->status_text(game_part, sizeof game_part);
        NXLOG_I(kTag, "status: ctx=%s in_game=%d aim=%s frames=%llu process_input=%llu aim_frames=%llu events=%llu pads=%zu captured=%d%s", ctx_name(ctx),
                in_game ? 1 : 0, gs.aim, static_cast<unsigned long long>(st.frames.load()),
                static_cast<unsigned long long>(st.pi_calls.load()), static_cast<unsigned long long>(st.aim_frames.load()),
                static_cast<unsigned long long>(st.events.load()), input::gamepad_count(), platform::relative_mouse() ? 1 : 0, game_part);
    }
    if (st.selftest.active()) st.selftest.tick(st.renderer, &st.limiter, selftest_host());
    return 0;
}

// The software keyboard applet blocks the game: the thread that asked waits for the answer, the main thread soon waits too, and no frame is
// presented (Pokopia's name entry: the freeze watchdog fired 10 s after the question, frame count still). update_hook, which pumps the window
// and feeds the overlay, does not run then. keyboard_service does that part on the window's thread while it is blocked in the kernel
// (kernel::set_blocked_hook) or about to make an SVC (busy_pump), every 8 ms at most: the window's events and the pads go to the overlay's
// on-screen keyboard, the mouse is free, typing is on, and the game's last frame is presented again so the keyboard is drawn. Nothing
// happens while no question is open, or from inside update_hook.
void keyboard_service() {
    static thread_local uint64_t last_ms = 0;
    const uint64_t t = SDL_GetTicks();
    if (t - last_ms < 8 || t_in_update) return;
    last_ms = t;
    G& st = g();
    if (!st.started) return;
    const bool swkbd = st.overlay && st.overlay->keyboard_open();
    if (!swkbd && st.hook_update != 0) return;
    const uint64_t now = now_ns();
    platform::pump_events();          // also headless (scripted runs): SDL's virtual pads update here
    if (st.window && st.quit_on_close && platform::quit_requested()) quit_now("exit requested");
    if (st.window) g_last_pump_ms.store(t, std::memory_order_relaxed);
    apply_pending_settings(now);
    apply_unfocused_mute();
    if (st.window) {
        if (swkbd) {
            platform::set_text_input(true);
            if (platform::relative_mouse()) platform::set_relative_mouse(false);
            platform::set_cursor_visible(true);
        } else if (st.overlay) {
            platform::set_text_input(false);
            if (st.overlay->blocks_game()) {
                if (platform::relative_mouse()) platform::set_relative_mouse(false);
                platform::set_cursor_visible(true);
            }
        }
    }
    input::Event buf[64];
    for (;;) {
        const size_t got = input::queue_drain(buf, 64);
        if (!got) break;
        for (size_t i = 0; i < got; ++i) handle_event(buf[i], now);
    }
    if (st.window && !platform::relative_mouse()) {
        const bool overlay_blocks = swkbd || (st.overlay && st.overlay->blocks_game());
        platform::set_cursor_visible(!should_autohide_fullscreen_cursor(now, overlay_blocks));
    }
    if (swkbd) {
        poll_pads(now);
        nvn::present_last_frame_again();
    } else if (st.ext_pad_active.load(std::memory_order_relaxed)) {
        // External input source (e.g. xc2_boot's LiveInput): feed menu gamepad navigation (- + + chord, A/B/D-pad)
        // and forward game vibration to connected SDL pads safely on the main thread.
        if (st.overlay) {
            st.overlay->set_keyboard_abxy(st.settings.pad.label_abxy);
            const uint64_t eb = st.ext_pad_buttons.load(std::memory_order_relaxed);
            const int elx = st.ext_pad_lx.load(std::memory_order_relaxed);
            const int ely = st.ext_pad_ly.load(std::memory_order_relaxed);
            const uint64_t mask = st.overlay->set_menu_pad(eb, elx, ely);
            st.ext_menu_mask.store(mask, std::memory_order_relaxed);
        }
        uint16_t lo = 0, hi = 0;
        {
            const float l = std::max(st.vib[0][0].load(), st.vib[1][0].load());
            const float h = std::max(st.vib[0][1].load(), st.vib[1][1].load());
            map_rumble(l, h, st.settings.pad, &lo, &hi);
        }
        const bool rumble_changed = lo != st.rumble_lo || hi != st.rumble_hi;
        const bool rumble_refresh = (lo || hi) && now - st.last_rumble_ns > 100 * kMs;
        if (rumble_changed || rumble_refresh) {
            input::GamepadInfo infos[8];
            const size_t n = input::gamepad_list(infos, 8);
            for (size_t i = 0; i < n; ++i) platform::gamepad_rumble(infos[i].id, lo, hi, 250);
            st.rumble_lo = lo;
            st.rumble_hi = hi;
            st.last_rumble_ns = now;
        }
    }
}

// ---- what the game module gets from play mode (play_game.h GameHost) ----------------------------------------------------------------------
class GameHostImpl : public GameHost {
public:
    const Settings& settings() const override { return g().settings; }
    bool action_active(Act a, uint64_t now_ns) const override { return g().engine && g().engine->act_active(a, now_ns); }
    uint64_t action_activations(Act a) const override { return g().engine ? g().engine->activations(a) : 0; }
    void right_stick(float* x, float* y) const override {
        *x = *y = 0;
        if (g().source_owner) g().source_owner->right_stick(x, y);
    }
    bool pad_look(bool gyro_enabled) const override { return g().source_owner && g().source_owner->pad_look(gyro_enabled); }
    bool has_input() const override { return g().source_owner != nullptr; }
    void input_tick(double* dx, double* dy) override {
        G& st = g();
        const uint64_t now = now_ns();
        st.last_pi_ns.store(now);
        st.last_pi_frame.store(st.frames.load());
        st.pi_calls++;
        *dx = *dy = 0;
        if (st.update_seq != st.aim_seq) {                // the first input tick since the frame boundary takes the mouse counts
            st.aim_seq = st.update_seq;
            // Late sample: the frame boundary pumped the events at the start of the frame update; the tick that reaches the input function is a little
            // later (the whole update body runs first). Pump again so the counts include whatever the mouse did in between. Main thread only, like the
            // pump in the frame-boundary hook; the key and button events it queues are handled at the next boundary.
            if (st.late_sample && st.window && std::this_thread::get_id() == st.pump_thread) platform::pump_events();
            if (extras::LowLatency* ll = extras::lowlat()) ll->input_sample();
            const input::MouseDelta d = input::mouse_delta_take();
            *dx = d.dx;
            *dy = d.dy;
        }
    }
    void aim_applied(double dx, double dy) override {
        G& st = g();
        st.aim_frames++;
        st.counts += static_cast<uint64_t>(std::fabs(dx) + std::fabs(dy));
    }
    double pitch_limit_deg() const override { return g().pitch_limit_deg; }
    bool log_status() const override { return g().log_status; }
};
GameHostImpl g_game_host;

// ---- the settings page host -----------------------------------------------------------------------------------------------------------
class Host : public UiHost {
public:
    Settings settings() const override { G& st = g(); std::lock_guard<std::mutex> lk(st.mu); return st.have_pending ? st.pending : st.settings; }
    Settings factory() const override { return default_settings(); }
    void apply(const Settings& s) override { G& st = g(); std::lock_guard<std::mutex> lk(st.mu); st.pending = s; st.have_pending = true; }
    LiveStatus status() const override {
        G& st = g();
        LiveStatus s;
        s.ctx = st.engine ? st.engine->context() : Ctx::Menu;
        GameStatus gs;
        if (const GamePlay* gp = game_play(); gp && gp->status) gp->status(&gs);
        s.aim = gs.aim;
        s.prompt_now = prompts_effective_name();
        if (st.pi_calls.load() == 0) s.aim = "not in game";
        s.mouse_captured = st.window && platform::relative_mouse();
        input::GamepadInfo infos[4];
        s.pads = static_cast<int>(input::gamepad_count());
        if (input::gamepad_list(infos, 4) > 0) {
            s.pad_name = infos[0].name;
            s.pad_gyro = infos[0].gyro;
            s.pad_rumble = infos[0].rumble;
        }
        if (st.renderer) {
            s.present_mode = gfx::present_mode_name(st.renderer->present_mode());
            const uint64_t now = now_ns();
            const uint64_t frames = st.renderer->stats().frames;
            std::lock_guard<std::mutex> lk(st.st_mu);
            if (now - st.fps_last_ns > 1000 * kMs) {
                if (st.fps_last_ns) st.fps = static_cast<double>(frames - st.fps_last_frames) * 1e9 / static_cast<double>(now - st.fps_last_ns);
                st.fps_last_frames = frames;
                st.fps_last_ns = now;
            }
            s.fps = st.fps;
        }
        if (st.window) {
            s.window_mode = window_mode_name(platform::fullscreen_mode());
            s.refresh_hz = platform::display_refresh_hz();
            platform::pixel_size(&s.width, &s.height);
        }
        if (st.ext_pad_active.load(std::memory_order_relaxed)) {
            s.out.buttons = st.ext_pad_buttons.load(std::memory_order_relaxed);
            s.out.lx = static_cast<float>(st.ext_pad_lx.load(std::memory_order_relaxed)) / 32767.0f;
            s.out.ly = static_cast<float>(st.ext_pad_ly.load(std::memory_order_relaxed)) / 32767.0f;
            s.out.rx = static_cast<float>(st.ext_pad_rx.load(std::memory_order_relaxed)) / 32767.0f;
            s.out.ry = static_cast<float>(st.ext_pad_ry.load(std::memory_order_relaxed)) / 32767.0f;
        } else {
            s.out = st.engine->output(now_ns());
        }
        s.pitch_deg = gs.pitch_rad * 180.0 / kPi;
        s.input_calls = st.pi_calls.load();
        return s;
    }
    std::string settings_path() const override { return g().path; }
    void begin_capture() override {
        G& st = g();
        std::lock_guard<std::mutex> lk(st.cap_mu);
        st.cap_have = false;
        st.capture_state.store(1);
    }
    bool capturing() const override { return g().capture_state.load() == 1; }
    bool capture_result(InputId* out) override {
        G& st = g();
        std::lock_guard<std::mutex> lk(st.cap_mu);
        if (!st.cap_have) return false;
        st.cap_have = false;
        if (out) *out = st.cap_id;
        return true;
    }
    void cancel_capture() override { g().capture_state.store(0); }
    bool reserved_key(const InputId& id) const override {
        G& st = g();
        return st.overlay && id.kind == InKind::Key && id.code == st.overlay->toggle_scancode();
    }
    void request_exit() override {
        diag_set_exiting();
        platform::request_quit();
    }
};
Host g_host;

void toggle_fps_counter() {
    Settings s = g_host.settings();
    s.fps.show = !s.fps.show;
    g_host.apply(s);
}

// The self-test's window onto play mode: the same entry points the settings page uses.
class SelfTestHostImpl : public SelfTestHost {
public:
    bool set(const std::string& key, const std::string& value) override {
        G& st = g();
        Settings s = g_host.settings();
        const double d = std::atof(value.c_str());
        if (key == "vsync") s.display.vsync = d != 0;
        else if (key == "fps_cap" || key == "fps_limit") s.display.fps_cap = static_cast<int>(d);
        else if (key == "window_mode") s.display.window_mode = value;
        else if (key == "yaw_source") s.mouse.yaw_rad = rad_per_count(d);
        else if (key == "pitch_source") s.mouse.pitch_rad = rad_per_count(d);
        else if (key == "invert_y") s.mouse.invert_y = d != 0;
        else if (key == "invert_x") s.mouse.invert_x = d != 0;
        else if (key == "fps") s.fps.show = d != 0;
        else if (key == "fps_in_shots") s.fps.in_screenshots = d != 0;
        else if (key == "fps_corner") {
            if (!parse_corner(value, &s.fps.corner)) return false;
        } else if (key == "fps_detail") s.fps.detail = std::clamp(static_cast<int>(d), 0, 2);
        else if (key == "prompt_style") {
            if (!parse_prompt_style(value, &s.prompts.style)) return false;
        } else return false;
        g_host.apply(s);
        (void)st;
        return true;
    }
    std::string present_mode() override { G& st = g(); return st.renderer ? gfx::present_mode_name(st.renderer->present_mode()) : ""; }
    std::string settings_text() override {
        G& st = g();
        std::string text;
        if (FILE* f = std::fopen(st.path.c_str(), "rb")) {
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
            std::fclose(f);
        }
        return text;
    }
};

SelfTestHost* selftest_host() {
    static SelfTestHostImpl h;
    return &h;
}

// mpr.ini and the environment override the settings file for the display extras. A key that is set there ([hdr] mode in mpr.ini, or NXRT_HDR_MODE)
// replaces the file's value with the one extras::settings_from_config() reads; other keys keep the file's value (the F1 page edits the file).
extras::Settings merge_env(const extras::Settings& from_file) {
    const extras::Settings env = extras::settings_from_config();
    extras::Settings r = from_file;
    auto set = [](const char* name) {
        const char* v = std::getenv(name);
        return v != nullptr && v[0] != '\0';
    };
    if (config_has("hdr", "mode")) r.hdr.mode = env.hdr.mode;
    if (config_has("hdr", "format")) r.hdr.format = env.hdr.format;
    if (config_has("hdr", "peak_nits")) r.hdr.peak_nits = env.hdr.peak_nits;
    if (config_has("hdr", "paper_white_nits")) r.hdr.paper_white_nits = env.hdr.paper_white_nits;
    if (config_has("hdr", "highlight_boost")) r.hdr.highlight_boost = env.hdr.highlight_boost;
    if (config_has("hdr", "ui_white_nits")) r.hdr.ui_white_nits = env.hdr.ui_white_nits;
    if (config_has("hdr", "deband")) r.hdr.deband = env.hdr.deband;
    if (config_has("hdr", "style")) r.hdr.style = env.hdr.style;
    if (config_has("upscale", "mode")) r.upscale.mode = env.upscale.mode;
    if (config_has("upscale", "sharpness")) r.upscale.sharpness = env.upscale.sharpness;
    if (config_has("upscale", "quality")) r.upscale.quality = env.upscale.quality;
    if (config_has("upscale", "scene_scale")) r.upscale.scene_scale = env.upscale.scene_scale;
    if (config_has("upscale", "fsr_dll")) r.upscale.fsr_dll = env.upscale.fsr_dll;
    if (config_has("upscale", "dlss_dll")) r.upscale.dlss_dll = env.upscale.dlss_dll;
    if (config_has("upscale", "xess_dll")) r.upscale.xess_dll = env.upscale.xess_dll;
    if (config_has("upscale", "frame_generation")) r.upscale.frame_generation = env.upscale.frame_generation;
    if (config_has("latency", "mode")) r.latency.mode = env.latency.mode;
    if (config_has("latency", "log_timings")) r.latency.log_timings = env.latency.log_timings;
    // [picture]: a preset or any of the four keys in mpr.ini / the environment replaces the file's picture controls as a whole
    if (config_has("picture", "preset") || config_has("picture", "contrast") || config_has("picture", "saturation") || config_has("picture", "black") ||
        config_has("picture", "gamma"))
        r.picture = env.picture;
    return r;
}

}  // namespace

// ---- public API ---------------------------------------------------------------------------------------------------------------------------
bool load() {
    G& st = g();
    if (st.loaded) return st.load_ok;
    st.loaded = true;
    st.path = config_get_str("play", "controls", "controls.ini");
    std::string err;
    const bool existed = GetFileAttributesA(st.path.c_str()) != INVALID_FILE_ATTRIBUTES;
    st.load_ok = load_settings(st.path, &st.settings, &err);
    if (!existed) {
        std::string e2;
        if (save_settings(st.path, st.settings, &e2)) NXLOG_I(kTag, "wrote the factory settings to %s", st.path.c_str());
    } else if (!st.load_ok) {
        NXLOG_W(kTag, "%s: %s (the factory settings are in use)", st.path.c_str(), err.c_str());
    } else {
        NXLOG_I(kTag, "settings loaded from %s (%zu bindings)", st.path.c_str(), st.settings.bindings.size());
    }
    // An older controls.ini got the newer factory bindings (play_config.cpp, kDefaultAdditions): keep the old file once as .bak, then
    // write the upgraded one. CopyFileA with bFailIfExists = TRUE leaves an existing .bak alone.
    if (existed && st.load_ok && st.settings.migrated) {
        const std::string bak = st.path + ".bak";
        if (CopyFileA(st.path.c_str(), bak.c_str(), TRUE)) NXLOG_I(kTag, "kept the previous %s as %s", st.path.c_str(), bak.c_str());
        std::string e3;
        if (save_settings(st.path, st.settings, &e3))
            NXLOG_I(kTag, "controls.ini updated to defaults revision %d (added: %s)", st.settings.defaults_rev,
                    st.settings.migration_added.empty() ? "nothing new" : st.settings.migration_added.c_str());
        else
            NXLOG_W(kTag, "cannot save the upgraded %s: %s", st.path.c_str(), e3.c_str());
    }
    // Display extras: a value set in mpr.ini or in the environment (NXRT_HDR_MODE ...) overrides the file; the file overrides the built-in defaults.
    st.settings.extras = merge_env(st.settings.extras);
    st.settings.game3d = hydrate_game3d_from_config();
    publish_live_pad_config(st.settings.pad);
    publish_live_mouse_config(st.settings.mouse);
    extras::set_settings(st.settings.extras);
    // S38 GPU paths: the file's choice becomes [nvn] pool_mirror / pass_barriers unless mpr.ini or the environment set them. The replay
    // reads both once, at its first pool and its first submission, after this.
    if (!config_has("nvn", "pool_mirror")) config_set("nvn", "pool_mirror", st.settings.display.fast_gpu_memory ? "on" : "off");
    if (!config_has("nvn", "pass_barriers")) config_set("nvn", "pass_barriers", st.settings.display.fewer_gpu_waits ? "tracked" : "full");
    // Prewarm: the file's choice becomes [nvn] prewarm unless the ini or the environment set it (read once, by the pipeline prewarm).
    if (!config_has("nvn", "prewarm")) config_set("nvn", "prewarm", st.settings.display.shader_prewarm ? "true" : "false");
    // ASTC texture quality: the file's choice becomes [gfx] astc_quality unless the ini or the environment set it (the texture back end reads it at its first ASTC texture).
    if (!config_has("gfx", "astc_quality")) config_set("gfx", "astc_quality", st.settings.display.astc_quality.c_str());
    // Audio: a volume or device set in the ini or the environment wins over the file; the page then shows what is in force.
    if (config_has("audio", "volume")) st.settings.audio.volume = audio::master_volume();
    else audio::set_master_volume(static_cast<float>(st.settings.audio.volume));
    if (config_has("audio", "device")) st.settings.audio.device = config_get_str("audio", "device", "default");
    else config_set("audio", "device", st.settings.audio.device.c_str());
    st.capture_mouse = config_get_bool("play", "capture_mouse", true);
    st.late_sample = config_get_bool("play", "late_sample", true);
    st.quit_on_close = config_get_bool("play", "quit_on_close", true);
    st.log_status = config_get_bool("play", "log_status", false);
    st.pointer_draw = config_get_bool("play", "pointer_draw", false);
    st.pitch_limit_deg = config_get_double("play", "pitch_limit_deg", 0.0);
    st.base = static_cast<uint64_t>(config_get_int("play", "game_base", static_cast<int64_t>(game::info().main_base)));
    apply_frame_cap(st.settings);
    return st.load_ok;
}

DisplayConfig display() {
    G& st = g();
    load();
    DisplayConfig d;
    d.width = st.settings.display.width;
    d.height = st.settings.display.height;
    d.window_mode = st.settings.display.window_mode;
    d.vsync = st.settings.display.vsync;
    d.fps_cap = st.settings.display.fps_cap;
    d.keep_aspect = st.settings.display.keep_aspect;
    d.present_mode = present_mode_for(st.settings.display);
    return d;
}

FrameLimiter* limiter() {
    load();
    return &g().limiter;
}

std::string settings_path() {
    load();
    return g().path;
}

bool start(const StartOptions& opt) {
    G& st = g();
    load();
    if (st.started) return true;
    st.renderer = opt.renderer;
    st.device = opt.device;
    st.window = opt.window && platform::has_window();
    apply_frame_cap(st.settings);                      // the display's refresh rate is known now (vsync cap)
    kernel::sched_set_process_priority(st.window);     // a person is playing: keep the frame out of the background load
    st.pump_thread = std::this_thread::get_id();
    g_pump_tid.store(GetCurrentThreadId(), std::memory_order_relaxed);
    st.engine = std::make_unique<Engine>(st.settings);
    st.engine->set_context(Ctx::Menu, now_ns());
    st.source_owner = std::make_unique<PlaySource>(st.engine.get());
    st.source = std::shared_ptr<hid::InputSource>(st.source_owner.get(), [](hid::InputSource*) {});   // owned by G
    hid::set_input_source(st.source);
    hid::set_vibration_sink(&vib_sink, nullptr);
    tt::set_mouse_sink(&input::mouse_delta_add);
    tt::set_key_sink(&key_sink);
    if (st.renderer) tt::set_screenshot_provider(&screenshot_provider, nullptr);
    if (const std::string script = config_get_str("play", "selftest", ""); !script.empty()) {
        std::string err;
        if (!st.selftest.parse(script, &err)) {
            NXLOG_E(kTag, "[play] selftest: %s", err.c_str());
            nxrt::log_flush();
            TerminateProcess(GetCurrentProcess(), 2);
        }
        NXLOG_I(kTag, "selftest script: %s", script.c_str());
    }

    if (st.renderer) {
        st.limiter.set_pre_frame_hook(&pre_frame, nullptr);
        if (!st.renderer->headless()) st.renderer->set_present_mode(present_mode_for(st.settings.display));
        // [applet] swkbd_prompt = force makes the overlay for the on-screen keyboard even where the host asked for none (headless test runs)
        const bool swkbd_force = config_get_str("applet", "swkbd_prompt", "true") == "force";
        if ((opt.overlay || swkbd_force) && opt.device) {
            gfx::Overlay::Context oc;
            oc.device = opt.device;
            oc.renderer = opt.renderer;
            oc.limiter = &st.limiter;
            oc.has_window = st.window;
            std::string err;
            st.overlay = gfx::Overlay::create(oc, &err);
            if (!st.overlay) {
                NXLOG_W(kTag, "no overlay: %s", err.c_str());
            } else {
                gfx::Overlay* ov = st.overlay.get();
                if (const GamePlay* gp = game_play(); gp && gp->settings_menu) {   // the game's own menu (GamePlay::settings_menu)
                    st.overlay->set_menu([gp, ov] { gp->settings_menu(g_host, *ov); });
                } else {
                    st.overlay->set_menu([ov] { draw_settings_menu(g_host, *ov); });
                }
                st.overlay->install_on_renderer();
                apply_fps_counter(st.settings.fps);
            }
        }
    }
    // The software keyboard asks the player on the overlay's on-screen keyboard while there is a game window ([applet] swkbd_prompt = false: never;
    // force: also without a window, for scripted headless runs).
    {
        const bool force = config_get_str("applet", "swkbd_prompt", "true") == "force";
        services::swkbdui::enable_prompt(st.overlay && (force || (st.window && config_get_bool("applet", "swkbd_prompt", true))));
    }
    // the window keeps answering while the game boots or loads (no frame boundary for seconds), and the on-screen keyboard works while the
    // game waits for it (keyboard_service; also headless with swkbd_prompt = force, which has an overlay and no window)
    if (st.window || st.overlay) svc_set_busy_hook(&busy_pump);
    kernel::set_blocked_hook(GetCurrentThreadId(), &keyboard_service);
    if (st.window && st.settings.display.monitor > 0) platform::move_to_display(st.settings.display.monitor);
    if (st.window && st.settings.display.window_mode != "windowed") set_window_mode(st.settings.display.window_mode);

    const GamePlay* gp = game_play();
    {   // display options: the internal resolution comes from the window as it is now (after the window mode change above)
        int ww = 0, wh = 0;                    // no window (headless, test runs): the configured window size, not a made-up one
        if (st.window) platform::pixel_size(&ww, &wh);
        if (ww <= 0 || wh <= 0) {
            ww = st.settings.display.width > 0 ? st.settings.display.width : 1280;
            wh = st.settings.display.height > 0 ? st.settings.display.height : 720;
        }
        display_tuning(st.settings.display);
        nvn::set_keep_aspect(st.settings.display.keep_aspect);
        if (gp && gp->install_render) gp->install_render(st.base, st.settings.display, ww, wh);   // the game's display options, tone curve
    }
    // the frame boundary, then the game's input, camera, lock-on and menu hooks (hooks of one address run in registration order)
    if (gp && gp->frame_update) st.hook_update = hook_add_entry(st.base + gp->frame_update, &update_hook);
    else NXLOG_W(kTag, "the game (%s) names no frame update: play mode is not pumped", game::info().id);
    if (gp && gp->install) gp->install(st.base, &g_game_host, st.settings);
    nxrt::cheats::init_from_config(st.base);
    st.started = true;
    NXLOG_I(kTag, "play mode started (%s, overlay %s, base 0x%llx, game %s, frame hook %u)", st.window ? "window" : "no window", st.overlay ? "on" : "off",
            static_cast<unsigned long long>(st.base), game::info().id, st.hook_update);
    return true;
}

void stop() {
    G& st = g();
    if (!st.started) return;
    if (st.overlay) {
        st.overlay->set_hud_callback({});
        st.overlay->set_picker({});
    }
    nxrt::cheats::clear();
    vpad_shutdown();      // a virtual pad from a test script does not outlive play mode
    if (const GamePlay* gp = game_play(); gp && gp->remove) gp->remove();
    kernel::set_blocked_hook(0, nullptr);
    if (st.hook_update) hook_remove(st.hook_update);
    st.hook_update = 0;
    hid::set_vibration_sink(nullptr, nullptr);
    if (hid::input_source() == st.source) hid::set_input_source(nullptr);
    if (st.dirty) {
        std::string err;
        std::lock_guard<std::mutex> lk(st.mu);
        save_settings(st.path, st.settings, &err);
        st.dirty = false;
    }
    st.limiter.set_pre_frame_hook(nullptr, nullptr);
    st.started = false;
}

bool overlay_blocks_game() {
    const G& st = g();
    return st.started && st.overlay && st.overlay->blocks_game();
}

void set_hud_callback(std::function<void(const gfx::HudCanvas&)> fn) {
    G& st = g();
    if (st.overlay) st.overlay->set_hud_callback(std::move(fn));
}

void set_picker(gfx::PickerCallbacks cb) {
    G& st = g();
    if (st.overlay) st.overlay->set_picker(std::move(cb));
}

void toggle_picker() {
    G& st = g();
    if (st.overlay) st.overlay->toggle_picker();
}

uint64_t create_hud_texture_bc3(uint32_t w, uint32_t h, const void* bc3_linear, size_t bytes) {
    G& st = g();
    return st.overlay ? st.overlay->create_hud_texture_bc3(w, h, bc3_linear, bytes) : 0;
}

uint64_t create_hud_texture_bc7(uint32_t w, uint32_t h, const void* bc7_linear, size_t bytes) {
    G& st = g();
    return st.overlay ? st.overlay->create_hud_texture_bc7(w, h, bc7_linear, bytes) : 0;
}

void set_external_mouse_capture(bool on) {
    const bool was = g_external_mouse.exchange(on, std::memory_order_relaxed);
    // Hand the pointer back cleanly on the way out: the caller captures it again on its own terms.
    if (was && !on && platform::relative_mouse()) platform::set_relative_mouse(false);
}

void publish_raw_right_stick(float x, float y) {
    G& st = g();
    st.pad_raw_rs_x.store(x, std::memory_order_relaxed);
    st.pad_raw_rs_y.store(y, std::memory_order_relaxed);
}

void publish_raw_left_stick(float x, float y) {
    G& st = g();
    st.pad_raw_ls_x.store(x, std::memory_order_relaxed);
    st.pad_raw_ls_y.store(y, std::memory_order_relaxed);
}

bool binding_output(uint64_t now_ns, uint64_t* buttons, float* lx, float* ly, float* rx, float* ry) {
    const G& st = g();
    if (!st.started || !st.engine) return false;
    const EngineOut o = st.engine->output(now_ns);
    if (buttons) *buttons = o.buttons;
    if (lx) *lx = o.lx;
    if (ly) *ly = o.ly;
    if (rx) *rx = o.rx;
    if (ry) *ry = o.ry;
    return true;
}

LivePadConfig live_pad_config() {
    const G& st = g();
    LivePadConfig c;
    c.label_abxy = st.pad_cfg_label_abxy.load(std::memory_order_relaxed);
    c.stick_deadzone = st.pad_cfg_ls_dz.load(std::memory_order_relaxed);
    c.right_deadzone = st.pad_cfg_rs_dz.load(std::memory_order_relaxed);
    c.stick_max = st.pad_cfg_stick_max.load(std::memory_order_relaxed);
    c.look_sens = st.pad_cfg_look_sens.load(std::memory_order_relaxed);
    c.mouse_stick_scale = st.mouse_cfg_stick_scale.load(std::memory_order_relaxed);
    c.mouse_camera_capture = st.mouse_cfg_camera_capture.load(std::memory_order_relaxed);
    c.ls_center_x = st.pad_cfg_ls_cx.load(std::memory_order_relaxed);
    c.ls_center_y = st.pad_cfg_ls_cy.load(std::memory_order_relaxed);
    c.raw_ls_x = st.pad_raw_ls_x.load(std::memory_order_relaxed);
    c.raw_ls_y = st.pad_raw_ls_y.load(std::memory_order_relaxed);
    c.rs_center_x = st.pad_cfg_rs_cx.load(std::memory_order_relaxed);
    c.rs_center_y = st.pad_cfg_rs_cy.load(std::memory_order_relaxed);
    c.raw_rs_x = st.pad_raw_rs_x.load(std::memory_order_relaxed);
    c.raw_rs_y = st.pad_raw_rs_y.load(std::memory_order_relaxed);
    c.invert_y = st.pad_cfg_invert_y.load(std::memory_order_relaxed);
    c.rumble = st.pad_cfg_rumble.load(std::memory_order_relaxed);
    c.rumble_scale = st.pad_cfg_rumble_scale.load(std::memory_order_relaxed);
    return c;
}

uint64_t note_external_pad(uint64_t buttons, int lx, int ly, int rx, int ry) {
    G& st = g();
    st.ext_pad_buttons.store(buttons, std::memory_order_relaxed);
    st.ext_pad_lx.store(lx, std::memory_order_relaxed);
    st.ext_pad_ly.store(ly, std::memory_order_relaxed);
    st.ext_pad_rx.store(rx, std::memory_order_relaxed);
    st.ext_pad_ry.store(ry, std::memory_order_relaxed);
    st.ext_pad_active.store(true, std::memory_order_release);
    return st.ext_menu_mask.load(std::memory_order_relaxed);
}

Stats stats() {
    G& st = g();
    Stats s;
    s.frames = st.frames.load();
    s.process_input_calls = st.pi_calls.load();
    s.aim_frames = st.aim_frames.load();
    s.events = st.events.load();
    s.mouse_counts_applied = st.counts.load();
    s.pump_thread_mismatch = st.thread_mismatch.load();
    return s;
}

}  // namespace nxrt::play
