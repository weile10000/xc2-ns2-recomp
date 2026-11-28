// SPDX-License-Identifier: GPL-3.0-or-later
// xc2_boot: the first real runtime harness for Xenoblade Chronicles 2 (Switch 2 Edition).
//
// It does what the console's loader does, in the order the FEFW base documents:
//   1. map the four NSOs (rtld, main, subsdk0, sdk) at the bases the lifter baked into the generated code,
//   2. register the lifted module tables with the runtime (each library carries the ABI version it was built with),
//   3. link: the runtime's linker ([boot] link = host) or the game's own rtld (link = rtld),
//   4. adopt the host main thread as the guest main thread,
//   5. enter rtld at offset 0 with x0 = 0, x1 = the main thread handle.
//
// Graphics is off in this build ([boot] nvn = 0): the goal is the first SVC / host_import, not a frame.
//
//   xc2_boot --exefs DIR [--link host|rtld] [--max-seconds N] [--log FILE]
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <string>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/render_quality.h"
#include "nxrt/dispatch.h"
#include "nxrt/fiber.h"
#include "nxrt/frame_limiter.h"
#include "nxrt/gauge.h"
#include "nxrt/hooks.h"
#include "nxrt/input.h"
#include "nxrt/input_source.h"
#include "nxrt/kernel.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"
#include "nxrt/module.h"
#include "nxrt/movie_vdec.h"
#include "nxrt/nxmodule.h"
#include "nxrt/runtime.h"
#include "nxrt/services_api.h"
#include "nxrt/svc.h"
#include "nxrt/gfx.h"
#include "nxrt/gfx_tuning.h"
#include "nxrt/nvn.h"
#include "nxrt/nvn_backend.h"
#include "nxrt/platform.h"
#include "nxrt/play.h"

#include <memory>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "xc2ns2/xc2ns2_game.h"
#include "xc2_modules.h"
#include "xc2_nav_overlay.h"
#include "voice/xc2_voice_overlay.h"

#include "play_types.h"

// ---- the keyboard/mouse scheme the game starts with ---------------------------------------------------------------
// Play mode already feeds the binding engine (src/gfx/play/play_engine.h) from the SDL event pump and merges its
// Switch-pad output into every snapshot, for any game. XC2 had no factory bindings, so the engine had nothing to
// apply and the hardcoded keys in LiveInput below were the only way in. These are the defaults a fresh controls.ini
// is written with, and every one of them can be changed in the F1 page (controls.ini is the file that page edits).
static const char* const kXc2SettingsText = R"INI(
[meta]
defaults_rev = 13

[bindings.all]
# movement: WASD and the arrow keys
W = move_forward
S = move_back
A = move_left
D = move_right
Up = move_forward
Down = move_back
Left = move_left
Right = move_right
# camera by keyboard; the mouse drives the stick separately (see LiveInput)
I = look_up
K = look_down
J = look_left
L = look_right
# attack / confirm (A): E, and the left mouse button the way a PC port does it
E = btn_a
Mouse1 = btn_a
Return = btn_a
# jump / cancel (B)
Space = btn_b
Backspace = btn_b
Escape = btn_b
# lock-on (R): right click. SDL numbers the buttons 1 left, 2 middle, 3 right, so Mouse3 is the right one.
Mouse3 = btn_r
# the face buttons on the number row; Q/R page the arts palette (ZL/ZR)
1 = btn_y
2 = btn_a
3 = btn_b
4 = btn_x
Q = btn_zl
R = btn_zr
# the shoulders
F = btn_l
G = btn_r
# menu (Tab), map / minus (V); Esc stays cancel (B), which is what people expect. M carries the left-stick click.
Tab = btn_plus
V = btn_minus
M = btn_lstick
# the wheel scrolls up/down; Z/C drive d-pad left/right (blade switch, party commands, menus)
WheelUp = dpad_up
WheelDown = dpad_down
Z = dpad_left
C = dpad_right
# stick clicks (the left one sits on M, see above)
Left Ctrl = btn_rstick

[mouse]
# The mouse drives the camera and its buttons are bound above, so it must not also act as a menu pointer: with
# menu_cursor on, the game's touch path takes the left click before the binding sees it, and Mouse1 = btn_a does
# nothing (issue #69). Set it back to 1 to use the mouse as a pointer in menus.
menu_cursor = 0

[display]
# 16x anisotropic filtering keeps angled ground/distant textures sharp at negligible GPU cost (issue #79).
anisotropy = 16
)INI";

// The `all` layer, not `play`: play mode takes the engine's context from the game table's frame callback, and XC2
// declares no frame-step site, so the context never leaves `menu` - a `play` binding would never apply.
//
// controls.ini files written before those bindings existed carry empty [bindings.*] sections, and an empty section
// means "this file is the whole truth", so the factory text above never reaches them. The additions are what play
// mode appends to such a file (defaults_rev 3 -> 4), keeping the old one as .bak.
static const nxrt::play::GameDefaultAddition kXc2Additions[] = {
    {7, nxrt::play::Ctx::All, "W", nxrt::play::Act::MoveForward},
    {7, nxrt::play::Ctx::All, "S", nxrt::play::Act::MoveBack},
    {7, nxrt::play::Ctx::All, "A", nxrt::play::Act::MoveLeft},
    {7, nxrt::play::Ctx::All, "D", nxrt::play::Act::MoveRight},
    {7, nxrt::play::Ctx::All, "Up", nxrt::play::Act::MoveForward},
    {7, nxrt::play::Ctx::All, "Down", nxrt::play::Act::MoveBack},
    {7, nxrt::play::Ctx::All, "Left", nxrt::play::Act::MoveLeft},
    {7, nxrt::play::Ctx::All, "Right", nxrt::play::Act::MoveRight},
    {7, nxrt::play::Ctx::All, "I", nxrt::play::Act::LookUp},
    {7, nxrt::play::Ctx::All, "K", nxrt::play::Act::LookDown},
    {7, nxrt::play::Ctx::All, "J", nxrt::play::Act::LookLeft},
    {7, nxrt::play::Ctx::All, "L", nxrt::play::Act::LookRight},
    {7, nxrt::play::Ctx::All, "E", nxrt::play::Act::BtnA},
    {7, nxrt::play::Ctx::All, "Mouse1", nxrt::play::Act::BtnA},
    {7, nxrt::play::Ctx::All, "Return", nxrt::play::Act::BtnA},
    {7, nxrt::play::Ctx::All, "Space", nxrt::play::Act::BtnB},
    {7, nxrt::play::Ctx::All, "Backspace", nxrt::play::Act::BtnB},
    {7, nxrt::play::Ctx::All, "Escape", nxrt::play::Act::BtnB},
    {7, nxrt::play::Ctx::All, "Mouse3", nxrt::play::Act::BtnR},
    {11, nxrt::play::Ctx::All, "M", nxrt::play::Act::BtnLStick},
    {10, nxrt::play::Ctx::All, "WheelUp", nxrt::play::Act::DpadUp},
    {10, nxrt::play::Ctx::All, "WheelDown", nxrt::play::Act::DpadDown},
    {13, nxrt::play::Ctx::All, "Z", nxrt::play::Act::DpadLeft},
    {13, nxrt::play::Ctx::All, "C", nxrt::play::Act::DpadRight},
    {13, nxrt::play::Ctx::All, "V", nxrt::play::Act::BtnMinus},
    {9, nxrt::play::Ctx::All, "1", nxrt::play::Act::BtnY},
    {9, nxrt::play::Ctx::All, "2", nxrt::play::Act::BtnA},
    {9, nxrt::play::Ctx::All, "3", nxrt::play::Act::BtnB},
    {9, nxrt::play::Ctx::All, "4", nxrt::play::Act::BtnX},
    {7, nxrt::play::Ctx::All, "Q", nxrt::play::Act::BtnZL},
    {7, nxrt::play::Ctx::All, "R", nxrt::play::Act::BtnZR},
    {7, nxrt::play::Ctx::All, "F", nxrt::play::Act::BtnL},
    {7, nxrt::play::Ctx::All, "G", nxrt::play::Act::BtnR},
    {7, nxrt::play::Ctx::All, "Tab", nxrt::play::Act::BtnPlus},
    {7, nxrt::play::Ctx::All, "M", nxrt::play::Act::BtnMinus},
    {8, nxrt::play::Ctx::All, "V", nxrt::play::Act::BtnLStick},
    {7, nxrt::play::Ctx::All, "Left Ctrl", nxrt::play::Act::BtnRStick},
};

// Readable names for the slots the F1 rebind page lists. The settings-file key stays act_info(a).name ("btn_a"); these
// are only what the page shows, so "1 = Y" reads as "Y" and "Mouse3 = R" as "锁定 (R)".
static const nxrt::play::GameActLabel kXc2Labels[] = {
    {nxrt::play::Act::MoveForward, "前进", "移动"},
    {nxrt::play::Act::MoveBack, "后退", "移动"},
    {nxrt::play::Act::MoveLeft, "左移", "移动"},
    {nxrt::play::Act::MoveRight, "右移", "移动"},
    {nxrt::play::Act::LookUp, "视角上", "视角"},
    {nxrt::play::Act::LookDown, "视角下", "视角"},
    {nxrt::play::Act::LookLeft, "视角左", "视角"},
    {nxrt::play::Act::LookRight, "视角右", "视角"},
    {nxrt::play::Act::BtnA, "确认 / 攻击 (A)", "按键"},
    {nxrt::play::Act::BtnB, "取消 / 跳跃 (B)", "按键"},
    {nxrt::play::Act::BtnX, "X", "按键"},
    {nxrt::play::Act::BtnY, "Y", "按键"},
    {nxrt::play::Act::BtnL, "L", "按键"},
    {nxrt::play::Act::BtnR, "锁定 (R)", "按键"},
    {nxrt::play::Act::BtnZL, "ZL (技能盘翻页)", "按键"},
    {nxrt::play::Act::BtnZR, "ZR (技能盘翻页)", "按键"},
    {nxrt::play::Act::BtnPlus, "菜单 (+)", "按键"},
    {nxrt::play::Act::BtnMinus, "地图 (-)", "按键"},
    {nxrt::play::Act::BtnLStick, "左摇杆按下", "按键"},
    {nxrt::play::Act::BtnRStick, "右摇杆按下", "按键"},
    {nxrt::play::Act::DpadUp, "十字键 上", "按键"},
    {nxrt::play::Act::DpadDown, "十字键 下", "按键"},
    {nxrt::play::Act::DpadLeft, "十字键 左", "按键"},
    {nxrt::play::Act::DpadRight, "十字键 右", "按键"},
};

static const nxrt::play::GameVocabulary& xc2_vocabulary() {
    static const nxrt::play::GameVocabulary v = [] {
        nxrt::play::GameVocabulary g;
        g.labels = kXc2Labels;
        g.label_count = static_cast<unsigned>(sizeof kXc2Labels / sizeof kXc2Labels[0]);
        return g;
    }();
    return v;
}
// Settings a later revision changed, for a controls.ini that still holds the old factory value: the mouse pointer
// (rev 12) was on by default until the keyboard/mouse scheme made it take the left click.
static const nxrt::play::GameDefaultChange kXc2Changes[] = {
    {12, "mouse", "menu_cursor", "1", "0"},
    {13, "display", "anisotropy", "0", "16"},
};

static const nxrt::play::GameDefaults& xc2_play_defaults() {
    static const nxrt::play::GameDefaults d = [] {
        nxrt::play::GameDefaults g;
        g.settings_text = kXc2SettingsText;
        g.defaults_rev = 12;
        g.additions = kXc2Additions;
        g.addition_count = static_cast<unsigned>(sizeof kXc2Additions / sizeof kXc2Additions[0]);
        g.vocabulary = &xc2_vocabulary();
        g.changes = kXc2Changes;
        g.change_count = static_cast<unsigned>(sizeof kXc2Changes / sizeof kXc2Changes[0]);
        return g;
    }();
    return d;
}

static const nxrt::game::Info& xc2_game_info() {
    static const nxrt::game::Info info = [] {
        nxrt::game::Info g = nxrt::games::xc2ns2::core_info();
        g.play_defaults = &xc2_play_defaults();
        return g;
    }();
    return info;
}

// The build the log reports. Both build paths fill this in: tools\build\build-boot.ps1 and xc2\release\CMakeLists.txt
// run `git describe --tags --always --dirty`, so a player's log says which build ran (tag, commit, and whether the
// tree was dirty) instead of a literal that goes stale the moment a release ships.
#ifndef XC2_BUILD_STAMP
#define XC2_BUILD_STAMP "dev"
#endif

extern "C" {
void xc2_opus301_install();    // x2_opus301.cpp: hardware Opus -> the sdk's software OpusDecoder (HANDOVER 90)
void xc2_audio301_install();   // x2_audio301.cpp: nn::audio OutputStream -> the runtime's audio sink (HANDOVER 90)
#ifndef XC2_RT_DLL
// The lifted modules live in the player's exe, not in this DLL: see xc2_modules.h.
extern const NxModule nxmod_main, nxmod_sdk, nxmod_subsdk0, nxmod_rtld;
extern const uint32_t nxmod_main_abi, nxmod_sdk_abi, nxmod_subsdk0_abi, nxmod_rtld_abi;
#ifdef XC2_MODULES_7
// The retail 3.0.1 ExeFS carries three more modules (HANDOVER 89): subsdk1 (42 MB, mostly a 34 MB data segment), subsdk2
// (18 MB rodata) and subsdk3. They are linked only when the build passes -Defines XC2_MODULES_7, so the dev/playable exe
// (4 modules) keeps building from the same source.
extern const NxModule nxmod_subsdk1, nxmod_subsdk2, nxmod_subsdk3;
extern const uint32_t nxmod_subsdk1_abi, nxmod_subsdk2_abi, nxmod_subsdk3_abi;
#endif
#endif  // XC2_RT_DLL
}

namespace {

static uint32_t g_main_handle = 0;   // set after main_thread_adopt; the SDK boot-parameter hook needs it

constexpr uint64_t kMainBase = 0x7100000000ull;

// Module bases, filled from the table the exe hands over. File scope because the hook callbacks below are plain
// function pointers and cannot capture anything.
static uint64_t g_main_base = 0, g_sdk_base = 0, g_rtld_base = 0;

// The module table: the exe owns it (it links the lifted libraries) and the DLL is handed it, so a small range
// wrapper lets the loop below read the same in both builds.
struct ModuleTable {
    const ModEntry *first;
    int count;
    const ModEntry *begin() const { return first; }
    const ModEntry *end() const { return first + count; }
};

#ifdef XC2_RT_DLL
static const ModEntry *g_modules = nullptr;
static int g_module_count = 0;
extern "C" __declspec(dllexport) void xc2_rt_set_modules(const ModEntry *mods, int count) {
    g_modules = mods;
    g_module_count = count;
}
static ModuleTable module_table() { return ModuleTable{g_modules, g_module_count}; }
#else
// The console loader's order: rtld first (it is the entry), then main, the subsdks, sdk.
#ifdef XC2_MODULES_7
const ModEntry kModules[] = {
    {"rtld", &nxmod_rtld, &nxmod_rtld_abi},
    {"main", &nxmod_main, &nxmod_main_abi},
    {"subsdk0", &nxmod_subsdk0, &nxmod_subsdk0_abi},
    {"subsdk1", &nxmod_subsdk1, &nxmod_subsdk1_abi},
    {"subsdk2", &nxmod_subsdk2, &nxmod_subsdk2_abi},
    {"subsdk3", &nxmod_subsdk3, &nxmod_subsdk3_abi},
    {"sdk", &nxmod_sdk, &nxmod_sdk_abi},
};
#else
const ModEntry kModules[] = {
    {"rtld", &nxmod_rtld, &nxmod_rtld_abi},
    {"main", &nxmod_main, &nxmod_main_abi},
    {"subsdk0", &nxmod_subsdk0, &nxmod_subsdk0_abi},
    {"sdk", &nxmod_sdk, &nxmod_sdk_abi},
};
#endif
static ModuleTable module_table() { return ModuleTable{kModules, int(sizeof(kModules) / sizeof(kModules[0]))}; }
#endif  // XC2_RT_DLL

[[noreturn]] void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
    nxrt::log_flush();
    std::exit(2);
}

bool file_exists(const std::string &p) {
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

// SDL3: the pad state (and newly plugged pads) only move when its event queue is pumped, which nothing does for this
// game; the live source pumps it below. No SDL header here, only the two calls that are needed.
extern "C" void SDL_UpdateGamepads(void);

// Nothing pumps SDL's queue for this game, so window and pad events would pile up forever. Pump it from the input
// source on a 5 ms rhythm: it keeps the pad state fresh, opens pads that were plugged in, and makes the close button and
// Alt+F4 work. XC2_NO_SDL_PUMP=1 leaves it alone (then only the keyboard is live).
void pump_platform_events(bool any_pad) {
    static const bool off = std::getenv("XC2_NO_SDL_PUMP") != nullptr;
    if (off) return;
    static std::chrono::steady_clock::time_point next{};
    const auto now = std::chrono::steady_clock::now();
    if (now < next) return;
    next = now + (any_pad ? std::chrono::milliseconds(5) : std::chrono::milliseconds(500));   // discovery can be slow
    nxrt::platform::pump_events(false, 0);
    if (nxrt::platform::quit_requested()) {
        NXLOG_I("input", "window close requested: exiting");
        nxrt::log_flush();
        std::_Exit(0);
    }
}

nxrt::gfx::Renderer *g_renderer = nullptr;   // for the F12 screenshot hotkey

// ---- live input: keyboard and the first gamepad ---------------------------------------------------------------
// The base runtime puts its real input into the hid service from play mode, and play mode pumps the SDL queue on the
// game's main thread - a site XC2 never declares ("the game lists no frame-step sites", docs/FORK-PLAN). So the SDL
// queue is never pumped and its input source never updates: the window shows a game that ignores the keyboard.
// The hid service samples an InputSource every 5 ms from its own thread, so this source reads the state itself:
//   keyboard  Win32 GetAsyncKeyState (thread safe, needs no SDL; only while the window has the focus)
//   gamepad   platform::gamepad_poll (SDL's cached pad state, refreshed here with SDL_UpdateGamepads)
// XC2_PRESS presses stay on top of it, so a scripted run and a human can drive one process. XC2_NO_INPUT=1 turns the
// live part off (a scripted run that must not see the keyboard).
void route_record(const nxrt::hid::InputSnapshot &s);   // XC2_ROUTE_OUT (see "replay routes" below)

class LiveInput final : public nxrt::hid::ManualInputSource {
public:
    void poll(nxrt::hid::InputSnapshot &out) override {
        nxrt::hid::ManualInputSource::poll(out);   // scripted presses / connected pads from the harness
        live(out);
        route_record(out);                         // what the game saw, for XC2_ROUTE_OUT
    }

private:
    static bool down(int vk) { return (::GetAsyncKeyState(vk) & 0x8000) != 0; }
    // A hotkey is polled once per presented frame, so a quick tap can fall between two polls (20-30 fps in the port). The low
    // bit is set when the key went down since the previous call and is cleared by it, which catches that tap.
    static bool tapped(int vk) { return (::GetAsyncKeyState(vk) & 0x0001) != 0; }
    void live(nxrt::hid::InputSnapshot &out);

    unsigned shots = 0;
    int f12_left = 0;              // frames still to write from one F12 press (NXRT_SHOT_BURST)
    std::vector<uint8_t> shot_raw_buf;     // NXRT_SHOT_RAW: the last captured frame, RGBA
    uint32_t shot_raw_w = 0, shot_raw_h = 0;
    bool shot_raw_have = false;
    unsigned shot_raw_idx = 0;
    void write_raw_frame();
    bool f12_was_down = false;
    bool f3_was_down = false;
    bool f11_was_down = false;
};

// Half resolution, raw RGBA, header of two uint32 (w, h): a burst that comes out in order and consecutive.
void LiveInput::write_raw_frame() {
    if (shot_raw_w < 2 || shot_raw_h < 2) return;
    // The capture fills the vector and the dimensions together; anything else means a frame was skipped and the two no
    // longer describe each other. Reading it without this check faulted at once (2026-11-21).
    const size_t want = static_cast<size_t>(shot_raw_w) * shot_raw_h * 4;
    if (shot_raw_buf.size() != want) {
        static bool said = false;
        if (!said) {
            said = true;
            NXLOG_E("input", "raw burst: buffer %zu bytes, but %ux%u needs %zu: skipped", shot_raw_buf.size(), shot_raw_w, shot_raw_h, want);
        }
        return;
    }
    const uint32_t w = shot_raw_w / 2, h = shot_raw_h / 2;
    std::vector<uint8_t> half(static_cast<size_t>(w) * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* r0 = shot_raw_buf.data() + static_cast<size_t>(2 * y) * shot_raw_w * 4;
        const uint8_t* r1 = r0 + static_cast<size_t>(shot_raw_w) * 4;
        uint8_t* o = half.data() + static_cast<size_t>(y) * w * 4;
        for (uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 4; ++c)
                o[x * 4 + c] = static_cast<uint8_t>((r0[static_cast<size_t>(2 * x) * 4 + c] + r0[static_cast<size_t>(2 * x + 1) * 4 + c] +
                                                     r1[static_cast<size_t>(2 * x) * 4 + c] + r1[static_cast<size_t>(2 * x + 1) * 4 + c]) / 4);
    }
    CreateDirectoryA("shots_raw", nullptr);
    char path[64];
    std::snprintf(path, sizeof path, "shots_raw\\frame_%04u.bin", shot_raw_idx++);
    if (FILE* f = std::fopen(path, "wb")) {
        const uint32_t head[2] = {w, h};
        std::fwrite(head, sizeof head, 1, f);
        std::fwrite(half.data(), 1, half.size(), f);
        std::fclose(f);
    }
}

void LiveInput::live(nxrt::hid::InputSnapshot &out) {
    using namespace nxrt::hid;
    static const bool off = std::getenv("XC2_NO_INPUT") != nullptr;
    if (off) return;
    int kx = 0, ky = 0, klx = 0, kly = 0;       // keyboard sticks (full deflection)
    uint64_t kb = 0;
    // Keys belong to the window that has the focus. SDL's own focus flag never changes here (nothing pumps its event
    // queue), so ask Win32 for the foreground window: the same answer without needing an event pump.
    const HWND own = static_cast<HWND>(nxrt::platform::native_hwnd());
    const HWND fg = ::GetForegroundWindow();
    const bool focused = own && fg == own;
    const bool menu_blocked = nxrt::play::overlay_blocks_game();
    // Mouse camera: capture the pointer (relative mode, raw and unaccelerated counts) while the game has the focus and no
    // page is open, and turn the counts into right-stick deflection for this frame. The game's own camera curve runs on top,
    // which is why this drives the stick instead of a yaw angle. [xc2] mouse_capture = 0 leaves the pointer to the game.
    static const bool ini_mouse_cam = nxrt::config_get_bool("xc2", "mouse_capture", true);
    const bool mouse_cam = ini_mouse_cam && nxrt::play::live_pad_config().mouse_camera_capture;
    // The gain is the F1 page's [mouse] stick_scale (controls.ini), read live so its slider needs no restart.
    static const bool mouse_invert_y = nxrt::config_get_bool("xc2", "mouse_invert_y", false);
    static bool mouse_held = false;
    const bool want_mouse = mouse_cam && own != nullptr && focused && !menu_blocked;
    if (want_mouse != mouse_held) {
        mouse_held = want_mouse;
        nxrt::play::set_external_mouse_capture(mouse_held);
        nxrt::platform::set_relative_mouse(mouse_held);
        if (mouse_held) nxrt::input::mouse_delta_take();   // drop whatever piled up before the capture
        NXLOG_I("input", "mouse camera %s", mouse_held ? "on (pointer captured)" : "off");
    }
    // one line the first time a mapped key is down, so a hands-on session can be told from "the keyboard is not seen"
    static bool said_focus = false;
    if (!said_focus && own && (down('W') || down(VK_SPACE) || down(VK_RETURN) || down(VK_ESCAPE) || down('A'))) {
        said_focus = true;
        NXLOG_I("input", "keyboard seen: window %p, foreground %p (%s)", static_cast<void *>(own), static_cast<void *>(fg),
                focused ? "keys go to the game" : "the game does not have the focus: keys are ignored");
    }
    if (focused && !menu_blocked) {
        // The keyboard and mouse buttons are not read here: play mode's binding engine owns them (kXc2SettingsText at the
        // top of this file, rebindable in the F1 page). PlaySource would merge the engine's output itself, but this source
        // replaces PlaySource on purpose (see the set_input_source call below), so ask the engine directly.
        uint64_t eb = 0;
        float blx = 0, bly = 0, brx = 0, bry = 0;
        if (nxrt::play::binding_output(nxrt::FrameLimiter::now_ns(), &eb, &blx, &bly, &brx, &bry)) {
            kb |= eb;
            if (blx != 0.0f || bly != 0.0f) {
                kx += static_cast<int>(std::lround(blx * 32767.0f));
                ky += static_cast<int>(std::lround(bly * 32767.0f));
            }
            if (brx != 0.0f || bry != 0.0f) {
                klx += static_cast<int>(std::lround(brx * 32767.0f));
                kly += static_cast<int>(std::lround(bry * 32767.0f));
            }
        }
        if (mouse_held) {
            // This source is polled every 5 ms but the game samples the pad once a frame (~16.7 ms), so the counts have to
            // outlive the tick that read them: summing them into a decaying accumulator keeps roughly a frame of motion
            // instead of handing the game only the last 5 ms of it (which is what made the camera crawl).
            static double acc_x = 0.0, acc_y = 0.0;
            const nxrt::input::MouseDelta d = nxrt::input::mouse_delta_take();
            acc_x += d.dx;
            acc_y += d.dy;
            const double full = (std::max)(1.0, static_cast<double>(nxrt::play::live_pad_config().mouse_stick_scale));
            if (acc_x != 0.0 || acc_y != 0.0) {
                const double dy = mouse_invert_y ? -acc_y : acc_y;
                klx += static_cast<int>(std::lround(std::clamp(acc_x / full, -1.0, 1.0) * 32767.0));
                kly += static_cast<int>(std::lround(std::clamp(-dy / full, -1.0, 1.0) * 32767.0));
            }
            acc_x *= 0.55;      // ~10 ms of memory: a frame's worth at 60 fps, and no leftover after a stop
            acc_y *= 0.55;
        }
    }
    if (focused) {
        // F12: write the next presented frame as shots_manual/shot_NNNN.png (RGBA8). Works any time, also mid-cutscene.
        // NXRT_SHOT_BURST=N makes one press take the next N frames: a flicker that lasts a few frames cannot be caught by
        // hand, and a run of consecutive PNGs shows whether a part vanishes, moves or tears.
        static const int shot_burst = [] {
            const char* s = std::getenv("NXRT_SHOT_BURST");
            const int n = s ? std::atoi(s) : 1;
            return n > 0 ? n : 1;
        }();
        // NXRT_SHOT_RAW=1: the burst writes half-resolution raw RGBA instead of PNG. A 1440p PNG takes 200-300 ms to encode
        // and the capture is one slot, so at 60 fps only every tenth frame of a burst survives; raw frames come out in order
        // and consecutive, which is what a flicker that alternates per frame needs.
        static const bool shot_raw = std::getenv("NXRT_SHOT_RAW") != nullptr;
        const bool f12 = down(VK_F12) || tapped(VK_F12);
        if (f12 && !f12_was_down) {
            f12_left = shot_burst;
            shot_raw_have = false;
            shot_raw_idx = 0;
            NXLOG_I("input", "F12: %d frame(s) into %s", shot_burst, shot_raw ? "shots_raw" : "shots_manual");
        }
        f12_was_down = f12;
        const bool f3 = down(VK_F3) || tapped(VK_F3);
        if (f3 && !f3_was_down) xc2::nav::toggle();
        f3_was_down = f3;
        static bool f4_was_down = false;
        const bool f4 = down(VK_F4) || tapped(VK_F4);
        if (f4 && !f4_was_down) nxrt::play::toggle_picker();
        f4_was_down = f4;
        static bool f5_was_down = false;
        const bool f5 = down(VK_F5) || tapped(VK_F5);
        if (f5 && !f5_was_down) {
            if (down(VK_CONTROL))    xc2::voice::toggle_log();
            else if (down(VK_SHIFT)) xc2::voice::cycle_density();
            else                     xc2::voice::toggle();
        }
        f5_was_down = f5;
        if (xc2::voice::debug_enabled()) {
            static bool f6_was_down = false;
            const bool f6 = down(VK_F6) || tapped(VK_F6);
            if (f6 && !f6_was_down) xc2::voice::trigger_mock(0);
            f6_was_down = f6;
            static bool f7_was_down = false;
            const bool f7 = down(VK_F7) || tapped(VK_F7);
            if (f7 && !f7_was_down) xc2::voice::trigger_mock(1);
            f7_was_down = f7;
            static bool f8_was_down = false;
            const bool f8 = down(VK_F8) || tapped(VK_F8);
            if (f8 && !f8_was_down) xc2::voice::toggle_mock_combat_mode();
            f8_was_down = f8;
        }
        if (shot_raw) {
            if (f12_left > 0 && shot_raw_have) {
                write_raw_frame();     // the buffer filled at the end of the previous frame
                --f12_left;
            }
            if (f12_left > 0 && g_renderer) {
                g_renderer->capture_next_frame(&shot_raw_buf, &shot_raw_w, &shot_raw_h);
                shot_raw_have = true;
                if (f12_left == 1) NXLOG_I("input", "F12: %d raw frame(s) written", shot_raw_idx);
            }
        } else if (f12_left > 0 && g_renderer) {
            char path[64];
            std::snprintf(path, sizeof path, "shots_manual\\shot_%04u.png", ++shots);
            CreateDirectoryA("shots_manual", nullptr);
            g_renderer->request_screenshot(path);
            if (--f12_left == 0) NXLOG_I("input", "F12: %u screenshots written", shots);
        }
        // F9 or Ctrl+F11: trace the next presented frame (plain F11 is fullscreen toggle in play mode).
        const bool trace_key = (down(VK_F9) || tapped(VK_F9)) || (down(VK_CONTROL) && (down(VK_F11) || tapped(VK_F11)));
        if (trace_key && !f11_was_down) {
            static const int trace_span = [] {
                const char* s = std::getenv("NXRT_TRACE_SPAN");
                return s ? std::atoi(s) : 0;
            }();
            if (trace_span > 0) {
                nxrt::nvn::backend_trace_frames(trace_span);
                NXLOG_I("input", "F9/Ctrl+F11: tracing the next %d frames", trace_span);
            } else {
                nxrt::nvn::backend_trace_next_frame();
                NXLOG_I("input", "F9/Ctrl+F11: tracing the next frame (render targets -> trace_t<id>_<fmt>.png here)");
            }
        }
        f11_was_down = trace_key;
    } else if (down(VK_F12) || down(VK_F9)) {
        // a hotkey is held but the window does not have the focus: say so once, so a session log can tell it apart from
        // "the hotkey is broken"
        static bool said = false;
        if (!said) {
            said = true;
            NXLOG_I("input", "hotkey seen, but the game window is not the foreground window: click the window first");
        }
    }
    // mouse -> touch (HANDOVER 86.14): the 3.0.0 guest ignores the pad's button bits, so the menu can still be driven by clicking
    // it. Left button down = one finger at the pointer, scaled from the window's client area to the 1280x720 touch screen.
    // XC2_NO_TOUCH=1 turns it off.
    static const bool touch_off = std::getenv("XC2_NO_TOUCH") != nullptr;
    if (!touch_off && focused && own && !menu_blocked) {
        POINT pt{};
        RECT rc{};
        if (::GetCursorPos(&pt) && ::ScreenToClient(own, &pt) && ::GetClientRect(own, &rc)) {
            const int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
            const bool lmb = (::GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
            if (lmb && cw > 0 && ch > 0) {
                int tx = static_cast<int>(static_cast<long long>(pt.x) * 1280 / cw);
                int ty = static_cast<int>(static_cast<long long>(pt.y) * 720 / ch);
                tx = std::clamp(tx, 0, 1279);
                ty = std::clamp(ty, 0, 719);
                out.touch.count = 1;
                out.touch.fingers[0].id = 0;
                out.touch.fingers[0].x = static_cast<uint32_t>(tx);
                out.touch.fingers[0].y = static_cast<uint32_t>(ty);
                static bool said = false;
                if (!said) {
                    said = true;
                    NXLOG_I("input", "touch: mouse -> touch at %d,%d (client %dx%d)", tx, ty, cw, ch);
                }
            }
        }
    }
    // gamepad: play mode pumps SDL's event queue on the window-owning main thread (play.cpp keyboard_service),
    // which keeps pad discovery and state fresh without cross-thread SDL_PollEvent races.
    const nxrt::play::LivePadConfig pcfg = nxrt::play::live_pad_config();
    int px = 0, py = 0, prx = 0, pry = 0;
    uint64_t pb = 0;
    uint32_t raw_buttons = 0;   // the SDL button mask as the pads report it (XC2_PAD_DEBUG)
    uint32_t raw_unmapped = 0;  // the device's own mask, before SDL's gamepad mapping
    if (nxrt::input::gamepad_count()) {
        // SDL hands a pad that went away and came back a NEW instance id (a power-save drop, a Bluetooth hiccup, the
        // window losing the foreground and getting it again: the log shows "gamepad 3 added ... removed ... gamepad 4
        // added ..."). A cached id nothing answers to means no pad input for the rest of the run, which reads as "the
        // controller stopped working although it is fine", so keep the cached id only while the list still has it.
        nxrt::input::GamepadInfo list[8];
        const size_t n = nxrt::input::gamepad_list(list, 8);
        // Poll EVERY pad and merge the results; never let the first entry of the list veto the rest. Virtual Desktop
        // (and similar software) installs an "XBOX Synthetic Gamepad" that enumerates BEFORE the real pad and answers
        // every poll with zeros - taking list[0] then reads as "the controller is dead" while it is perfectly fine
        // (HANDOVER 40.43). Merging also survives SDL handing a re-attached pad a new instance id.
        for (size_t i = 0; i < n; ++i) {
            nxrt::platform::GamepadRaw raw;
            if (!nxrt::platform::gamepad_poll(list[i].id, &raw)) continue;
            // SDL's gamepad mapping can report axes but no buttons at all (a synthetic pad, or a device it has no mapping for):
            // then the unmapped mask is the only thing left, so fall back to it (HANDOVER 86.11). The bit positions differ:
            // SDL's gamepad order has BACK 4 / START 6 / LB 9 / RB 10 / L3 7 / R3 8, a device's own order is the HID one
            // (LB 4 / RB 5 / BACK 6 / START 7 / L3 8 / R3 9), so the fallback maps by that.
            const uint32_t m = raw.buttons;
            const bool raw_mode = (m == 0 && raw.raw_buttons != 0);
            raw_buttons |= raw_mode ? raw.raw_buttons : m;
            raw_unmapped |= raw.raw_buttons;
            auto has = [&](int bit) { return (m >> bit) & 1u; };
            auto hasr = [&](int bit) { return (raw.raw_buttons >> bit) & 1u; };
            auto map_face = [&](bool south, bool east, bool west, bool north) {
                if (pcfg.label_abxy) {
                    if (south) pb |= btn(kBtnA);
                    if (east)  pb |= btn(kBtnB);
                    if (west)  pb |= btn(kBtnX);
                    if (north) pb |= btn(kBtnY);
                } else {
                    if (east)  pb |= btn(kBtnA);
                    if (south) pb |= btn(kBtnB);
                    if (north) pb |= btn(kBtnX);
                    if (west)  pb |= btn(kBtnY);
                }
            };
            if (raw_mode) {
                map_face(hasr(0), hasr(1), hasr(2), hasr(3));
                if (hasr(4)) pb |= btn(kBtnL);
                if (hasr(5)) pb |= btn(kBtnR);
                if (hasr(6)) pb |= btn(kBtnMinus);
                if (hasr(7)) pb |= btn(kBtnPlus);
                if (hasr(8)) pb |= btn(kBtnStickL);
                if (hasr(9)) pb |= btn(kBtnStickR);
                if (hasr(11)) pb |= btn(kBtnUp);
                if (hasr(12)) pb |= btn(kBtnDown);
                if (hasr(13)) pb |= btn(kBtnLeft);
                if (hasr(14)) pb |= btn(kBtnRight);
            } else {
                map_face(has(0), has(1), has(2), has(3));
                if (has(9)) pb |= btn(kBtnL);
                if (has(10)) pb |= btn(kBtnR);
                if (has(6)) pb |= btn(kBtnPlus);
                if (has(4)) pb |= btn(kBtnMinus);
                if (has(7)) pb |= btn(kBtnStickL);
                if (has(8)) pb |= btn(kBtnStickR);
                if (has(11)) pb |= btn(kBtnUp);
                if (has(12)) pb |= btn(kBtnDown);
                if (has(13)) pb |= btn(kBtnLeft);
                if (has(14)) pb |= btn(kBtnRight);
            }
            if (raw.axes[4] > 8000) pb |= btn(kBtnZL);
            if (raw.axes[5] > 8000) pb |= btn(kBtnZR);
            // sticks: keep the largest deflection any pad reports, so a second (silent) pad cannot zero a real one
            if (raw.axes[0] * raw.axes[0] > px * px) px = raw.axes[0];
            { const int v = -raw.axes[1]; if (v * v > py * py) py = v; }   // SDL y is down-positive, hid y is up-positive
            if (raw.axes[2] * raw.axes[2] > prx * prx) prx = raw.axes[2];
            { const int v = -raw.axes[3]; if (v * v > pry * pry) pry = v; }
        }
        // What the pad reports, before anything is applied: the F1 calibration shows it and stores the rest position.
        nxrt::play::publish_raw_left_stick(static_cast<float>(px) / 32767.0f, static_cast<float>(py) / 32767.0f);
        // A worn stick rests off zero, and a radial dead zone cannot fix that: it removes the small signal but the axis
        // still leans one way, so "push back" becomes a slow walk (issue #76). Subtract the measured centre first.
        nxrt::play::publish_raw_right_stick(static_cast<float>(prx) / 32767.0f, static_cast<float>(pry) / 32767.0f);
        px = std::clamp(px - pcfg.ls_center_x, -32767, 32767);
        py = std::clamp(py - pcfg.ls_center_y, -32767, 32767);
        prx = std::clamp(prx - pcfg.rs_center_x, -32767, 32767);
        pry = std::clamp(pry - pcfg.rs_center_y, -32767, 32767);
        auto apply_deadzone = [](int &sx, int &sy, float dz, float max_travel) {
            const float fx = static_cast<float>(sx) / 32767.0f;
            const float fy = static_cast<float>(sy) / 32767.0f;
            const float mag = std::sqrt(fx * fx + fy * fy);
            if (mag <= dz || dz >= 0.99f) { sx = 0; sy = 0; return; }
            const float span = (std::max)(0.05f, max_travel - dz);
            const float sc = (std::min)(1.0f, (mag - dz) / span) / mag;
            sx = std::clamp(static_cast<int>(std::lround(fx * sc * 32767.0f)), -32767, 32767);
            sy = std::clamp(static_cast<int>(std::lround(fy * sc * 32767.0f)), -32767, 32767);
        };
        apply_deadzone(px, py, pcfg.stick_deadzone, pcfg.stick_max);
        apply_deadzone(prx, pry, pcfg.right_deadzone, 1.0f);
        if (prx || pry) {
            prx = std::clamp(static_cast<int>(std::lround(static_cast<float>(prx) * pcfg.look_sens)), -32767, 32767);
            pry = std::clamp(static_cast<int>(std::lround(static_cast<float>(pry) * pcfg.look_sens)), -32767, 32767);
            if (pcfg.invert_y) pry = -pry;
        }
    }
    // XC2_PAD_DEBUG: what the pads report BEFORE the mapping, logged when the raw button mask changes (and every 3 s while a
    // pad is deflected). A pad whose sticks move but whose mask stays 0 is an SDL-level problem (mapping/instance), not ours;
    // a non-zero mask that does not reach the game is our mapping. HANDOVER 86.11.
    static const bool pad_dbg = std::getenv("XC2_PAD_DEBUG") != nullptr;
    if (pad_dbg && nxrt::input::gamepad_count()) {
        static uint32_t last_mask = 0xffffffffu;
        static uint64_t last_ns = 0;
        static uint32_t shown = 0;
        const uint64_t now = nxrt::platform::ticks_ns();
        const bool moved = px || py || prx || pry;
        if (raw_buttons != last_mask || (moved && now - last_ns > 3000000000ull)) {
            last_mask = raw_buttons;
            last_ns = now;
            if (shown++ < 400)
                NXLOG_I("input", "pad raw: mask=0x%x (unmapped 0x%x) pads=%zu mapped=0x%llx sticks %d,%d %d,%d",
                        raw_buttons, raw_unmapped, nxrt::input::gamepad_count(), static_cast<unsigned long long>(pb), px, py, prx, pry);
        }
    }
    // XC2_PAD_SWAP=1: diagnostic. Map the four face buttons onto the four direction bits, so "pressing A moves the cursor up"
    // tells a pad that never arrives apart from a guest that ignores bits 0..3 (HANDOVER 86.12).
    // XC2_PAD_ALL=1: diagnostic. Any face button also sets EVERY button bit 0..15, so one press tells "the guest's confirm bit is
    // somewhere in 0..15" apart from "the guest never looks at this field" (HANDOVER 86.13).
    static const bool pad_all = std::getenv("XC2_PAD_ALL") != nullptr;
    if (pad_all && (pb & (btn(kBtnA) | btn(kBtnB) | btn(kBtnX) | btn(kBtnY)))) pb |= 0xffffffull;   // 0..23: buttons and stick directions
    static const bool pad_swap = std::getenv("XC2_PAD_SWAP") != nullptr;
    if (pad_swap) {
        const uint64_t face = pb & (btn(kBtnA) | btn(kBtnB) | btn(kBtnX) | btn(kBtnY));
        if (face & btn(kBtnA)) pb |= btn(kBtnUp);
        if (face & btn(kBtnB)) pb |= btn(kBtnDown);
        if (face & btn(kBtnX)) pb |= btn(kBtnLeft);
        if (face & btn(kBtnY)) pb |= btn(kBtnRight);
    }
    const uint64_t b_all = kb | pb;
    const int sx = kx ? kx : px, sy = ky ? ky : py, rx = klx ? klx : prx, ry = kly ? kly : pry;
    // Gamepad chords for 3D Navigation (conflict-free with XC2 field controls and Minus+Plus F1 menu):
    //   ZL + L3 = toggle 3D nav mode (F3)
    //   ZL + R3 = open/close 3D nav destination picker (F4)
    static uint64_t nav_pad_prev = 0;
    static uint64_t nav_pad_mask = 0;
    {
        const uint64_t chord_f3 = btn(kBtnZL) | btn(kBtnStickL);
        const uint64_t chord_f4 = btn(kBtnZL) | btn(kBtnStickR);
        const uint64_t fresh_pb = pb & ~nav_pad_prev;
        if (!menu_blocked && (pb & chord_f3) == chord_f3 && (fresh_pb & chord_f3)) {
            xc2::nav::toggle();
            nav_pad_mask |= chord_f3;
        }
        if ((pb & chord_f4) == chord_f4 && (fresh_pb & chord_f4)) {
            nxrt::play::toggle_picker();
            nav_pad_mask |= chord_f4;
        }
        nav_pad_mask &= pb;
        nav_pad_prev = pb;
    }
    const uint64_t menu_mask = nxrt::play::note_external_pad(b_all, sx, sy, rx, ry) | nav_pad_mask;
    if (menu_blocked) return;
    const uint64_t b = b_all & ~menu_mask;
    if (!b && !sx && !sy && !rx && !ry) return;
    // player 1 and the handheld entry the service also fills (XC2 asks for both)
    for (uint32_t slot : {0u, 9u}) {
        NpadInput &p = out.npad[slot];
        p.buttons |= b;
        if (sx || sy) { p.left.x = sx; p.left.y = sy; }
        if (rx || ry) { p.right.x = rx; p.right.y = ry; }
    }
}


// ---- replay routes (tools/run/route.ps1) ---------------------------------------------------------------
// A route is the input of one run written down: XC2_ROUTE_OUT=<file> records what the game saw (one line per change,
// milliseconds since the recorder started), XC2_ROUTE=<file> plays it back. Comparing two builds needs the same scene,
// and the same buttons at the same milliseconds is what gets there: two runs of the same XC2_PRESS script landed 4100
// and 736 draws/frame apart on 2026-11-20 (HANDOVER 228).
//
//   +1200 A            press A at 1.2 s
//   -1450 A            release A at 1.45 s
//   L3000 0 32767      left stick to (0, 32767) at 3.0 s, held until the next L/R line
namespace route {
struct Bit { const char *name; uint64_t bit; };
const Bit kBits[] = {{"A", nxrt::hid::btn(nxrt::hid::kBtnA)},     {"B", nxrt::hid::btn(nxrt::hid::kBtnB)},
                     {"X", nxrt::hid::btn(nxrt::hid::kBtnX)},     {"Y", nxrt::hid::btn(nxrt::hid::kBtnY)},
                     {"PLUS", nxrt::hid::btn(nxrt::hid::kBtnPlus)}, {"MINUS", nxrt::hid::btn(nxrt::hid::kBtnMinus)},
                     {"L", nxrt::hid::btn(nxrt::hid::kBtnL)},     {"R", nxrt::hid::btn(nxrt::hid::kBtnR)},
                     {"ZL", nxrt::hid::btn(nxrt::hid::kBtnZL)},   {"ZR", nxrt::hid::btn(nxrt::hid::kBtnZR)},
                     {"DPD", nxrt::hid::btn(nxrt::hid::kBtnDown)}, {"DPU", nxrt::hid::btn(nxrt::hid::kBtnUp)},
                     {"DPL", nxrt::hid::btn(nxrt::hid::kBtnLeft)}, {"DPR", nxrt::hid::btn(nxrt::hid::kBtnRight)}};
struct Rec {
    std::mutex mu;
    FILE *f = nullptr;
    bool init = false;
    std::chrono::steady_clock::time_point t0;
    uint64_t btn = 0;
    int32_t lx = 0, ly = 0, rx = 0, ry = 0;
};
Rec &rec() {
    static Rec r;
    return r;
}
// One origin for recording and playback, set when the input source goes live in main: the hid service only samples
// the source once the game asks for input (about 6 s in), so a recorder that started its own clock at the first
// sample wrote +880 for a press scheduled at 7 s, and a playback that started at its thread's start fired 6 s early.
std::chrono::steady_clock::time_point g_route_t0;
long long ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}
struct Ev { long long ms = 0; char kind = 'B'; bool down = false; uint64_t bit = 0; int32_t x = 0, y = 0; };
}  // namespace route

void route_record(const nxrt::hid::InputSnapshot &s) {
    static const char *path = [] {
        const char *p = std::getenv("XC2_ROUTE_OUT");
        return (p && *p) ? p : nullptr;
    }();
    if (!path) return;
    route::Rec &r = route::rec();
    std::lock_guard<std::mutex> lk(r.mu);
    const uint64_t buttons = s.npad[0].buttons;
    const int32_t lx = s.npad[0].left.x, ly = s.npad[0].left.y, rx = s.npad[0].right.x, ry = s.npad[0].right.y;
    if (!r.init) {
        r.init = true;
        r.t0 = std::chrono::steady_clock::now();
        r.f = std::fopen(path, "wb");
        if (r.f) std::fprintf(r.f, "# xc2 route v1, recorded by XC2_ROUTE_OUT; play it with XC2_ROUTE\n");
        r.btn = buttons;
        r.lx = lx; r.ly = ly; r.rx = rx; r.ry = ry;
        return;
    }
    if (!r.f) return;
    const long long ms = route::ms_since(route::g_route_t0);
    for (const route::Bit &b : route::kBits)
        if ((buttons ^ r.btn) & b.bit) std::fprintf(r.f, "%c%lld %s\n", (buttons & b.bit) ? '+' : '-', ms, b.name);
    r.btn = buttons;
    if (lx != r.lx || ly != r.ly) { std::fprintf(r.f, "L%lld %d %d\n", ms, lx, ly); r.lx = lx; r.ly = ly; }
    if (rx != r.rx || ry != r.ry) { std::fprintf(r.f, "R%lld %d %d\n", ms, rx, ry); r.rx = rx; r.ry = ry; }
    std::fflush(r.f);   // a route whose run ends with the window closed still has its tail on disk
}

// Reads the file and drives the source on its own thread. Unknown lines are skipped (a route is data, not a program).
static void route_start(const std::shared_ptr<nxrt::hid::ManualInputSource> &src, const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "route: cannot open %s\n", path);
        return;
    }
    std::vector<route::Ev> evs;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (!*p || *p == '#' || *p == '\r' || *p == '\n') continue;
        route::Ev e;
        if (*p == '+' || *p == '-') {          // a button edge
            e.down = *p == '+';
            char name[32] = {};
            if (std::sscanf(p + 1, "%lld %31s", &e.ms, name) != 2) continue;
            for (const route::Bit &b : route::kBits)
                if (!std::strcmp(b.name, name)) e.bit = b.bit;
            if (!e.bit) continue;
        } else if (*p == 'L' || *p == 'R') {   // a stick position
            e.kind = *p;
            if (std::sscanf(p + 1, "%lld %d %d", &e.ms, &e.x, &e.y) != 3) continue;
        } else {
            continue;
        }
        evs.push_back(e);
    }
    std::fclose(f);
    long long delay = 0;
    if (const char *d = std::getenv("XC2_ROUTE_DELAY")) delay = std::atoll(d);
    std::fprintf(stderr, "route: %zu events from %s, starting at +%lld ms\n", evs.size(), path, delay);
    std::thread([src, evs, delay] {
        const auto start = route::g_route_t0;
        for (const route::Ev &e : evs) {
            std::this_thread::sleep_until(start + std::chrono::milliseconds(delay + e.ms));
            if (e.kind == 'B') {
                if (e.down) src->press(0, e.bit);
                else src->release(0, e.bit);
            } else {
                src->stick(0, e.kind == 'R', e.x, e.y);
            }
        }
    }).detach();
}

}  // namespace

// Model-spawn probe (XC2_MODEL_PROBE=1): log the engine's model creation and resource loading with their raw arguments, and
// for every argument that points into the guest's memory the first bytes as hex text. A model that never appears in the
// frame is either never created (this probe stays silent for it) or created and culled (it shows up here). Answers the
// "did the game even ask for it?" half of HANDOVER 19.9 before anything in the renderer is touched.
static void model_probe_dump(const char *what, int n, Ctx *c) {
    std::fprintf(stderr, "model probe %s #%d:", what, n);
    for (int i = 0; i < 4; ++i) {
        const uint64_t v = c->x[i];
        std::fprintf(stderr, " x%d=0x%llx", i, static_cast<unsigned long long>(v));
        if (v > 0x7000000000ull && v < 0x8000000000ull) {   // a guest pointer: show what it points at
            const uint8_t *p = reinterpret_cast<const uint8_t *>(v);
            char txt[33];
            for (int k = 0; k < 32; ++k) {
                const uint8_t ch = p[k];
                txt[k] = (ch >= 32 && ch < 127) ? static_cast<char>(ch) : '.';
            }
            txt[32] = 0;
            std::fprintf(stderr, " [%s]", txt);
        }
    }
    std::fprintf(stderr, "\n");
}

// Arms the probe: hits the site hooks for ml::Scn::createMdl and the launcher's resource loader, both of which are already
// in build\sites-main.txt (so no re-lift is needed), and prints a running count every second so the log shows WHEN a scene
// created its models.
static void arm_model_probe(uint64_t main_base) {
    static std::atomic<int> n_mdl{0}, n_res{0}, n_alloc{0};
    nxrt::hook_add_entry(main_base + 0x7409ac, [](Ctx *c, uint64_t, void *) -> int {   // ml::Scn::createMdl
        const int n = n_mdl.fetch_add(1) + 1;
        if (n <= 120) model_probe_dump("createMdl", n, c);
        return 0;
    });
    nxrt::hook_add_entry(main_base + 0x1254, [](Ctx *c, uint64_t, void *) -> int {     // apps::FrameworkLauncher::mlthFnLoadResource
        const int n = n_res.fetch_add(1) + 1;
        if (n <= 120) model_probe_dump("loadResource", n, c);
        return 0;
    });
    std::fprintf(stderr, "model probe armed (createMdl 0x7409ac, loadResource 0x1254)\n");
    std::thread([&] {
        for (int t = 1; t <= 600; ++t) {
            Sleep(1000);
            std::fprintf(stderr, "model probe t=%ds: createMdl=%d loadResource=%d\n", t, n_mdl.load(), n_res.load());
        }
    }).detach();
}

// ---- frame rate (HANDOVER 55) -----------------------------------------------------------------------------------------------
// The game steps `interval x vsync period` per frame: the interval is 2 (set at boot, main+0xb75c40) and the period is the 1/60 the
// graphics device's virtual returns, a single float in rodata (0x7100f65a44). One frame therefore advances 1/30 s whatever the present
// rate, so 60 Hz runs the game at 2x and 120 Hz at 4x (the stairs run ends off the platform). [xc2] fps = 60|120 rewrites that float to
// 1 / (2 x fps): the step becomes 1/fps per frame and the game keeps real time. 0 (the default) leaves the game exactly as it is.
// (NS1 does the same through a patch page: frame_rate.c xc2_rate_store, patches-v301.txt set xc2fps. Their build reads the period through
// the virtual every frame; ours has to be checked - if the game caches it at boot, this write lands before the read either way.)
static nxrt::FrameLimiter *g_frame_limiter = nullptr;   // the play-mode limiter: its measured interval is the display time
static std::atomic<float> g_fps_limiter_step{1.0f / 60.0f};
static std::atomic<float> g_fps_game_step{1.0f / 60.0f};
static uint64_t g_fps_main_base = 0;

// Retail 3.0.1 vsync-period virtual override (main+0xba1320, vtable slot +0xc8 at main+0xe62780; zero direct BL callers).
// Separates NS1-style dual words without binary patching:
//   - renderSyncSys limiter (lr = main+0xb8eb4c / 0xb8ebd0) receives exact target step (1/90 gameplay, 1/60 cutscene), never self-locking;
//   - getFrameStep (0xb865bc, 0xb8e540) receives the 5-frame real-time moving average so gameplay/hair/cloth keep 1.0x real time.
static void host_vsync_period_301(::Ctx *c) {
    const uint64_t mb = g_fps_main_base;
    int32_t iv = 1;
    const uint64_t iv_addr = mb + 0xe95460;   // retail 3.0.1 vsync interval (written to 1 at boot by 0x735300)
    if (nxrt::mem::is_mapped(iv_addr, 4)) {
        std::memcpy(&iv, reinterpret_cast<const void *>(iv_addr), 4);
    }
    if (iv <= 0) iv = 2;   // matches cbz w8 -> 2*period in renderSyncSys (0xb8eba0)
    const uint64_t lr = c->x[30];
    const bool is_limiter = (lr == mb + 0xb8eb4c || lr == mb + 0xb8ebd0);
    static std::atomic<uint64_t> seen[8] = {};
    static std::atomic<int> nseen{0};
    bool known = false;
    const int n = nseen.load(std::memory_order_relaxed);
    for (int i = 0; i < n && i < 8; ++i) {
        if (seen[i].load(std::memory_order_relaxed) == lr) { known = true; break; }
    }
    if (!known && n < 8) {
        const int idx = nseen.fetch_add(1, std::memory_order_relaxed);
        if (idx < 8) {
            seen[idx].store(lr, std::memory_order_relaxed);
            NXLOG_I("xc2", "vsync_period(0xba1320): caller lr=0x%llx (main+0x%llx) role=%s iv=%d",
                    static_cast<unsigned long long>(lr), static_cast<unsigned long long>(lr - mb),
                    is_limiter ? "renderSyncSys-limiter" : "game-step", iv);
        }
    }
    const float step = is_limiter ? g_fps_limiter_step.load(std::memory_order_relaxed)
                                  : g_fps_game_step.load(std::memory_order_relaxed);
    c->v[0].u64[0] = 0;
    c->v[0].u64[1] = 0;
    c->v[0].f[0] = step / static_cast<float>(iv);
}

static std::atomic<int64_t> g_teardown_fps{60};
static nxrt::GuestFn g_orig_821b40 = nullptr;
static nxrt::GuestFn g_orig_822570 = nullptr;
static nxrt::GuestFn g_orig_822600 = nullptr;

// Scale guest-83's frame-counted 0x27d0 resource-bank teardown (main+0x820200 via 0x821b40/0x822570, and main+0x822600) back to <=60 Hz wall time.
// Main retires consumer nodes at 30 Hz (33.3 ms), while 60 fps teardown waits 4 rendered frames per stage (66.7 ms); above 60 fps,
// unmodified 2/4-frame thresholds finish in <=33.3 ms and unmap active pools during cutscene exit or fast travel.
static bool hold_teardown_counter(uint64_t bank, uint16_t off, uint16_t trigger_minus_one, uint16_t hold_limit, uint16_t& held) {
    if (held < hold_limit) {
        ++held;
        const uint16_t back = static_cast<uint16_t>(trigger_minus_one - 1);
        std::memcpy(reinterpret_cast<void*>(bank + off), &back, 2);
        return true;
    }
    return false;
}

static void handle_teardown_tick(uint64_t bank, bool is_820200) {
    if (!bank || !nxrt::mem::is_mapped(bank, 0x10c0)) return;
    const int64_t cur_fps = g_teardown_fps.load(std::memory_order_relaxed);
    const int64_t scale = (cur_fps > 60) ? ((cur_fps + 59) / 60) : 1;

    struct TeardownHold {
        uint64_t bank = 0;
        uint16_t h4184_3 = 0;
        uint16_t h4182_1 = 0;
        uint16_t h4182_3 = 0;
    };
    static std::mutex mu;
    static std::vector<TeardownHold> holds;
    static std::atomic<bool> logged_once{false};

    uint32_t flags = 0;
    std::memcpy(&flags, reinterpret_cast<const void*>(bank + 4284), 4);
    const bool phase4184 = is_820200 && ((flags & 0x40000u) == 0);
    const bool phase4182 = !is_820200 || ((flags & 0x40000u) != 0 && (flags & 0x8000000u) != 0);
    if (!phase4184 && !phase4182) return;

    std::lock_guard<std::mutex> lk(mu);
    auto it = std::find_if(holds.begin(), holds.end(), [bank](const TeardownHold& e) { return e.bank == bank; });
    if (scale <= 1) {
        if (it != holds.end()) holds.erase(it);
        return;
    }

    const uint16_t extra2 = static_cast<uint16_t>(2 * (scale - 1));
    const uint16_t extra4 = static_cast<uint16_t>(4 * (scale - 1));
    bool held_now = false;
    if (phase4184) {
        uint16_t c4184 = 0;
        std::memcpy(&c4184, reinterpret_cast<const void*>(bank + 4184), 2);
        if (c4184 == 0) {
            if (it != holds.end()) it->h4184_3 = 0;
            return;
        }
        if (c4184 != 3) return;
        if (it == holds.end()) {
            holds.push_back({bank, 0, 0, 0});
            it = holds.end() - 1;
        }
        held_now = hold_teardown_counter(bank, 4184, 3, extra4, it->h4184_3);
    } else {
        uint16_t c4182 = 0;
        std::memcpy(&c4182, reinterpret_cast<const void*>(bank + 4182), 2);
        if (c4182 == 0) {
            if (it != holds.end()) {
                it->h4182_1 = 0;
                it->h4182_3 = 0;
            }
            return;
        }
        if (c4182 == 1) {
            if (it == holds.end()) {
                holds.push_back({bank, 0, 0, 0});
                it = holds.end() - 1;
            }
            held_now = hold_teardown_counter(bank, 4182, 1, extra2, it->h4182_1);
        } else if (c4182 == 3) {
            if (it == holds.end()) {
                holds.push_back({bank, 0, 0, 0});
                it = holds.end() - 1;
            }
            held_now = hold_teardown_counter(bank, 4182, 3, extra2, it->h4182_3);
            if (!held_now) holds.erase(it);
        }
    }
    if (held_now && !logged_once.exchange(true, std::memory_order_relaxed)) {
        NXLOG_I("xc2", "teardown cadence: scaling bank 0x%llx cleanup at %lld fps (%lldx)",
                static_cast<unsigned long long>(bank), static_cast<long long>(cur_fps), static_cast<long long>(scale));
    }
}

static void wrap_teardown_821b40(::Ctx* c) {
    if (c) handle_teardown_tick(c->x[0], true);
    if (g_orig_821b40) g_orig_821b40(c);
}

static void wrap_teardown_822570(::Ctx* c) {
    if (c) handle_teardown_tick(c->x[0], true);
    if (g_orig_822570) g_orig_822570(c);
}

static void wrap_teardown_822600(::Ctx* c) {
    if (c) handle_teardown_tick(c->x[0], false);
    if (g_orig_822600) g_orig_822600(c);
}

static void arm_teardown_cadence_301(uint64_t main_base, int64_t fps) {
    if (fps <= 60) return;
    g_teardown_fps.store(fps, std::memory_order_relaxed);
    g_orig_821b40 = nxrt::fn_lookup(main_base + 0x821b40ull);
    g_orig_822570 = nxrt::fn_lookup(main_base + 0x822570ull);
    g_orig_822600 = nxrt::fn_lookup(main_base + 0x822600ull);
    if (g_orig_821b40) nxrt::fn_register(main_base + 0x821b40ull, wrap_teardown_821b40);
    if (g_orig_822570) nxrt::fn_register(main_base + 0x822570ull, wrap_teardown_822570);
    if (g_orig_822600) nxrt::fn_register(main_base + 0x822600ull, wrap_teardown_822600);
    NXLOG_I("xc2", "teardown cadence: wrapped indirect dispatch 821b40=%d 822570=%d 822600=%d for >60 fps safety (target %lld fps, scale %lldx)",
            g_orig_821b40 != nullptr, g_orig_822570 != nullptr, g_orig_822600 != nullptr,
            static_cast<long long>(fps), static_cast<long long>((fps + 59) / 60));
}

static void arm_frame_rate(uint64_t main_base) {
    const int64_t fps = nxrt::config_get_int("xc2", "fps", 0);
    // [nvn] present_ahead (HANDOVER 61): the guest used to be kept from leading the replay because it rewrote data the GPU was
    // still reading (torn / scrambled parts of the picture). [nvn] mirror_capture gives every submission the buffer bytes of its own
    // submit call and the GPU no longer reads guest memory live, so the guest may lead by one frame: replay and game overlap, which
    // is what holds 60 fps. The capture needs whole-pool copies in video memory, so all three are "auto": the runtime turns the
    // capture on when the copies' budget can hold them (a 16 GB card, not a 12 GB one, an APU or a Steam Deck), and then leads by one
    // frame without A7's per-draw instance snapshots; otherwise all stays as it was (no lead, snapshots on). Settings the player made win.
    // A stamp the log can be checked against: with two ways to build the runtime (this exe or the package's xc2_rt.dll),
    // "which binary ran" decided several tests on 2026-11-21. XC2_BUILD_STAMP is the tag and commit the build came from
    // (see the top of this file); the media model below still needs its own bump whenever film/decode changes.
    NXLOG_I("xc2", "build stamp: %s media (end of stream is a separate output; the placeholder track ends with the picture)",
            XC2_BUILD_STAMP);
    // The NXRT_* switches this run got, one line, so a log says which launch chain produced it. 2026-11-26: a dev
    // profile that had lost NXRT_NVN_EARLYZ_FS=auto looked like a rendering regression in the code, and nothing in the
    // log said the switch was missing. Paths are left out (they are long and say nothing about behaviour).
    {
        std::string switches;
        if (LPCH block = ::GetEnvironmentStringsA()) {
            for (const char* e = block; *e; e += std::strlen(e) + 1) {
                if (std::strncmp(e, "NXRT_", 5) != 0) continue;
                const char* eq = std::strchr(e, '=');
                if (!eq) continue;
                const std::string key(e, static_cast<std::size_t>(eq - e));
                if (key == "NXRT_NVN_SHADER_CACHE" || key == "NXRT_FS_ROMFS" || key == "NXRT_FS_SAVE_DIR" ||
                    key == "NXRT_AOC_DIR" || key == "NXRT_NVN_DUMP_MISSING" || key == "NXRT_INI" ||
                    key == "NXRT_DLSS_HOT_FILE" || key == "NXRT_NVN_SKIP_FS_FILE" || key == "NXRT_NVN_HOT_STATE_FILE")
                    continue;   // paths
                if (!switches.empty()) switches += ' ';
                switches += e;
            }
            ::FreeEnvironmentStringsA(block);
        }
        NXLOG_I("xc2", "launch switches: %s", switches.empty() ? "(none)" : switches.c_str());
    }
    if (!nxrt::config_has("nvn", "mirror_capture")) nxrt::config_set("nvn", "mirror_capture", "auto");
    if (!nxrt::config_has("nvn", "snapshot_instances")) nxrt::config_set("nvn", "snapshot_instances", "auto");
    if (!nxrt::config_has("nvn", "present_ahead")) nxrt::config_set("nvn", "present_ahead", "auto");
    // The effects' particle updates (vertex shaders writing storage) stay in the capture's copy: written back, their whole ranges
    // went over the CPU's next-frame words and the waterfall mist blinked (nvn_replay.cpp, [nvn] mirror_gfx_storage_writeback).
    if (!nxrt::config_has("nvn", "mirror_gfx_storage_writeback")) nxrt::config_set("nvn", "mirror_gfx_storage_writeback", "0");
    NXLOG_I("xc2", "present_ahead = %s (auto = 1 while [nvn] mirror_capture is on, else 0; [nvn] present_ahead overrides this default)",
            nxrt::config_get_str("nvn", "present_ahead", "").c_str());
    if (fps <= 0) return;
    const uint64_t period_addr = main_base + 0xf65a44;      // the 1/60 the vsync-period virtual loads (0x3C888889) on 3.0.0
    float old = 0.0f;
    const bool has_300_slot = nxrt::mem::is_mapped(period_addr, 4) &&
                              (std::memcpy(&old, reinterpret_cast<const void*>(period_addr), 4), std::fabs(old - (1.0f / 60.0f)) <= 1e-4f);
    if (!has_300_slot) {
        g_fps_main_base = main_base;
        const float init_step = static_cast<float>(1.0 / static_cast<double>(fps));
        g_fps_limiter_step.store(init_step, std::memory_order_relaxed);
        g_fps_game_step.store(init_step, std::memory_order_relaxed);
        nxrt::fn_register(main_base + 0xba1320, host_vsync_period_301);
        arm_teardown_cadence_301(main_base, fps);
        const int64_t cutscene_fps = nxrt::config_get_int("xc2", "cutscene_fps", fps > 60 ? 60 : 0);
        NXLOG_I("xc2", "frame rate: hooked retail 3.0.1 vsync_period virtual at main+0xba1320 (target %lld fps, cutscene %lld fps, step %.2f ms)",
                static_cast<long long>(fps), static_cast<long long>(cutscene_fps), static_cast<double>(init_step) * 1000.0);
        nxrt::set_present_interval_hz(static_cast<double>(fps));
        nxrt::services::viapi::set_display_vsync_hz(static_cast<double>(fps));
        const bool follow = nxrt::config_get_bool("xc2", "fps_follow", true);
        if ((!follow && cutscene_fps <= 0) || !g_frame_limiter) return;
        std::thread([main_base, fps, cutscene_fps, follow] {
            constexpr uint64_t kCutsceneFlagOff = 0xedf1a0ull;
            int64_t cur_fps = fps;
            bool was_rt_cutscene = false;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
                const bool in_movie = nxrt::media::is_movie_playing();
                bool in_rt_cutscene = false;
                const uint64_t flag_addr = main_base + kCutsceneFlagOff;
                if (nxrt::mem::is_mapped(flag_addr, 1)) {
                    uint8_t flag = 0;
                    std::memcpy(&flag, reinterpret_cast<const void *>(flag_addr), 1);
                    in_rt_cutscene = (flag & 1u) != 0;
                    if (in_rt_cutscene != was_rt_cutscene) {
                        was_rt_cutscene = in_rt_cutscene;
                        NXLOG_I("xc2", "real-time cutscene %s (main+0xedf1a0=0x%02x)", in_rt_cutscene ? "starts" : "ends", static_cast<unsigned>(flag));
                    }
                }
                const bool in_cutscene = in_movie || in_rt_cutscene;
                const int64_t want_fps = (in_cutscene && cutscene_fps > 0 && cutscene_fps < fps) ? cutscene_fps : fps;
                if (want_fps != cur_fps) {
                    cur_fps = want_fps;
                    g_teardown_fps.store(cur_fps, std::memory_order_relaxed);
                    nxrt::set_present_interval_hz(static_cast<double>(cur_fps));
                    nxrt::services::viapi::set_display_vsync_hz(static_cast<double>(cur_fps));
                    g_frame_limiter->set_target_fps(static_cast<double>(cur_fps));
                    NXLOG_I("xc2", "frame rate -> %lld fps (%s)", static_cast<long long>(cur_fps),
                            in_movie ? "AV1 movie" : in_rt_cutscene ? "real-time cutscene" : "gameplay");
                }
                const double target = 1.0 / static_cast<double>(cur_fps);
                g_fps_limiter_step.store(static_cast<float>(target), std::memory_order_relaxed);
                double step = target;
                if (follow) {
                    const double recent_ms = g_frame_limiter->recent_mean_ms(5, 250.0);
                    if (recent_ms > 0.0) step = (std::max)(target, recent_ms / 1000.0);
                    step = (std::min)(step, 1.0 / 15.0);
                }
                g_fps_game_step.store(static_cast<float>(step), std::memory_order_relaxed);
            }
        }).detach();
        return;
    }
    // [xc2] fps_interval (default 2): the game's "vsyncs per frame". 2 with period 1/(2 fps) is the shipping arrangement;
    // 1 makes the period 1/fps itself, as the NS1 patch page does (frame_rate.c interval_store), which also changes what
    // the code that counts vsyncs (present requests, lip sync) sees.
    const int64_t nvsync = (std::max)(1LL, nxrt::config_get_int("xc2", "fps_interval", 2));
    const float period = static_cast<float>(1.0 / (static_cast<double>(nvsync) * static_cast<double>(fps)));
    if (!nxrt::mem::protect(period_addr, 4, nxrt::mem::Prot::RW)) {
        NXLOG_W("xc2", "frame rate: cannot make the vsync period writable: the game keeps its own 30 fps step");
        return;
    }
    std::memcpy(reinterpret_cast<void*>(period_addr), &period, 4);
    NXLOG_I("xc2", "frame rate: vsync period %.6f -> %.6f s, %lld vsync(s) a frame, step %.2f ms (target %lld fps)",
            static_cast<double>(old), static_cast<double>(period), static_cast<long long>(nvsync),
            static_cast<double>(period) * static_cast<double>(nvsync) * 1000.0, static_cast<long long>(fps));
    // The game keeps the interval in the render object: [main+0x11c5378] is a pointer, its +0 a 32-bit vsync count (the
    // limiter f_7100722a04 reads it: wait = period x interval x 0.997, interval 0 meaning 2). Written again every 4 ms
    // because the game sets it itself while it boots and when a cutscene starts.
    if (nvsync != 2) {
        const uint64_t holder = main_base + 0x11c5378;
        std::thread([holder, nvsync] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
                if (!nxrt::mem::is_mapped(holder, 8)) continue;
                uint64_t obj = 0;
                std::memcpy(&obj, reinterpret_cast<const void*>(holder), 8);
                if (!obj || !nxrt::mem::is_mapped(obj, 4)) continue;
                uint32_t have = 0;
                std::memcpy(&have, reinterpret_cast<const void*>(obj), 4);
                if (have == static_cast<uint32_t>(nvsync)) continue;
                nxrt::mem::protect(obj, 4, nxrt::mem::Prot::RW);
                const uint32_t v = static_cast<uint32_t>(nvsync);
                std::memcpy(reinterpret_cast<void*>(obj), &v, 4);
            }
        }).detach();
    }
    // [xc2] fps_follow (default on): the step follows the *measured* display interval, never below the target - the game's own limiter waits
    // interval x period x 0.997, so a frame that misses the target shows up in the recent average and raises the step: the game keeps real
    // time instead of running slow (NS1's xc2_rate_store does the same with a 5-frame average, at most 1/15 s).
    // Uses lock-free frame_times_recent (last 8 frames, ignoring loading stalls > 250 ms) instead of FrameLimiter::stats() (which locked and
    // sorted an 8192-frame ring 250 times/s), and only re-applies RW protection when the loader actually restored RO on that page.
    if (!nxrt::config_get_bool("xc2", "fps_follow", true) || !g_frame_limiter) return;
    const double target = 1.0 / static_cast<double>(fps);
    std::thread([period_addr, target, nvsync] {
        float last = static_cast<float>(target / static_cast<double>(nvsync));
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            float recent[8];
            const size_t n = nxrt::frame_times_recent(recent, 8);
            if (n < 2) continue;
            double sum = 0.0;
            size_t cnt = 0;
            for (size_t i = 0; i < n; ++i) {
                if (recent[i] > 0.0f && recent[i] < 250.0f) {
                    sum += static_cast<double>(recent[i]);
                    ++cnt;
                }
            }
            if (cnt == 0) continue;
            double step = (std::max)(target, (sum / static_cast<double>(cnt)) / 1000.0);
            step = (std::min)(step, 1.0 / 15.0);
            const float p = static_cast<float>(step / static_cast<double>(nvsync));
            if (std::fabs(p - last) <= 1e-5f) continue;
            MEMORY_BASIC_INFORMATION mbi{};
            if (VirtualQuery(reinterpret_cast<const void*>(period_addr), &mbi, sizeof mbi) != sizeof mbi || mbi.State != MEM_COMMIT) continue;
            const DWORD prot = mbi.Protect & 0xFF;
            if (prot != PAGE_READWRITE && prot != PAGE_EXECUTE_READWRITE) {
                if (!nxrt::mem::protect(period_addr, 4, nxrt::mem::Prot::RW)) continue;
            }
            std::memcpy(reinterpret_cast<void*>(period_addr), &p, 4);
            last = p;
        }
    }).detach();
}

struct Xc2RenderIntent {
    int64_t mode = 0;
    int64_t rcls = 0;
    int32_t w = 2560;
    int32_t h = 1440;
    bool from_quality = false;
    const char* quality_key = "";
};

// Resolve effective XC2 render mode & target dimensions:
// 1. Explicit [xc2] native_res > 0 takes precedence with [xc2] native_w/native_h / [xc2] res_class.
// 2. When [xc2] native_res == 0 (the default), [upscale] render_quality uses the NVN attachment scaling path in
//    nvn_gpu_texture.cpp (quality_scales) together with xc2_lock_render_size() (#47).
static Xc2RenderIntent resolve_xc2_render_intent() {
    static const Xc2RenderIntent resolved = [] {
        Xc2RenderIntent r;
        int32_t am_w = 1920, am_h = 1080;
        const bool have_am_res = [&] {
            const std::string am_res = nxrt::config_get_str("am", "resolution", "");
            const size_t xpos = am_res.find_first_of("xX");
            if (xpos == std::string::npos) return false;
            const int32_t pw = std::atoi(am_res.c_str());
            const int32_t ph = std::atoi(am_res.c_str() + xpos + 1);
            if (pw >= 1280 && ph >= 720 && pw <= 7680 && ph <= 4320) {
                am_w = pw;
                am_h = ph;
                return true;
            }
            return false;
        }();
        const int64_t explicit_mode = nxrt::config_get_int("xc2", "native_res", 0);
        const int64_t explicit_rcls = nxrt::config_get_int("xc2", "res_class", 0);
        if (explicit_mode > 0) {
            r.mode = explicit_mode;
            r.rcls = explicit_rcls;
            const int32_t def_w = have_am_res ? am_w : 2560;
            const int32_t def_h = have_am_res ? am_h : 1440;
            const int64_t cw = nxrt::config_get_int("xc2", "native_w", 0);
            const int64_t ch = nxrt::config_get_int("xc2", "native_h", 0);
            r.w = cw > 0 ? static_cast<int32_t>(cw) : def_w;
            r.h = ch > 0 ? static_cast<int32_t>(ch) : def_h;
        } else {
            r.rcls = explicit_rcls;
            r.w = am_w;
            r.h = am_h;
        }
        return r;
    }();
    return resolved;
}

// The dynamic-resolution lock: when native_res is off, any render quality but the console's implies it; otherwise [upscale] lock_render_size decides.
static bool xc2_lock_render_size() {
    static const bool on = [] {
        if (resolve_xc2_render_intent().mode > 0) return false;
        const nxrt::RenderQualityMode* m = nxrt::render_quality_mode(nxrt::config_get_str("upscale", "render_quality", "").c_str());
        if (m && m->share > 0.0) return true;
        return nxrt::config_get_bool("upscale", "lock_render_size", false);
    }();
    return on;
}

// True when the upscaler the runtime is actually running keeps no temporal history. Only a spatial upscaler
// needs the game's Halton(2,3) sub-pixel jitter removed: with no accumulation every frame is a resample at a
// different sub-pixel offset, which is the wobbling moire in #94. TAAU / FSR / DLSS consume the jitter and must
// keep it. The answer comes from the runtime's live setting (F1 hot-switches it mid-session), not from the raw
// ini string, so the three mismatches reported in #94 - "spatial" not recognised, a case-sensitive compare, and
// a default that disagreed with the runtime's - cannot come back.
static bool upscaler_is_spatial() {
    std::string up = nxrt::nvn::dlss_live_info().upscaler;
    for (char& ch : up) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return up == "bilinear" || up == "spatial" || up == "nearest" || up == "none";
}

// ---- native resolution override (3.0.1) -------------------------------------------------------------------------------------
// [xc2] native_res = 0 (off) | 1 (native 100%, DLSS off) | 2 (DLSS/FSR 1:1 Native AA at [xc2] native_w x native_h)
// [xc2] native_w / native_h: target 3D render size for native_res=2 (defaults to [am] resolution, fallback 2560x1440)
// [xc2] res_class  = 0 (auto) | 2 (force mode 5: 2560x1440) | 1 (force mode 1: 3840x2160)
static void arm_native_res(uint64_t main_base) {
    const Xc2RenderIntent ri = resolve_xc2_render_intent();
    static int64_t g_nat_mode = 0;
    static int64_t g_rcls = 0;
    g_nat_mode = ri.mode;
    const int64_t mode = g_nat_mode;
    g_rcls = ri.rcls;
    const int64_t rcls = g_rcls;
    static int32_t g_nat_w = 2560;
    static int32_t g_nat_h = 1440;
    g_nat_w = ri.w;
    g_nat_h = ri.h;
    // Always attach lightweight RT rebuild logger (logs first 8 calls or size changes) so logs confirm actual 3D resolution.
    nxrt::hook_add_entry(main_base + 0x844120, [](::Ctx* c, uint64_t, void*) -> int {
        static int last_rw = -1, last_rh = -1, last_bw = -1, last_bh = -1, count = 0;
        const uint64_t rtmgr = c->x[0];
        int32_t d[4] = {};
        bool have_dims = false;
        if (rtmgr && nxrt::mem::is_mapped(rtmgr + 8, 8)) {
            uint64_t ctx = 0;
            std::memcpy(&ctx, reinterpret_cast<const void*>(rtmgr + 8), 8);
            if (ctx && nxrt::mem::is_mapped(ctx + 0x5e0, 8)) {
                uint64_t view = 0;
                std::memcpy(&view, reinterpret_cast<const void*>(ctx + 0x5e0), 8);
                if (view && nxrt::mem::is_mapped(view + 0x18, 16)) {
                    std::memcpy(d, reinterpret_cast<const void*>(view + 0x18), 16);
                    have_dims = (d[2] > 0 && d[3] > 0);
                    // Clamp dynamic-resolution drops BEFORE updating viewport info or logging rt_rebuild (#55),
                    // so locked 1280x720 runs never report or publish transient 1216x684 / 960x540 rungs.
                    static uint32_t g_pin_w = 0, g_pin_h = 0;
                    static bool g_pin_init = false;
                    if (!g_pin_init) {
                        g_pin_init = true;
                        if (const char* s = std::getenv("XC2_RENDER_SIZE_PIN")) {
                            unsigned a = 0, b = 0;
                            if (std::sscanf(s, "%ux%u", &a, &b) == 2 && a && b) { g_pin_w = a; g_pin_h = b; }
                        }
                        if (!g_pin_w && xc2_lock_render_size()) { g_pin_w = 1280; g_pin_h = 720; }
                    }
                    if (g_nat_mode == 0 && g_pin_w && d[0] > 0 && d[1] > 0 && d[0] <= 1920 && d[1] <= 1080 &&
                        (d[0] < static_cast<int32_t>(g_pin_w) || d[1] < static_cast<int32_t>(g_pin_h))) {
                        const uint32_t pinned[2] = {g_pin_w, g_pin_h};
                        std::memcpy(reinterpret_cast<void*>(view + 0x18), pinned, 8);
                        d[0] = static_cast<int32_t>(g_pin_w);
                        d[1] = static_cast<int32_t>(g_pin_h);
                    }
                    if (have_dims && d[0] > 0 && d[1] > 0) {
                        nxrt::nvn::set_game_viewport_info(static_cast<uint32_t>(d[0]), static_cast<uint32_t>(d[1]),
                                                          static_cast<uint32_t>(d[2]), static_cast<uint32_t>(d[3]),
                                                          g_rcls > 0, g_nat_mode == 1);
                    }
                    if (d[0] != last_rw || d[1] != last_rh || d[2] != last_bw || d[3] != last_bh || count < 3) {
                        last_rw = d[0]; last_rh = d[1]; last_bw = d[2]; last_bh = d[3];
                        ++count;
                        NXLOG_I("xc2", "native_res: rt_rebuild #%d render=%dx%d base=%dx%d w1=%llu", count, d[0], d[1], d[2], d[3],
                                static_cast<unsigned long long>(c->x[1]));
                    }
                }
            }
        }
        if (g_nat_mode == 2 && rtmgr && nxrt::mem::is_mapped(rtmgr + 0x928, 4)) {
            const float denom = (have_dims && d[2] > 0) ? static_cast<float>(d[2]) : 3840.0f;
            const float s = (std::min)(1.0f, static_cast<float>(g_nat_w) / denom);
            std::memcpy(reinterpret_cast<void*>(rtmgr + 0x928), &s, 4);
            // Switch dlss_rt+28 to XC2's direct-to-swapchain DLSS/FSR mode (0) with output = base (d[2] x d[3]),
            // so both 1:1 true native (4K/1440p/1080p) AND locked temporal upscaling (e.g. 1440p -> 4K) feed
            // FSR/TAAU cleanly into the full output swapchain without the spatial-blit black-screen bug.
            if (have_dims && d[2] > 0 && d[3] > 0 && nxrt::mem::is_mapped(rtmgr + 2316, 28)) {
                const uint32_t zero32 = 0;
                std::memcpy(reinterpret_cast<void*>(rtmgr + 2316), &zero32, 4);
                uint64_t dlss_rt = 0;
                std::memcpy(&dlss_rt, reinterpret_cast<const void*>(rtmgr + 2336), 8);
                if (dlss_rt && nxrt::mem::is_mapped(dlss_rt, 88)) {
                    const uint8_t one8 = 1;
                    const uint32_t bw = static_cast<uint32_t>(d[2]);
                    const uint32_t bh = static_cast<uint32_t>(d[3]);
                    std::memcpy(reinterpret_cast<void*>(dlss_rt + 28), &zero32, 4);
                    std::memcpy(reinterpret_cast<void*>(dlss_rt + 48), &bw, 4);
                    std::memcpy(reinterpret_cast<void*>(dlss_rt + 52), &bh, 4);
                    std::memcpy(reinterpret_cast<void*>(dlss_rt + 64), &one8, 1);
                }
            }
        }
        return 0;
    });
    if (rcls > 0) {
        // XC2 3.0.1 docked mode table (c_00a6.cpp:5208-5223): 0x8d9ce0 returning 0 selects mode 1 (3840x2160),
        // returning 1 selects mode 4 (3072x1728), and returning 2 selects mode 5 (2560x1440).
        // For res_class=1 we return 0 (mode 1: 3840x2160); for res_class=3 (Native 1080p with full docked quality)
        // we select mode 5 (return 2) and patch its rodata dimensions at main+0xc346cc from 2560x1440 to 1920x1080.
        int64_t effective_rcls = rcls;
        if (rcls == 1) {
            effective_rcls = 0;
        } else if (rcls == 3) {
            effective_rcls = 2;
            const uint64_t dim_addr = main_base + 0xc346ccULL;
            if (nxrt::mem::is_mapped(dim_addr, 8)) {
                uint32_t cur[2] = {};
                std::memcpy(cur, reinterpret_cast<const void*>(dim_addr), 8);
                if (cur[0] == 2560 && cur[1] == 1440) {
                    nxrt::mem::protect(dim_addr, 8, nxrt::mem::Prot::RW);
                    const uint32_t target_1080[2] = {1920, 1080};
                    std::memcpy(reinterpret_cast<void*>(dim_addr), target_1080, 8);
                    NXLOG_I("xc2", "native_res: patched docked mode 5 base dimensions 2560x1440 -> 1920x1080");
                }
            }
        }
        nxrt::hook_add_entry(main_base + 0x8d9ce0, [](::Ctx* c, uint64_t, void* u) -> int {
            c->x[0] = static_cast<uint64_t>(reinterpret_cast<intptr_t>(u));
            return 1;
        }, reinterpret_cast<void*>(static_cast<intptr_t>(effective_rcls)));
        NXLOG_I("xc2", "native_res: display res_class forced to %lld (effective %lld)",
                static_cast<long long>(rcls), static_cast<long long>(effective_rcls));
    }
    // Per-frame hook on getHaltonJitter (0x83f660, x0 = dlss_rt). Installed for every native_res mode: the jitter
    // the game adds is the same whether or not we are driving the render size, and #94 measured it on the
    // default native_res = 0 path, where nothing used to be hooked at all.
    // 1) mode 2 only: when target == base (e.g. true native 4K), keep dlss_rt+28 = 0 (direct-to-swapchain
    //    output) even if f_710084d250 re-sets mode 1/3 on a scene or display transition.
    // 2) always: zero the Halton(2,3) sub-pixel projection jitter when the upscaler that is actually running is
    //    spatial, and leave it alone for TAAU / FSR / DLSS, which need it.
    nxrt::hook_add_entry(main_base + 0x83f660, [](::Ctx* c, uint64_t, void*) -> int {
        if (g_nat_mode == 2) {
            const uint64_t dlss_rt = c->x[0];
            if (dlss_rt && nxrt::mem::is_mapped(dlss_rt, 88)) {
                uint64_t ctx = 0;
                std::memcpy(&ctx, reinterpret_cast<const void*>(dlss_rt), 8);
                if (ctx && nxrt::mem::is_mapped(ctx + 0x5e0, 8)) {
                    uint64_t view = 0;
                    std::memcpy(&view, reinterpret_cast<const void*>(ctx + 0x5e0), 8);
                    if (view && nxrt::mem::is_mapped(view + 40, 8)) {
                        int32_t bw = 0, bh = 0;
                        std::memcpy(&bw, reinterpret_cast<const void*>(view + 40), 4);
                        std::memcpy(&bh, reinterpret_cast<const void*>(view + 44), 4);
                        if (bw > 0 && bh > 0 && g_nat_w >= bw && g_nat_h >= bh) {
                            uint32_t cur_mode = 0;
                            std::memcpy(&cur_mode, reinterpret_cast<const void*>(dlss_rt + 28), 4);
                            if (cur_mode != 0) {
                                const uint32_t zero32 = 0;
                                const uint8_t one8 = 1;
                                const uint32_t ubw = static_cast<uint32_t>(bw);
                                const uint32_t ubh = static_cast<uint32_t>(bh);
                                std::memcpy(reinterpret_cast<void*>(dlss_rt + 28), &zero32, 4);
                                std::memcpy(reinterpret_cast<void*>(dlss_rt + 48), &ubw, 4);
                                std::memcpy(reinterpret_cast<void*>(dlss_rt + 52), &ubh, 4);
                                std::memcpy(reinterpret_cast<void*>(dlss_rt + 64), &one8, 1);
                                uint64_t rtmgr = 0;
                                std::memcpy(&rtmgr, reinterpret_cast<const void*>(dlss_rt + 8), 8);
                                if (rtmgr && nxrt::mem::is_mapped(rtmgr + 2316, 4)) {
                                    std::memcpy(reinterpret_cast<void*>(rtmgr + 2316), &zero32, 4);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (upscaler_is_spatial()) {
            const uint64_t out = c->x[8];
            if (out && nxrt::mem::is_mapped(out, 8)) {
                const uint64_t zero = 0;
                std::memcpy(reinterpret_cast<void*>(out), &zero, 8);
            }
            return 1;
        }
        return 0;
    });
    if (mode <= 0) return;
    if (mode == 1) {
        nxrt::hook_add_entry(main_base + 0x84d230, [](::Ctx* c, uint64_t, void*) -> int {
            c->x[0] = 0;
            return 1;
        });
        nxrt::hook_add_entry(main_base + 0x85f090, [](::Ctx* c, uint64_t, void*) -> int {
            c->x[2] = 0;
            return 0;
        });
        nxrt::hook_add_entry(main_base + 0x85f3c0, [](::Ctx* c, uint64_t, void*) -> int {
            const uint64_t obj = c->x[0];
            if (obj && nxrt::mem::is_mapped(obj + 0x58c, 8)) {
                const float one = 1.0f;
                std::memcpy(reinterpret_cast<void*>(obj + 0x58c), &one, 4);
                std::memcpy(reinterpret_cast<void*>(obj + 0x590), &one, 4);
            }
            return 1;
        });
    } else if (mode == 2) {
        // When native_w x native_h == base (3840x2160), f_71008442e0 sets [fb+440]=0 (render==base),
        // causing f_710084d230 to return 0 and the tonemapper to write into t10 instead of the DLSS
        // input RT (t54), leaving DLSS reading a black t54. Keep f_710084d230 returning 1 so the
        // tonemapper always feeds DLSS cleanly, including at true native 4K (3840x2160).
        nxrt::hook_add_entry(main_base + 0x84d230, [](::Ctx* c, uint64_t, void*) -> int {
            c->x[0] = 1;
            return 1;
        });
        // Keep DLSS/FSR slot active and let 0x85f090 execute normally (including 0x85ee70 state sync and
        // half-res scale copies at +0x5a0/+0x5a8), feeding it our locked target scale via dynres (+0x24/+0x28).
        nxrt::hook_add_entry(main_base + 0x85f090, [](::Ctx* c, uint64_t, void*) -> int {
            const uint64_t obj = c->x[0];
            const uint64_t dynres = c->x[1];
            const int32_t bw = static_cast<int32_t>(c->x[3]);
            const int32_t bh = static_cast<int32_t>(c->x[4]);
            if (!obj || bw <= 0 || bh <= 0 || !nxrt::mem::is_mapped(obj + 0x57c, 0x40)) return 0;
            const int32_t tw = (std::min)(bw, g_nat_w) & ~1;
            const int32_t th = (std::min)(bh, g_nat_h) & ~1;
            const float sx = static_cast<float>(tw) / static_cast<float>(bw);
            const float sy = static_cast<float>(th) / static_cast<float>(bh);
            const uint8_t one_b = 1;
            const float one_f = 1.0f;
            std::memcpy(reinterpret_cast<void*>(obj + 0x57e), &one_b, 1);
            std::memcpy(reinterpret_cast<void*>(obj + 0x584), &sx, 4);
            std::memcpy(reinterpret_cast<void*>(obj + 0x588), &sy, 4);
            std::memcpy(reinterpret_cast<void*>(obj + 0x598), &one_f, 4);
            std::memcpy(reinterpret_cast<void*>(obj + 0x59c), &one_f, 4);
            c->x[2] = 1;
            if (dynres && nxrt::mem::is_mapped(dynres, 0x44)) {
                const uint32_t zero_w = 0;
                const uint32_t preset3 = 3;
                std::memcpy(reinterpret_cast<void*>(dynres + 0x00), &one_b, 1);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x24), &sx, 4);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x28), &sy, 4);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x38), &zero_w, 4);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x3c), &zero_w, 4);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x40), &preset3, 4);
            }
            return 0;
        });
        // Lock dynres::update (0x7b8b30) so heavy scenes (e.g. Gormott) never step [dynres+0x38] (0..4)
        // or toggle U_POST ENABLE_SHARPEN/ENABLE_SAMPLE_BLOOM shader permutations mid-frame.
        nxrt::hook_add_entry(main_base + 0x7b8b30, [](::Ctx* c, uint64_t, void*) -> int {
            const uint64_t dynres = c->x[0];
            if (dynres && nxrt::mem::is_mapped(dynres, 0x44)) {
                const uint8_t one_b = 1;
                const uint32_t zero_w = 0;
                const uint32_t preset3 = 3;
                std::memcpy(reinterpret_cast<void*>(dynres + 0x00), &one_b, 1);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x38), &zero_w, 4);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x3c), &zero_w, 4);
                std::memcpy(reinterpret_cast<void*>(dynres + 0x40), &preset3, 4);
            }
            return 1;
        });
        // Ensure nvn_dlss::getPresetInputSize (0x8eab90) returns native_w x native_h even on Frame 1
        // (when 0x844ad0 queries mode_idx=7 -> 1920x1080), so secondary RTs [x27+0x190..0x1b0] match
        // the 2560x1440 primary RTs and preserve NVN memory-pool VkImage aliasing.
        nxrt::hook_add_entry(main_base + 0x8eab90, [](::Ctx* c, uint64_t, void*) -> int {
            const uint64_t out = c->x[1];
            const int32_t bw = static_cast<int32_t>(c->x[3]);
            const int32_t bh = static_cast<int32_t>(c->x[4]);
            if (out && bw > 0 && bh > 0 && nxrt::mem::is_mapped(out, 8)) {
                const int32_t tw = (std::min)(bw, g_nat_w) & ~1;
                const int32_t th = (std::min)(bh, g_nat_h) & ~1;
                std::memcpy(reinterpret_cast<void*>(out), &tw, 4);
                std::memcpy(reinterpret_cast<void*>(out + 4), &th, 4);
                return 1;
            }
            return 0;
        });
        nxrt::hook_add_entry(main_base + 0x85f3c0, [](::Ctx* c, uint64_t, void*) -> int {
            const uint64_t obj = c->x[0], pw = c->x[1], ph = c->x[2];
            if (pw && ph && nxrt::mem::is_mapped(pw, 4) && nxrt::mem::is_mapped(ph, 4)) {
                int32_t bw = 0, bh = 0;
                std::memcpy(&bw, reinterpret_cast<const void*>(pw), 4);
                std::memcpy(&bh, reinterpret_cast<const void*>(ph), 4);
                if (bw > 0 && bh > 0) {
                    const int32_t tw = (std::min)(bw, g_nat_w) & ~1;
                    const int32_t th = (std::min)(bh, g_nat_h) & ~1;
                    std::memcpy(reinterpret_cast<void*>(pw), &tw, 4);
                    std::memcpy(reinterpret_cast<void*>(ph), &th, 4);
                    if (obj && nxrt::mem::is_mapped(obj + 0x58c, 8)) {
                        const float sx = static_cast<float>(tw) / static_cast<float>(bw);
                        const float sy = static_cast<float>(th) / static_cast<float>(bh);
                        std::memcpy(reinterpret_cast<void*>(obj + 0x58c), &sx, 4);
                        std::memcpy(reinterpret_cast<void*>(obj + 0x590), &sy, 4);
                    }
                }
            }
            return 1;
        });
        nxrt::hook_add_inline(main_base + 0x7cfc2c, [](::Ctx* c, uint64_t, void*) {
            // s0 was 0.5f; override to native_w / base_w (e.g. 2560/3840 = 0.6667f, or 1.0f at 1:1 native).
            float denom = 3840.0f;
            if (c->x[19] && nxrt::mem::is_mapped(c->x[19] + 1504, 8)) {
                uint64_t view = 0;
                std::memcpy(&view, reinterpret_cast<const void*>(c->x[19] + 1504), 8);
                if (view && nxrt::mem::is_mapped(view + 40, 4)) {
                    int32_t bw = 0;
                    std::memcpy(&bw, reinterpret_cast<const void*>(view + 40), 4);
                    if (bw > 0) denom = static_cast<float>(bw);
                }
            }
            c->v[0].f[0] = (std::min)(1.0f, static_cast<float>(g_nat_w) / denom);
        });
    }
    if (ri.from_quality) {
        NXLOG_I("xc2", "native_res: armed mode %lld from [upscale] render_quality=%s (target %dx%d, res_class %lld)",
                static_cast<long long>(mode), ri.quality_key, g_nat_w, g_nat_h, static_cast<long long>(rcls));
    } else {
        NXLOG_I("xc2", "native_res: armed mode %lld (target %dx%d)", static_cast<long long>(mode), g_nat_w, g_nat_h);
    }
}

// ---- delta scan (diagnostic, HANDOVER 57) -----------------------------------------------------------------------------------
// [xc2] delta_scan_at = N (seconds, 0 = off): N seconds after boot, scan the main module's data and bss for floats that look like a frame
// delta (1/60 or 1/30) and log them. Run once at 60 fps and once at 30: an address that holds 0.0167 in one and 0.0333 in the other is a
// per-frame delta the game updates itself - the candidate for the animation clock that still runs at 30 Hz steps at 60 fps.
static void arm_delta_scan(uint64_t main_base) {
    const int64_t at = nxrt::config_get_int("xc2", "delta_scan_at", 0);
    if (at <= 0) return;
    std::thread([main_base, at] {
        std::this_thread::sleep_for(std::chrono::seconds(at));
        const uint64_t data_beg = main_base + 0x1171000, data_end = data_beg + 0x578e8;
        const uint64_t bss_beg = data_end, bss_end = bss_beg + 0x29af8750ull;
        int hits = 0;
        const uint64_t kStep = 0x10000;   // bss pages are committed on use: check every 64 KiB before reading it
        for (uint64_t base = data_beg; base < bss_end && hits < 120; base += kStep) {
            const uint64_t len = (std::min)(kStep, bss_end - base);
            if (!nxrt::mem::is_mapped(base, len)) continue;
            const uint8_t* q = reinterpret_cast<const uint8_t*>(base);
            for (uint64_t i = 0; i + 4 <= len; i += 4) {
                float f;
                std::memcpy(&f, q + i, 4);
                if (!(f > 0.0149f && f < 0.0182f) && !(f > 0.0320f && f < 0.0352f)) continue;
                NXLOG_I("xc2", "delta scan: %s 0x%llx = %.6f", base + i < data_end ? "data" : "bss",
                        static_cast<unsigned long long>(base + i), static_cast<double>(f));
                if (++hits >= 120) break;
            }
        }
        NXLOG_I("xc2", "delta scan done: %d hits", hits);
    }).detach();
}

// ---- period getter probe (diagnostic, HANDOVER 57) ---------------------------------------------------------------------------
// [xc2] period_probe = 1: log who calls the vsync period virtual (main+0x73a8f4) and how often. The callers are the frame-time getter
// (the step) and, through it, everything that consumes the frame delta; the return address names the lifted function to read.
static void arm_period_probe(uint64_t main_base) {
    if (!nxrt::config_get_bool("xc2", "period_probe", false)) return;
    static std::atomic<uint64_t> n{0};
    static std::atomic<uint64_t> seen[64];
    static std::atomic<int> nseen{0};
    nxrt::hook_add_entry(main_base + 0x73a8f4, [](Ctx *c, uint64_t, void *) -> int {
        const uint64_t k = n.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t lr = c->x[30];
        bool known = false;
        for (int i = 0; i < nseen.load(std::memory_order_relaxed); ++i) if (seen[i].load(std::memory_order_relaxed) == lr) { known = true; break; }
        if (!known) {
            const int i = nseen.fetch_add(1, std::memory_order_relaxed);
            if (i < 64) {
                seen[i].store(lr, std::memory_order_relaxed);
                NXLOG_I("xc2", "period getter: new caller lr=0x%llx (call #%llu)", static_cast<unsigned long long>(lr), static_cast<unsigned long long>(k));
            }
        }
        return 0;   // let the lifted function run: this only observes
    });
    std::thread([] {
        uint64_t last = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            const uint64_t c = n.load();
            NXLOG_I("xc2", "period getter: %.1f calls/s, %d distinct callers", static_cast<double>(c - last) / 2.0, nseen.load());
            last = c;
        }
    }).detach();
}

// ---- graphics object scan (diagnostic, HANDOVER 57) ---------------------------------------------------------------------------
// [xc2] obj_scan_at = N (seconds, 0 = off): N seconds after boot, follow [main+0x11c5888] (the object whose vtable slot +0x208 is the vsync
// period virtual) and log the floats that are exactly 1.0 in its first 0x4000 bytes - NS1's "graphics step" is a pair of 1.0 floats
// (+0x1c84 x +0x1c88) in that object, and its product is what still runs per frame at 30 Hz steps.
static void arm_obj_scan(uint64_t main_base) {
    const int64_t at = nxrt::config_get_int("xc2", "obj_scan_at", 0);
    if (at <= 0) return;
    std::thread([main_base, at] {
        std::this_thread::sleep_for(std::chrono::seconds(at));
        const uint64_t slot = main_base + 0x11c5000 + 888;
        uint64_t obj = 0;
        std::memcpy(&obj, reinterpret_cast<const void*>(slot), 8);
        NXLOG_I("xc2", "obj scan: [0x%llx] = 0x%llx", static_cast<unsigned long long>(slot), static_cast<unsigned long long>(obj));
        if (!obj || !nxrt::mem::is_mapped(obj, 0x4000)) return;
        const uint8_t* q = reinterpret_cast<const uint8_t*>(obj);
        for (uint32_t off = 0; off + 8 <= 0x4000; off += 4) {
            float a, b;
            std::memcpy(&a, q + off, 4);
            std::memcpy(&b, q + off + 4, 4);
            if (a == 1.0f && b == 1.0f)
                NXLOG_I("xc2", "obj scan: two 1.0 floats at +0x%x and +0x%x", off, off + 4);
        }
        NXLOG_I("xc2", "obj scan done");
    }).detach();
}

// ---- NS1 3.0.0 guest probes (diagnostic, HANDOVER 86) ------------------------------------------------------------------------
// The 2026 NS1 3.0.0 exefs (lift-ns1, HANDOVER 85) dies in DevFileThread once the menu resources have been read:
//   f_71006b3358 -> (indirect, vtable slot 2 of [x19+48]) f_710032d4bc -> f_71006e9f44 -> f_71006e927c -> svcBreak,
// and the guest log shows mlStackTrace 1..12 right before the break. Measured here: f_71006e927c(out, size, align, flag, name,
// line) is the region allocator every mtl::MemManager allocation goes through (f_71006e8f18 = create calls it with the region's
// size and name, f_71006e9f44 = allocate calls it with the request), so the 12-frame trace is printed by the allocator when the
// request does not fit: the last line this hook writes before svcBreak is the request that failed. The addresses are NS1-only: on
// the NS2 build no lifted function carries a site at those offsets, so the hooks never fire. Needs build\sites-main-ns1.txt and a
// lift that passed --sites (see HANDOVER 86).
static void ns1_read_str(uint64_t p, char* out, size_t cap) {
    out[0] = 0;
    if (!p || !nxrt::mem::is_mapped(p, 1)) return;
    const char* s = reinterpret_cast<const char*>(p);
    size_t i = 0;
    for (; i + 1 < cap; ++i) {
        const char ch = s[i];
        if (!ch) break;
        out[i] = (ch >= 0x20 && ch < 0x7f) ? ch : '.';
    }
    out[i] = 0;
}

struct Ns1Site {
    const char* tag;
    uint64_t off;
    int cap = 6;
    std::atomic<int> seen{0};
};


// ---- write watch on the dynamic-resolution record (HANDOVER 172, 189) --------------------------------------------------
// The live 3D size is a pair of int32 at [*(this+8)+0x5E0] + 24/+28. The read-point pin (XC2_RENDER_SIZE_PIN) writes that
// pair back just before the frame reads it, so the game's own switch bookkeeping never ran with the new value and the scene
// came out black (HANDOVER 189). This watches the store instead: the page holding the record is made read-only, the first
// write faults, the handler logs the store's host instruction pointer and single-steps the store, and - when a pin is
// configured - clamps what landed in the record. The game's handlers then run with the value the experiment asked for,
// which is exactly the case 189 could not test.
namespace rswatch {

struct Entry {
    uint64_t rip = 0;            // the host instruction that stored (x2_boot.exe + offset; the map file names it)
    uint64_t addr = 0;           // the address the store targeted
    uint32_t before[2] = {0, 0};
    uint32_t after[2] = {0, 0};
};

struct State {
    std::atomic<uint64_t> record{0};    // p: the struct that holds +24/+28
    uint64_t page[2] = {0, 0};
    int pages = 0;
    std::atomic<bool> armed{false};
    std::atomic<uint32_t> arms{0};
    std::atomic<uint32_t> other{0};     // faults on the page that were not the record
    bool pin = false;
    bool pin_checked = false;
    uint32_t pin_w = 0, pin_h = 0;
    std::atomic<uint32_t> head{0};
    Entry ring[32]{};
    bool started = false;
};

State& state() { static State s; return s; }

thread_local bool stepping = false;
thread_local Entry pending{};

bool watched(const State& st, uint64_t addr) {
    for (int i = 0; i < st.pages; ++i)
        if (addr >= st.page[i] && addr < st.page[i] + 0x1000) return true;
    return false;
}

void set_protect(const State& st, DWORD prot) {
    for (int i = 0; i < st.pages; ++i) {
        DWORD old = 0;
        if (!VirtualProtect(reinterpret_cast<void*>(st.page[i]), 0x1000, prot, &old)) {
            MEMORY_BASIC_INFORMATION mbi{};
            VirtualQuery(reinterpret_cast<void*>(st.page[i]), &mbi, sizeof mbi);
            NXLOG_W("rsize", "protect %llx -> %lx failed (state %lx protect %lx, error %lu)",
                    static_cast<unsigned long long>(st.page[i]), prot, mbi.State, mbi.Protect, GetLastError());
        }
    }
}

void push(const Entry& e) {
    State& st = state();
    const uint32_t i = st.head.fetch_add(1) % 32;
    st.ring[i] = e;
}

LONG CALLBACK veh(EXCEPTION_POINTERS* ep) {
    State& st = state();
    if (!st.started) return EXCEPTION_CONTINUE_SEARCH;
    EXCEPTION_RECORD* er = ep->ExceptionRecord;
    if (er->ExceptionCode == EXCEPTION_SINGLE_STEP) {
        if (!stepping) return EXCEPTION_CONTINUE_SEARCH;
        stepping = false;
        ep->ContextRecord->ContextFlags |= CONTEXT_CONTROL;
        ep->ContextRecord->EFlags &= ~0x100u;
        const uint64_t p = st.record.load();
        const void* q = reinterpret_cast<const void*>(p + 24);
        std::memcpy(pending.after, q, 8);
        if (st.pin && pending.after[0] <= 1920 && pending.after[1] <= 1080 &&
            (pending.after[0] < st.pin_w || pending.after[1] < st.pin_h)) {
            const uint32_t clamped[2] = {st.pin_w, st.pin_h};
            std::memcpy(reinterpret_cast<void*>(p + 24), clamped, 8);
            pending.after[0] = st.pin_w;
            pending.after[1] = st.pin_h;
        }
        push(pending);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2 || er->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;
    const uint64_t addr = er->ExceptionInformation[1];
    if (!watched(st, addr)) return EXCEPTION_CONTINUE_SEARCH;
    st.armed.store(false);
    set_protect(st, PAGE_READWRITE);
    const uint64_t p = st.record.load();
    if (addr >= p + 24 && addr < p + 32) {
        pending = Entry{};
        pending.rip = reinterpret_cast<uint64_t>(er->ExceptionAddress);
        pending.addr = addr;
        std::memcpy(pending.before, reinterpret_cast<const void*>(p + 24), 8);
        stepping = true;
        ep->ContextRecord->ContextFlags |= CONTEXT_CONTROL;
        ep->ContextRecord->EFlags |= 0x100u;    // trap flag: the store completes, then SINGLE_STEP fires
    } else {
        st.other.fetch_add(1);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

// Called from the frame hook once per frame: arm the page again after the previous frame's fault, and print what the last
// writes did. Logging here (not in the handler) keeps the vectored handler out of the logger's locks.
void frame(uint64_t p) {
    State& st = state();
    if (!st.started) {
        st.started = true;
        st.record.store(p);
        for (int i = 0; i < 2; ++i) {
            const uint64_t pg = (p + 24 + i * 4) & ~0xFFFULL;
            if (st.pages == 0 || st.page[st.pages - 1] != pg) {
                if (st.pages < 2) st.page[st.pages++] = pg;
            }
        }
        if (const char* s = std::getenv("XC2_RENDER_SIZE_WRITE_PIN")) {
            unsigned a = 0, b = 0;
            if (std::sscanf(s, "%ux%u", &a, &b) == 2 && a && b) { st.pin = true; st.pin_w = a; st.pin_h = b; }
        }
        AddVectoredExceptionHandler(1, veh);
        NXLOG_I("rsize", "write watch: record %llx, %d page(s), pin %s",
                static_cast<unsigned long long>(p), st.pages,
                st.pin ? "on" : "off");
    }
    static uint32_t printed = 0;
    const uint32_t head = st.head.load();
    while (printed != head) {
        const Entry& e = st.ring[printed % 32];
        ++printed;
        NXLOG_I("rsize", "write %llx at %llx+%llx: %u x %u -> %u x %u%s", static_cast<unsigned long long>(e.addr),
                static_cast<unsigned long long>(reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr))),
                static_cast<unsigned long long>(e.rip), e.before[0], e.before[1], e.after[0], e.after[1],
                st.pin ? " (clamped)" : "");
    }
    // Re-arm every frame, not only after our own fault: something else in the runtime protects heap pages as it needs
    // them (the GPU back end before a texture upload, memory.cpp's page bookkeeping), and a single VirtualProtect there
    // silently undoes ours. Asking on every frame tells us which it is.
    if (st.arms.load() < 20000) {
        MEMORY_BASIC_INFORMATION mbi{};
        VirtualQuery(reinterpret_cast<void*>(st.page[0]), &mbi, sizeof mbi);
        if (!st.armed.load() || mbi.Protect != PAGE_READONLY) {
            if (st.armed.load() && st.other.load() == 0 && !st.pin_checked) {
                st.pin_checked = true;
                NXLOG_I("rsize", "page %llx was %lx, not read-only: something else re-protected it",
                        static_cast<unsigned long long>(st.page[0]), mbi.Protect);
            }
            st.arms.fetch_add(1);
            set_protect(st, PAGE_READONLY);
            st.armed.store(true);
        }
        if (st.arms.load() == 1) {
            MEMORY_BASIC_INFORMATION mbi{};
            VirtualQuery(reinterpret_cast<void*>(st.page[0]), &mbi, sizeof mbi);
            NXLOG_I("rsize", "armed page %llx: state %lx protect %lx", static_cast<unsigned long long>(st.page[0]), mbi.State,
                    mbi.Protect);
            if (std::getenv("XC2_RENDER_SIZE_WATCH_SELFTEST")) {   // does a write we make fault at all?
                const uint64_t q = st.record.load() + 24;
                uint32_t v[2] = {0, 0};
                std::memcpy(v, reinterpret_cast<const void*>(q), 8);
                NXLOG_I("rsize", "selftest: writing %u x %u to %llx", v[0], v[1], static_cast<unsigned long long>(q));
                std::memcpy(reinterpret_cast<void*>(q), v, 8);
                NXLOG_I("rsize", "selftest: write returned");
            }
        }
    }
}


}  // namespace rswatch

static void arm_ns1_probe(uint64_t main_base) {
    // XC2_NS1_PROBE turns on the logging sites; the two memory overrides below have their own switches and run without it.
    static const bool probe = std::getenv("XC2_NS1_PROBE") != nullptr;
    static Ns1Site s_log{"log+stacktrace", 0x6e9f44};
    static Ns1Site s_stack{"mlStackTrace", 0x6e927c};
    static Ns1Site s_virt{"failing-virtual", 0x32d4bc};
    static Ns1Site s_disp{"dispatcher", 0x6b3358};
    static Ns1Site s_caller{"caller", 0x6b1608};
    static Ns1Site s_file{"file-call", 0x6bc000};
    static Ns1Site s_create{"memmgr-create", 0x6e8f18};
    for (Ns1Site* s : {&s_log, &s_stack, &s_virt, &s_disp, &s_caller, &s_file}) {
        if (!probe) break;
        // 0x6e927c is the allocator every region goes through: log every call, so the last line before svcBreak is the request
        // that did not fit. The rest are context only.
        s->cap = (s == &s_stack || s == &s_log) ? 200000 : 6;
        nxrt::hook_add_entry(main_base + s->off, [](Ctx* c, uint64_t addr, void* user) -> int {
            Ns1Site* st = static_cast<Ns1Site*>(user);
            if (st->seen.fetch_add(1) >= st->cap) return 0;
            char s1[96] = {}, s2[96] = {}, s4[96] = {};
            ns1_read_str(c->x[1], s1, sizeof s1);
            ns1_read_str(c->x[2], s2, sizeof s2);
            ns1_read_str(c->x[4], s4, sizeof s4);
            // x0 points at the u32 the allocator reads and writes: its value on entry is the ALLOC_HANDLE (which region),
            // the value on return is the block. Logging it names the region the request went to.
            uint32_t handle = 0;
            if (nxrt::mem::is_mapped(c->x[0], 4)) std::memcpy(&handle, reinterpret_cast<const void*>(c->x[0]), 4);
            // Resolve handle -> the manager object the way the allocator does (handle table at [0x7100b8d328] + 8192/+8208),
            // then dump that object for big requests: its size/used fields are what the failure is about.
            uint64_t region = 0;
            const uint64_t tabp = 0x7100b8d328ull;
            if (nxrt::mem::is_mapped(tabp, 8)) {
                uint64_t base = 0;
                std::memcpy(&base, reinterpret_cast<const void*>(tabp), 8);
                if (base && nxrt::mem::is_mapped(base + 8192, 32)) {
                    uint32_t cnt = 0, mask = 0;
                    std::memcpy(&cnt, reinterpret_cast<const void*>(base + 8192), 4);
                    std::memcpy(&mask, reinterpret_cast<const void*>(base + 8208), 4);
                    if (cnt) {
                        const uint32_t key = handle & mask;
                        const uint64_t ent = base + static_cast<uint64_t>(key % cnt) * 16;
                        uint32_t k = 0;
                        std::memcpy(&k, reinterpret_cast<const void*>(ent + 8), 4);
                        if (k == key) std::memcpy(&region, reinterpret_cast<const void*>(ent), 8);
                    }
                }
            }
            static const uint64_t dump_min = [] {
                const char* v = std::getenv("XC2_NS1_DUMP_MIN");
                return v ? static_cast<uint64_t>(std::strtoull(v, nullptr, 0)) : 0x100000ull;
            }();
            if (region && c->x[1] >= dump_min && nxrt::mem::is_mapped(region, 0x100)) {
                const uint64_t* q = reinterpret_cast<const uint64_t*>(region);
                NXLOG_I("ns1", "  region %llx: +0=%llx +8=%llx +10=%llx +18=%llx +20=%llx +28=%llx +30=%llx +38=%llx +40=%llx +48=%llx +50=%llx +58=%llx +60=%llx +68=%llx +70=%llx +78=%llx +a8=%llx +b0=%llx",
                        static_cast<unsigned long long>(region), static_cast<unsigned long long>(q[0]), static_cast<unsigned long long>(q[1]),
                        static_cast<unsigned long long>(q[2]), static_cast<unsigned long long>(q[3]), static_cast<unsigned long long>(q[4]),
                        static_cast<unsigned long long>(q[5]), static_cast<unsigned long long>(q[6]), static_cast<unsigned long long>(q[7]),
                        static_cast<unsigned long long>(q[8]), static_cast<unsigned long long>(q[9]), static_cast<unsigned long long>(q[10]),
                        static_cast<unsigned long long>(q[11]), static_cast<unsigned long long>(q[12]), static_cast<unsigned long long>(q[13]),
                        static_cast<unsigned long long>(q[14]), static_cast<unsigned long long>(q[15]), static_cast<unsigned long long>(q[21]),
                        static_cast<unsigned long long>(q[22]));
            }
            NXLOG_I("ns1", "%s f_%llx: h=%u x0=%llx x1=%llx[%s] x2=%llx[%s] x3=%llx x4=%llx[%s] x5=%llx lr=%llx", st->tag,
                    static_cast<unsigned long long>(addr & 0xffffffffffULL), handle, static_cast<unsigned long long>(c->x[0]),
                    static_cast<unsigned long long>(c->x[1]), s1, static_cast<unsigned long long>(c->x[2]), s2,
                    static_cast<unsigned long long>(c->x[3]), static_cast<unsigned long long>(c->x[4]), s4,
                    static_cast<unsigned long long>(c->x[5]), static_cast<unsigned long long>(c->x[30]));
            return 0;   // observe only
        }, s);
    }
    // XC2_NS1_BUMP=Name=bytes[,Name=bytes]: raise a compiled-in mtl::MemManager region budget at create time. The NS1 3.0.0
    // build sizes GfFontRsc at 20 MiB (main+0x32d808, mov w1,#0x1400000) and pkg7's fonts need more than that: one 19.3 MB
    // allocation leaves ~1.4 MB, and the next 2.64 MB request is what aborts (HANDOVER 86). x1 is the size, x2 the name.
    struct Bump { std::string name; uint64_t bytes; };
    static std::vector<Bump> bumps;
    if (const char* spec = std::getenv("XC2_NS1_BUMP")) {
        const char* p = spec;
        while (*p) {
            const char* eq = std::strchr(p, '=');
            if (!eq) break;
            const char* end = std::strchr(eq + 1, ',');
            const std::string name(p, static_cast<size_t>(eq - p));
            const uint64_t bytes = std::strtoull(eq + 1, nullptr, 0);
            if (!name.empty() && bytes) bumps.push_back({name, bytes});
            if (!end) break;
            p = end + 1;
        }
    }
    // XC2_NS1_SCALE=<n>: multiply every region budget by n (diagnostic). The compiled budgets fit the NS1 asset set; pkg7's
    // menu/font data needs several times the space (HANDOVER 86). Nested regions scale with their parent, so this stays
    // self-consistent; the root arena has to be raised with XC2_NS1_ROOT_ARENA.
    static const uint64_t scale = [] {
        const char* v = std::getenv("XC2_NS1_SCALE");
        const uint64_t n = v ? std::strtoull(v, nullptr, 0) : 0;
        return n ? n : 1ull;
    }();
    s_create.cap = 200;
    nxrt::hook_add_entry(main_base + s_create.off, [](Ctx* c, uint64_t, void*) -> int {
        char nm[64];
        ns1_read_str(c->x[2], nm, sizeof nm);
        if (probe && s_create.seen.fetch_add(1) < s_create.cap)
            NXLOG_I("ns1", "memmgr-create %s: size 0x%llx = %llu KB", nm, static_cast<unsigned long long>(c->x[1]),
                    static_cast<unsigned long long>(c->x[1] >> 10));
        if (scale > 1 && c->x[1] > 0 && c->x[1] < 0x40000000ull) c->x[1] *= scale;
        for (const Bump& b : bumps) {
            if (b.name != nm) continue;
            NXLOG_W("ns1", "memmgr-create %s: budget 0x%llx -> 0x%llx (XC2_NS1_BUMP)", nm,
                    static_cast<unsigned long long>(c->x[1]), static_cast<unsigned long long>(b.bytes));
            c->x[1] = b.bytes;
            break;
        }
        return 0;   // let the lifted create run with the (possibly raised) size
    }, nullptr);
    // XC2_NS1_ROOT_ARENA=<bytes>: raise the root arena at MemManager::initialize(setup const*). The struct carries three
    // (address, size) pairs plus a u32; the 3.0.0 NS1 build asks for 0x8c400000 = 2244 MiB (main+0x6421b4) and every region is
    // carved out of it, so the compiled budgets only fit the NS1 asset set (HANDOVER 86). x0 = &struct, the root size is at +16.
    // The struct carries three arenas: +16 = 2244 MiB (MLSYSHANDLE), +32 = 216 MiB, +48 = 512 MiB (the GLIB* arena). Regions
    // are carved out of them, so both have to grow together with XC2_NS1_SCALE.
    static const uint64_t arena_scale = [] {
        const char* v = std::getenv("XC2_NS1_ARENA_SCALE");
        const uint64_t n = v ? std::strtoull(v, nullptr, 0) : 0;
        return n ? n : 1ull;
    }();
    static const uint64_t root_arena = [] {
        const char* v = std::getenv("XC2_NS1_ROOT_ARENA");
        return v ? static_cast<uint64_t>(std::strtoull(v, nullptr, 0)) : 0ull;
    }();
    if (arena_scale > 1 || root_arena) {
        nxrt::hook_add_entry(main_base + 0x6e3f34, [](Ctx* c, uint64_t, void*) -> int {
            if (!nxrt::mem::is_mapped(c->x[0], 64)) return 0;
            for (uint64_t off : {16ull, 32ull, 48ull}) {
                uint64_t old = 0;
                std::memcpy(&old, reinterpret_cast<const void*>(c->x[0] + off), 8);
                if (!old) continue;
                uint64_t want = old * arena_scale;
                if (off == 16 && root_arena) want = root_arena;
                std::memcpy(reinterpret_cast<void*>(c->x[0] + off), &want, 8);
                NXLOG_W("ns1", "MemManager::initialize: arena at +0x%llx 0x%llx -> 0x%llx", static_cast<unsigned long long>(off),
                        static_cast<unsigned long long>(old), static_cast<unsigned long long>(want));
            }
            return 0;   // let initialize run with the raised sizes
        }, nullptr);
    }
    // XC2_NS1_NPAD_PROBE=1: what the guest is handed by nn::hid::detail::GetNpadStates (sdk+0x203a4c). An entry hook has no return
    // hook, so it dumps the PREVIOUS call's out buffer; the game calls this every frame, so that is last frame's state. This is the
    // last link of the input chain: shared memory -> SDK -> guest (HANDOVER 86.16).
    if (std::getenv("XC2_NS1_NPAD_PROBE")) {
        // The game's own call: nn::hid::GetNpadState(out, &id) at sdk+0x21406c, ret at sdk+0x2140a4. arg0 is the caller's buffer,
        // so this shows exactly what the guest is handed (HANDOVER 86.17).
        static uint64_t pub_out = 0, pub_lr = 0;
        static uint32_t pub_id = 0, pub_calls = 0, pub_shown = 0;
        nxrt::hook_add_entry(0x750021406cull, [](Ctx* c, uint64_t, void*) -> int {
            pub_out = c->x[0];
            pub_lr = c->x[30];
            pub_id = nxrt::mem::is_mapped(c->x[1], 4) ? *reinterpret_cast<const uint32_t*>(c->x[1]) : 0;
            ++pub_calls;
            return 0;
        }, nullptr);
        nxrt::hook_add_mid(0x75002140a4ull, [](Ctx*, uint64_t, void*) -> int {
            if (!pub_out || !nxrt::mem::is_mapped(pub_out, 40) || pub_shown >= 60) return 0;
            const uint64_t buttons = *reinterpret_cast<const uint64_t*>(pub_out + 8);
            int32_t lx = 0, ly = 0;
            std::memcpy(&lx, reinterpret_cast<const void*>(pub_out + 16), 4);
            std::memcpy(&ly, reinterpret_cast<const void*>(pub_out + 20), 4);
            if (!buttons && !lx && !ly) return 0;
            ++pub_shown;
            NXLOG_I("ns1", "npad pub: buttons=%llx lx=%d ly=%d id=%u calls=%u lr=%llx", static_cast<unsigned long long>(buttons), lx, ly,
                    pub_id, pub_calls, static_cast<unsigned long long>(pub_lr));
            return 0;
        }, nullptr);
        // At the function's ret the out buffer holds the result; [sp+8] is where the prologue saved it.
        nxrt::hook_add_mid(0x7500203cb0ull, [](Ctx* c, uint64_t, void*) -> int {
            static uint32_t shown = 0;
            const uint64_t sp = c->x[31];
            if (!nxrt::mem::is_mapped(sp + 8, 8) || shown >= 80) return 0;
            uint64_t out = 0;
            std::memcpy(&out, reinterpret_cast<const void*>(sp + 8), 8);
            if (!out || !nxrt::mem::is_mapped(out, 40)) return 0;
            const uint64_t sampling = *reinterpret_cast<const uint64_t*>(out);
            const uint64_t buttons = *reinterpret_cast<const uint64_t*>(out + 8);
            int32_t lx = 0, ly = 0;
            std::memcpy(&lx, reinterpret_cast<const void*>(out + 16), 4);
            std::memcpy(&ly, reinterpret_cast<const void*>(out + 20), 4);
            uint32_t attr = 0;
            std::memcpy(&attr, reinterpret_cast<const void*>(out + 32), 4);
            if (!buttons && !lx && !ly) return 0;
            ++shown;
            NXLOG_I("ns1", "npad ret: buttons=%llx sampling=%llu lx=%d ly=%d attr=%x", static_cast<unsigned long long>(buttons),
                    static_cast<unsigned long long>(sampling), lx, ly, attr);
            return 0;
        }, nullptr);
        nxrt::hook_add_entry(0x7500203a4cull, [](Ctx* c, uint64_t, void*) -> int {
            static uint64_t last_out = 0;
            static uint32_t shown = 0, calls = 0;
            ++calls;
            if (last_out && nxrt::mem::is_mapped(last_out, 40) && shown < 60) {
                const uint64_t sampling = *reinterpret_cast<const uint64_t*>(last_out);
                const uint64_t buttons = *reinterpret_cast<const uint64_t*>(last_out + 8);
                int32_t lx = 0, ly = 0;
                std::memcpy(&lx, reinterpret_cast<const void*>(last_out + 16), 4);
                std::memcpy(&ly, reinterpret_cast<const void*>(last_out + 20), 4);
                uint32_t attr = 0;
                std::memcpy(&attr, reinterpret_cast<const void*>(last_out + 32), 4);
                uint32_t id = 0;
                if (nxrt::mem::is_mapped(c->x[3], 4)) std::memcpy(&id, reinterpret_cast<const void*>(c->x[3]), 4);
                if (buttons || lx || ly) {
                    ++shown;
                    NXLOG_I("ns1", "npad out: buttons=%llx sampling=%llu lx=%d ly=%d attr=%x (id %u, count %u, call %u)",
                            static_cast<unsigned long long>(buttons), static_cast<unsigned long long>(sampling), lx, ly, attr, id,
                            static_cast<unsigned>(c->x[2] & 0xffffffffu), calls);
                }
            }
            last_out = c->x[1];
            return 0;
        }, nullptr);
    }
    // XC2_RENDER_SIZE_PROBE=1: where the dynamic resolution actually lives. f_7100844120 (main+0x844120) reads four int32 at
    // [*(this+8) + 0x5E0] + 24..39, converts them to float and hands them to main+0x8442E0, which multiplies them by 1.0 or
    // 0.5 to size the twenty render targets. Those four are the live 3D size - not the lib_nx.ini settings object, which was
    // searched and came up empty (HANDOVER 95.16). Logging them (and, once they match the ladder, writing 1280x720 back) is
    // the host-side equivalent of what the NS1 quality mods do with one branch instruction.
    // Registered for the lock as well as the probe: [upscale] lock_render_size is what the launcher writes, and the hook
    // is the only place the pin can run. With the lock alone the hook stays quiet (no per-frame logging).
    static const bool rs_probe = std::getenv("XC2_RENDER_SIZE_PROBE") != nullptr;   // static: the hook is a plain fn pointer
    static const bool rs_watch = std::getenv("XC2_RENDER_SIZE_WATCH") != nullptr ||
                                 std::getenv("XC2_RENDER_SIZE_WRITE_PIN") != nullptr;
    const bool rs_lock = xc2_lock_render_size();
    if (rs_probe || rs_lock || rs_watch) {
        static int shown = 0;
        static uint32_t last0 = 0xFFFFFFFFu, last1 = 0xFFFFFFFFu;
        nxrt::hook_add_entry(main_base + 0x844120, [](Ctx* c, uint64_t, void*) -> int {
            const uint64_t self = c->x[0];
            if (!nxrt::mem::is_mapped(self + 8, 8)) return 0;
            uint64_t obj = 0;
            std::memcpy(&obj, reinterpret_cast<const void*>(self + 8), 8);
            if (!obj || !nxrt::mem::is_mapped(obj + 0x5E0, 8)) return 0;
            uint64_t p = 0;
            std::memcpy(&p, reinterpret_cast<const void*>(obj + 0x5E0), 8);
            if (!p || !nxrt::mem::is_mapped(p + 24, 16)) return 0;
            uint32_t v[4] = {};
            std::memcpy(v, reinterpret_cast<const void*>(p + 24), 16);
            // XC2_RENDER_SIZE_PIN=1280x720: pin the 3D render size instead of only logging it. The ladder only ever goes
            // down from 1280x720 (the console saves time, a desktop does not need to), and this record is what the twenty
            // render targets below are sized from. Pin only while the game is inside the 3D range: the first calls carry
            // the 4K output size, which must not be touched.
            static uint32_t pin_w = 0, pin_h = 0;
            static bool pin_read = false;
            if (!pin_read) {
                pin_read = true;
                if (const char* s = std::getenv("XC2_RENDER_SIZE_PIN")) {
                    unsigned a = 0, b = 0;
                    if (std::sscanf(s, "%ux%u", &a, &b) == 2 && a && b) { pin_w = a; pin_h = b; }
                }
                // The launcher's Picture page writes this: lock the ladder at its top rung (the game's fixed 1280x720).
                if (!pin_w && xc2_lock_render_size()) { pin_w = 1280; pin_h = 720; }
            }
            if (pin_w && v[0] <= 1920 && v[1] <= 1080 && (v[0] < pin_w || v[1] < pin_h)) {
                const uint32_t pinned[2] = {pin_w, pin_h};
                std::memcpy(reinterpret_cast<void*>(p + 24), pinned, 8);
                v[0] = pin_w;
                v[1] = pin_h;
            }
            // XC2_RENDER_SIZE_WATCH=1 (log) or XC2_RENDER_SIZE_WRITE_PIN=WxH (clamp): watch the store that sets the record
            // instead of writing the record back here, so the game's own switch handling runs with the pinned value (189).
            if (rs_watch && v[0] && v[1] && v[0] <= 1920 && v[1] <= 1080) rswatch::frame(p);
            // XC2_TAA_SHARP=2.0: the game's own per-rung TAA sharpening (tmaaSharpness_0..5, found at +0x3114 by
            // matching the lib_nx.ini defaults 0.4/0.6/0.6/0.72/0.8/0.9). Raising all six to one value answers the
            // question the values cannot: is that pass still in the chain when our FSR replaces the DLSS step? Written
            // every call, because the settings object is re-read each frame.
            static float taa_sharp = -1.0f;
            static bool taa_read = false;
            if (!taa_read) {
                taa_read = true;
                if (const char* s = std::getenv("XC2_TAA_SHARP")) taa_sharp = static_cast<float>(std::atof(s));
            }
            if (taa_sharp >= 0.0f && nxrt::mem::is_mapped(obj + 0x3114, 24)) {
                const float six[6] = {taa_sharp, taa_sharp, taa_sharp, taa_sharp, taa_sharp, taa_sharp};
                std::memcpy(reinterpret_cast<void*>(obj + 0x3114), six, sizeof six);
            }
            if (std::getenv("XC2_SETTINGS_DUMP")) {
                static bool dumped = false;
                if (!dumped) {
                    dumped = true;
                    // One contiguous window around the block the value scan pointed at (tmaa +0x3114, AA_Sharpness
                    // +0x3100), printed as offset: value so the keys can be matched by hand against lib_nx.ini order.
                    const uint32_t lo = 0x2F80, hi = 0x3180;
                    if (nxrt::mem::is_mapped(obj + lo, hi - lo)) {
                        for (uint32_t off = lo; off < hi; off += 16) {
                            float f[4];
                            std::memcpy(f, reinterpret_cast<const void*>(obj + off), 16);
                            NXLOG_I("settings", "+0x%04x  %11.6g %11.6g %11.6g %11.6g", off, static_cast<double>(f[0]),
                                    static_cast<double>(f[1]), static_cast<double>(f[2]), static_cast<double>(f[3]));
                        }
                    }
                }
            }
            if (std::getenv("XC2_SETTINGS_SCAN")) {
                static bool scanned = false;
                if (!scanned) {
                    scanned = true;
                    // The lib_nx.ini defaults are known from the archive itself (work/tmp/lib_nx.ini), so the settings
                    // object can be matched by value instead of by disassembling the parser: six consecutive sharpness
                    // rungs, the red_* block, and a few values nothing else uses. Offsets are from the object base.
                    const uint64_t base = obj;
                    const uint32_t span = 0x4000;
                    if (nxrt::mem::is_mapped(base, span)) {
                        std::vector<float> f(span / 4);
                        std::memcpy(f.data(), reinterpret_cast<const void*>(base), span);
                        auto dump = [&](uint32_t off, const char* what) {
                            NXLOG_I("settings", "%s at +0x%x (%.6g)", what, off, static_cast<double>(f[off / 4]));
                        };
                        for (uint32_t off = 0; off + 40 <= span; off += 4) {
                            const float* q = &f[off / 4];
                            if (q[0] == 0.4f && q[1] == 0.6f && q[2] == 0.6f && q[3] == 0.72f && q[4] == 0.8f && q[5] == 0.9f)
                                dump(off, "tmaaSharpness_0..5 run");
                            if (q[0] == 1.0f && q[1] == 1.0f && q[2] == 1.0f && q[3] == 1.0f && q[5] == 1.0f && q[6] == 1.0f &&
                                q[7] == 0.7f && q[8] == 0.7f && q[9] == 97.0f)
                                dump(off, "red_sclX..red_AtRate block");
                        }
                        for (uint32_t off = 0; off + 4 <= span; off += 4) {
                            const float x = f[off / 4];
                            if (x == 80000.0f) dump(off, "cam_far");
                            else if (x == 8.0f) dump(off, "AA_Sharpness");
                            else if (x == 0.75f) dump(off, "ao_blurScale");
                            else if (x == 0.35f) dump(off, "tssaoOutputAO");
                            else if (x == 0.2f) dump(off, "blurStr / godray-ish");
                        }
                    }
                }
            }
            if (rs_probe && (v[0] != last0 || v[1] != last1) && shown < 40) {
                ++shown;
                last0 = v[0];
                last1 = v[1];
                NXLOG_I("rsize", "render size: %u x %u (v2 %u v3 %u) obj %llx +5E0 %llx", v[0], v[1], v[2], v[3],
                        static_cast<unsigned long long>(obj), static_cast<unsigned long long>(p));
            }
            return 0;
        }, nullptr);
    }
    // XC2_NS1_UI_PROBE=1: the UI's own view of the input. f_7100830b18 (main+0x830b18) is a cursor widget update that reads the
    // per-player input word at [this + 160 + idx*4] (idx = *(u32*)(*(u64*)0x7100b8e000+1656)), tests the four direction bits (0xf000)
    // and moves its own cursor state at [this+168]. The pad flags carry A as 0x4 (HANDOVER 86.19), so this shows whether the UI is
    // handed that word at all: never 0x4 while A is down means the copy from the pad entries drops it; 0x4 present means the confirm
    // consumer (elsewhere) is gated.
    if (std::getenv("XC2_NS1_UI_PROBE")) {
        static uint32_t ui_last = 0xFFFFFFFFu, ui_shown = 0, ui_calls = 0;
        nxrt::hook_add_entry(main_base + 0x830b18, [](Ctx* c, uint64_t, void*) -> int {
            ++ui_calls;
            uint32_t idx = 0;
            const uint64_t gp = 0x7100b8e000ull + 1656;
            if (nxrt::mem::is_mapped(gp, 8)) {
                uint64_t p = 0;
                std::memcpy(&p, reinterpret_cast<const void*>(gp), 8);
                if (p && nxrt::mem::is_mapped(p, 4)) std::memcpy(&idx, reinterpret_cast<const void*>(p), 4);
            }
            const uint64_t thisp = c->x[0];
            const uint64_t wa = thisp + 160 + static_cast<uint64_t>(idx) * 4;
            uint32_t word = 0, st = 0;
            if (nxrt::mem::is_mapped(wa, 4)) std::memcpy(&word, reinterpret_cast<const void*>(wa), 4);
            if (nxrt::mem::is_mapped(thisp + 168, 4)) std::memcpy(&st, reinterpret_cast<const void*>(thisp + 168), 4);
            if (word != ui_last && ui_shown < 120) {
                ++ui_shown;
                ui_last = word;
                NXLOG_I("ns1", "ui word: this=%llx idx=%u word=0x%x state+168=0x%x calls=%u", static_cast<unsigned long long>(thisp),
                        idx, word, st, ui_calls);
            }
            return 0;
        }, nullptr);
        static uint32_t pad_last = 0xFFFFFFFFu, pad_shown = 0;
        nxrt::hook_add_entry(main_base + 0x6ac804, [](Ctx* c, uint64_t, void*) -> int {
            const uint64_t entry = c->x[1];
            uint32_t flags = 0;
            if (nxrt::mem::is_mapped(entry, 4)) std::memcpy(&flags, reinterpret_cast<const void*>(entry), 4);
            if (flags != pad_last && pad_shown < 80) {
                ++pad_shown;
                pad_last = flags;
                NXLOG_I("ns1", "pad flags: entry=%llx flags=0x%x", static_cast<unsigned long long>(entry), flags);
            }
            return 0;
        }, nullptr);
        NXLOG_W("ns1", "UI input probe armed at main+0x830b18 and main+0x6ac804 (XC2_NS1_UI_PROBE)");
    }
    if (probe) NXLOG_W("ns1", "NS1 guest probes armed at 6 entry sites (XC2_NS1_PROBE)");
}

// ---- NS2 dev build + pkg7: memory budgets (HANDOVER 87.12) -------------------------------------------------------------------
// The NS2 dev build creates its regions with mtl::MemManager::create(handle, size, name, bool, type, align, name2, line)
// (main+0x77ff84, already an entry site in build\sites-main.txt). HANDOVER 84 bumped ONE region through the MenuResource call
// site (midhook 0x6fa790) and the run aborted 0.6 ms later: the region bases are laid out ahead of time, so a single region that
// grows pushes into the next. Scaling EVERY region keeps that layout self-consistent - the trick HANDOVER 86 used on the NS1
// build. initialize() (main+0x77ef28) takes no argument here: the regions are carved from the game heap, so there is no separate
// arena struct to grow.
// XC2_NS2_SCALE_NAMES=GfMenu,MenuResource,... : only these regions are grown (empty: all of them).
// The name the create entry hook saw, per thread, so the return hook can pair it with the handle the call returns.
static thread_local char t_create_name[48];

static bool ns2_scale_name(const char* nm) {
    static const std::string list = [] {
        const char* v = std::getenv("XC2_NS2_SCALE_NAMES");
        return std::string(v ? v : "");
    }();
    if (list.empty()) return true;
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        const size_t end = comma == std::string::npos ? list.size() : comma;
        const std::string frag = list.substr(pos, end - pos);
        if (!frag.empty() && std::strcmp(frag.c_str(), nm) == 0) return true;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return false;
}

// XC2_NS2_ROOT=MLSYSHANDLE=4096M,MLGRAHANDLE=1G : absolute sizes for the root arenas (K/M/G suffixes; 0 or absent = keep the
// size the game asks for). XC2_NS2_SIZE=<name>=<size>,... does the same for any region (AssetMain=1180M, MenuResource=600M):
// the regions the compiled budgets are too small for, one absolute number each, instead of one multiplier for a whole list.
static uint64_t ns2_env_bytes(const char* env, const char* nm) {
    static const std::string root_list = [] {
        const char* v = std::getenv("XC2_NS2_ROOT");
        return std::string(v ? v : "");
    }();
    static const std::string size_list = [] {
        const char* v = std::getenv("XC2_NS2_SIZE");
        return std::string(v ? v : "");
    }();
    const std::string& list = std::strcmp(env, "XC2_NS2_SIZE") == 0 ? size_list : root_list;
    if (list.empty() || !nm || !nm[0]) return 0;
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        const size_t end = comma == std::string::npos ? list.size() : comma;
        const std::string frag = list.substr(pos, end - pos);
        const size_t eq = frag.find('=');
        if (eq != std::string::npos && frag.compare(0, eq, nm) == 0) {
            const char* v = frag.c_str() + eq + 1;
            char* stop = nullptr;
            uint64_t n = std::strtoull(v, &stop, 0);
            if (stop && (*stop == 'K' || *stop == 'k')) n <<= 10;
            else if (stop && (*stop == 'M' || *stop == 'm')) n <<= 20;
            else if (stop && (*stop == 'G' || *stop == 'g')) n <<= 30;
            return n;
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return 0;
}

// A block of our own for a root arena. The four root arenas are created with
// mtl::MemManager::create(void* block, size, name, bool, type) (main+0x77fc6c) and the game hands in a block it allocated from its
// own heap. Those blocks sit back to back (measured: MLGRAHANDLE 0x780d35f000 + 512 MB = MLSYSHANDLE 0x782d35f000 + 2244 MB =
// MLDEBHANDLE 0x78b975f000 + 216 MB = sdkFileMem 0x78c6f5f000), so a region that grows in place runs into the next one - which is
// why AssetMain x2 moved the crash instead of fixing it (HANDOVER 84.3, 87.12).
//
// The block has to be HEAP memory even though the guest allocator must never hand it out: the engine maps thread stacks out of a
// region with svcMapMemory, and that call refuses a source outside the heap (svc_memory.cpp map_memory: in_heap(src, size, heap_now)).
// A block carved from the section above the guest's grant - the first thing tried - dies there: 20 MapMemory attempts at stack
// addresses, all 0xD401, then svcBreak at 0.44 s (measured). So the heap view itself is grown (nxrt::svc_heap_extend) and the block
// comes from the part above what SetHeapSize granted: heap memory as far as the SVCs are concerned, and invisible to the guest's
// allocator, which owns [kHeapBase, granted) only.
static uint64_t ns2_own_block(uint64_t size, const char* nm) {
    if (!nxrt::mem::heap_exists()) {
        NXLOG_E("ns2", "root %s: no guest heap yet, cannot place a block of our own", nm);
        return 0;
    }
    size = (size + 0xffffull) & ~0xffffull;
    const uint64_t addr = nxrt::svc_heap_extend(size);
    if (!addr) {
        NXLOG_E("ns2", "root %s: the heap cannot grow by %llu MB (heap %llu MB of the %llu MB section)", nm,
                static_cast<unsigned long long>(size >> 20), static_cast<unsigned long long>(nxrt::mem::heap_granted() >> 20),
                static_cast<unsigned long long>(nxrt::mem::heap_reserved() >> 20));
        return 0;
    }
    return addr;
}

static void arm_ns2_mem_probe(uint64_t main_base) {
    static const bool probe = std::getenv("XC2_NS2_PROBE") != nullptr;
    static const uint64_t scale = [] {
        const char* v = std::getenv("XC2_NS2_SCALE");
        const uint64_t n = v ? std::strtoull(v, nullptr, 0) : 0;
        return n ? n : 1ull;
    }();
    static const bool root_set = std::getenv("XC2_NS2_ROOT") != nullptr || std::getenv("XC2_NS2_SIZE") != nullptr ||
                                 std::getenv("XC2_NS2_QUEST_SKIP") != nullptr;
    if (!probe && scale == 1 && !root_set) return;
    static std::atomic<int> shown{0};
    nxrt::hook_add_entry(main_base + 0x77ff84, [](Ctx* c, uint64_t, void*) -> int {
        char nm[48] = {};
        if (nxrt::mem::is_mapped(c->x[2], 1)) {
            for (int i = 0; i < 40; ++i) {
                const char ch = *reinterpret_cast<const char*>(static_cast<uintptr_t>(c->x[2] + i));
                if (!ch) break;
                nm[i] = ch;
            }
        }
        const uint64_t size = c->x[1];
        if (probe) std::memcpy(t_create_name, nm, sizeof t_create_name);
        if (probe && shown.fetch_add(1) < 200)
            NXLOG_I("ns2", "memmgr create %-14s 0x%llx = %llu KB (align 0x%llx)", nm, static_cast<unsigned long long>(size),
                    static_cast<unsigned long long>(size >> 10), static_cast<unsigned long long>(c->x[5]));
        // Scaling EVERY region overflows the arenas that are fixed blocks (GLIBTOP 478 MB lives in MLGRAHANDLE, 512 MB):
        // XC2_NS2_SCALE_NAMES picks the regions to grow. The nested ones must grow together (GfMenu is the parent of
        // MenuResource, HANDOVER 86: 0x500000 + 0x5500000 = 0x5A00000 exactly), and their sum has to stay inside MLSYSHANDLE.
        const uint64_t abs = ns2_env_bytes("XC2_NS2_SIZE", nm);
        if (abs) {
            if (abs != size) NXLOG_W("ns2", "region %-14s %llu MB -> %llu MB (XC2_NS2_SIZE)", nm,
                                     static_cast<unsigned long long>(size >> 20), static_cast<unsigned long long>(abs >> 20));
            c->x[1] = abs;
        } else if (scale > 1 && size > 0 && size < 0x40000000ull && ns2_scale_name(nm)) {
            c->x[1] = size * scale;
        }
        return 0;
    }, nullptr);
    // The other two overloads: create(char const* name) and create(void* block, size, name, bool). The second one takes a fixed
    // memory block, which is how the root region is usually made; it is scaled too, so the children have room.
    nxrt::hook_add_entry(main_base + 0x77f770, [](Ctx* c, uint64_t, void*) -> int {
        char nm[48] = {};
        if (nxrt::mem::is_mapped(c->x[0], 1)) {
            for (int i = 0; i < 40; ++i) {
                const char ch = *reinterpret_cast<const char*>(static_cast<uintptr_t>(c->x[0] + i));
                if (!ch) break;
                nm[i] = ch;
            }
        }
        if (probe) NXLOG_I("ns2", "memmgr create(name) %s", nm);
        return 0;
    }, nullptr);
    nxrt::hook_add_entry(main_base + 0x77fc6c, [](Ctx* c, uint64_t, void*) -> int {
        char nm[48] = {};
        if (nxrt::mem::is_mapped(c->x[2], 1)) {
            for (int i = 0; i < 40; ++i) {
                const char ch = *reinterpret_cast<const char*>(static_cast<uintptr_t>(c->x[2] + i));
                if (!ch) break;
                nm[i] = ch;
            }
        }
        const uint64_t size = c->x[1];
        if (probe) NXLOG_I("ns2", "memmgr create(block) %-14s ptr=0x%llx size=0x%llx = %llu KB", nm,
                           static_cast<unsigned long long>(c->x[0]), static_cast<unsigned long long>(size),
                           static_cast<unsigned long long>(size >> 10));
        // Root arenas: replace (block, size) with a block of our own. Scaling the size in place cannot work here - the blocks are
        // adjacent, so a bigger one runs into its neighbour (HANDOVER 84.3). XC2_NS2_ROOT names the absolute size,
        // XC2_NS2_SCALE * SCALE_NAMES still multiplies (which is how the children are grown: AssetMain x2 needs MLSYSHANDLE x2).
        uint64_t want = ns2_env_bytes("XC2_NS2_ROOT", nm);
        if (!want && scale > 1 && ns2_scale_name(nm)) want = size * scale;
        if (want > size) {
            const uint64_t blk = ns2_own_block(want, nm);
            if (blk) {
                NXLOG_W("ns2", "root %-12s block 0x%llx (%llu MB) -> own block 0x%llx (%llu MB)", nm,
                        static_cast<unsigned long long>(c->x[0]), static_cast<unsigned long long>(size >> 20),
                        static_cast<unsigned long long>(blk), static_cast<unsigned long long>(want >> 20));
                c->x[0] = blk;
                c->x[1] = want;
            } else {
                NXLOG_W("ns2", "root %-12s keeps the game's block of %llu MB (no own block)", nm,
                        static_cast<unsigned long long>(size >> 20));
            }
        } else if (scale > 1 && size > 0 && size < 0x40000000ull && ns2_scale_name(nm)) {
            c->x[1] = size * scale;
        }
        return 0;
    }, nullptr);
    // Which region a big allocation goes to: allocate(ALLOC_HANDLE handle, size, align, bool, name, line) at main+0x781798
    // (an entry site already). Handles are handed out in creation order, so the Nth create in the log is handle N.
    static std::atomic<int> alloc_logs{0};
    nxrt::hook_add_entry(main_base + 0x781798, [](Ctx* c, uint64_t, void*) -> int {
        const uint64_t size = c->x[1];
        if (size >= 0x100000ull && alloc_logs.fetch_add(1) < 200) {
            char nm[32] = {};
            if (nxrt::mem::is_mapped(c->x[4], 1)) {
                for (int i = 0; i < 24; ++i) {
                    const char ch = *reinterpret_cast<const char*>(static_cast<uintptr_t>(c->x[4] + i));
                    if (!ch) break;
                    nm[i] = ch;
                }
            }
            NXLOG_I("ns2", "alloc h=%u size=%llu KB (%s)", static_cast<unsigned>(c->x[0]),
                    static_cast<unsigned long long>(size >> 10), nm);
        }
        return 0;
    }, nullptr);
    // The return of create (main+0x7802e4, the function's only ret): x0 is the handle the new region got. Pairing it with the name
    // the entry hook saved says which region an allocation handle belongs to (the handles are opaque tokens, not indices).
    nxrt::hook_add_mid(main_base + 0x7802e4, [](Ctx* c, uint64_t, void*) -> int {
        if (probe && t_create_name[0])
            NXLOG_I("ns2", "create ret %-14s -> handle 0x%x (%u)", t_create_name, static_cast<unsigned>(c->x[0]),
                    static_cast<unsigned>(c->x[0]));
        return 0;
    }, nullptr);
    // The NS2 dev build cannot hold 3.0.0's order-quest list: gf::GfQuestManager::reqistLinkQuest (main+0x36e594) pushes into a
    // mm::fixed_vector of capacity 16 (entries 0x48 bytes at this+0x4380000, count at this+0x8B80000) and asserts
    // "size() < capacity()" (mm_mtl/fixed_vector.h:227) on the 17th. pkg7's quest data has more than 16 that pass the filters;
    // the same program on pkg0 data never asserts (measured: 40 s clean), so this is the dev build being older than the data, the
    // same wall as HANDOVER 79 (btl WpnType) and 87.18 (sound/*.pck). XC2_NS2_QUEST_SKIP=1 skips the registration once the vector
    // is full instead of aborting - a diagnostic that lets the title applet finish so the NS2 UI/audio/input can be looked at, at
    // the cost of a truncated order-quest list. Every lifted function has an entry hook, so no re-lift is needed.
    if (std::getenv("XC2_NS2_QUEST_SKIP")) {
        static std::atomic<int> skipped{0};
        nxrt::hook_add_entry(main_base + 0x36e594, [](Ctx* c, uint64_t, void*) -> int {
            const uint64_t count_at = c->x[0] + 0x8B80000ull;   // movz 0x40000 + movk 0x8b8: movk replaces bits 31..16
            uint64_t n = 0;
            const bool mapped = nxrt::mem::is_mapped(count_at, 8);
            if (mapped) std::memcpy(&n, reinterpret_cast<const void*>(static_cast<uintptr_t>(count_at)), 8);
            static std::atomic<int> calls{0};
            const int call = calls.fetch_add(1);
            if (call < 4)
                NXLOG_I("ns2", "reqistLinkQuest #%d this=0x%llx id=0x%llx mapped=%d n=0x%llx", call,
                        static_cast<unsigned long long>(c->x[0]), static_cast<unsigned long long>(c->x[1]), mapped ? 1 : 0,
                        static_cast<unsigned long long>(n));
            // With pkg7's data the field holds 0x24924924924900ff (the 0x2492 fill pattern with 255 in the low bits); with pkg0
            // data the same address reads 0. Clearing it to 0 only moves the assert 1.7 ms later (the entry's own 16-slot inner
            // vector is uninitialised too), so the diagnostic skips the registrations outright.
            if (n < 16) return 0;                                    // room left: let the game register it
            if (skipped.fetch_add(1) == 0)
                NXLOG_W("ns2", "quest list full (%llu entries): skipping further registrations (XC2_NS2_QUEST_SKIP)", static_cast<unsigned long long>(n));
            return 1;                                                // handled: the function returns without pushing
        }, nullptr);
        NXLOG_W("ns2", "quest registration guard armed at main+0x36e594 (XC2_NS2_QUEST_SKIP)");
    }
    NXLOG_W("ns2", "memory probe armed at mtl::MemManager::create (scale %llu, probe %d, root %s)",
            static_cast<unsigned long long>(scale), probe ? 1 : 0, root_set ? std::getenv("XC2_NS2_ROOT") : "-");
}

// The release split (upstream's fefw_rt.dll / fefw_stub.cpp): the runtime and this boot code live in a DLL and
// the player's exe only hands it the process. Built with XC2_RT_DLL the entry is exported under a C name so the
// stub can find it; without the define this is still the exe entry it always was.
static bool run_shader_prepare(bool interactive, bool force, const std::string& programs_file, bool* cancelled = nullptr) {
    if (cancelled) *cancelled = false;
    nxrt::nvn::ShaderPrepareOptions o;
    o.programs_file = programs_file;
    o.only_if_needed = !force;
    o.force = force;

    std::atomic<uint64_t> done{0}, total{0};
    std::atomic<bool> stop{false}, finished{false};
    std::mutex mu;
    std::condition_variable cv;
    bool ok = false;
    nxrt::nvn::ShaderPrepareResult res;

    auto run_body = [&] {
        std::atomic<uint64_t> next_log{1000};
        o.progress = [&](uint64_t d, uint64_t t) -> bool {
            done.store(d, std::memory_order_relaxed);
            total.store(t, std::memory_order_relaxed);
            nxrt::diag_progress();
            uint64_t m = next_log.load(std::memory_order_relaxed);
            if ((d >= m || (t > 0 && d == t)) && next_log.compare_exchange_strong(m, d + 1000, std::memory_order_relaxed)) {
                NXLOG_I("xc2", "shader prepare: %llu / %llu programs", static_cast<unsigned long long>(d),
                        static_cast<unsigned long long>(t));
                if (!interactive) {
                    std::fprintf(stderr, "shader prepare: %llu / %llu\n", static_cast<unsigned long long>(d),
                                 static_cast<unsigned long long>(t));
                }
            }
            return !stop.load(std::memory_order_relaxed);
        };
        ok = nxrt::nvn::prepare_shaders(o, &res);
        {
            std::lock_guard<std::mutex> lk(mu);
            finished.store(true, std::memory_order_release);
        }
        cv.notify_all();
    };

    if (!interactive) {
        run_body();
    } else {
        std::thread worker(run_body);
        uint64_t shown_done = ~0ull;
        bool title_changed = false;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::milliseconds(5), [&] { return finished.load(std::memory_order_acquire); });
        }
        while (!finished.load(std::memory_order_acquire)) {
            nxrt::diag_progress();
            nxrt::platform::pump_events(false, 0);
            if (nxrt::platform::quit_requested()) stop.store(true, std::memory_order_relaxed);
            const uint64_t d = done.load(std::memory_order_relaxed);
            const uint64_t t = total.load(std::memory_order_relaxed);
            if (t > 0 && d != shown_done) {
                shown_done = d;
                title_changed = true;
                char buf[160];
                const int pct = static_cast<int>(d * 100 / t);
                std::snprintf(buf, sizeof buf, "%s - Preparing Shaders %d%% (%llu / %llu)", nxrt::game::info().title, pct,
                              static_cast<unsigned long long>(d), static_cast<unsigned long long>(t));
                nxrt::platform::set_window_title(buf);
            }
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::milliseconds(100), [&] { return finished.load(std::memory_order_acquire); });
        }
        worker.join();
        if (title_changed) nxrt::platform::set_window_title(nxrt::game::info().title);
    }

    if (cancelled) *cancelled = res.cancelled;
    if (!ok) {
        if (interactive && res.error.rfind("cannot open ", 0) == 0) {
            NXLOG_I("xc2", "shader prepare: skipped (%s)", res.error.c_str());
            return true;
        }
        NXLOG_W("xc2", "shader prepare failed: %s", res.error.c_str());
        if (!interactive) std::fprintf(stderr, "shader prepare failed: %s\n", res.error.c_str());
        return false;
    }
    if (res.already_ready) {
        NXLOG_I("xc2", "shader prepare: cache already up to date (%llu programs)",
                static_cast<unsigned long long>(res.programs));
        if (!interactive) {
            std::fprintf(stderr, "shader prepare: already ready (%llu programs)\n",
                         static_cast<unsigned long long>(res.programs));
        }
    } else {
        NXLOG_I("xc2", "shader prepare: %s (%llu/%llu prepared, %llu parts, %.2f s)",
                res.cancelled ? "cancelled" : "done", static_cast<unsigned long long>(res.prepared),
                static_cast<unsigned long long>(res.programs), static_cast<unsigned long long>(res.parts), res.seconds);
        if (!interactive) {
            std::fprintf(stderr, "shader prepare: %llu/%llu prepared, %llu missing, %llu failed, %llu parts in %.2f s\n",
                         static_cast<unsigned long long>(res.prepared), static_cast<unsigned long long>(res.programs),
                         static_cast<unsigned long long>(res.missing_shaders), static_cast<unsigned long long>(res.failed),
                         static_cast<unsigned long long>(res.parts), res.seconds);
        }
    }
    return !res.cancelled && res.failed == 0;
}

#ifdef XC2_RT_DLL
extern "C" __declspec(dllexport) int xc2_rt_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
    std::string exefs;
    std::string link = "host";
    std::string entry_mode = "rtld";
    std::string programs_file;
    bool prepare_only = false;
    bool force_prepare = false;
    uint64_t addr_override = 0;
    const char *logp = "xc2_boot.log";
    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char *n) { return !std::strcmp(argv[i], n) && i + 1 < argc; };
        if (arg("--exefs")) exefs = argv[++i];
        else if (arg("--link")) link = argv[++i];
        else if (arg("--log")) logp = argv[++i];
        else if (arg("--gfx")) ++i;   // parsed again below (window | headless)
        else if (arg("--entry")) entry_mode = argv[++i];
        else if (arg("--addr")) addr_override = std::strtoull(argv[++i], nullptr, 0);
        else if (!std::strcmp(argv[i], "--prepare-shaders")) prepare_only = true;
        else if (!std::strcmp(argv[i], "--force-prepare")) force_prepare = true;
        else if (arg("--programs")) programs_file = argv[++i];
        else die("usage: xc2_boot --exefs DIR [--link host|rtld] [--gfx window|headless] [--prepare-shaders] [--log FILE]");
    }
    if (!prepare_only && exefs.empty()) die("--exefs is required: point it at the ExeFS you dumped from your own copy");
    if (!exefs.empty() && exefs.back() == '\\') exefs.pop_back();

    // The game boundary: the core facts plus the factory bindings above, so play mode's binding engine has something to
    // apply. Before nxrt::init reads the title facts.
    nxrt::game::set_current(&xc2_game_info());

    nxrt::InitOptions opt;
    opt.log_path = logp;
    if (!nxrt::init(opt)) die("runtime start-up failed (address-space layout)");
    // Load play settings BEFORE creating the window, Vulkan device/renderer and NVN back end (nxrt/play.h start-up order),
    // so controls.ini's HDR, low latency, keep_aspect, ASTC quality and windowed size all take effect at boot.
    nxrt::play::load();
    const nxrt::play::DisplayConfig play_disp = nxrt::play::display();
    // ---- graphics: window (default) or headless; the NVN back end replays the game's command lists on Vulkan ----
    const char* gfxmode = "window";
    for (int i = 1; i < argc; i++)
        if (!std::strcmp(argv[i], "--gfx") && i + 1 < argc) gfxmode = argv[++i];
    const bool headless = prepare_only || !std::strcmp(gfxmode, "headless");
    std::unique_ptr<nxrt::gfx::Device> gfx_device;
    std::unique_ptr<nxrt::gfx::Renderer> gfx_renderer;
    nxrt::nvn::install();
    bool start_borderless = false;
    bool start_exclusive = false;
    if (!headless) {
        nxrt::platform::init();
        nxrt::platform::WindowDesc wd;
        wd.title = nxrt::game::info().title;
        start_borderless = (play_disp.window_mode == "borderless");
        start_exclusive = (play_disp.window_mode == "exclusive");
        const bool start_fullscreen = start_borderless || start_exclusive;
        int win_w = static_cast<int>(nxrt::config_get_int("gfx", "window_width", 0));
        int win_h = static_cast<int>(nxrt::config_get_int("gfx", "window_height", 0));
        const char* size_src = "[gfx] window_width/height";
        if (!start_fullscreen && (win_w <= 0 || win_h <= 0) && play_disp.width >= 320 && play_disp.height >= 240) {
            win_w = play_disp.width;
            win_h = play_disp.height;
            size_src = "controls.ini";
        }
        if (start_fullscreen || win_w <= 0 || win_h <= 0) {
            win_w = ::GetSystemMetrics(SM_CXSCREEN);
            win_h = ::GetSystemMetrics(SM_CYSCREEN);
            if (win_w < 320 || win_h < 240) { win_w = 1280; win_h = 720; }   // no desktop mode to ask (session 0)
            // Cap default windowed size at 1440p preserving aspect ratio so 4K desktops stay sharp without blowing VRAM at first boot.
            if (!start_fullscreen && win_h > 1440) {
                win_w = (std::max)(320, static_cast<int>(static_cast<long long>(win_w) * 1440 / win_h));
                win_h = 1440;
            }
            size_src = "auto display";
        }
        wd.width = win_w;
        wd.height = win_h;
        NXLOG_I("boot", "window %dx%d (%s, %s)", win_w, win_h, play_disp.window_mode.c_str(), size_src);
        if (!nxrt::platform::create_window(wd)) die("cannot create the window");
        // Paint the window background black immediately and apply a custom local .ico (if present) for the
        // taskbar, Alt+Tab switcher and title bar before entering fullscreen and creating the swapchain.
        if (HWND hwnd = static_cast<HWND>(nxrt::platform::native_hwnd())) {
            ::SetClassLongPtrW(hwnd, GCLP_HBRBACKGROUND, reinterpret_cast<LONG_PTR>(::GetStockObject(BLACK_BRUSH)));
            const int big_cx = ::GetSystemMetrics(SM_CXICON);
            const int big_cy = ::GetSystemMetrics(SM_CYICON);
            const int sm_cx = ::GetSystemMetrics(SM_CXSMICON);
            const int sm_cy = ::GetSystemMetrics(SM_CYSMICON);
            HINSTANCE hInst = ::GetModuleHandleW(nullptr);
            HICON hBig = static_cast<HICON>(::LoadImageW(hInst, MAKEINTRESOURCEW(1), IMAGE_ICON, big_cx, big_cy, LR_DEFAULTCOLOR));
            HICON hSm = static_cast<HICON>(::LoadImageW(hInst, MAKEINTRESOURCEW(1), IMAGE_ICON, sm_cx, sm_cy, LR_DEFAULTCOLOR));
            std::string icon_src = (hBig || hSm) ? "embedded PE resource #1" : "";
            if (!hBig && !hSm) {
                static const char* kIconNames[] = {"xc2_icon.ico", "mythra_ns2_301.ico", "mythra_icon.ico"};
                char exe_path[MAX_PATH] = {};
                std::string exe_dir;
                if (::GetModuleFileNameA(nullptr, exe_path, MAX_PATH) > 0) {
                    exe_dir = exe_path;
                    const size_t sl = exe_dir.find_last_of("\\/");
                    if (sl != std::string::npos) exe_dir.resize(sl + 1);
                    else exe_dir.clear();
                }
                for (const char* nm : kIconNames) {
                    if (file_exists(nm)) { icon_src = nm; break; }
                    if (!exe_dir.empty() && file_exists(exe_dir + nm)) { icon_src = exe_dir + nm; break; }
                }
                if (!icon_src.empty()) {
                    hBig = static_cast<HICON>(::LoadImageA(nullptr, icon_src.c_str(), IMAGE_ICON, big_cx, big_cy, LR_LOADFROMFILE));
                    hSm = static_cast<HICON>(::LoadImageA(nullptr, icon_src.c_str(), IMAGE_ICON, sm_cx, sm_cy, LR_LOADFROMFILE));
                }
            }
            if (hBig) {
                ::SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hBig));
                ::SetClassLongPtrW(hwnd, GCLP_HICON, reinterpret_cast<LONG_PTR>(hBig));
            }
            if (hSm) {
                ::SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hSm));
                ::SetClassLongPtrW(hwnd, GCLP_HICONSM, reinterpret_cast<LONG_PTR>(hSm));
            }
            if (!icon_src.empty()) {
                NXLOG_I("boot", "window icon loaded from %s", icon_src.c_str());
            }
            if (HDC hdc = ::GetDC(hwnd)) {
                RECT rc{};
                ::GetClientRect(hwnd, &rc);
                ::FillRect(hdc, &rc, static_cast<HBRUSH>(::GetStockObject(BLACK_BRUSH)));
                ::ReleaseDC(hwnd, hdc);
            }
        }
    }
    {
        std::string err;
        nxrt::gfx::DeviceDesc dd;
        dd.headless = headless ? 1 : 0;
        gfx_device = nxrt::gfx::Device::create(dd, &err);
        if (!gfx_device) die("graphics device: %s", err.c_str());
        nxrt::gfx::RendererDesc rd;
        rd.headless_width = 1280; rd.headless_height = 720;
        gfx_renderer = nxrt::gfx::Renderer::create(*gfx_device, rd, &err);
        if (!gfx_renderer) die("renderer: %s", err.c_str());
        // Enter borderless/exclusive fullscreen only after the Vulkan device and initial swapchain exist so Windows
        // display/driver hooks cannot deadlock on an un-pumped borderless HWND before Vulkan enumeration (issue #18).
        if (!headless) {
            if (start_borderless) nxrt::platform::set_fullscreen(nxrt::platform::FullscreenMode::Borderless);
            else if (start_exclusive) nxrt::platform::set_fullscreen(nxrt::platform::FullscreenMode::Exclusive);
        }
        g_renderer = gfx_renderer.get();   // the F12 hotkey writes the next presented frame through it
        // Use play mode's shared FrameLimiter so the F1 overlay, corner FPS counter, present-mode switches and
        // arm_frame_rate all observe the same presented frames.
        nxrt::FrameLimiter *frame_limiter = nxrt::play::limiter();
        g_frame_limiter = frame_limiter;
        const Xc2RenderIntent ri = resolve_xc2_render_intent();
        if (ri.mode > 0 && !nxrt::config_has("gfx", "render_scale") && !nxrt::config_has("gfx", "output_scale")) {
            nxrt::gfx::tuning().game_handles_render_quality.store(true, std::memory_order_relaxed);
        }
        nxrt::nvn::BackendDesc bd;
        bd.device = gfx_device.get();
        bd.renderer = gfx_renderer.get();
        bd.limiter = frame_limiter;
        bd.keep_aspect = play_disp.keep_aspect;
        if (!nxrt::nvn::install_backend(bd)) die("NVN back end failed to start (see the log)");
        if (prepare_only) {
            const bool ok = run_shader_prepare(false, force_prepare, programs_file);
            nxrt::nvn::shutdown_backend();
            gfx_renderer.reset();
            gfx_device.reset();
            nxrt::shutdown();
            return ok ? 0 : 1;
        }
        if (!headless && nxrt::config_get_bool("nvn", "auto_prepare", true)) {
            bool prep_cancelled = false;
            run_shader_prepare(true, force_prepare, programs_file, &prep_cancelled);
            if (prep_cancelled || nxrt::platform::quit_requested()) {
                nxrt::nvn::shutdown_backend();
                gfx_renderer.reset();
                gfx_device.reset();
                if (!headless) nxrt::platform::shutdown();
                nxrt::shutdown();
                return 0;
            }
        }
        nxrt::play::StartOptions po;
        po.device = gfx_device.get();
        po.renderer = gfx_renderer.get();
        po.window = !headless;
        po.overlay = !headless;
        nxrt::play::start(po);
        if (!headless) {
            xc2::nav::init(kMainBase);
            xc2::voice::init(kMainBase);
            nxrt::play::set_hud_callback([](const nxrt::gfx::HudCanvas& c) {
                xc2::nav::draw(c);
                xc2::voice::draw(c);
            });
            nxrt::play::set_picker(xc2::nav::make_picker_callbacks());
        }
    }
    // Input: the hid service reads this source every 5 ms. It is live (keyboard + first gamepad, see LiveInput above)
    // and carries the scripted XC2_PRESS presses on top, so a run can be driven by a script, by a human, or both.
    // Must be set after play::start, which installs its own source.
    auto input_src = std::make_shared<LiveInput>();
    {
        nxrt::hid::InputSnapshot snap{};
        snap.npad[0].connected = true;
        snap.npad[0].style = nxrt::hid::NpadStyle::FullKey;
        snap.npad[9].connected = true;                     // handheld slot: keep it a valid pad too
        snap.npad[9].style = nxrt::hid::NpadStyle::Handheld;
        input_src->set(snap);
        route::g_route_t0 = std::chrono::steady_clock::now();   // the origin XC2_ROUTE_OUT and XC2_ROUTE share
        nxrt::hid::set_input_source(input_src);
        std::fprintf(stderr,
                     "input: live keyboard/gamepad on (WASD or arrows = move, IJKL = camera, Space/Enter = A, "
                     "Backspace/Esc = B, U = X, O = Y, Q = L, E = R, 1/2 = ZL/ZR, Tab or + = PLUS, - = MINUS, "
                     "Shift/Ctrl = stick clicks, F12 = screenshot into shots_manual\\\\); XC2_NO_INPUT=1 turns it off\n");
    }
    if (const char* press = std::getenv("XC2_PRESS")) {
        std::shared_ptr<nxrt::hid::ManualInputSource> src = input_src;   // presses land in the live source too
        std::thread([src, spec = std::string(press)] {
            // A button press at <t> is held 120 ms. A stick deflection (RSU/RSD/RSL/RSR/LSU/...) takes "<t>+<seconds>": held that
            // long and then centred. Without a window focus nothing can be driven by the keyboard, so this is how a run tilts the
            // camera on its own (a bug that only shows at a camera angle has to be reproducible in a scripted run).
            struct Ev { uint64_t at_ms; uint64_t btn; int stick; int32_t x, y; double hold_ms; };
            std::vector<Ev> evs;
            size_t i = 0;
            while (i < spec.size()) {
                size_t j = spec.find_first_of(",; ", i);
                if (j == std::string::npos) j = spec.size();
                std::string item = spec.substr(i, j - i);
                i = j + 1;
                if (item.empty()) continue;
                const size_t at = item.find_first_of("@#");
                if (at == std::string::npos) continue;
                const std::string name = item.substr(0, at);
                const double when = std::atof(item.c_str() + at + 1);
                double hold = 0.12;
                if (const size_t plus = item.find('+', at); plus != std::string::npos) hold = std::atof(item.c_str() + plus + 1);
                const uint64_t ms = static_cast<uint64_t>(when * (item[at] == '@' ? 1000.0 : 16.0));
                uint64_t bit = ~0ull;
                int stick = -1;
                int32_t sx = 0, sy = 0;
                if (name == "RSU") { stick = 1; sy = 32767; }
                else if (name == "RSD") { stick = 1; sy = -32767; }
                else if (name == "RSL") { stick = 1; sx = -32767; }
                else if (name == "RSR") { stick = 1; sx = 32767; }
                else if (name == "LSU") { stick = 0; sy = 32767; }
                else if (name == "LSD") { stick = 0; sy = -32767; }
                else if (name == "LSL") { stick = 0; sx = -32767; }
                else if (name == "LSR") { stick = 0; sx = 32767; }
                else if (name == "A") bit = nxrt::hid::btn(nxrt::hid::kBtnA);
                else if (name == "B") bit = nxrt::hid::btn(nxrt::hid::kBtnB);
                else if (name == "X") bit = nxrt::hid::btn(nxrt::hid::kBtnX);
                else if (name == "Y") bit = nxrt::hid::btn(nxrt::hid::kBtnY);
                else if (name == "PLUS") bit = nxrt::hid::btn(nxrt::hid::kBtnPlus);
                else if (name == "MINUS") bit = nxrt::hid::btn(nxrt::hid::kBtnMinus);
                else if (name == "L") bit = nxrt::hid::btn(nxrt::hid::kBtnL);
                else if (name == "R") bit = nxrt::hid::btn(nxrt::hid::kBtnR);
                else if (name == "ZL") bit = nxrt::hid::btn(nxrt::hid::kBtnZL);
                else if (name == "ZR") bit = nxrt::hid::btn(nxrt::hid::kBtnZR);
                else if (name == "DPD") bit = nxrt::hid::btn(nxrt::hid::kBtnDown);
                else if (name == "DPU") bit = nxrt::hid::btn(nxrt::hid::kBtnUp);
                else if (name == "DPL") bit = nxrt::hid::btn(nxrt::hid::kBtnLeft);
                else if (name == "DPR") bit = nxrt::hid::btn(nxrt::hid::kBtnRight);
                if (bit != ~0ull) evs.push_back({ms, bit, -1, 0, 0, hold * 1000.0});
                else if (stick >= 0) evs.push_back({ms, 0, stick, sx, sy, hold * 1000.0});
            }
            std::sort(evs.begin(), evs.end(), [](const Ev& a, const Ev& b) { return a.at_ms < b.at_ms; });
            const auto start = std::chrono::steady_clock::now();
            for (const Ev& e : evs) {
                std::this_thread::sleep_until(start + std::chrono::milliseconds(e.at_ms));
                if (e.stick < 0) {
                    src->press(0, e.btn);
                    std::this_thread::sleep_for(std::chrono::milliseconds(120));
                    src->release(0, e.btn);
                    std::fprintf(stderr, "input: pressed button 0x%llx at %llu ms\n", (unsigned long long)e.btn,
                                 (unsigned long long)e.at_ms);
                } else {
                    src->stick(0, e.stick == 1, e.x, e.y);
                    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(e.hold_ms)));
                    src->stick(0, e.stick == 1, 0, 0);
                    std::fprintf(stderr, "input: %s stick (%d,%d) held %.0f ms at %llu ms\n", e.stick == 1 ? "right" : "left",
                                 e.x, e.y, e.hold_ms, (unsigned long long)e.at_ms);
                }
            }
        }).detach();
        std::fprintf(stderr, "scripted input armed: %s\n", press);
    }
    if (const char *route = std::getenv("XC2_ROUTE")) {   // a recorded run replayed (tools/run/route.ps1)
        std::shared_ptr<nxrt::hid::ManualInputSource> src = input_src;
        route_start(src, route);
    }
    nxrt::config_set("boot", "link", link.c_str());
    // The guest rtld relocates the images itself; when we skip it (host link + direct entry) the host's relocation must stay.
    nxrt::config_set("boot", "guest_relinks", (link == "host" && entry_mode == "rtld" && !addr_override) ? "1" : "0");
    nxrt::config_set("process", "program_id", "0x0100E95004038000");
    // The "Switch 2 Edition" runs the Switch 1 graphics stack (NVN 3.5.0, Maxwell shaders), so its GPU memory
    // keeps Tegra X1 GOB order: with the Switch 2 order every block-linear texture is scrambled (the 16^3 colour
    // LUT came out green and turned the whole scene cyan - HANDOVER section 15).
    nxrt::config_set("nvn", "gob_switch1", "1");
    // Pacing. The game steps its logic 30 times a second but presents as fast as the host runs (48.9 fps here), so an animated layer
    // (the cloud sea) is sampled at 30 Hz and shown at 48.9 Hz: the cadence never divides, and the eye reads that as judder. The
    // console paces such a title by its present interval; XC2 never asks for one, so [nvn] present_interval_hz supplies the rate and
    // the frame limiter (above) holds it. NXRT_NVN_PRESENT_INTERVAL_HZ=0 turns pacing off, =60 asks for 60.
    if (!nxrt::config_has("nvn", "present_interval_hz")) nxrt::config_set("nvn", "present_interval_hz", "30");
    nxrt::config_set("system", "language", "AmericanEnglish");   // Monolith dev_setting refuses the zh-Hans the host reports
    // The game's own languages (an older launch.env listed American English, which crashes the Event Theater); env wins over config_set.
    _putenv_s("NXRT_TITLE_SUPPORTED_LANGUAGES", "Japanese,BritishEnglish,French,German,Italian,Spanish,Chinese,Taiwanese,Korean,SimplifiedChinese,TraditionalChinese");
    nxrt::config_set("system", "region", "USA");
    // F11 (see LiveInput) traces the next frame; with this default the trace also writes every sampled texture's prepared
    // bytes next to the render target dumps (texdump_f11\t<id>.bin/.txt/.raw), so a hands-on bug report carries its own
    // textures. NXRT_NVN_TRACE_DUMP_TEX overrides the folder; "off" / "-" / an empty value turns the texture dump off (a multi-frame
    // trace would otherwise write tens of thousands of images).
    if (const char *dump_env = std::getenv("NXRT_NVN_TRACE_DUMP_TEX")) {
        const bool off = !*dump_env || !std::strcmp(dump_env, "off") || !std::strcmp(dump_env, "-");
        if (off) nxrt::config_set("nvn", "trace_dump_tex", "");
    } else {
        nxrt::config_set("nvn", "trace_dump_tex", "texdump_f11");
    }

    // 1+2. map and register
    for (const ModEntry &g : module_table()) {
        const std::string path = exefs + "\\" + g.file;
        if (!file_exists(path)) die("module file not found: %s", path.c_str());
        nxrt::LoadOptions lo;
        lo.name = g.file;
        lo.base = g.nx->base;
        std::string err;
        nxrt::Module *m = nxrt::module_load_nso_file(path.c_str(), lo, &err);
        if (!m) die("cannot load %s: %s", path.c_str(), err.c_str());
        if (m->base() != g.nx->base) die("module %s mapped at 0x%llx, lifted for 0x%llx", g.file,
                                         static_cast<unsigned long long>(m->base()), static_cast<unsigned long long>(g.nx->base));
        const nxrt::RegResult r = nxrt::nxmodule_register(m, g.nx, *g.abi);
        if (!r.ok()) die("lifted module %s refused: %s", g.file, r.message.c_str());
        NXLOG_I("boot", "module %s at 0x%llx", g.file, static_cast<unsigned long long>(m->base()));
    }

    // In the DLL build the lifted modules come from the exe, so ask the table rather than naming nxmod_*.
    const auto module_base = [&](const char *name) -> uint64_t {
        for (const ModEntry &g : module_table())
            if (!std::strcmp(g.file, name)) return g.nx->base;
        die("the module table handed to the runtime has no %s", name);
    };
    const uint64_t main_base = module_base("main");
    g_main_base = main_base;
    g_sdk_base = module_base("sdk");
    g_rtld_base = module_base("rtld");

    // Model-spawn probe (XC2_MODEL_PROBE=1): "did the engine ever create this model?", logged with its arguments. Works
    // without a re-lift: both site addresses are already in build\sites-main.txt (HANDOVER 19.9, step P3).
    if (std::getenv("XC2_MODEL_PROBE")) arm_model_probe(main_base);
    arm_frame_rate(main_base);
    arm_native_res(main_base);
    // XC2_BUMP_MENU_RESOURCE=<bytes>: raise the compiled-in MenuResource budget (HANDOVER 84, DIAGNOSTIC). The 3.0.0 content
    // needs more than the 76,800 KB this build was built with, and the 64 region budgets use only 5,440 MiB of the 10,642 MiB
    // heap, so there is ~5.1 GiB of slack. The midhook sits on the bl that creates the region: x1 is the size.
    if (const char* v = std::getenv("XC2_BUMP_MENU_RESOURCE")) {
        const uint64_t bytes = std::strtoull(v, nullptr, 0);
        if (bytes) {
            nxrt::hook_add_mid(main_base + 0x6fa790, [](Ctx* c, uint64_t, void* u) -> int {
                static bool logged = false;
                const uint64_t want = reinterpret_cast<uint64_t>(u);
                if (!logged) {
                    logged = true;
                    NXLOG_W("dbg", "MenuResource budget 0x%llx -> 0x%llx (diagnostic, XC2_BUMP_MENU_RESOURCE)",
                            static_cast<unsigned long long>(c->x[1]), static_cast<unsigned long long>(want));
                }
                c->x[1] = want;
                return 0;
            }, reinterpret_cast<void*>(bytes));
        }
    }
    arm_delta_scan(main_base);
    arm_period_probe(main_base);
    arm_obj_scan(main_base);
    arm_ns1_probe(main_base);
    arm_ns2_mem_probe(main_base);

    // Bring-up probe (XC2_TRACE_LAUNCHER=1): which apps::FrameworkLauncher work function runs each frame. The engine draws a
    // frame with no geometry; this tells a standby (waiting for the applet/user) apart from a scene that failed to build.
    // An entry hook only fires where the lifter emitted nx_hook_entry, so these addresses must also be in build\sites-main.txt
    // (`entryhook 7100000658` ...) and main re-lifted; until then every counter stays 0.
    // Addresses are main-relative (tools/analysis/symat.py resolves them from .dynsym; the old
    // work/symresolve.py is archived under work/_archive/tools-20261112/ and needs lz4).
    if (std::getenv("XC2_TRACE_LAUNCHER")) {
        struct Site { const char* name; uint64_t off; unsigned* n; };
        static Site sites[] = {
            {"FW::update", 0x658, new unsigned(0)},          {"Scn::createMdl", 0x7409ac, new unsigned(0)},
            {"MT::update", 0x7570d4, new unsigned(0)},       {"MT::tail", 0x75731c, new unsigned(0)},
            {"MT::sleep", 0x756ef4, new unsigned(0)},        {"MT::switchThread", 0x756728, new unsigned(0)},
            {"MT::switchMain", 0x756720, new unsigned(0)},   {"MT::setupThread", 0x7564cc, new unsigned(0)},
            {"TLapp::initialize", 0x142c50, new unsigned(0)}, {"TLapp::update", 0x142cf0, new unsigned(0)},
            {"TLapp::execute", 0x1427b8, new unsigned(0)},   {"TLapp::nowLoading", 0x1420a0, new unsigned(0)},
            {"TLmain::create", 0x142f44, new unsigned(0)},   {"TLmain::update", 0x1439f4, new unsigned(0)},
            {"App::wkRender", 0x1fc4, new unsigned(0)},      {"App::wkUpdate", 0x1fc0, new unsigned(0)},
        };
        for (Site& s : sites) {
            nxrt::hook_add_entry(main_base + s.off, [](Ctx*, uint64_t, void* user) -> int {
                ++*static_cast<unsigned*>(user);
                return 0;
            }, s.n);
        }
        // ml::MicroThread::update(MLTHG*): which groups the frame loop actually pumps. The build microthread sleeps in a
        // group; a group that is never updated parks every microthread in it for good.
        static uint64_t g_groups[8] = {};
        static int g_group_n = 0;
        static uint64_t g_mt_this = 0, g_mt_group = 0;
        nxrt::hook_add_entry(main_base + 0x7570d4, [](Ctx* c, uint64_t, void*) -> int {
            const uint64_t g = c->x[1];
            g_mt_this = c->x[0];
            g_mt_group = g;
            for (int i = 0; i < g_group_n; i++)
                if (g_groups[i] == g) return 0;
            if (g_group_n < 8) g_groups[g_group_n++] = g;
            return 0;
        });
        // ml::MicroThread::createHandle(MLTHG*, unsigned int): the groups microthreads are actually created in.
        static uint64_t g_handles[8] = {};
        static int g_handle_n = 0;
        nxrt::hook_add_entry(main_base + 0x756908, [](Ctx* c, uint64_t, void*) -> int {
            const uint64_t g = c->x[0];
            for (int i = 0; i < g_handle_n; i++)
                if (g_handles[i] == g) return 0;
            if (g_handle_n < 8) g_handles[g_handle_n++] = g;
            return 0;
        });
        // Was the sleeping microthread ever released (removed from its group) or destroyed?
        static unsigned g_rel = 0, g_dst = 0;
        nxrt::hook_add_entry(main_base + 0x756bdc, [](Ctx*, uint64_t, void*) -> int { ++g_rel; return 0; });
        nxrt::hook_add_entry(main_base + 0x756614, [](Ctx*, uint64_t, void*) -> int { ++g_dst; return 0; });
        // XC2_TRACE_MEMMGR=1: log the big requests into mtl::MemManager (HANDOVER 80/81). pkg7 dies because
        // FileHandleTh::setReadBuffer asks a 320 KB region for up to 1.79 MB, so the interesting datum is which ALLOC_HANDLE
        // (which named region) the request goes to and how big it is. mtl::MemManager::allocate(ALLOC_HANDLE handle, size,
        // align, bool, name, line): x0 points at the handle word, x1 = size, x2 = align.
        if (std::getenv("XC2_TRACE_MEMMGR")) {
            nxrt::hook_add_entry(main_base + 0x781798, [](Ctx* c, uint64_t, void*) -> int {
                static std::atomic<int> n{0};
                const uint64_t hp = c->x[0];
                const uint64_t size = c->x[1], align = c->x[2];
                const bool okp = hp > 0x7000000000ull && hp < 0x8000000000ull;
                const uint32_t id = okp ? *reinterpret_cast<const uint32_t*>(hp) : 0u;
                if (size >= 0x100000 && n.fetch_add(1) < 2000)   // >= 1 MiB: the 512 KiB read buffers would drown it
                    NXLOG_I("dbg", "mtl::MemManager::allocate(handle %u, size 0x%llx = %llu KB, align 0x%llx, from 0x%llx)", id,
                            static_cast<unsigned long long>(size), static_cast<unsigned long long>(size >> 10),
                            static_cast<unsigned long long>(align), static_cast<unsigned long long>(c->x[30] & 0xffffffffffULL));
                return 0;
            });
            // Every region the game creates, with its name: mtl::MemManager::create(void*/HANDLE, size, char const* name, bool)
            // keeps the size in x1 and the name in x2, so base and pkg7 can be compared region by region.
            for (uint64_t a : {main_base + 0x77fc6c, main_base + 0x77ff84}) {
                nxrt::hook_add_entry(a, [](Ctx* c, uint64_t, void*) -> int {
                    const uint64_t size = c->x[1];
                    const uint64_t np = c->x[2];
                    const char* nm = (np > 0x7000000000ull && np < 0x8000000000ull) ? reinterpret_cast<const char*>(np) : "?";
                    NXLOG_I("dbg", "mtl::MemManager::create(%s) size 0x%llx = %llu KB  from 0x%llx", nm, static_cast<unsigned long long>(size),
                            static_cast<unsigned long long>(size >> 10), static_cast<unsigned long long>(c->x[30] & 0xffffffffffULL));
                    return 0;
                });
            }
        }
        // NXRT_DEBUG_ENTRY=0x71002a0000[,0x...]: log the arguments of the first calls of those entry-hooked functions.
        // Used to see what the resource manager is asked for (HANDOVER 77: pkg7 asserts on a table index of 526).
        if (const char* list = std::getenv("NXRT_DEBUG_ENTRY")) {
            for (const char* p = list; *p;) {
                char* end = nullptr;
                const unsigned long long a = std::strtoull(p, &end, 0);
                if (end == p) break;
                p = (*end == ',') ? end + 1 : end;
                if (!a) continue;
                nxrt::hook_add_entry(a, [](Ctx* c, uint64_t addr, void*) -> int {
                    static std::atomic<int> n{0};
                    if (n.fetch_add(1) < 64)
                        NXLOG_I("dbg", "entry f_%llx: x0=%llx x1=%llx x2=%llx x3=%llx lr=%llx", static_cast<unsigned long long>(addr & 0xffffffffffULL),
                                static_cast<unsigned long long>(c->x[0]), static_cast<unsigned long long>(c->x[1]),
                                static_cast<unsigned long long>(c->x[2]), static_cast<unsigned long long>(c->x[3]),
                                static_cast<unsigned long long>(c->x[30]));
                    return 0;
                });
            }
        }
        // XC2_SKIP_MT_SLEEP=1: make ml::MicroThread::sleep a no-op. Experiment only: if the launcher's build thread then
        // reaches the applet, the microthread scheduler is the blocker; it does not prove the rest of the boot is right.
        if (std::getenv("XC2_SKIP_MT_SLEEP")) {
            nxrt::hook_add_entry(main_base + 0x756ef4, [](Ctx* c, uint64_t, void*) -> int {
                c->x[0] = 0;
                return 1;   // handled: the lifted function returns at once
            });
            std::fprintf(stderr, "microthread sleep bypassed (XC2_SKIP_MT_SLEEP)\n");
        }
        // switchThread: after the "current fiber" call (x0 = its result, x19 = the target fiber) and at the bail target.
        static unsigned g_sw_bail = 0;
        static uint64_t g_sw_curfib = 0, g_sw_tgt = 0;
        nxrt::hook_add_mid(main_base + 0x75680c, [](Ctx* c, uint64_t, void*) -> int {
            g_sw_curfib = c->x[0];
            g_sw_tgt = c->x[19];
            return 0;
        });
        nxrt::hook_add_mid(main_base + 0x756824, [](Ctx*, uint64_t, void*) -> int { ++g_sw_bail; return 0; });
        // ml::MicroThreadNx::switchThread(MTHREAD* cur, MTHREAD* next): what update() picked. x1 = cur, x2 = next.
        static uint64_t g_sw_last = 0, g_sw_cur = 0;
        static unsigned g_sw_n = 0;
        nxrt::hook_add_entry(main_base + 0x756728, [](Ctx* c, uint64_t, void*) -> int {
            g_sw_cur = c->x[1];
            g_sw_last = c->x[2];
            ++g_sw_n;
            return 0;
        });
        // ml::MicroThreadNx::setupThread(MTHREAD*, entry): x1 is the MTHREAD the launcher's build thread lives in.
        // Watching its fields over time tells whether the sleep counter ever expires and whether the state changes.
        static uint64_t g_mt = 0;
        nxrt::hook_add_entry(main_base + 0x7564cc, [](Ctx* c, uint64_t, void*) -> int {
            if (!g_mt) g_mt = c->x[1];
            return 0;
        });
        // ml::MicroThread::createThread(MLTHG* group, entry, arg): the group the launcher's build thread is created in.
        static uint64_t g_cthreads[8] = {};
        static int g_cthread_n = 0;
        nxrt::hook_add_entry(main_base + 0x75749c, [](Ctx* c, uint64_t, void*) -> int {
            const uint64_t g = c->x[1];   // member function: x0 = this (the manager), x1 = the MLTHG* group
            for (int i = 0; i < g_cthread_n; i++)
                if (g_cthreads[i] == g) return 0;
            if (g_cthread_n < 8) g_cthreads[g_cthread_n++] = g;
            return 0;
        });
        // midhook inside ml::MicroThread::update (update+0x128, site `midhook 71007571fc`): x27 = the MTHREAD under
        // test, [x27+0x28] = sleep frames left, [sp+12] = elapsed time. Shows why a sleeping microthread is not picked.
        nxrt::hook_add_mid(main_base + 0x7571fc, [](Ctx* c, uint64_t, void*) -> int {
            static int n = 0;
            if (n++ < 40) {
                const uint64_t mt = c->x[27];
                const bool ok = mt > 0x7000000000ull && mt < 0x8000000000ull;
                const uint32_t frames = ok ? *reinterpret_cast<uint32_t*>(mt + 0x28) : 0;
                const float elapsed = *reinterpret_cast<float*>(c->x[31] + 12);
                std::fprintf(stderr, "mt scan #%d: mt=0x%llx frames=%u elapsed=%.6f\n", n, (unsigned long long)mt, frames,
                             static_cast<double>(elapsed));
            }
            return 0;
        });
        std::fprintf(stderr, "launcher probe armed (%zu sites)\n", sizeof(sites) / sizeof(sites[0]));
        std::thread([sites = &sites] {
            for (int t = 1; t <= 60; t++) {
                Sleep(1000);
                std::fprintf(stderr, "launcher probe t=%ds:", t);
                for (auto& s : *sites) std::fprintf(stderr, " %s=%u", s.name, *s.n);
                std::fprintf(stderr, "\n");
                if (t == 8 || t == 16) {   // where every guest fiber is suspended (LR = the guest address it resumes at)
                    std::fprintf(stderr, "group probe: %d distinct MicroThread::update group(s):", g_group_n);
                    for (int i = 0; i < g_group_n; i++) std::fprintf(stderr, " 0x%llx", (unsigned long long)g_groups[i]);
                    std::fprintf(stderr, "\n");
                    // The gate update() checks before it scans for a ready microthread: [head+8] (0 = no scan at all).
                    if (g_mt_this > 0x7000000000ull && g_mt_this < 0x8000000000ull && g_mt_group < 8) {
                        const uint64_t head = *reinterpret_cast<uint64_t*>(g_mt_this + g_mt_group * 8 + 8);
                        std::fprintf(stderr, "group probe: this=0x%llx group=%llu head=0x%llx",
                                     (unsigned long long)g_mt_this, (unsigned long long)g_mt_group, (unsigned long long)head);
                        if (head > 0x7000000000ull && head < 0x8000000000ull)
                            std::fprintf(stderr, " [head+8]=%u [head+537]=%u",
                                         *reinterpret_cast<uint32_t*>(head + 8), *reinterpret_cast<uint8_t*>(head + 537));
                        std::fprintf(stderr, "\n");
                    }
                    std::fprintf(stderr, "group probe: createHandle groups:");
                    for (int i = 0; i < g_handle_n; i++) std::fprintf(stderr, " 0x%llx", (unsigned long long)g_handles[i]);
                    std::fprintf(stderr, " | releaseThread=%u destroyThread=%u", g_rel, g_dst);
                    std::fprintf(stderr, " | createThread groups:");
                    for (int i = 0; i < g_cthread_n; i++) std::fprintf(stderr, " 0x%llx", (unsigned long long)g_cthreads[i]);
                    if (g_mt > 0x7000000000ull && g_mt < 0x8000000000ull) {
                        std::fprintf(stderr, " | mt=0x%llx state=%u sleep=%u bool=%u stamp=%llx [0x38]=%llx",
                                     (unsigned long long)g_mt, *reinterpret_cast<uint32_t*>(g_mt),
                                     *reinterpret_cast<uint32_t*>(g_mt + 0x28), *reinterpret_cast<uint8_t*>(g_mt + 0x2d),
                                     (unsigned long long)*reinterpret_cast<uint64_t*>(g_mt + 0x20),
                                     (unsigned long long)*reinterpret_cast<uint64_t*>(g_mt + 0x38));
                        std::fprintf(stderr, " fiber=[mt+0x10]=0x%llx", (unsigned long long)*reinterpret_cast<uint64_t*>(g_mt + 0x10));
                        const uint64_t fib = *reinterpret_cast<uint64_t*>(g_mt + 0x10);
                        if (fib > 0x7000000000ull && fib < 0x8000000000ull)
                            std::fprintf(stderr, " [fiber+0x60]=%u [fiber+0x64]=%u", *reinterpret_cast<uint32_t*>(fib + 0x60),
                                         *reinterpret_cast<uint32_t*>(fib + 0x64));
                    }
                    if (g_mt_this > 0x7000000000ull && g_mt_this < 0x8000000000ull) {
                        const uint64_t cur = *reinterpret_cast<uint64_t*>(g_mt_this + 704);   // current group head
                        std::fprintf(stderr, " | cur_group=0x%llx", (unsigned long long)cur);
                        std::fprintf(stderr, " | switchThread n=%u cur=0x%llx next=0x%llx", g_sw_n, (unsigned long long)g_sw_cur,
                                     (unsigned long long)g_sw_last);
                        std::fprintf(stderr, " tgt=0x%llx curfib=0x%llx bail=%u", (unsigned long long)g_sw_tgt,
                                     (unsigned long long)g_sw_curfib, g_sw_bail);
                        if (cur > 0x7000000000ull && cur < 0x8000000000ull)
                            std::fprintf(stderr, " next=[cur+1648]=0x%llx", (unsigned long long)*reinterpret_cast<uint64_t*>(cur + 1648));
                        const uint64_t vt = *reinterpret_cast<uint64_t*>(g_mt_this);
                        std::fprintf(stderr, " | manager vtable=0x%llx slots:", (unsigned long long)vt);
                        for (int i = 0; i < 6; i++)
                            std::fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)*reinterpret_cast<uint64_t*>(vt + i * 8));
                    }
                    if (g_mt_this > 0x7000000000ull && g_mt_this < 0x8000000000ull) {
                        std::fprintf(stderr, " | manager[+8..+32]:");
                        for (int i = 0; i < 4; i++)
                            std::fprintf(stderr, " 0x%llx", (unsigned long long)*reinterpret_cast<uint64_t*>(g_mt_this + 8 + i * 8));
                        std::fprintf(stderr, "\n");
                        for (int i = 0; i < 2; i++) {   // both groups: the run queue and its count
                            const uint64_t h = *reinterpret_cast<uint64_t*>(g_mt_this + 8 + i * 8);
                            if (h < 0x7000000000ull || h >= 0x8000000000ull) continue;
                            std::fprintf(stderr, "group probe:   group %d head=0x%llx [+8]=%u [+537]=%u [+528]=0x%llx",
                                         i, (unsigned long long)h, *reinterpret_cast<uint32_t*>(h + 8),
                                         *reinterpret_cast<uint8_t*>(h + 537), (unsigned long long)*reinterpret_cast<uint64_t*>(h + 528));
                            const uint64_t first = *reinterpret_cast<uint64_t*>(h + 528);
                            if (first > 0x7000000000ull && first < 0x8000000000ull)
                                std::fprintf(stderr, " first->[+0x28]=%u", *reinterpret_cast<uint32_t*>(first + 0x28));
                            std::fprintf(stderr, "\n");
                        }
                    }
                    nxrt::fiber_dump([](void*, uint64_t guest, uint64_t fn, uint64_t lr, uint64_t sp, int state) {
                        std::fprintf(stderr, "fiber probe: fiber 0x%llx entry 0x%llx state %d resume 0x%llx sp 0x%llx\n",
                                     (unsigned long long)guest, (unsigned long long)fn, state,
                                     (unsigned long long)lr, (unsigned long long)sp);
                    }, nullptr);
                }
            }
        }).detach();
    }

    // Debug: log the arguments of Monolith's allocator (the site is compiled in by --sites build\sites-main.txt).
    nxrt::hook_add_entry(g_sdk_base + 0x8f514, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "heap impl: SetHeapSize(out=0x%llx size=0x%llx)\n", (unsigned long long)c->x[1], (unsigned long long)c->x[2]);
        return 0;
    });
    nxrt::hook_add_mid(g_sdk_base + 0x8f52c, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "heap impl: svcSetHeapSize result=0x%x\n", static_cast<uint32_t>(c->x[0]));
        return 0;
    });    nxrt::hook_add_entry(g_sdk_base + 0x8e514, [](Ctx*, uint64_t, void*) -> int {
        *reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(g_sdk_base + 0x92ccf0)) = g_main_handle;
        std::fprintf(stderr, "boot: boot-parameter hook wrote g_OsBootParamter[0] = 0x%x\\n", g_main_handle);
        return 0;
    });
    // XC2_301_BOOTPARAM=1 (HANDOVER 89): the retail 3.0.1 sdk keeps the main thread's handle in a boot parameter block. Its
    // thread manager ctor (sdk+0x170030, an entry site in build\sites-sdk.txt) reads *(*(sdk+0xe75408)) as that handle, and the
    // block's first word is 0 on our loader path, so every handle comparison degenerates: GetThreadPriority(0)/GetThreadId(0)
    // fail, and once 0 is answered with the calling thread,
    // nn::os::detail::InternalCriticalSectionImplByHorizon::IsLockedByCurrentThread() says "locked by me" and
    // nn::diag::detail::AbortImpl runs (measured: the abort predicate resolves through import slot 0xe7cba0). Writing the handle
    // main_thread_adopt gave the main thread into the block before the ctor reads it makes both correct.
    if (std::getenv("XC2_301_BOOTPARAM")) {
        nxrt::hook_add_entry(g_sdk_base + 0x170030, [](Ctx*, uint64_t, void*) -> int {
            const uint64_t pptr = g_sdk_base + 0xe75408;
            uint64_t block = 0;
            if (nxrt::mem::is_mapped(pptr, 8))
                std::memcpy(&block, reinterpret_cast<const void*>(static_cast<uintptr_t>(pptr)), 8);
            if (block && nxrt::mem::is_mapped(block, 4)) {
                std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(block)), &g_main_handle, 4);
                static std::atomic<int> n{0};
                if (n.fetch_add(1) == 0)
                    NXLOG_W("boot", "3.0.1 boot parameter block 0x%llx: main thread handle 0x%x written",
                            static_cast<unsigned long long>(block), g_main_handle);
            }
            return 0;
        }, nullptr);
        NXLOG_W("boot", "3.0.1 boot parameter hook armed at sdk+0x170030 (XC2_301_BOOTPARAM)");
    }
    // XC2_301_VAMM=1 (HANDOVER 90): the retail sdk's virtual address memory manager is enabled (GetInfo(16)
    // SystemResourceSize is non-zero), but nothing in this ExeFS calls nn::os::detail::VammManager::InitializeIfEnabled: it is
    // exported for other modules and no module here imports it, so VammManager::AllocateAddressRegion (sdk+0x16dcc0) reads the
    // address-space allocator at manager+0x40 == null and the game ends up memsetting the null region it got back (host AV).
    // Call the initializer once, right before the first AllocateAddressRegion, with the manager that call was handed - what
    // nn::os::Initialize does on the console. The whole register context is put back afterwards.
    if (std::getenv("XC2_301_VAMM")) {
        nxrt::hook_add_entry(g_sdk_base + 0x16dcc0, [](Ctx* c, uint64_t, void*) -> int {
            const uint64_t mgr = c->x[0];
            uint64_t inited = 0;
            if (nxrt::mem::is_mapped(mgr + 0x40, 8))
                std::memcpy(&inited, reinterpret_cast<const void*>(static_cast<uintptr_t>(mgr + 0x40)), 8);
            {
                static std::atomic<int> n{0};
                if (n.fetch_add(1) < 4)
                    NXLOG_I("boot", "VammManager::AllocateAddressRegion(mgr 0x%llx, out 0x%llx, size 0x%llx), +0x40 = 0x%llx",
                            static_cast<unsigned long long>(mgr), static_cast<unsigned long long>(c->x[1]),
                            static_cast<unsigned long long>(c->x[2]), static_cast<unsigned long long>(inited));
            }
            if (!inited) {
                static std::atomic<int> n{0};
                const Ctx saved = *c;
                c->x[0] = mgr;
                nxrt::dispatch_call(c, g_sdk_base + 0x16daa0);
                *c = saved;
                uint64_t after = 0;
                if (nxrt::mem::is_mapped(mgr + 0x40, 8))
                    std::memcpy(&after, reinterpret_cast<const void*>(static_cast<uintptr_t>(mgr + 0x40)), 8);
                if (n.fetch_add(1) == 0)
                    NXLOG_W("boot", "VammManager 0x%llx: InitializeIfEnabled called, +0x40 0x%llx -> 0x%llx",
                            static_cast<unsigned long long>(mgr), static_cast<unsigned long long>(inited),
                            static_cast<unsigned long long>(after));
            }
            return 0;
        }, nullptr);
        NXLOG_W("boot", "VAMM init hook armed at sdk+0x16dcc0 (XC2_301_VAMM)");
    }
    // rtld 0x1470: flag-setting helper (writes 1 at rtld+0x3000, returns to its caller). Handled in the host.
    nxrt::hook_add_entry(g_rtld_base + 0x1470, [](Ctx*, uint64_t, void*) -> int {
        *reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(g_rtld_base + 0x3000)) = 1;
        std::fprintf(stderr, "rtld: flag helper -> [rtld+0x3000] = 1\n");
        return 1;
    });    nxrt::hook_add_entry(main_base + 0x77ef28, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "MM: initialize(this=0x%llx)\n", (unsigned long long)c->x[0]); return 0; });
    nxrt::hook_add_entry(main_base + 0x77f770, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "MM: create(name=0x%llx)\n", (unsigned long long)c->x[0]); return 0; });
    nxrt::hook_add_entry(main_base + 0x77ff84, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "MM: create(h=0x%x size=0x%llx name=0x%llx type=0x%llx)\n",
                     static_cast<uint32_t>(c->x[0]), (unsigned long long)c->x[1], (unsigned long long)c->x[2], (unsigned long long)c->x[5]); return 0; });
    nxrt::hook_add_entry(main_base + 0x77fc6c, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "MM: create x0=0x%llx x1=0x%llx x2=0x%llx x3=0x%llx x4=0x%llx x5=0x%llx\n",
                     (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[2],
                     (unsigned long long)c->x[3], (unsigned long long)c->x[4], (unsigned long long)c->x[5]); return 0; });
    nxrt::hook_add_entry(main_base + 0x77fb44, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "MM: Region::initialize x0=0x%llx x1=0x%llx x2=0x%llx x3=0x%llx x4=0x%llx x5=0x%llx\n",
                     (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[2],
                     (unsigned long long)c->x[3], (unsigned long long)c->x[4], (unsigned long long)c->x[5]); return 0; });    nxrt::hook_add_entry(0x7100781798ull, [](Ctx* c, uint64_t, void*) -> int {
        std::fprintf(stderr, "ml alloc: x0=%llx x1=%llx x2=%llx x3=%llx x4=%llx lr=%llx\n",
                     (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[2],
                     (unsigned long long)c->x[3], (unsigned long long)c->x[4], (unsigned long long)c->x[30]);
        return 0;
    });
    // XC2_301_OPUS=1: the hardware Opus decoder replaced by the SDK's own software one (x2_opus301.cpp, HANDOVER 90).
    if (std::getenv("XC2_301_OPUS")) xc2_opus301_install();
    if (std::getenv("XC2_301_AUDIO")) xc2_audio301_install();
    // The game reaches NVN only through `nvnBootstrapLoader` (an sdk export): force it to the host NVN shell.
    nxrt::module_override_symbol("nvnBootstrapLoader");
    if (const uint64_t def = nxrt::nxmodule_find_export("nvnBootstrapLoader"))
        if (nxrt::HleFn fn = nxrt::hle_lookup("nvnBootstrapLoader")) nxrt::fn_register(def, fn);

    // 3. link
    if (link == "host") {
        if (!nxrt::module_link_all()) die("linking the modules failed (see the log)");
    } else {
        nxrt::module_protect_all();
    }

    // 4. the guest main thread (main.npdm: 1 MiB, priority 44, core 0)
    const uint64_t stack = static_cast<uint64_t>(nxrt::config_get_int("boot", "main_stack_kb", 1024)) << 10;
    const uint32_t h = nxrt::kernel::main_thread_adopt(stack, "main");
    if (!h) die("cannot create the main thread");
    Ctx *ctx = nxrt::kernel::current_ctx();

    // 5. The console path enters rtld at 0x7700000000; the experiments below skip rtld.
    uint64_t entry = entry_mode == "main" ? (main_base + 0x69fb7c) : module_base("rtld");
    if (addr_override) entry = addr_override;
    g_main_handle = h;   // the SDK boot-parameter hook (ThreadManagerImplByHorizon) needs it
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (entry_mode == "start") {
        // What the console loader does after relocation: run main's DT_INIT (the C++ constructors), then
        // nn::init::Start(0, main thread handle, nnMain, UserExceptionHandler).
        entry = module_base("sdk") + 0x68668;             // nn::init::Start
        a1 = h;
        a2 = main_base + 0x69fb7c;                  // nnMain
        a3 = module_base("sdk") + 0x8fea4;                // nn::os::detail::UserExceptionHandler
    }
    if (!nxrt::fn_lookup(entry)) die("the lifted module has no function at 0x%llx", (unsigned long long)entry);
    ctx->x[0] = a0;
    ctx->x[1] = a1;
    ctx->x[2] = a2;
    ctx->x[3] = a3;
    ctx->x[30] = 0;
    NXLOG_I("boot", "entering guest at 0x%llx (x1 = main thread handle 0x%x)", static_cast<unsigned long long>(entry), h);    nxrt::dispatch_call(ctx, entry);
    NXLOG_I("boot", "the entry function returned (x0 = 0x%llx)", static_cast<unsigned long long>(ctx->x[0]));
    nxrt::kernel::main_thread_release();
    nxrt::shutdown();
    return static_cast<int>(ctx->x[0] & 0xff);
}



























