// SPDX-License-Identifier: GPL-3.0-or-later
// Dear ImGui overlay (declarations in include/nxrt/overlay.h). ImGui and its Vulkan backend are pinned in cmake/gfx_deps.cmake.
#include "nxrt/overlay.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <windows.h>
#include <psapi.h>
#include <intrin.h>

#include <SDL3/SDL_clipboard.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "nxrt/config.h"
#include "nxrt/gauge.h"
#include "nxrt/input_source.h"
#include "nxrt/log.h"
#include "nxrt/nvn.h"
#include "nxrt/nvn_backend.h"
#include "nxrt/platform.h"
#include "nxrt/services_api.h"
#include "extras/framegen.h"
#include "extras/nav_ribbon.h"
#include "extras/overlay_composite.h"
#include "fps_stats.h"

namespace nxrt::gfx {
namespace {
constexpr const char* kTag = "overlay";

ImGuiKey map_key(int sc) {
    if (sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z) return static_cast<ImGuiKey>(ImGuiKey_A + (sc - SDL_SCANCODE_A));
    if (sc >= SDL_SCANCODE_1 && sc <= SDL_SCANCODE_9) return static_cast<ImGuiKey>(ImGuiKey_1 + (sc - SDL_SCANCODE_1));
    if (sc == SDL_SCANCODE_0) return ImGuiKey_0;
    if (sc >= SDL_SCANCODE_F1 && sc <= SDL_SCANCODE_F12) return static_cast<ImGuiKey>(ImGuiKey_F1 + (sc - SDL_SCANCODE_F1));
    switch (sc) {
        case SDL_SCANCODE_RETURN: return ImGuiKey_Enter;
        case SDL_SCANCODE_ESCAPE: return ImGuiKey_Escape;
        case SDL_SCANCODE_BACKSPACE: return ImGuiKey_Backspace;
        case SDL_SCANCODE_TAB: return ImGuiKey_Tab;
        case SDL_SCANCODE_SPACE: return ImGuiKey_Space;
        case SDL_SCANCODE_INSERT: return ImGuiKey_Insert;
        case SDL_SCANCODE_DELETE: return ImGuiKey_Delete;
        case SDL_SCANCODE_HOME: return ImGuiKey_Home;
        case SDL_SCANCODE_END: return ImGuiKey_End;
        case SDL_SCANCODE_PAGEUP: return ImGuiKey_PageUp;
        case SDL_SCANCODE_PAGEDOWN: return ImGuiKey_PageDown;
        case SDL_SCANCODE_RIGHT: return ImGuiKey_RightArrow;
        case SDL_SCANCODE_LEFT: return ImGuiKey_LeftArrow;
        case SDL_SCANCODE_DOWN: return ImGuiKey_DownArrow;
        case SDL_SCANCODE_UP: return ImGuiKey_UpArrow;
        case SDL_SCANCODE_LCTRL: return ImGuiKey_LeftCtrl;
        case SDL_SCANCODE_RCTRL: return ImGuiKey_RightCtrl;
        case SDL_SCANCODE_LSHIFT: return ImGuiKey_LeftShift;
        case SDL_SCANCODE_RSHIFT: return ImGuiKey_RightShift;
        case SDL_SCANCODE_LALT: return ImGuiKey_LeftAlt;
        case SDL_SCANCODE_RALT: return ImGuiKey_RightAlt;
        default: return ImGuiKey_None;
    }
}

const double kFpsChoices[] = {0, 30, 60, 90, 120, 144, 165, 240, 360};

// ---- on-screen keyboard: text helpers and the key grid ----
// UTF-8 to code points (a bad byte becomes U+FFFD) and back. The applet's max_len counts characters, so the text is kept as code points.
std::u32string utf8_to_u32(const std::string& s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size()) { out.push_back(0xFFFD); ++i; continue; }
        char32_t v = n == 1 ? c : (c & (0x7Fu >> n));
        bool ok = true;
        for (size_t k = 1; k < n; ++k) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            v = (v << 6) | (cc & 0x3Fu);
        }
        out.push_back(ok ? v : 0xFFFD);
        i += ok ? n : 1;
    }
    return out;
}

std::string u32_to_utf8(const std::u32string& s, size_t from = 0, size_t to = std::u32string::npos) {
    std::string out;
    to = std::min(to, s.size());
    auto put = [&](uint32_t b) { out.push_back(static_cast<char>(b)); };
    for (size_t i = from; i < to; ++i) {
        const uint32_t c = s[i];
        if (c < 0x80) put(c);
        else if (c < 0x800) { put(0xC0 | (c >> 6)); put(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { put(0xE0 | (c >> 12)); put(0x80 | ((c >> 6) & 0x3F)); put(0x80 | (c & 0x3F)); }
        else { put(0xF0 | (c >> 18)); put(0x80 | ((c >> 12) & 0x3F)); put(0x80 | ((c >> 6) & 0x3F)); put(0x80 | (c & 0x3F)); }
    }
    return out;
}

// Typographic punctuation the overlay's font has no glyph for, as plain ASCII, for the game's header and guide text (Pokopia's name question
// is "What\u2019s your name?", drawn as "What?s" otherwise).
std::string fold_punct(const std::string& s) {
    std::u32string in = utf8_to_u32(s), out;
    for (char32_t c : in) {
        switch (c) {
            case 0x2018: case 0x2019: case 0x201A: case 0x2032: out += U'\''; break;
            case 0x201C: case 0x201D: case 0x201E: case 0x2033: out += U'"'; break;
            case 0x2010: case 0x2011: case 0x2012: case 0x2013: case 0x2014: case 0x2212: out += U'-'; break;
            case 0x2026: out += U"..."; break;
            case 0x00A0: case 0x2002: case 0x2003: case 0x2009: case 0x3000: out += U' '; break;
            default: out += c;
        }
    }
    return u32_to_utf8(out);
}

// The grid: rows of 10 column units, a key spans `w` of them. Shift turns letters upper case and punctuation into its second character (alt).
enum class KbOp : uint8_t { Char, Shift, Space, Back, Ok, Cancel };
struct KbKey { char32_t ch; char32_t alt; int w; KbOp op; };
constexpr int kKbUnits = 10;
const std::vector<std::vector<KbKey>>& kb_rows() {
    static const std::vector<std::vector<KbKey>> rows = [] {
        std::vector<std::vector<KbKey>> r;
        auto row = [&](const char* chars, const char* alts) {
            std::vector<KbKey> v;
            for (int i = 0; chars[i]; ++i) v.push_back({static_cast<char32_t>(chars[i]), static_cast<char32_t>(alts[i]), 1, KbOp::Char});
            r.push_back(v);
        };
        row("1234567890", "1234567890");
        row("qwertyuiop", "QWERTYUIOP");
        row("asdfghjkl-", "ASDFGHJKL_");
        row("zxcvbnm,.?", "ZXCVBNM;:/");
        r.push_back({{0, 0, 2, KbOp::Shift}, {U' ', U' ', 4, KbOp::Space}, {U'!', U'@', 1, KbOp::Char}, {U'\'', U'"', 1, KbOp::Char}, {0, 0, 2, KbOp::Back}});
        r.push_back({{0, 0, 5, KbOp::Cancel}, {0, 0, 5, KbOp::Ok}});
        return r;
    }();
    return rows;
}
// The key of a row that covers column unit u (past the end: the last key); *start = its first unit.
int kb_key_at(const std::vector<KbKey>& row, int u, int* start) {
    int x = 0;
    for (int i = 0; i < static_cast<int>(row.size()); ++i) {
        if (u < x + row[i].w || i + 1 == static_cast<int>(row.size())) {
            if (start) *start = x;
            return i;
        }
        x += row[i].w;
    }
    return 0;
}

// Pad directions the keyboard moves on: the d-pad, or the left stick past half way (set_keyboard_pad adds the stick bits).
constexpr uint64_t kKbDirBits[4] = {
    hid::btn(hid::kBtnUp) | hid::btn(hid::kBtnStickLUp), hid::btn(hid::kBtnDown) | hid::btn(hid::kBtnStickLDown),
    hid::btn(hid::kBtnLeft) | hid::btn(hid::kBtnStickLLeft), hid::btn(hid::kBtnRight) | hid::btn(hid::kBtnStickLRight)};
constexpr double kKbRepeatDelay = 0.40, kKbRepeatEvery = 0.07;   // seconds: a held direction or B repeats

// Drops pressed keys and mouse buttons so nothing stays latched in ImGui (the caller holds the context's mutex).
void clear_imgui_input(ImGuiContext* ctx) {
    ImGuiContext* prev = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputKeys();
    io.ClearInputMouse();
    ImGui::SetCurrentContext(prev);
}
}  // namespace

namespace {
std::atomic<int> g_shader_backlog{0};
inline ImU32 rgba_to_imu32(uint32_t rgba) {
    return IM_COL32((rgba >> 24) & 0xFF, (rgba >> 16) & 0xFF, (rgba >> 8) & 0xFF, rgba & 0xFF);
}
}
void set_shader_backlog(int pending) { g_shader_backlog.store(pending < 0 ? 0 : pending, std::memory_order_relaxed); }
int shader_backlog() { return g_shader_backlog.load(std::memory_order_relaxed); }

void HudCanvas::rect_filled(float x0, float y0, float x1, float y1, uint32_t rgba, float rounding) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), rgba_to_imu32(rgba), rounding);
}
void HudCanvas::rect_filled_hgrad(float x0, float y0, float x1, float y1, uint32_t rgba_left, uint32_t rgba_right) const {
    if (!draw_list) return;
    const ImU32 cl = rgba_to_imu32(rgba_left);
    const ImU32 cr = rgba_to_imu32(rgba_right);
    static_cast<ImDrawList*>(draw_list)->AddRectFilledMultiColor(ImVec2(x0, y0), ImVec2(x1, y1), cl, cr, cr, cl);
}
void HudCanvas::rect(float x0, float y0, float x1, float y1, uint32_t rgba, float rounding, float thickness) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), rgba_to_imu32(rgba), rounding, 0, thickness);
}
void HudCanvas::line(float x0, float y0, float x1, float y1, uint32_t rgba, float thickness) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), rgba_to_imu32(rgba), thickness);
}
void HudCanvas::triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t rgba) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddTriangleFilled(ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x2, y2), rgba_to_imu32(rgba));
}
void HudCanvas::quad_filled(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3, uint32_t rgba) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddQuadFilled(ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x2, y2), ImVec2(x3, y3), rgba_to_imu32(rgba));
}
void HudCanvas::quad(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3, uint32_t rgba, float thickness) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddQuad(ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x2, y2), ImVec2(x3, y3), rgba_to_imu32(rgba), thickness);
}
void HudCanvas::convex_poly_filled(const float* xy, int count, uint32_t rgba) const {
    if (!draw_list || !xy || count < 3) return;
    if (count > 32) count = 32;
    ImVec2 pts[32];
    for (int i = 0; i < count; ++i) pts[i] = ImVec2(xy[i * 2], xy[i * 2 + 1]);
    static_cast<ImDrawList*>(draw_list)->AddConvexPolyFilled(pts, count, rgba_to_imu32(rgba));
}
void HudCanvas::convex_poly_hgrad(const float* xy, int count, float x_left, float x_right, uint32_t rgba_left, uint32_t rgba_right) const {
    if (!draw_list || !xy || count < 3) return;
    if (count > 32) count = 32;
    auto* dl = static_cast<ImDrawList*>(draw_list);
    const float span = (x_right > x_left + 1e-4f) ? (x_right - x_left) : 1.0f;
    const float lr = static_cast<float>((rgba_left >> 24) & 0xFF);
    const float lg = static_cast<float>((rgba_left >> 16) & 0xFF);
    const float lb = static_cast<float>((rgba_left >> 8) & 0xFF);
    const float la = static_cast<float>(rgba_left & 0xFF);
    const float rr = static_cast<float>((rgba_right >> 24) & 0xFF);
    const float rg = static_cast<float>((rgba_right >> 16) & 0xFF);
    const float rb = static_cast<float>((rgba_right >> 8) & 0xFF);
    const float ra = static_cast<float>(rgba_right & 0xFF);
    const ImVec2 uv = ImGui::GetFontTexUvWhitePixel();
    const int idx_count = (count - 2) * 3;
    dl->PrimReserve(idx_count, count);
    const ImDrawIdx vtx_base = static_cast<ImDrawIdx>(dl->_VtxCurrentIdx);
    for (int i = 0; i < count; ++i) {
        const float px = xy[i * 2];
        const float py = xy[i * 2 + 1];
        float t = (px - x_left) / span;
        if (t < 0.f) t = 0.f;
        if (t > 1.f) t = 1.f;
        const ImU32 col = IM_COL32(
            static_cast<int>(lr + (rr - lr) * t + 0.5f),
            static_cast<int>(lg + (rg - lg) * t + 0.5f),
            static_cast<int>(lb + (rb - lb) * t + 0.5f),
            static_cast<int>(la + (ra - la) * t + 0.5f));
        dl->PrimWriteVtx(ImVec2(px, py), uv, col);
    }
    for (int i = 2; i < count; ++i) {
        dl->PrimWriteIdx(vtx_base);
        dl->PrimWriteIdx(static_cast<ImDrawIdx>(vtx_base + i - 1));
        dl->PrimWriteIdx(static_cast<ImDrawIdx>(vtx_base + i));
    }
}
void HudCanvas::polyline(const float* xy, int count, uint32_t rgba, bool closed, float thickness) const {
    if (!draw_list || !xy || count < 2) return;
    if (count > 32) count = 32;
    ImVec2 pts[32];
    for (int i = 0; i < count; ++i) pts[i] = ImVec2(xy[i * 2], xy[i * 2 + 1]);
    static_cast<ImDrawList*>(draw_list)->AddPolyline(pts, count, rgba_to_imu32(rgba), closed ? ImDrawFlags_Closed : ImDrawFlags_None, thickness);
}
void HudCanvas::circle_filled(float cx, float cy, float r, uint32_t rgba, int segments) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddCircleFilled(ImVec2(cx, cy), r, rgba_to_imu32(rgba), segments);
}
void HudCanvas::circle(float cx, float cy, float r, uint32_t rgba, float thickness, int segments) const {
    if (draw_list) static_cast<ImDrawList*>(draw_list)->AddCircle(ImVec2(cx, cy), r, rgba_to_imu32(rgba), segments, thickness);
}
void HudCanvas::text(float x, float y, float size, uint32_t rgba, const char* str) const {
    if (draw_list && font && str && *str) {
        static_cast<ImDrawList*>(draw_list)->AddText(static_cast<ImFont*>(font), size, ImVec2(x, y), rgba_to_imu32(rgba), str);
    }
}
void HudCanvas::text_outlined(float x, float y, float size, uint32_t rgba, uint32_t outline_rgba, const char* str) const {
    if (!draw_list || !font || !str || !*str) return;
    auto* dl = static_cast<ImDrawList*>(draw_list);
    auto* f = static_cast<ImFont*>(font);
    const ImU32 oc = rgba_to_imu32(outline_rgba);
    dl->AddText(f, size, ImVec2(x - 1.f, y), oc, str);
    dl->AddText(f, size, ImVec2(x + 1.f, y), oc, str);
    dl->AddText(f, size, ImVec2(x, y - 1.f), oc, str);
    dl->AddText(f, size, ImVec2(x, y + 1.f), oc, str);
    dl->AddText(f, size, ImVec2(x, y), rgba_to_imu32(rgba), str);
}
void HudCanvas::measure_text(float size, const char* str, float* out_w, float* out_h) const {
    if (!font || !str || !*str) {
        if (out_w) *out_w = 0.f;
        if (out_h) *out_h = 0.f;
        return;
    }
    const ImVec2 sz = static_cast<ImFont*>(font)->CalcTextSizeA(size, FLT_MAX, 0.0f, str);
    if (out_w) *out_w = sz.x;
    if (out_h) *out_h = sz.y;
}
void HudCanvas::image(uint64_t tex_id, float x0, float y0, float x1, float y1,
                      float u0, float v0, float u1, float v1, uint32_t tint_rgba) const {
    if (!draw_list || !tex_id) return;
    static_cast<ImDrawList*>(draw_list)->AddImage(
        static_cast<ImTextureID>(tex_id), ImVec2(x0, y0), ImVec2(x1, y1),
        ImVec2(u0, v0), ImVec2(u1, v1), rgba_to_imu32(tint_rgba));
}
void HudCanvas::image_rounded(uint64_t tex_id, float x0, float y0, float x1, float y1,
                              float u0, float v0, float u1, float v1,
                              float rounding, uint32_t tint_rgba) const {
    if (!draw_list || !tex_id) return;
    static_cast<ImDrawList*>(draw_list)->AddImageRounded(
        static_cast<ImTextureID>(tex_id), ImVec2(x0, y0), ImVec2(x1, y1),
        ImVec2(u0, v0), ImVec2(u1, v1), rgba_to_imu32(tint_rgba), rounding);
}

namespace {
struct RibbonSink {
    std::vector<Ribbon3DVertex> verts;
    Ribbon3DParams params;
    bool active = false;
};
}  // namespace

void HudCanvas::set_ribbon_3d(const Ribbon3DVertex* verts, uint32_t count, const Ribbon3DParams& params) const {
    if (!ribbon_sink || !verts || count < 4) return;
    auto* sink = static_cast<RibbonSink*>(ribbon_sink);
    sink->verts.assign(verts, verts + count);
    sink->params = params;
    sink->active = true;
}

struct Overlay::Impl {
    Context ctx;
    ImGuiContext* imgui = nullptr;
    ImGuiStyle base_style;            // the style at scale 1; style = base_style scaled by ui_scale (update_scale)
    float ui_scale = 0.f;             // 0: not applied yet
    bool backend_ready = false;
    std::atomic<bool> visible{false};
    std::mutex mu;                    // guards the ImGui context: events arrive on the game's main thread, drawing happens on the presenting thread
    int toggle_sc = SDL_SCANCODE_F1;
    bool restore_relative = false;
    std::chrono::steady_clock::time_point last_frame = std::chrono::steady_clock::now();
    uint64_t drawn = 0;

    struct Panel { std::string title; std::function<void()> fn; };
    std::vector<Panel> panels;
    std::function<void()> menu;           // set_menu: the game's own menu replaces the Diagnostics window and the panels
    std::mutex hud_mu;
    std::function<void(const HudCanvas&)> hud_cb;
    std::atomic<bool> has_hud_cb{false};
    struct HudTex {
        gfx::Image img;
        VkDescriptorSet ds = VK_NULL_HANDLE;
    };
    std::mutex hud_tex_mu;
    VkSampler hud_tex_sampler = VK_NULL_HANDLE;
    std::vector<HudTex> hud_textures;
    std::mutex picker_mu;
    PickerCallbacks picker_cb;
    std::atomic<bool> picker_open{false};
    std::atomic<bool> want_picker_close{false};
    std::atomic<bool> want_picker_clear{false};
    std::atomic<int> picker_tab_delta{0};
    bool picker_restore_relative = false;
    char picker_filter[128] = {};
    int picker_tab = 0;  // 0=All, 1=Landmarks, 2=NPCs
    std::vector<PickerItem> picker_items;
    bool picker_has_pin = false;
    std::string picker_pin_name;
    double picker_last_refresh_s = -1e9;
    std::atomic<bool> want_hide{false};   // request_hide: applied at the start of the next draw (the menu runs with mu held)

    // The pad for the menu (set_menu_pad, on the thread that polls the pads)
    uint64_t menu_chord = 0;              // [gfx] overlay_pad as npad bits; 0 = no chord
    uint64_t menu_prev = 0;               // the buttons at the last poll
    uint64_t menu_mask = 0;               // held back from the game until released (the press that opened or closed the menu)
    std::atomic<bool> hold_on_hide{false};   // the overlay was hidden: what is held at the next poll stays away from the game
    uint64_t menu_stick = 0;              // d-pad bits the left stick pressed at the last poll

    // menu state applied by apply_pending()
    bool want_mode = false;
    PresentMode new_mode = PresentMode::Fifo;
    bool want_fullscreen = false;
    platform::FullscreenMode new_fullscreen = platform::FullscreenMode::Windowed;

    float frame_ms[120] = {};
    int frame_ms_pos = 0;
    int fps_choice = 0;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    bool hdr = false;                 // HDR swapchain: the overlay is drawn into an RGBA8 target and composited (draw)
    extras::OverlayComposite composite;
    extras::NavRibbonPass nav_ribbon;
    bool nav_ribbon_ok = false;
    RibbonSink ribbon_sink;

    // Corner frame-rate counter. set_counter() stores the settings from any thread; the rest is used by the presenting thread only.
    std::atomic<bool> ptr_show{false}, ptr_in_shots{true};
    std::atomic<float> ptr_x{0}, ptr_y{0};
    std::atomic<bool> counter_show{false};
    std::atomic<int> counter_corner{0};
    std::atomic<int> counter_detail{1};
    std::atomic<bool> counter_in_shots{false};
    std::atomic<bool> counter_lang_zh{true};
    bool counter_was_on = false;
    bool counter_have_prev = false;
    std::chrono::steady_clock::time_point counter_prev{};
    FpsStats stats;
    double text_s = -1e9;                 // time of the last text refresh, seconds (far past: refresh at once)
    FpsStats::Snapshot snap;              // the snapshot taken at text_s
    std::string counter_lines[4];
    int counter_n = 0;

    // Detailed Performance HUD (counter_detail == 3) state: refreshed every 250-500 ms with minimal overhead.
    struct PerfHudState {
        std::string cpu_name;
        bool cpu_name_init = false;
        // Rolling frame times for sparkline graph (newest at graph_pos - 1)
        static constexpr int kGraphN = 120;
        float graph_ms[kGraphN] = {};
        int graph_pos = 0;
        int graph_count = 0;
        // Hardware telemetry (sampled at <= 2 Hz)
        double hw_last_s = -1e9;
        uint64_t prev_sys_idle = 0, prev_sys_kernel = 0, prev_sys_user = 0;
        uint64_t prev_proc_kernel = 0, prev_proc_user = 0;
        bool have_cpu_prev = false;
        int cpu_total_pct = -1;
        int cpu_game_pct = -1;
        double ram_used_gb = 0.0, ram_total_gb = 0.0, ram_game_gb = 0.0;
        double vram_used_gb = 0.0, vram_total_gb = 0.0;
        int gpu_util_pct = -1;
        int gpu_temp_c = -1;
        int gpu_power_w = -1;
        int gpu_clock_mhz = -1;
        // Dynamic vendor telemetry handles (loaded lazily on first detailed HUD sample)
        bool vendor_init = false;
        HMODULE adl_dll = nullptr;
        void* adl_ctx = nullptr;
        int adl_adapter = -1;
        using AdlDestroyFn = int(__cdecl*)(void*);
        using AdlPmLogFn = int(__cdecl*)(void*, int, void*);
        AdlDestroyFn adl_destroy = nullptr;
        AdlPmLogFn adl_pmlog = nullptr;
        HMODULE nvml_dll = nullptr;
        void* nvml_dev = nullptr;
        using NvmlShutdownFn = int(__cdecl*)();
        using NvmlUtilFn = int(__cdecl*)(void*, void*);
        using NvmlTempFn = int(__cdecl*)(void*, unsigned int, unsigned int*);
        using NvmlPowerFn = int(__cdecl*)(void*, unsigned int*);
        using NvmlClockFn = int(__cdecl*)(void*, unsigned int, unsigned int*);
        NvmlShutdownFn nvml_shutdown = nullptr;
        NvmlUtilFn nvml_util = nullptr;
        NvmlTempFn nvml_temp = nullptr;
        NvmlPowerFn nvml_power = nullptr;
        NvmlClockFn nvml_clock = nullptr;
        // Per-frame backend telemetry (delta between 250 ms samples)
        bool have_be_prev = false;
        uint64_t prev_presents = 0;
        uint64_t prev_tex_uploads = 0;
        uint64_t prev_gpu_busy_ns = 0;
        double prev_prep_ms = 0.0;
        double prev_pipe_ms = 0.0;
        double prev_shader_ms = 0.0;
        double prev_upload_ms = 0.0;
        double gpu_prep_ms = 0.0;
        double gpu_busy_ms = 0.0;
        double pipe_wait_ms = 0.0;
        double shader_wait_ms = 0.0;
        double tex_uploads_pf = 0.0;
        double tex_upload_ms = 0.0;
    } perf;

    void build_ui();
    void update_scale(const VkExtent2D& extent);
    void refresh_counter_text(double now_s);
    void refresh_perf_hud(double now_s);
    void shutdown_perf_hud();
    void draw_counter(const FrameInfo& f, double now_s);
    void draw_perf_hud(const FrameInfo& f, double now_s);
    // The "Preparing shaders" note: shown once the backlog has stood for a second and a half, gone a second after it is empty.
    bool note_enabled = true;
    bool note_read = false;
    double note_since_s = -1.0, note_empty_s = -1.0;
    int note_peak = 0;
    bool note_visible(double now_s);
    void draw_note(const FrameInfo& f);

    // The on-screen keyboard (services_api.h swkbdui). kb_mu guards the question and its text: key events edit it on the game's main thread,
    // the pad and the mouse in draw() on the presenting thread. Lock order: mu (ImGui), then kb_mu, then the swkbdui question's own lock.
    std::mutex kb_mu;
    std::atomic<bool> kb_open{false};
    std::atomic<bool> kb_by_label{false};
    uint64_t kb_id = 0;                   // the question on screen, 0 = none
    uint64_t kb_answered = 0;             // the last question this keyboard answered (ids only grow): never reopened
    services::swkbdui::Request kb_req;
    std::u32string kb_text;               // code points
    size_t kb_caret = 0;
    int kb_shift = 0;                     // 0 off, 1 the next character only, 2 locked
    int kb_row = 1, kb_unit = 0;          // the pad's key: row, column unit
    double kb_rep[5] = {};                // the next repeat of up, down, left, right, B (seconds; DBL_MAX: not held since a press)
    // The pad (set_keyboard_pad, on the thread that polls it): what is held, and presses not yet taken by draw().
    std::atomic<uint64_t> pad_held{0}, pad_new{0};
    uint64_t pad_prev = 0, pad_mask = 0;  // set_keyboard_pad only

    bool kb_poll();                       // follows swkbdui::pending(): opens a new question, drops one answered elsewhere; true while open
    bool kb_finish(bool ok);              // kb_mu held: answers the question; false when there is none or OK is not allowed yet (min_len)
    void kb_insert(const std::u32string& t);   // kb_mu held: at the caret, control characters dropped, up to max_len
    void kb_backspace();                  // kb_mu held
    bool kb_press(const KbKey& k);        // kb_mu held: an on-screen key; true when it answered the question
    bool kb_key_event(const input::Event& e);   // a physical key or text; kb_mu and mu not held
    void draw_keyboard(double now_s);     // mu held, inside an ImGui frame
};

void Overlay::Impl::build_ui() {
    if (menu) {
        menu();
        return;
    }
    Renderer& r = *ctx.renderer;
    // First-use layout, in units of the UI scale (update_scale): the panels (the settings menu) sit at the top left, the diagnostics window
    // at the top right and folded up. The player moves them; nothing is remembered between runs.
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const float k = ui_scale > 0.f ? ui_scale : 1.f;
    const float diag_w = std::min(420.f * k, std::max(200.f, disp.x - 32.f * k));
    ImGui::SetNextWindowPos(ImVec2(std::max(16.f * k, disp.x - diag_w - 16.f * k), 16.f * k), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(diag_w, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Diagnostics")) {
        const GpuInfo& g = ctx.device->gpu();
        ImGui::Text("GPU: %s", g.name.c_str());
        ImGui::Text("Vulkan 1.3, driver 0x%08x, validation %s%s", g.driver_version, ctx.device->validation_enabled() ? "on" : "off",
                    ctx.device->headless() ? ", headless" : "");
        const VkExtent2D e = r.extent();
        ImGui::Text("Target: %ux%u, %u images, %u frames in flight", e.width, e.height, r.image_count(), r.frames_in_flight());

        ImGui::SeparatorText("Presentation");
        const PresentMode modes[] = {PresentMode::Fifo, PresentMode::FifoRelaxed, PresentMode::Mailbox, PresentMode::Immediate};
        int cur = static_cast<int>(r.requested_present_mode());
        if (ImGui::BeginCombo("Present mode", present_mode_name(modes[cur]))) {
            for (int i = 0; i < 4; ++i) {
                const bool ok = r.present_mode_supported(modes[i]);
                ImGui::BeginDisabled(!ok);
                if (ImGui::Selectable(present_mode_name(modes[i]), i == cur)) { want_mode = true; new_mode = modes[i]; }
                ImGui::EndDisabled();
            }
            ImGui::EndCombo();
        }
        if (r.present_mode() != r.requested_present_mode()) ImGui::TextDisabled("in use: %s (requested mode unavailable)", present_mode_name(r.present_mode()));

        if (ctx.limiter) {
            char label[32];
            const double t = ctx.limiter->target_fps();
            std::snprintf(label, sizeof label, t > 0 ? "%.0f fps" : "unlimited", t);
            if (ImGui::BeginCombo("Frame cap", label)) {
                for (double v : kFpsChoices) {
                    std::snprintf(label, sizeof label, v > 0 ? "%.0f fps" : "unlimited", v);
                    if (ImGui::Selectable(label, v == t)) ctx.limiter->set_target_fps(v);
                }
                ImGui::EndCombo();
            }
            const float hz = platform::has_window() ? platform::display_refresh_hz() : 0.f;
            if (hz > 0) {
                ImGui::TextDisabled("display %.2f Hz; VRR-friendly cap %.1f fps", hz, suggest_vrr_cap(hz));
                if (ImGui::SmallButton("Use VRR cap")) ctx.limiter->set_target_fps(suggest_vrr_cap(hz));
            }
        }

        if (ctx.has_window && platform::has_window()) {
            ImGui::SeparatorText("Window");
            const auto fm = platform::fullscreen_mode();
            if (ImGui::RadioButton("Windowed", fm == platform::FullscreenMode::Windowed)) { want_fullscreen = true; new_fullscreen = platform::FullscreenMode::Windowed; }
            ImGui::SameLine();
            if (ImGui::RadioButton("Borderless", fm == platform::FullscreenMode::Borderless)) { want_fullscreen = true; new_fullscreen = platform::FullscreenMode::Borderless; }
            ImGui::SameLine();
            if (ImGui::RadioButton("Exclusive", fm == platform::FullscreenMode::Exclusive)) { want_fullscreen = true; new_fullscreen = platform::FullscreenMode::Exclusive; }
            bool rel = platform::relative_mouse();
            if (ImGui::Checkbox("Raw mouse (relative mode)", &rel)) platform::set_relative_mouse(rel);
            ImGui::SameLine();
            ImGui::TextDisabled(platform::relative_mouse_is_raw() ? "(unaccelerated)" : "(system scaling!)");
        }

        ImGui::SeparatorText("Timing");
        const RendererStats rs = r.stats();
        const float dt_ms = ImGui::GetIO().DeltaTime * 1000.f;
        frame_ms[frame_ms_pos] = dt_ms;
        frame_ms_pos = (frame_ms_pos + 1) % 120;
        ImGui::Text("CPU frame %.2f ms (%.0f fps)   GPU %.2f ms (avg %.2f)", dt_ms, dt_ms > 0 ? 1000.f / dt_ms : 0.f, rs.gpu_ms_last, rs.gpu_ms_avg);
        ImGui::PlotLines("##ft", frame_ms, 120, frame_ms_pos, nullptr, 0.f, 33.f, ImVec2(0, 48));
        ImGui::Text("frames %llu, swapchains %llu, out-of-date %llu, screenshots %llu", static_cast<unsigned long long>(rs.frames),
                    static_cast<unsigned long long>(rs.swapchain_creates), static_cast<unsigned long long>(rs.out_of_date),
                    static_cast<unsigned long long>(rs.screenshots));
        const ValidationStats vs = ctx.device->validation_stats();
        ImGui::Text("validation: %llu errors, %llu warnings", static_cast<unsigned long long>(vs.errors), static_cast<unsigned long long>(vs.warnings));

        ImGui::SeparatorText("NVN front end");
        ImGui::Text("%zu entry points, %zu implemented; stub calls %llu (%zu distinct)", nvn::entry_count(), nvn::implemented_count(),
                    static_cast<unsigned long long>(nvn::stub_calls_total()), nvn::stubs_called_distinct());
        if (ImGui::TreeNode("Most called stubs")) {
            auto calls = nvn::called_entries();
            int shown = 0;
            for (auto& c : calls) {
                if (c.implemented) continue;
                ImGui::Text("%8llu  %s", static_cast<unsigned long long>(c.calls), c.entry->name);
                if (++shown >= 12) break;
            }
            if (!shown) ImGui::TextDisabled("none called");
            ImGui::TreePop();
        }
    }
    ImGui::End();
    for (auto& p : panels) {
        ImGui::SetNextWindowPos(ImVec2(16.f * k, 16.f * k), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(std::min(760.f * k, std::max(240.f, disp.x - 32.f * k)), std::min(680.f * k, std::max(200.f, disp.y - 32.f * k))),
                                 ImGuiCond_FirstUseEver);
        if (ImGui::Begin(p.title.c_str())) p.fn();
        ImGui::End();
    }
}

// DPI: the UI is laid out for a 1080 pixel high frame at 16 px text and grows with the frame (4K: twice the size; a Windows display scaling
// of 150% shows up as more pixels in the same window, so it is covered too). Never below 1: a 720p window keeps the 16 px text.
void Overlay::Impl::update_scale(const VkExtent2D& extent) {
    const float want = std::round(std::clamp(static_cast<float>(extent.height) / 1080.f, 1.0f, 3.0f) * 20.f) / 20.f;   // steps of 0.05
    if (want == ui_scale) return;
    ui_scale = want;
    ImGuiStyle& style = ImGui::GetStyle();
    style = base_style;
    style.ScaleAllSizes(want);
    style.FontScaleDpi = want;
}

std::unique_ptr<Overlay> Overlay::create(const Context& ctx, std::string* error) {
    if (!ctx.device || !ctx.renderer) {
        if (error) *error = "overlay needs a device and a renderer";
        return nullptr;
    }
    std::unique_ptr<Overlay> o(new Overlay());
    o->impl_ = new Impl();
    Impl& s = *o->impl_;
    s.ctx = ctx;
    s.color_format = ctx.renderer->format();
    if (ctx.renderer->output_kind() != extras::OutputKind::Sdr8 || extras::hdr_state().scene_hdr) {
        // HDR swapchain: ImGui draws into an RGBA8 target that the composite maps onto the swapchain (see draw)
        if (s.composite.init(*ctx.device)) {
            s.hdr = true;
            s.color_format = VK_FORMAT_R8G8B8A8_UNORM;
        } else {
            NXLOG_W(kTag, "HDR overlay composite failed to initialise: the overlay is drawn straight into the HDR swapchain (colours will look wrong)");
        }
    }
    s.nav_ribbon_ok = s.nav_ribbon.init(*ctx.device);
    const std::string key = config_get_str("gfx", "overlay_key", "F1");
    if (const SDL_Scancode sc = SDL_GetScancodeFromName(key.c_str()); sc != SDL_SCANCODE_UNKNOWN) s.toggle_sc = sc;
    else NXLOG_W(kTag, "unknown [gfx] overlay_key '%s': using F1", key.c_str());
    {   // [gfx] overlay_pad: Switch buttons joined by '+' (after the player's mapping), "none" = no chord
        std::string chord = config_get_str("gfx", "overlay_pad", "minus+plus");
        for (char& c : chord) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        static const std::pair<const char*, hid::NpadButtonBit> kNames[] = {
            {"a", hid::kBtnA},       {"b", hid::kBtnB},       {"x", hid::kBtnX},         {"y", hid::kBtnY},         {"l", hid::kBtnL},
            {"r", hid::kBtnR},       {"zl", hid::kBtnZL},     {"zr", hid::kBtnZR},       {"plus", hid::kBtnPlus},   {"minus", hid::kBtnMinus},
            {"up", hid::kBtnUp},     {"down", hid::kBtnDown}, {"left", hid::kBtnLeft},   {"right", hid::kBtnRight}, {"lstick", hid::kBtnStickL},
            {"rstick", hid::kBtnStickR}};
        uint64_t bits = 0;
        bool ok = true;
        size_t from = 0;
        while (chord != "none" && !chord.empty() && from <= chord.size()) {
            size_t plus = chord.find('+', from);
            if (plus == std::string::npos) plus = chord.size();
            const std::string name = chord.substr(from, plus - from);
            bool found = false;
            for (const auto& kv : kNames)
                if (name == kv.first) { bits |= hid::btn(kv.second); found = true; }
            ok = ok && found;
            from = plus + 1;
        }
        if (!ok) {
            NXLOG_W(kTag, "unknown [gfx] overlay_pad '%s': using minus+plus", chord.c_str());
            bits = hid::btn(hid::kBtnMinus) | hid::btn(hid::kBtnPlus);
        }
        s.menu_chord = bits;
    }

    IMGUI_CHECKVERSION();
    s.imgui = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;     // no imgui.ini next to the exe
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;      // set_menu_pad feeds it
    // XC2 Aegis Theme (Deep Ocean Slate + Aegis Emerald / Warm Gold accents), adapted from totk-pc-recomp / SavageWindRecomp (MIT).
    {
        ImGuiStyle& st = ImGui::GetStyle();
        ImGui::StyleColorsDark(&st);
        st.WindowRounding = 14.0f;
        st.ChildRounding = 10.0f;
        st.FrameRounding = 8.0f;
        st.PopupRounding = 8.0f;
        st.ScrollbarRounding = 8.0f;
        st.GrabRounding = 8.0f;
        st.TabRounding = 8.0f;
        st.WindowPadding = ImVec2(24.0f, 18.0f);
        st.FramePadding = ImVec2(12.0f, 7.0f);
        st.ItemSpacing = ImVec2(12.0f, 10.0f);
        st.ScrollbarSize = 14.0f;
        st.WindowBorderSize = 1.0f;
        st.ChildBorderSize = 0.0f;
        st.PopupBorderSize = 1.0f;
        st.TabBarBorderSize = 1.0f;
        ImVec4* c = st.Colors;
        const ImVec4 gold(0.94f, 0.78f, 0.36f, 1.0f);
        const ImVec4 teal(0.14f, 0.42f, 0.44f, 1.0f);
        const ImVec4 teal_hi(0.18f, 0.56f, 0.54f, 1.0f);
        c[ImGuiCol_WindowBg] = ImVec4(0.055f, 0.085f, 0.135f, 0.94f);
        c[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        c[ImGuiCol_PopupBg] = ImVec4(0.06f, 0.10f, 0.16f, 0.97f);
        c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.02f, 0.035f, 0.06f, 0.60f);   // behind a dialog: the window colour, darker (the default is a light grey veil)
        c[ImGuiCol_Border] = ImVec4(0.22f, 0.46f, 0.54f, 0.45f);
        c[ImGuiCol_Text] = ImVec4(0.94f, 0.96f, 0.98f, 1.0f);
        c[ImGuiCol_TextDisabled] = ImVec4(0.58f, 0.66f, 0.74f, 1.0f);
        c[ImGuiCol_TitleBg] = ImVec4(0.045f, 0.075f, 0.125f, 0.98f);
        c[ImGuiCol_TitleBgActive] = ImVec4(0.055f, 0.095f, 0.155f, 0.98f);
        c[ImGuiCol_FrameBg] = ImVec4(0.09f, 0.15f, 0.22f, 0.95f);
        c[ImGuiCol_FrameBgHovered] = ImVec4(0.13f, 0.25f, 0.35f, 1.0f);
        c[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.33f, 0.43f, 1.0f);
        c[ImGuiCol_Button] = ImVec4(0.11f, 0.21f, 0.30f, 1.0f);
        c[ImGuiCol_ButtonHovered] = teal;
        c[ImGuiCol_ButtonActive] = teal_hi;
        c[ImGuiCol_Header] = ImVec4(0.12f, 0.28f, 0.36f, 0.90f);
        c[ImGuiCol_HeaderHovered] = teal;
        c[ImGuiCol_HeaderActive] = teal_hi;
        c[ImGuiCol_Tab] = ImVec4(0.09f, 0.15f, 0.22f, 0.90f);
        c[ImGuiCol_TabHovered] = teal;
        c[ImGuiCol_TabSelected] = ImVec4(0.14f, 0.36f, 0.42f, 1.0f);
        c[ImGuiCol_TabSelectedOverline] = gold;
        c[ImGuiCol_SliderGrab] = ImVec4(0.22f, 0.74f, 0.68f, 1.0f);
        c[ImGuiCol_SliderGrabActive] = gold;
        c[ImGuiCol_CheckMark] = ImVec4(0.30f, 0.88f, 0.78f, 1.0f);
        c[ImGuiCol_NavCursor] = gold;
        c[ImGuiCol_Separator] = ImVec4(0.24f, 0.44f, 0.52f, 0.50f);
        c[ImGuiCol_ScrollbarBg] = ImVec4(0.04f, 0.07f, 0.11f, 0.60f);
        c[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f, 0.38f, 0.46f, 0.80f);
        c[ImGuiCol_ScrollbarGrabHovered] = teal;
        c[ImGuiCol_ScrollbarGrabActive] = gold;
    }
    // Load crisp system CJK font (ImGui 1.92 dynamic atlas rasterizes glyphs on demand at native DPI).
    {
        bool font_loaded = false;
        static const char* const kCjkFonts[] = {
            "C:\\Windows\\Fonts\\msyh.ttc",
            "C:\\Windows\\Fonts\\Deng.ttf",
            "C:\\Windows\\Fonts\\simhei.ttf",
        };
        std::error_code ec;
        for (const char* path : kCjkFonts) {
            if (std::filesystem::exists(path, ec) && !ec) {
                ImFontConfig fc;
                fc.FontNo = 0;
                if (io.Fonts->AddFontFromFileTTF(path, 0.0f, &fc) != nullptr) {
                    NXLOG_I(kTag, "overlay font loaded: %s", path);
                    font_loaded = true;
                    break;
                }
            }
        }
        if (!font_loaded) {
            io.Fonts->AddFontDefaultVector();
            NXLOG_I(kTag, "overlay font: default vector fallback");
        }
    }
    ImGui::GetStyle().FontSizeBase = 18.0f;
    s.base_style = ImGui::GetStyle();
    const VkExtent2D e = ctx.renderer->extent();
    io.DisplaySize = ImVec2(static_cast<float>(e.width), static_cast<float>(e.height));

    VkPipelineRenderingCreateInfoKHR rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &s.color_format;
    ImGui_ImplVulkan_InitInfo ii = {};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = ctx.device->instance();
    ii.PhysicalDevice = ctx.device->physical();
    ii.Device = ctx.device->device();
    ii.QueueFamily = ctx.device->queue_family();
    ii.Queue = ctx.device->queue();
    ii.DescriptorPoolSize = 16;
    ii.MinImageCount = 2;
    ii.ImageCount = std::max(2u, ctx.renderer->image_count());
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo = rci;
    ii.CheckVkResultFn = [](VkResult r) { if (r != VK_SUCCESS) NXLOG_E(kTag, "ImGui Vulkan call failed (%d)", static_cast<int>(r)); };
    if (!ImGui_ImplVulkan_Init(&ii)) {
        if (error) *error = "ImGui_ImplVulkan_Init failed";
        ImGui::DestroyContext(s.imgui);
        s.imgui = nullptr;
        return nullptr;
    }
    s.backend_ready = true;
    NXLOG_I(kTag, "overlay ready (toggle key %s)", SDL_GetScancodeName(static_cast<SDL_Scancode>(s.toggle_sc)));
    return o;
}

Overlay::~Overlay() {
    if (!impl_) return;
    impl_->shutdown_perf_hud();
    if (impl_->ctx.device) impl_->ctx.device->wait_idle();
    {
        std::lock_guard<std::mutex> tlk(impl_->hud_tex_mu);
        for (auto& ht : impl_->hud_textures) {
            if (ht.ds && impl_->backend_ready) ImGui_ImplVulkan_RemoveTexture(ht.ds);
            if (ht.img && impl_->ctx.device) impl_->ctx.device->destroy_image(ht.img);
        }
        impl_->hud_textures.clear();
        if (impl_->hud_tex_sampler && impl_->ctx.device) {
            vkDestroySampler(impl_->ctx.device->device(), impl_->hud_tex_sampler, nullptr);
            impl_->hud_tex_sampler = VK_NULL_HANDLE;
        }
    }
    impl_->nav_ribbon.shutdown();
    impl_->composite.shutdown();
    if (impl_->backend_ready) ImGui_ImplVulkan_Shutdown();
    if (impl_->imgui) ImGui::DestroyContext(impl_->imgui);
    delete impl_;
}

void Overlay::set_visible(bool v) {
    Impl& s = *impl_;
    if (s.visible == v) return;
    s.visible = v;
    if (v) {
        s.restore_relative = platform::has_window() && platform::relative_mouse();
        if (s.restore_relative) platform::set_relative_mouse(false);
    } else {
        s.hold_on_hide.store(true);
        if (s.restore_relative && platform::has_window()) platform::set_relative_mouse(true);
        s.restore_relative = false;
        std::lock_guard<std::mutex> lk(s.mu);
        ImGuiContext* prev = ImGui::GetCurrentContext();
        ImGui::SetCurrentContext(s.imgui);
        ImGuiIO& io = ImGui::GetIO();   // drop pressed keys and buttons so nothing stays latched
        io.ClearInputKeys();
        io.ClearInputMouse();
        ImGui::SetCurrentContext(prev);
    }
    NXLOG_I(kTag, "overlay %s", v ? "shown" : "hidden");
}

bool Overlay::visible() const { return impl_->visible; }
void Overlay::toggle() { set_visible(!impl_->visible); }
int Overlay::toggle_scancode() const { return impl_->toggle_sc; }
void Overlay::set_toggle_scancode(int sc) { impl_->toggle_sc = sc; }
uint64_t Overlay::frames_drawn() const { return impl_->drawn; }

void Overlay::set_pointer(bool show, float x, float y, bool in_screenshots) {
    Impl& s = *impl_;
    s.ptr_x.store(x, std::memory_order_relaxed);
    s.ptr_y.store(y, std::memory_order_relaxed);
    s.ptr_in_shots.store(in_screenshots, std::memory_order_relaxed);
    s.ptr_show.store(show, std::memory_order_relaxed);
}

void Overlay::set_counter(const CounterConfig& c) {
    Impl& s = *impl_;
    s.counter_corner.store(std::clamp(c.corner, 0, 3), std::memory_order_relaxed);
    s.counter_detail.store(std::clamp(c.detail, 0, 3), std::memory_order_relaxed);
    s.counter_in_shots.store(c.in_screenshots, std::memory_order_relaxed);
    s.counter_lang_zh.store(c.lang_zh, std::memory_order_relaxed);
    s.counter_show.store(c.show, std::memory_order_relaxed);   // last: the other settings are in place when it turns on
}

namespace {
bool imgui_wants(std::mutex& mu, ImGuiContext* ctx, bool mouse, bool keyboard) {
    std::lock_guard<std::mutex> lk(mu);
    ImGuiContext* prev = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(ctx);
    const ImGuiIO& io = ImGui::GetIO();
    const bool w = (mouse && io.WantCaptureMouse) || (keyboard && io.WantCaptureKeyboard);
    ImGui::SetCurrentContext(prev);
    return w;
}
}  // namespace

bool Overlay::Impl::kb_poll() {
    namespace sk = services::swkbdui;
    const uint64_t id = sk::pending(nullptr);
    {
        std::lock_guard<std::mutex> lk(kb_mu);
        if (id == kb_id || (id && id <= kb_answered)) return kb_id != 0;
    }
    sk::Request r;
    uint64_t now_id = sk::pending(&r);
    std::lock_guard<std::mutex> lk(kb_mu);
    if (now_id && now_id <= kb_answered) now_id = 0;    // answered here meanwhile
    if (now_id == kb_id) return kb_id != 0;
    if (kb_id && !now_id) NXLOG_I(kTag, "on-screen keyboard: question %llu closed elsewhere", static_cast<unsigned long long>(kb_id));
    kb_id = now_id;
    if (now_id) {
        kb_req = std::move(r);
        for (std::string* t : {&kb_req.title, &kb_req.header, &kb_req.sub, &kb_req.guide, &kb_req.ok_text}) *t = fold_punct(*t);
        kb_text = utf8_to_u32(kb_req.initial);
        if (kb_req.max_len && kb_text.size() > kb_req.max_len) kb_text.resize(kb_req.max_len);
        kb_caret = kb_text.size();
        kb_shift = 0;
        kb_row = 1;
        kb_unit = 0;
        for (double& t : kb_rep) t = DBL_MAX;
        pad_new.store(0);
        NXLOG_I(kTag, "on-screen keyboard: question %llu (max %u characters%s)", static_cast<unsigned long long>(now_id), kb_req.max_len,
                kb_req.password ? ", password" : "");
    }
    kb_open = now_id != 0;
    return kb_open;
}

bool Overlay::Impl::kb_finish(bool ok) {
    if (!kb_id || (ok && kb_text.size() < kb_req.min_len)) return false;
    const uint64_t id = kb_id;
    kb_answered = id;
    kb_id = 0;
    kb_open = false;
    NXLOG_I(kTag, "on-screen keyboard: question %llu %s", static_cast<unsigned long long>(id), ok ? "answered" : "cancelled");
    services::swkbdui::answer(id, ok, ok ? u32_to_utf8(kb_text) : std::string());
    return true;
}

void Overlay::Impl::kb_insert(const std::u32string& t) {
    for (const char32_t c : t) {
        if (c < 0x20 || c == 0x7F) continue;
        if (kb_req.max_len && kb_text.size() >= kb_req.max_len) break;
        kb_text.insert(kb_caret++, 1, c);
    }
}

void Overlay::Impl::kb_backspace() {
    if (kb_caret == 0) return;
    kb_text.erase(--kb_caret, 1);
}

bool Overlay::Impl::kb_press(const KbKey& k) {
    switch (k.op) {
        case KbOp::Char:
            kb_insert(std::u32string(1, kb_shift ? k.alt : k.ch));
            if (kb_shift == 1) kb_shift = 0;
            return false;
        case KbOp::Shift: kb_shift = (kb_shift + 1) % 3; return false;
        case KbOp::Space: kb_insert(U" "); return false;
        case KbOp::Back: kb_backspace(); return false;
        case KbOp::Ok: return kb_finish(true);
        case KbOp::Cancel: return kb_finish(false);
    }
    return false;
}

// The physical keyboard while the on-screen keyboard is up: typing goes straight into its text (SDL text events), Enter answers, Escape
// cancels. Every key is consumed, the game sees none.
bool Overlay::Impl::kb_key_event(const input::Event& e) {
    bool closed = false;
    {
        std::lock_guard<std::mutex> lk(kb_mu);
        if (e.type == input::EventType::Text) {
            kb_insert(utf8_to_u32(e.u.text.utf8));
        } else if (e.type == input::EventType::KeyDown && kb_id) {
            const bool rep = (e.flags & input::kFlagRepeat) != 0;
            switch (e.u.key.scancode) {
                case SDL_SCANCODE_RETURN:
                case SDL_SCANCODE_KP_ENTER: if (!rep) closed = kb_finish(true); break;
                case SDL_SCANCODE_ESCAPE: if (!rep) closed = kb_finish(false); break;
                case SDL_SCANCODE_BACKSPACE: kb_backspace(); break;
                case SDL_SCANCODE_DELETE: if (kb_caret < kb_text.size()) kb_text.erase(kb_caret, 1); break;
                case SDL_SCANCODE_LEFT: if (kb_caret > 0) --kb_caret; break;
                case SDL_SCANCODE_RIGHT: if (kb_caret < kb_text.size()) ++kb_caret; break;
                case SDL_SCANCODE_HOME: kb_caret = 0; break;
                case SDL_SCANCODE_END: kb_caret = kb_text.size(); break;
                case SDL_SCANCODE_V:
                    if (e.mods & SDL_KMOD_CTRL) {
                        if (char* clip = SDL_GetClipboardText()) {
                            kb_insert(utf8_to_u32(clip));
                            SDL_free(clip);
                        }
                    }
                    break;
                default: break;
            }
        }
    }
    if (closed) {
        std::lock_guard<std::mutex> lk(mu);
        clear_imgui_input(imgui);
    }
    return true;
}

void Overlay::Impl::draw_keyboard(double now_s) {
    std::lock_guard<std::mutex> lk(kb_mu);
    if (!kb_id) return;
    const auto& rows = kb_rows();
    const int nrows = static_cast<int>(rows.size());
    bool closed = false;

    // The pad: presses since the last frame; a direction or B held past the delay repeats.
    const uint64_t held = pad_held.load(), fresh = pad_new.exchange(0);
    auto fired = [&](int slot, uint64_t bits) {
        bool f = false;
        if (fresh & bits) { f = true; kb_rep[slot] = now_s + kKbRepeatDelay; }
        else if ((held & bits) && now_s >= kb_rep[slot]) { f = true; kb_rep[slot] = now_s + kKbRepeatEvery; }
        if (!(held & bits)) kb_rep[slot] = DBL_MAX;
        return f;
    };
    auto move_h = [&](int d) {
        const auto& row = rows[kb_row];
        const int n = static_cast<int>(row.size());
        const int i = (kb_key_at(row, kb_unit, nullptr) + d + n) % n;
        int x = 0;
        for (int j = 0; j < i; ++j) x += row[j].w;
        kb_unit = x;
    };
    if (fired(0, kKbDirBits[0])) kb_row = (kb_row + nrows - 1) % nrows;
    if (fired(1, kKbDirBits[1])) kb_row = (kb_row + 1) % nrows;
    if (fired(2, kKbDirBits[2])) move_h(-1);
    if (fired(3, kKbDirBits[3])) move_h(1);
    const int sel = kb_key_at(rows[kb_row], kb_unit, nullptr);
    if (fresh & hid::btn(hid::kBtnA)) closed = kb_press(rows[kb_row][sel]);
    if (!closed && fired(4, hid::btn(hid::kBtnB))) kb_backspace();
    if (!closed && (fresh & hid::btn(hid::kBtnY))) kb_insert(U" ");
    if (!closed && (fresh & hid::btn(hid::kBtnX))) kb_shift = (kb_shift + 1) % 3;
    if (!closed && (fresh & hid::btn(hid::kBtnL)) && kb_caret > 0) --kb_caret;
    if (!closed && (fresh & hid::btn(hid::kBtnR)) && kb_caret < kb_text.size()) ++kb_caret;
    if (!closed && (fresh & hid::btn(hid::kBtnPlus))) closed = kb_finish(true);
    if (!closed && (fresh & hid::btn(hid::kBtnMinus))) closed = kb_finish(false);
    if (closed) {
        clear_imgui_input(imgui);
        return;
    }

    // A centred panel over the dimmed game, sized in units of the UI scale like the menu.
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const float k = ui_scale > 0.f ? ui_scale : 1.f;
    const float w = std::max(320.f, std::min(760.f * k, disp.x - 32.f * k));
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), disp, IM_COL32(0, 0, 0, 110));
    ImGui::SetNextWindowPos(ImVec2(disp.x * 0.5f, disp.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(w, 0), ImGuiCond_Always);
    const std::string title = (kb_req.title.empty() ? std::string("Keyboard") : kb_req.title) + "###nxrt_swkbd";
    const ImGuiWindowFlags wf = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollbar;
    if (ImGui::Begin(title.c_str(), nullptr, wf)) {
        ImGui::PushTextWrapPos(0.f);
        if (!kb_req.header.empty()) ImGui::TextUnformatted(kb_req.header.c_str());
        if (!kb_req.sub.empty()) ImGui::TextDisabled("%s", kb_req.sub.c_str());
        ImGui::PopTextWrapPos();

        // the text field: drawn here, not an ImGui InputText, so the keyboard, the pad and the mouse all edit the same text and caret
        const ImGuiStyle& style = ImGui::GetStyle();
        ImFont* font = ImGui::GetFont();
        const float big = ImGui::GetFontSize() * 1.25f;
        const float fh = std::round(big * 1.7f), pad = style.FramePadding.x * 1.5f;
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const float fw = ImGui::GetContentRegionAvail().x;
        ImGui::Dummy(ImVec2(fw, fh));
        const ImVec2 p1(p0.x + fw, p0.y + fh);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p0, p1, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameRounding);
        dl->AddRect(p0, p1, ImGui::GetColorU32(ImGuiCol_Border), style.FrameRounding);
        std::string shown, before;
        if (kb_req.password) {
            shown.assign(kb_text.size(), '*');
            before.assign(kb_caret, '*');
        } else {
            shown = u32_to_utf8(kb_text);
            before = u32_to_utf8(kb_text, 0, kb_caret);
        }
        const float cx = font->CalcTextSizeA(big, FLT_MAX, 0.0f, before.c_str()).x;
        const float scroll = std::max(0.f, cx - (fw - 2.f * pad) + 2.f);   // keep the caret in view
        const float ty = p0.y + (fh - big) * 0.5f;
        dl->PushClipRect(ImVec2(p0.x + pad * 0.5f, p0.y), ImVec2(p1.x - pad * 0.5f, p1.y), true);
        if (shown.empty() && !kb_req.guide.empty()) dl->AddText(font, big, ImVec2(p0.x + pad, ty), ImGui::GetColorU32(ImGuiCol_TextDisabled), kb_req.guide.c_str());
        else dl->AddText(font, big, ImVec2(p0.x + pad - scroll, ty), ImGui::GetColorU32(ImGuiCol_Text), shown.c_str());
        if (std::fmod(now_s, 1.0) < 0.6) {
            const float x = p0.x + pad - scroll + cx;
            dl->AddLine(ImVec2(x, ty), ImVec2(x, ty + big), ImGui::GetColorU32(ImGuiCol_Text), std::max(1.f, 1.5f * k));
        }
        dl->PopClipRect();
        if (kb_req.max_len) {
            char count[32];
            std::snprintf(count, sizeof count, "%zu / %u", kb_text.size(), kb_req.max_len);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + fw - ImGui::CalcTextSize(count).x);
            ImGui::TextDisabled("%s", count);
        }
        ImGui::Spacing();

        // the keys; the pad's key has a bright frame, the mouse clicks any of them
        const float gap = std::round(style.ItemSpacing.x * 0.75f);
        const float unit = (ImGui::GetContentRegionAvail().x - gap * (kKbUnits - 1)) / kKbUnits;
        const float kh = std::round(ImGui::GetFontSize() * 2.3f);
        const bool ok_allowed = kb_text.size() >= kb_req.min_len;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, gap));
        for (int r = 0; r < nrows && !closed; ++r) {
            int x = 0;
            for (int i = 0; i < static_cast<int>(rows[r].size()) && !closed; ++i) {
                const KbKey& key = rows[r][i];
                if (i) ImGui::SameLine();
                std::string label;
                switch (key.op) {
                    case KbOp::Char: label = u32_to_utf8(std::u32string(1, kb_shift ? key.alt : key.ch)); break;
                    case KbOp::Shift: label = kb_shift == 2 ? "CAPS" : kb_shift ? "SHIFT" : "Shift"; break;
                    case KbOp::Space: label = "Space"; break;
                    case KbOp::Back: label = "Delete"; break;
                    case KbOp::Ok: label = kb_req.ok_text.empty() ? "OK" : kb_req.ok_text; break;
                    case KbOp::Cancel: label = "Cancel"; break;
                }
                ImGui::PushID(r * 32 + i);
                const bool is_sel = r == kb_row && i == sel;
                const bool lit = key.op == KbOp::Shift && kb_shift;
                if (lit) ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[ImGuiCol_ButtonActive]);
                ImGui::BeginDisabled(key.op == KbOp::Ok && !ok_allowed);
                if (ImGui::Button(label.c_str(), ImVec2(unit * key.w + gap * (key.w - 1), kh))) {
                    kb_row = r;
                    kb_unit = x;
                    closed = kb_press(key);
                }
                ImGui::EndDisabled();
                if (lit) ImGui::PopStyleColor();
                if (is_sel)
                    ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 200, 40, 255), style.FrameRounding,
                                                        0, std::max(2.f, 2.5f * k));
                ImGui::PopID();
                x += key.w;
            }
        }
        ImGui::PopStyleVar();
        if (!closed) {
            ImGui::Spacing();
            // the face buttons are Switch buttons after the player's mapping: by printed label they are named so, by position they sit
            // where a Switch pad has them
            const bool lab = kb_by_label.load(std::memory_order_relaxed);
            ImGui::PushTextWrapPos(0.f);
            ImGui::TextDisabled("Pad: %s type, %s delete, %s space, %s shift, L/R move, + OK, - cancel", lab ? "A" : "A (right)", lab ? "B" : "B (bottom)",
                                lab ? "Y" : "Y (left)", lab ? "X" : "X (top)");
            ImGui::TextDisabled("Keyboard: type, Enter OK, Esc cancel, Ctrl+V paste");
            ImGui::PopTextWrapPos();
        }
    }
    ImGui::End();
    if (closed) clear_imgui_input(imgui);
}

bool Overlay::keyboard_open() const { return impl_->kb_poll(); }
bool Overlay::blocks_game() const { return impl_->visible || impl_->picker_open.load(std::memory_order_relaxed) || impl_->kb_poll(); }
void Overlay::set_keyboard_abxy(bool by_label) { impl_->kb_by_label.store(by_label, std::memory_order_relaxed); }

uint64_t Overlay::set_keyboard_pad(uint64_t buttons, int lx, int ly) {
    Impl& s = *impl_;
    constexpr int kHalf = 16384;     // half of the hid stick range
    uint64_t b = buttons;
    if (lx <= -kHalf) b |= hid::btn(hid::kBtnStickLLeft);
    if (lx >= kHalf) b |= hid::btn(hid::kBtnStickLRight);
    if (ly >= kHalf) b |= hid::btn(hid::kBtnStickLUp);       // hid y: up = +
    if (ly <= -kHalf) b |= hid::btn(hid::kBtnStickLDown);
    const bool open = s.kb_poll();
    if (open) {
        s.pad_new.fetch_or(b & ~s.pad_prev);
        s.pad_mask = b;
    } else {
        s.pad_mask &= b;             // a button leaves the mask once released
    }
    s.pad_held.store(b);
    s.pad_prev = b;
    return open ? ~0ull : s.pad_mask;
}

bool Overlay::wants_input() const { return impl_->visible && imgui_wants(impl_->mu, impl_->imgui, true, true); }
bool Overlay::wants_mouse() const { return impl_->visible && imgui_wants(impl_->mu, impl_->imgui, true, false); }
bool Overlay::wants_keyboard() const { return impl_->visible && imgui_wants(impl_->mu, impl_->imgui, false, true); }

void Overlay::add_panel(const std::string& title, std::function<void()> fn) { impl_->panels.push_back({title, std::move(fn)}); }
void Overlay::set_menu(std::function<void()> fn) { impl_->menu = std::move(fn); }
void Overlay::request_hide() { impl_->want_hide.store(true); }
const Overlay::Context& Overlay::context() const { return impl_->ctx; }

uint64_t Overlay::set_menu_pad(uint64_t b, int lx, int ly) {
    Impl& s = *impl_;
    if (s.kb_poll()) {               // the on-screen keyboard has the pad (set_keyboard_pad)
        s.menu_prev = b;
        return 0;
    }
    if (s.hold_on_hide.exchange(false)) s.menu_mask |= b;   // closed (B, F1, the chord): what is still held stays away from the game
    const uint64_t fresh = b & ~s.menu_prev;
    if (s.menu_chord && (b & s.menu_chord) == s.menu_chord && (fresh & s.menu_chord)) {
        toggle();
        s.menu_mask |= b;
        NXLOG_I(kTag, "overlay %s by the pad chord", s.visible ? "opened" : "closed");
    }
    // ImGui moves with the d-pad only: the left stick past half way (with some hysteresis) presses the d-pad direction too
    auto dir = [](bool was, int v, bool positive) {
        const int t = was ? 12000 : 18000;
        return positive ? v >= t : v <= -t;
    };
    const uint64_t up = hid::btn(hid::kBtnUp), down = hid::btn(hid::kBtnDown), left = hid::btn(hid::kBtnLeft), right = hid::btn(hid::kBtnRight);
    uint64_t stick = 0;
    if (dir((s.menu_stick & up) != 0, ly, true)) stick |= up;      // hid y: up = +
    if (dir((s.menu_stick & down) != 0, ly, false)) stick |= down;
    if (dir((s.menu_stick & left) != 0, lx, false)) stick |= left;
    if (dir((s.menu_stick & right) != 0, lx, true)) stick |= right;
    const uint64_t eff = b | stick, eff_prev = s.menu_prev | s.menu_stick;
    s.menu_stick = stick;
    const bool p_open = s.picker_open.load(std::memory_order_relaxed);
    if (p_open && !s.visible) {
        // Picker gamepad controls: B closes, L/R cycles tabs, X clears custom pin, D-pad/L-stick/A drive ImGui list.
        const uint64_t fresh_unmasked = fresh & ~s.menu_mask;
        if (fresh_unmasked & hid::btn(hid::kBtnB)) {
            s.want_picker_close.store(true, std::memory_order_relaxed);
            s.menu_mask |= b;
        } else if (fresh_unmasked & hid::btn(hid::kBtnL)) {
            s.picker_tab_delta.fetch_sub(1, std::memory_order_relaxed);
        } else if (fresh_unmasked & hid::btn(hid::kBtnR)) {
            s.picker_tab_delta.fetch_add(1, std::memory_order_relaxed);
        } else if (fresh_unmasked & hid::btn(hid::kBtnX)) {
            s.want_picker_clear.store(true, std::memory_order_relaxed);
        }
    }
    if (s.visible || p_open) {
        // ImGui's gamepad keys, by the Switch button's meaning: A activates, B goes back (the game's own convention, after the player's
        // mapping); L / R and ZL / ZR are the shoulders (the menu switches its tabs with them). X, Y, + and - are not given to ImGui (Y would
        // be its windowing mode).
        static const std::pair<hid::NpadButtonBit, ImGuiKey> kMap[] = {
            {hid::kBtnA, ImGuiKey_GamepadFaceDown},     {hid::kBtnB, ImGuiKey_GamepadFaceRight},  {hid::kBtnL, ImGuiKey_GamepadL1},
            {hid::kBtnR, ImGuiKey_GamepadR1},           {hid::kBtnZL, ImGuiKey_GamepadL2},        {hid::kBtnZR, ImGuiKey_GamepadR2},
            {hid::kBtnUp, ImGuiKey_GamepadDpadUp},      {hid::kBtnDown, ImGuiKey_GamepadDpadDown}, {hid::kBtnLeft, ImGuiKey_GamepadDpadLeft},
            {hid::kBtnRight, ImGuiKey_GamepadDpadRight}};
        // a button held from before the menu opened (the chord) counts only once it is released and pressed again
        const uint64_t changed = (eff ^ eff_prev) & ~(s.menu_mask & b);
        if (changed) {
            std::lock_guard<std::mutex> lk(s.mu);
            ImGuiContext* prev = ImGui::GetCurrentContext();
            ImGui::SetCurrentContext(s.imgui);
            ImGuiIO& io = ImGui::GetIO();
            for (const auto& kv : kMap) {
                const uint64_t bit = hid::btn(kv.first);
                if (changed & bit) io.AddKeyEvent(kv.second, (eff & bit) != 0);
            }
            ImGui::SetCurrentContext(prev);
        }
    }
    s.menu_mask &= b;                // a button leaves the mask once released
    s.menu_prev = b;
    return (s.visible || p_open) ? ~0ull : s.menu_mask;
}

bool Overlay::handle_event(const input::Event& e) {
    Impl& s = *impl_;
    using input::EventType;
    if ((e.type == EventType::KeyDown || e.type == EventType::KeyUp) && e.u.key.scancode == s.toggle_sc) {
        if (e.type == EventType::KeyDown && !(e.flags & input::kFlagRepeat)) toggle();
        return true;   // the toggle key never reaches the game (press, repeat and release)
    }
    const bool kb = s.kb_poll();
    if (kb && (e.type == EventType::KeyDown || e.type == EventType::KeyUp || e.type == EventType::Text)) return s.kb_key_event(e);
    const bool p_open = s.picker_open.load(std::memory_order_relaxed);
    if (p_open && e.type == EventType::KeyDown && e.u.key.scancode == SDL_SCANCODE_ESCAPE) {
        toggle_picker();
        return true;
    }
    if (!s.visible && !p_open && !kb) return false;
    std::lock_guard<std::mutex> lk(s.mu);
    ImGuiContext* prev = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(s.imgui);
    ImGuiIO& io = ImGui::GetIO();
    bool consumed = false;
    switch (e.type) {
        case EventType::KeyDown:
        case EventType::KeyUp: {
            const ImGuiKey k = map_key(e.u.key.scancode);
            if (k != ImGuiKey_None) io.AddKeyEvent(k, e.type == EventType::KeyDown);
            consumed = io.WantCaptureKeyboard;
            break;
        }
        case EventType::Text:
            io.AddInputCharactersUTF8(e.u.text.utf8);
            consumed = io.WantTextInput;
            break;
        case EventType::MouseMotion:
            io.AddMousePosEvent(e.u.motion.x, e.u.motion.y);
            consumed = io.WantCaptureMouse;
            break;
        case EventType::MouseButtonDown:
        case EventType::MouseButtonUp: {
            const int b = e.u.button.button;   // SDL: 1 left, 2 middle, 3 right
            const int ib = b == 1 ? 0 : b == 3 ? 1 : b == 2 ? 2 : b - 1;
            io.AddMousePosEvent(e.u.button.x, e.u.button.y);
            io.AddMouseButtonEvent(ib, e.type == EventType::MouseButtonDown);
            consumed = io.WantCaptureMouse;
            break;
        }
        case EventType::MouseWheel:
            io.AddMouseWheelEvent(e.u.wheel.dx, e.u.wheel.dy);
            consumed = io.WantCaptureMouse;
            break;
        case EventType::WindowFocusLost: io.AddFocusEvent(false); break;
        case EventType::WindowFocusGained: io.AddFocusEvent(true); break;
        default: break;
    }
    ImGui::SetCurrentContext(prev);
    // the on-screen keyboard takes the whole mouse while it is up
    if (kb) consumed = e.type == EventType::MouseMotion || e.type == EventType::MouseButtonDown || e.type == EventType::MouseButtonUp || e.type == EventType::MouseWheel;
    if (p_open) consumed = true;
    return consumed;
}

void Overlay::apply_pending() {
    apply_present_pending();
    apply_window_pending();
}

void Overlay::apply_present_pending() {
    Impl& s = *impl_;
    if (s.want_mode) {
        s.want_mode = false;
        s.ctx.renderer->set_present_mode(s.new_mode);
    }
}

void Overlay::apply_window_pending() {
    Impl& s = *impl_;
    if (s.want_fullscreen) {
        s.want_fullscreen = false;
        platform::set_fullscreen(s.new_fullscreen);
    }
}

void Overlay::Impl::refresh_counter_text(double now_s) {
    if (now_s - text_s < 0.25) return;    // the text is rebuilt at most every 250 ms
    text_s = now_s;
    snap = stats.snapshot(now_s);
    const int detail = counter_detail.load(std::memory_order_relaxed);
    const extras::FramegenTelemetry fg = extras::framegen_telemetry();
    const bool fg_live = fg.active && fg.shown_fps > 0.5;
    char buf[64];
    counter_n = 0;
    if (snap.valid) {
        if (fg_live) std::snprintf(buf, sizeof buf, "%.0f -> %.0f FPS (FG)", snap.fps, fg.shown_fps);
        else std::snprintf(buf, sizeof buf, "%.0f FPS", snap.fps);
    } else {
        std::snprintf(buf, sizeof buf, "-- FPS");
    }
    counter_lines[counter_n++] = buf;
    if (detail >= 1) {
        if (snap.valid) std::snprintf(buf, sizeof buf, "%.1f ms", snap.avg_ms);
        else std::snprintf(buf, sizeof buf, "-- ms");
        counter_lines[counter_n++] = buf;
    }
    if (detail >= 2) {
        if (snap.valid) std::snprintf(buf, sizeof buf, "1%% low %.0f", snap.low1_fps);
        else std::snprintf(buf, sizeof buf, "1%% low --");
        counter_lines[counter_n++] = buf;
        if (snap.valid) std::snprintf(buf, sizeof buf, "max %.1f ms", snap.max_ms);
        else std::snprintf(buf, sizeof buf, "max -- ms");
        counter_lines[counter_n++] = buf;
    }
}

namespace {

uint64_t ft_to_u64(const FILETIME& ft) {
    return (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | static_cast<uint64_t>(ft.dwLowDateTime);
}

std::string query_cpu_brand_string() {
    int regs[4] = {};
    __cpuid(regs, static_cast<int>(0x80000000u));
    if (static_cast<uint32_t>(regs[0]) < 0x80000004u) return "CPU";
    char brand[49] = {};
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, static_cast<int>(0x80000002u + i));
        std::memcpy(brand + i * 16, regs, 16);
    }
    brand[48] = '\0';
    std::string s(brand);
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s.empty() ? "CPU" : s;
}

void* __cdecl adl_alloc_cb(int sz) {
    return std::malloc(static_cast<size_t>(sz > 0 ? sz : 1));
}

struct AdlSingleSensor {
    int supported;
    int value;
};
struct AdlPmLogOutput {
    int size;
    AdlSingleSensor sensors[256];
};
struct NvmlUtil {
    unsigned int gpu;
    unsigned int memory;
};

}  // namespace

void Overlay::Impl::shutdown_perf_hud() {
    if (perf.adl_ctx && perf.adl_destroy) {
        perf.adl_destroy(perf.adl_ctx);
        perf.adl_ctx = nullptr;
    }
    if (perf.adl_dll) {
        FreeLibrary(perf.adl_dll);
        perf.adl_dll = nullptr;
    }
    if (perf.nvml_dev && perf.nvml_shutdown) {
        perf.nvml_shutdown();
        perf.nvml_dev = nullptr;
    }
    if (perf.nvml_dll) {
        FreeLibrary(perf.nvml_dll);
        perf.nvml_dll = nullptr;
    }
    perf.vendor_init = false;
}

void Overlay::Impl::refresh_perf_hud(double now_s) {
    if (!perf.cpu_name_init) {
        perf.cpu_name = query_cpu_brand_string();
        perf.cpu_name_init = true;
    }
    if (now_s - text_s >= 0.25) {
        text_s = now_s;
        snap = stats.snapshot(now_s);

        // Backend per-frame pipeline/upload/GPU busy metrics
        const nvn::BackendStats bs = nvn::backend_stats();
        const double prep_ms = nxrt::hitch_phase_total_ms(nxrt::kHpRenderReplay);
        const double pipe_ms = nxrt::hitch_phase_total_ms(nxrt::kHpRenderPipeline);
        const double shdr_ms = nxrt::hitch_phase_total_ms(nxrt::kHpRenderLockWait);
        const double upl_ms = nxrt::hitch_phase_total_ms(nxrt::kHpRenderUpload);
        if (perf.have_be_prev && bs.presents > perf.prev_presents) {
            const double df = static_cast<double>(bs.presents - perf.prev_presents);
            perf.gpu_prep_ms = std::max(0.0, (prep_ms - perf.prev_prep_ms) / df);
            perf.gpu_busy_ms = std::max(0.0, static_cast<double>(bs.gpu_busy_ns - perf.prev_gpu_busy_ns) / (1e6 * df));
            perf.pipe_wait_ms = std::max(0.0, (pipe_ms - perf.prev_pipe_ms) / df);
            perf.shader_wait_ms = std::max(0.0, (shdr_ms - perf.prev_shader_ms) / df);
            perf.tex_uploads_pf = std::max(0.0, static_cast<double>(bs.texture_uploads - perf.prev_tex_uploads) / df);
            perf.tex_upload_ms = std::max(0.0, (upl_ms - perf.prev_upload_ms) / df);
        }
        perf.prev_presents = bs.presents;
        perf.prev_tex_uploads = bs.texture_uploads;
        perf.prev_gpu_busy_ns = bs.gpu_busy_ns;
        perf.prev_prep_ms = prep_ms;
        perf.prev_pipe_ms = pipe_ms;
        perf.prev_shader_ms = shdr_ms;
        perf.prev_upload_ms = upl_ms;
        perf.have_be_prev = true;
    }

    // Hardware telemetry sampled at 2 Hz (every 500 ms) so OS/driver queries stay negligible
    if (now_s - perf.hw_last_s < 0.50) return;
    perf.hw_last_s = now_s;

    // 1. CPU utilization (total & this game process)
    FILETIME s_idle{}, s_kern{}, s_user{}, p_create{}, p_exit{}, p_kern{}, p_user{};
    if (GetSystemTimes(&s_idle, &s_kern, &s_user) &&
        GetProcessTimes(GetCurrentProcess(), &p_create, &p_exit, &p_kern, &p_user)) {
        const uint64_t si = ft_to_u64(s_idle), sk = ft_to_u64(s_kern), su = ft_to_u64(s_user);
        const uint64_t pk = ft_to_u64(p_kern), pu = ft_to_u64(p_user);
        if (perf.have_cpu_prev) {
            const uint64_t di = si - perf.prev_sys_idle;
            const uint64_t dsys = (sk - perf.prev_sys_kernel) + (su - perf.prev_sys_user);
            const uint64_t dproc = (pk - perf.prev_proc_kernel) + (pu - perf.prev_proc_user);
            if (dsys > 0) {
                const double busy = (dsys > di) ? static_cast<double>(dsys - di) : 0.0;
                perf.cpu_total_pct = std::clamp(static_cast<int>(std::lround(100.0 * busy / static_cast<double>(dsys))), 0, 100);
                perf.cpu_game_pct = std::clamp(static_cast<int>(std::lround(100.0 * static_cast<double>(dproc) / static_cast<double>(dsys))), 0, 100);
            }
        }
        perf.prev_sys_idle = si;
        perf.prev_sys_kernel = sk;
        perf.prev_sys_user = su;
        perf.prev_proc_kernel = pk;
        perf.prev_proc_user = pu;
        perf.have_cpu_prev = true;
    }

    // 2. RAM (system used/total + game process working set)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        perf.ram_total_gb = static_cast<double>(ms.ullTotalPhys) / kGiB;
        perf.ram_used_gb = static_cast<double>(ms.ullTotalPhys - ms.ullAvailPhys) / kGiB;
    }
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        perf.ram_game_gb = static_cast<double>(pmc.WorkingSetSize) / kGiB;
    }

    // 3. VRAM via Vulkan VK_EXT_memory_budget (fallback: device vram_bytes)
    if (ctx.device && ctx.device->physical() != VK_NULL_HANDLE) {
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        if (ctx.device->features().memory_budget) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            props.pNext = &budget;
            vkGetPhysicalDeviceMemoryProperties2(ctx.device->physical(), &props);
            uint64_t total_vram = 0, used_vram = 0;
            for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
                if (props.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    total_vram += props.memoryProperties.memoryHeaps[i].size;
                    used_vram += budget.heapUsage[i];
                }
            }
            if (total_vram > 0) {
                perf.vram_total_gb = static_cast<double>(total_vram) / kGiB;
                perf.vram_used_gb = static_cast<double>(used_vram) / kGiB;
            }
        }
        if (perf.vram_total_gb <= 0.0) {
            perf.vram_total_gb = static_cast<double>(ctx.device->gpu().vram_bytes) / kGiB;
        }
    }

    // 4. GPU utilization / temp / power / clock via dynamic AMD ADL2 or NVIDIA NVML
    const uint32_t vid = ctx.device ? ctx.device->gpu().vendor_id : 0;
    if (!perf.vendor_init) {
        perf.vendor_init = true;
        if (vid == 0x1002) {
            perf.adl_dll = LoadLibraryA("atiadlxx.dll");
            if (perf.adl_dll) {
                using CreateFn = int(__cdecl*)(void*(__cdecl*)(int), int, void**);
                using NumFn = int(__cdecl*)(void*, int*);
                auto fn_create = reinterpret_cast<CreateFn>(GetProcAddress(perf.adl_dll, "ADL2_Main_Control_Create"));
                auto fn_num = reinterpret_cast<NumFn>(GetProcAddress(perf.adl_dll, "ADL2_Adapter_NumberOfAdapters_Get"));
                perf.adl_destroy = reinterpret_cast<PerfHudState::AdlDestroyFn>(GetProcAddress(perf.adl_dll, "ADL2_Main_Control_Destroy"));
                perf.adl_pmlog = reinterpret_cast<PerfHudState::AdlPmLogFn>(GetProcAddress(perf.adl_dll, "ADL2_New_QueryPMLogData_Get"));
                if (fn_create && fn_num && perf.adl_destroy && perf.adl_pmlog && fn_create(&adl_alloc_cb, 1, &perf.adl_ctx) == 0 && perf.adl_ctx) {
                    int n = 0;
                    if (fn_num(perf.adl_ctx, &n) == 0 && n > 0) {
                        for (int idx = 0; idx < std::min(n, 16); ++idx) {
                            AdlPmLogOutput out{};
                            out.size = sizeof(out);
                            if (perf.adl_pmlog(perf.adl_ctx, idx, &out) == 0 &&
                                (out.sensors[1].supported || out.sensors[8].supported || out.sensors[19].supported)) {
                                perf.adl_adapter = idx;
                                break;
                            }
                        }
                    }
                }
            }
        } else if (vid == 0x10DE) {
            perf.nvml_dll = LoadLibraryA("nvml.dll");
            if (perf.nvml_dll) {
                using InitFn = int(__cdecl*)();
                using GetHandleFn = int(__cdecl*)(unsigned int, void**);
                auto fn_init = reinterpret_cast<InitFn>(GetProcAddress(perf.nvml_dll, "nvmlInit_v2"));
                auto fn_handle = reinterpret_cast<GetHandleFn>(GetProcAddress(perf.nvml_dll, "nvmlDeviceGetHandleByIndex_v2"));
                perf.nvml_shutdown = reinterpret_cast<PerfHudState::NvmlShutdownFn>(GetProcAddress(perf.nvml_dll, "nvmlShutdown"));
                perf.nvml_util = reinterpret_cast<PerfHudState::NvmlUtilFn>(GetProcAddress(perf.nvml_dll, "nvmlDeviceGetUtilizationRates"));
                perf.nvml_temp = reinterpret_cast<PerfHudState::NvmlTempFn>(GetProcAddress(perf.nvml_dll, "nvmlDeviceGetTemperature"));
                perf.nvml_power = reinterpret_cast<PerfHudState::NvmlPowerFn>(GetProcAddress(perf.nvml_dll, "nvmlDeviceGetPowerUsage"));
                perf.nvml_clock = reinterpret_cast<PerfHudState::NvmlClockFn>(GetProcAddress(perf.nvml_dll, "nvmlDeviceGetClockInfo"));
                if (fn_init && fn_handle && fn_init() == 0) {
                    if (fn_handle(0, &perf.nvml_dev) != 0) perf.nvml_dev = nullptr;
                }
            }
        }
    }

    perf.gpu_util_pct = -1;
    perf.gpu_temp_c = -1;
    perf.gpu_power_w = -1;
    perf.gpu_clock_mhz = -1;
    if (perf.adl_ctx && perf.adl_pmlog && perf.adl_adapter >= 0) {
        AdlPmLogOutput out{};
        out.size = sizeof(out);
        if (perf.adl_pmlog(perf.adl_ctx, perf.adl_adapter, &out) == 0) {
            if (out.sensors[19].supported) perf.gpu_util_pct = std::clamp(out.sensors[19].value, 0, 100);
            if (out.sensors[8].supported) perf.gpu_temp_c = out.sensors[8].value;
            else if (out.sensors[27].supported) perf.gpu_temp_c = out.sensors[27].value;
            if (out.sensors[42].supported && out.sensors[42].value > 0) perf.gpu_power_w = out.sensors[42].value;
            else if (out.sensors[23].supported && out.sensors[23].value > 0) perf.gpu_power_w = out.sensors[23].value;
            if (out.sensors[1].supported && out.sensors[1].value > 0) perf.gpu_clock_mhz = out.sensors[1].value;
        }
    } else if (perf.nvml_dev) {
        NvmlUtil u{};
        if (perf.nvml_util && perf.nvml_util(perf.nvml_dev, &u) == 0) perf.gpu_util_pct = std::clamp(static_cast<int>(u.gpu), 0, 100);
        unsigned int t = 0;
        if (perf.nvml_temp && perf.nvml_temp(perf.nvml_dev, 0, &t) == 0) perf.gpu_temp_c = static_cast<int>(t);
        unsigned int mw = 0;
        if (perf.nvml_power && perf.nvml_power(perf.nvml_dev, &mw) == 0) perf.gpu_power_w = static_cast<int>((mw + 500) / 1000);
        unsigned int clk = 0;
        if (perf.nvml_clock && perf.nvml_clock(perf.nvml_dev, 0, &clk) == 0) perf.gpu_clock_mhz = static_cast<int>(clk);
    }
    if (perf.gpu_util_pct < 0 && snap.valid && snap.avg_ms > 0.1 && perf.gpu_busy_ms > 0.0) {
        perf.gpu_util_pct = std::clamp(static_cast<int>(std::lround(100.0 * perf.gpu_busy_ms / snap.avg_ms)), 0, 100);
    }
}

void Overlay::Impl::draw_perf_hud(const FrameInfo& f, double now_s) {
    refresh_perf_hud(now_s);
    ImFont* font = ImGui::GetFont();
    const bool zh = counter_lang_zh.load(std::memory_order_relaxed);
    const float sz = std::clamp(std::round(static_cast<float>(f.extent.height) / 68.f), 12.f, 26.f);
    const float line_h = sz * 1.24f;
    const float pad_x = sz * 0.75f;
    const float pad_y = sz * 0.60f;
    const float sec_gap = sz * 0.45f;
    const float graph_h = sz * 3.2f;
    const float margin = sz * 0.9f;
    const float box_w = std::round(sz * 21.5f);
    const extras::FramegenTelemetry fg = extras::framegen_telemetry();
    const bool fg_live = fg.active && fg.shown_fps > 0.5;
    const int disp_rows = fg.requested ? 7 : 6;

    // Count sections:
    // 4 hw rows + 1 fps row + graph + 6 timing rows + 6..7 display rows + 2 hw name rows
    const float box_h = pad_y * 2.f +
                        4.f * line_h +          // GPU, VRAM, CPU, RAM
                        1.f * line_h +          // FPS / avg ms / max ms
                        sec_gap * 0.5f + graph_h + sec_gap +
                        6.f * line_h + sec_gap + // GPU prep/busy, pipeline/shader waits, tex uploads/time
                        static_cast<float>(disp_rows) * line_h + sec_gap + // Window, Game image, 3D scene, Scaling, [Frame gen], Frame limit, Presentation
                        2.f * line_h;            // GPU name, CPU name

    const int corner = counter_corner.load(std::memory_order_relaxed);
    const bool right = (corner == 1 || corner == 3);
    const bool bottom = (corner >= 2);
    const float W = static_cast<float>(f.extent.width), H = static_cast<float>(f.extent.height);
    const float x0 = right ? std::max(margin, W - margin - box_w) : margin;
    const float y0 = bottom ? std::max(margin, H - margin - box_h) : margin;
    const float xL = x0 + pad_x;
    const float xR = x0 + box_w - pad_x;

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + box_w, y0 + box_h), IM_COL32(10, 14, 20, 188), sz * 0.45f);
    dl->AddRect(ImVec2(x0, y0), ImVec2(x0 + box_w, y0 + box_h), IM_COL32(180, 210, 235, 36), sz * 0.45f, 0, 1.0f);

    const ImU32 cAmber  = IM_COL32(238, 186, 76, 255);
    const ImU32 cGreen  = IM_COL32(118, 214, 114, 255);
    const ImU32 cBlue   = IM_COL32(130, 168, 255, 255);
    const ImU32 cPink   = IM_COL32(236, 126, 184, 255);
    const ImU32 cCyan   = IM_COL32(136, 206, 238, 255);
    const ImU32 cWhite  = IM_COL32(242, 246, 252, 255);
    const ImU32 cMuted  = IM_COL32(175, 188, 204, 220);
    const ImU32 cSep    = IM_COL32(255, 255, 255, 24);

    auto draw_r = [&](float y, ImU32 col, const char* txt) {
        const float tw = font->CalcTextSizeA(sz, FLT_MAX, 0.0f, txt).x;
        dl->AddText(font, sz, ImVec2(xR - tw, y), col, txt);
    };
    auto draw_c = [&](float x, float y, ImU32 col, const char* txt) {
        dl->AddText(font, sz, ImVec2(x, y), col, txt);
    };
    auto draw_kv = [&](float y, ImU32 lcol, const char* label, const char* val) {
        draw_c(xL, y, lcol, label);
        draw_r(y, cWhite, val);
    };

    float y = y0 + pad_y;
    const float col1 = xL + sz * 4.2f;
    const float col2 = xL + sz * 9.8f;
    char buf1[64], buf2[64], buf3[64];

    // Row 1: GPU | util% | temp C | power W
    draw_c(xL, y, cAmber, "GPU");
    if (perf.gpu_util_pct >= 0) std::snprintf(buf1, sizeof buf1, "%d%%", perf.gpu_util_pct);
    else std::snprintf(buf1, sizeof buf1, "--%%");
    draw_c(col1, y, cWhite, buf1);
    if (perf.gpu_temp_c >= 0) {
        std::snprintf(buf2, sizeof buf2, "%d °C", perf.gpu_temp_c);
        draw_c(col2, y, cWhite, buf2);
    }
    if (perf.gpu_power_w >= 0) {
        std::snprintf(buf3, sizeof buf3, "%d W", perf.gpu_power_w);
        draw_r(y, cWhite, buf3);
    }
    y += line_h;

    // Row 2: VRAM | used / total GB | clock MHz
    draw_c(xL, y, cGreen, "VRAM");
    if (perf.vram_total_gb > 0.0) std::snprintf(buf1, sizeof buf1, "%.1f / %.1f GB", perf.vram_used_gb, perf.vram_total_gb);
    else std::snprintf(buf1, sizeof buf1, "-- GB");
    draw_c(col1, y, cWhite, buf1);
    if (perf.gpu_clock_mhz > 0) {
        std::snprintf(buf3, sizeof buf3, "%d MHz", perf.gpu_clock_mhz);
        draw_r(y, cWhite, buf3);
    }
    y += line_h;

    // Row 3: CPU | total% | game XX%
    draw_c(xL, y, cBlue, "CPU");
    if (perf.cpu_total_pct >= 0) std::snprintf(buf1, sizeof buf1, "%d%%", perf.cpu_total_pct);
    else std::snprintf(buf1, sizeof buf1, "--%%");
    draw_c(col1, y, cWhite, buf1);
    if (perf.cpu_game_pct >= 0) {
        std::snprintf(buf3, sizeof buf3, zh ? "游戏 %d%%" : "game %d%%", perf.cpu_game_pct);
        draw_r(y, cWhite, buf3);
    }
    y += line_h;

    // Row 4: RAM | used / total GB | game X.X GB
    draw_c(xL, y, cPink, "RAM");
    if (perf.ram_total_gb > 0.0) std::snprintf(buf1, sizeof buf1, "%.1f / %.1f GB", perf.ram_used_gb, perf.ram_total_gb);
    else std::snprintf(buf1, sizeof buf1, "-- GB");
    draw_c(col1, y, cWhite, buf1);
    if (perf.ram_game_gb > 0.0) {
        std::snprintf(buf3, sizeof buf3, zh ? "游戏 %.1f GB" : "game %.1f GB", perf.ram_game_gb);
        draw_r(y, cWhite, buf3);
    }
    y += line_h;

    // Row 5: FPS | avg ms | max ms
    if (snap.valid) {
        if (fg_live) std::snprintf(buf1, sizeof buf1, "%.0f -> %.0f FPS", snap.fps, fg.shown_fps);
        else std::snprintf(buf1, sizeof buf1, "%.0f FPS", snap.fps);
        std::snprintf(buf2, sizeof buf2, "%.1f ms", snap.avg_ms);
        std::snprintf(buf3, sizeof buf3, "max %.1f ms", snap.max_ms);
    } else {
        std::snprintf(buf1, sizeof buf1, "-- FPS");
        std::snprintf(buf2, sizeof buf2, "-- ms");
        std::snprintf(buf3, sizeof buf3, "max -- ms");
    }
    const float w1 = font->CalcTextSizeA(sz, FLT_MAX, 0.0f, buf1).x;
    draw_c(xL, y, cWhite, buf1);
    draw_c(std::max(col1 + sz * 1.1f, xL + w1 + sz * 0.8f), y, cWhite, buf2);
    draw_r(y, cMuted, buf3);
    y += line_h + sec_gap * 0.5f;

    // Frame-time sparkline graph
    const ImVec2 g0(xL, y), g1(xR, y + graph_h);
    dl->AddRectFilled(g0, g1, IM_COL32(255, 255, 255, 12), sz * 0.2f);
    dl->AddRect(g0, g1, cSep, sz * 0.2f, 0, 1.0f);
    // 16.67 ms (60fps) and 33.33 ms (30fps) subtle guide lines on a 0..40 ms scale
    const float max_scale = 40.0f;
    const float y60 = g1.y - (16.67f / max_scale) * graph_h;
    const float y30 = g1.y - (33.33f / max_scale) * graph_h;
    dl->AddLine(ImVec2(g0.x, y60), ImVec2(g1.x, y60), IM_COL32(118, 214, 114, 48), 1.0f);
    dl->AddLine(ImVec2(g0.x, y30), ImVec2(g1.x, y30), IM_COL32(238, 186, 76, 42), 1.0f);
    if (perf.graph_count >= 2) {
        ImVec2 pts[PerfHudState::kGraphN];
        const int cnt = std::min(perf.graph_count, PerfHudState::kGraphN);
        const int start = (perf.graph_pos - cnt + PerfHudState::kGraphN) % PerfHudState::kGraphN;
        const float gw = g1.x - g0.x - 2.f;
        for (int i = 0; i < cnt; ++i) {
            const float ms_val = std::clamp(perf.graph_ms[(start + i) % PerfHudState::kGraphN], 0.f, max_scale);
            const float px = g0.x + 1.f + gw * static_cast<float>(i) / static_cast<float>(PerfHudState::kGraphN - 1);
            const float py = g1.y - 1.f - (ms_val / max_scale) * (graph_h - 2.f);
            pts[i] = ImVec2(px, py);
        }
        dl->AddPolyline(pts, cnt, IM_COL32(96, 178, 255, 235), 0, 1.5f);
    }
    y += graph_h + sec_gap;

    // Renderer / Pipeline timings section
    std::snprintf(buf1, sizeof buf1, "%.2f ms", perf.gpu_prep_ms);
    draw_kv(y, cAmber, zh ? "GPU 指令准备" : "GPU frame prep", buf1); y += line_h;
    std::snprintf(buf1, sizeof buf1, "%.2f ms", perf.gpu_busy_ms);
    draw_kv(y, cAmber, zh ? "GPU 渲染耗时" : "GPU busy per frame", buf1); y += line_h;
    std::snprintf(buf1, sizeof buf1, "%.2f ms", perf.pipe_wait_ms);
    draw_kv(y, cAmber, zh ? "管线编译等待" : "Pipeline waits", buf1); y += line_h;
    const int pending_sh = shader_backlog();
    if (pending_sh > 0) std::snprintf(buf1, sizeof buf1, "%.2f ms (%d)", perf.shader_wait_ms, pending_sh);
    else std::snprintf(buf1, sizeof buf1, "%.2f ms", perf.shader_wait_ms);
    draw_kv(y, cAmber, zh ? "着色器等待" : "Shader waits", buf1); y += line_h;
    std::snprintf(buf1, sizeof buf1, zh ? "%.1f 次/帧" : "%.1f per frame", perf.tex_uploads_pf);
    draw_kv(y, cAmber, zh ? "纹理上传次数" : "Texture uploads", buf1); y += line_h;
    std::snprintf(buf1, sizeof buf1, "%.2f ms", perf.tex_upload_ms);
    draw_kv(y, cAmber, zh ? "纹理上传耗时" : "Texture upload time", buf1); y += line_h + sec_gap;

    // Display & Resolution section
    const nvn::GameViewportInfo gvi = nvn::scaled_game_viewport_info();
    const uint32_t game_w = gvi.base_w > 0 ? gvi.base_w : f.extent.width;
    const uint32_t game_h = gvi.base_h > 0 ? gvi.base_h : f.extent.height;
    const uint32_t scn_w = gvi.render_w > 0 ? gvi.render_w : game_w;
    const uint32_t scn_h = gvi.render_h > 0 ? gvi.render_h : game_h;
    const int pct = game_w > 0 ? static_cast<int>(std::lround(100.0 * static_cast<double>(scn_w) / static_cast<double>(game_w))) : 100;

    std::snprintf(buf1, sizeof buf1, "%ux%u", f.extent.width, f.extent.height);
    draw_kv(y, cCyan, zh ? "窗口输出" : "Window", buf1); y += line_h;
    std::snprintf(buf1, sizeof buf1, "%ux%u", game_w, game_h);
    draw_kv(y, cCyan, zh ? "游戏画布" : "Game image", buf1); y += line_h;
    std::snprintf(buf1, sizeof buf1, "%ux%u, %d%%", scn_w, scn_h, pct);
    draw_kv(y, cCyan, zh ? "3D 场景" : "3D scene", buf1); y += line_h;

    const nvn::DlssLiveInfo dli = nvn::dlss_live_info();
    const bool is_1to1 = (scn_w == game_w && scn_h == game_h);
    const bool is_bi = (dli.upscaler == "bilinear");
    std::string scale_str;
    if (gvi.dlss_bypassed || (is_1to1 && is_bi)) {
        scale_str = zh ? "原生直出 (1:1)" : "Native (1:1)";
    } else if (!dli.active_name.empty()) {
        scale_str = dli.active_name;
        if (is_1to1) scale_str += " (Native AA)";
    } else if (!dli.upscaler.empty()) {
        scale_str = dli.upscaler;
    } else {
        scale_str = "bilinear";
    }
    draw_kv(y, cCyan, zh ? "超分 / 缩放" : "Scaling", scale_str.c_str()); y += line_h;

    if (fg.requested) {
        const char* fg_kind = fg.mode == 3 ? "FSR 3 FG" : "DLSS FG";
        if (fg_live) std::snprintf(buf1, sizeof buf1, zh ? "%s (%.0f 帧)" : "%s (%.0f FPS)", fg_kind, fg.shown_fps);
        else if (fg.active) std::snprintf(buf1, sizeof buf1, zh ? "%s (已开启)" : "%s (On)", fg_kind);
        else if (fg.available) std::snprintf(buf1, sizeof buf1, zh ? "%s (待机)" : "%s (Standby)", fg_kind);
        else std::snprintf(buf1, sizeof buf1, zh ? "%s (不可用)" : "%s (Unavailable)", fg_kind);
        draw_kv(y, cCyan, zh ? "帧生成" : "Frame gen", buf1); y += line_h;
    }

    const double cap = ctx.limiter ? ctx.limiter->target_fps() : 0.0;
    if (cap > 0.5) std::snprintf(buf1, sizeof buf1, "%.0f FPS", std::round(cap));
    else std::snprintf(buf1, sizeof buf1, zh ? "无上限" : "Unlimited");
    draw_kv(y, cCyan, zh ? "帧率上限" : "Frame limit", buf1); y += line_h;

    const PresentMode pm = ctx.renderer ? ctx.renderer->present_mode() : PresentMode::Fifo;
    const char* pm_str = (pm == PresentMode::Fifo || pm == PresentMode::FifoRelaxed) ? "VSync"
                       : (pm == PresentMode::Mailbox) ? "Mailbox" : "Immediate";
    draw_kv(y, cCyan, zh ? "呈现模式" : "Presentation", pm_str); y += line_h + sec_gap;

    // Hardware footer
    dl->AddLine(ImVec2(xL, y - sec_gap * 0.5f), ImVec2(xR, y - sec_gap * 0.5f), cSep, 1.0f);
    const char* gpu_nm = ctx.device ? ctx.device->gpu().name.c_str() : "Vulkan GPU";
    draw_c(xL, y, cMuted, gpu_nm); y += line_h;
    draw_c(xL, y, cMuted, perf.cpu_name.c_str());
}

// Draws the counter with the foreground draw list: no window, so it never takes mouse or keyboard input.
void Overlay::Impl::draw_counter(const FrameInfo& f, double now_s) {
    if (counter_detail.load(std::memory_order_relaxed) >= 3) {
        draw_perf_hud(f, now_s);
        return;
    }
    refresh_counter_text(now_s);
    ImFont* font = ImGui::GetFont();
    const float size = std::clamp(std::round(static_cast<float>(f.extent.height) / 45.f), 14.f, 40.f);
    const float pad = size * 0.4f;
    const float gap = size * 0.15f;
    const float margin = size * 0.8f;
    const int n = counter_n;
    float lines_w[4] = {};
    float line_h = 0, text_w = 0;
    for (int i = 0; i < n; ++i) {
        const ImVec2 sz = font->CalcTextSizeA(size, FLT_MAX, 0.0f, counter_lines[i].c_str());
        lines_w[i] = sz.x;
        line_h = std::max(line_h, sz.y);
        text_w = std::max(text_w, sz.x);
    }
    const float box_w = text_w + 2.f * pad;
    const float box_h = static_cast<float>(n) * line_h + static_cast<float>(n - 1) * gap + 2.f * pad;
    const int corner = counter_corner.load(std::memory_order_relaxed);
    const bool right = corner == 1 || corner == 3;
    const bool bottom = corner >= 2;
    const float W = static_cast<float>(f.extent.width), H = static_cast<float>(f.extent.height);
    const float x0 = right ? W - margin - box_w : margin;
    const float y0 = bottom ? H - margin - box_h : margin;

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + box_w, y0 + box_h), IM_COL32(0, 0, 0, 140), size * 0.3f);
    for (int i = 0; i < n; ++i) {
        const float x = right ? x0 + box_w - pad - lines_w[i] : x0 + pad;
        const float y = y0 + pad + static_cast<float>(i) * (line_h + gap);
        dl->AddText(font, size, ImVec2(x, y), IM_COL32(255, 255, 255, 255), counter_lines[i].c_str());
    }
}

bool Overlay::Impl::note_visible(double now_s) {
    if (!note_read) {
        note_read = true;
        note_enabled = nxrt::config_get_bool("nvn", "compile_note", true);
    }
    const int pending = shader_backlog();
    if (!note_enabled) return false;
    if (pending > 0) {
        if (note_since_s < 0) { note_since_s = now_s; note_peak = 0; }
        note_empty_s = -1.0;
        note_peak = std::max(note_peak, pending);
        return now_s - note_since_s >= 1.5;
    }
    if (note_since_s < 0) return false;
    if (note_empty_s < 0) note_empty_s = now_s;
    if (now_s - note_empty_s >= 1.0) { note_since_s = -1.0; return false; }
    return now_s - note_since_s >= 1.5;
}

void Overlay::Impl::draw_note(const FrameInfo& f) {
    char text[96];
    const int pending = shader_backlog();
    if (pending > 0) std::snprintf(text, sizeof text, "Preparing shaders (%d left)", pending);
    else std::snprintf(text, sizeof text, "Preparing shaders (done)");
    ImFont* font = ImGui::GetFont();
    const float size = std::clamp(std::round(static_cast<float>(f.extent.height) / 60.f), 12.f, 28.f);
    const float pad = size * 0.4f;
    const ImVec2 sz = font->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
    const float W = static_cast<float>(f.extent.width), H = static_cast<float>(f.extent.height);
    const float x0 = (W - sz.x) * 0.5f - pad, y0 = H - size * 0.8f - sz.y - 2.f * pad;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + sz.x + 2.f * pad, y0 + sz.y + 2.f * pad), IM_COL32(0, 0, 0, 120), size * 0.3f);
    dl->AddText(font, size, ImVec2(x0 + pad, y0 + pad), IM_COL32(255, 255, 255, 220), text);
}

void Overlay::draw(const FrameInfo& f) {
    Impl& s = *impl_;
    if (s.want_hide.exchange(false)) set_visible(false);   // the menu asked to close during the last frame (mu is not held here)
    if (s.want_picker_close.exchange(false) && s.picker_open.load(std::memory_order_relaxed)) toggle_picker();
    const bool p_open = s.picker_open.load(std::memory_order_relaxed);
    const bool counter_on = s.counter_show.load(std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const double now_s = std::chrono::duration<double>(now.time_since_epoch()).count();
    const bool note_on = s.note_visible(now_s) && (s.counter_in_shots.load(std::memory_order_relaxed) || !s.ctx.renderer->capture_pending());
    const bool ptr_on = s.ptr_show.load(std::memory_order_relaxed) && (s.ptr_in_shots.load(std::memory_order_relaxed) || !s.ctx.renderer->capture_pending());
    const bool kb_on = s.kb_poll();
    const bool hud_on = s.has_hud_cb.load(std::memory_order_relaxed);
    if (!s.visible && !p_open && !counter_on && !note_on && !ptr_on && !kb_on && !hud_on) {
        s.last_frame = now;
        s.counter_was_on = false;
        return;
    }
    if (counter_on) {
        if (!s.counter_was_on) {          // just switched on: start from an empty history
            s.counter_was_on = true;
            s.counter_have_prev = false;
            s.stats.reset();
            s.text_s = -1e9;
        }
        if (s.counter_have_prev) {
            const double dt_ms = std::chrono::duration<double, std::milli>(now - s.counter_prev).count();
            if (dt_ms <= 1000.0) {
                s.stats.push(now_s, dt_ms);   // over a second: a pause (debugger, window in the background), not a frame
                s.perf.graph_ms[s.perf.graph_pos] = static_cast<float>(dt_ms);
                s.perf.graph_pos = (s.perf.graph_pos + 1) % Impl::PerfHudState::kGraphN;
                if (s.perf.graph_count < Impl::PerfHudState::kGraphN) ++s.perf.graph_count;
            }
        }
        s.counter_prev = now;
        s.counter_have_prev = true;
    } else {
        s.counter_was_on = false;
    }
    // a screenshot leaves the counter out unless asked for; with the menu hidden there is then nothing to draw
    const bool show_counter = counter_on && (s.counter_in_shots.load(std::memory_order_relaxed) || !s.ctx.renderer->capture_pending());
    if (!s.visible && !p_open && !show_counter && !note_on && !ptr_on && !kb_on && !hud_on) {
        s.last_frame = now;
        return;
    }
    // made for HDR: the overlay always goes through the RGBA8 target, and the composite writes it as HDR or, after a switch to SDR, as SDR
    const bool hdr_frame = s.hdr;
    if (hdr_frame && !s.composite.ensure_target(f.extent)) return;
    std::lock_guard<std::mutex> lk(s.mu);
    ImGuiContext* prev = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(s.imgui);
    ImGuiIO& io = ImGui::GetIO();
    io.DeltaTime = std::clamp(std::chrono::duration<float>(now - s.last_frame).count(), 1.0f / 1000.f, 0.25f);
    s.last_frame = now;
    io.DisplaySize = ImVec2(static_cast<float>(f.extent.width), static_cast<float>(f.extent.height));

    s.update_scale(f.extent);
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    if (s.visible && !kb_on) s.build_ui();    // the keyboard is modal: the menu comes back once it is answered
    if (kb_on) s.draw_keyboard(now_s);
    if (p_open && !kb_on) {
        PickerCallbacks pcb;
        { std::lock_guard<std::mutex> plk(s.picker_mu); pcb = s.picker_cb; }
        const int td = s.picker_tab_delta.exchange(0, std::memory_order_relaxed);
        if (td != 0) s.picker_tab = ((s.picker_tab + td) % 3 + 3) % 3;
        if (s.want_picker_clear.exchange(false, std::memory_order_relaxed) && s.picker_has_pin) {
            if (pcb.clear) pcb.clear();
            s.picker_has_pin = false;
            s.picker_pin_name.clear();
            s.want_picker_close.store(true);
        }
        if (pcb.query && (now_s - s.picker_last_refresh_s > 0.35 || s.picker_items.empty())) {
            s.picker_items.clear();
            pcb.query(s.picker_items, s.picker_has_pin, s.picker_pin_name);
            s.picker_last_refresh_s = now_s;
        }
        const float W = static_cast<float>(f.extent.width);
        const float H = static_cast<float>(f.extent.height);
        const float win_w = std::clamp(W * 0.38f, 460.f, 720.f);
        const float win_h = std::clamp(H * 0.62f, 440.f, 760.f);
        ImGui::SetNextWindowPos(ImVec2((W - win_w) * 0.5f, (H - win_h) * 0.5f), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(win_w, win_h), ImGuiCond_Appearing);
        // XC2 native obsidian-navy glass + launcher sky-blue (#56C8FA) accent styling
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.025f, 0.065f, 0.115f, 0.93f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.337f, 0.784f, 0.980f, 0.78f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.045f, 0.115f, 0.190f, 0.96f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.055f, 0.155f, 0.255f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.18f, 0.55f, 0.80f, 0.42f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.337f, 0.784f, 0.980f, 0.35f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.337f, 0.784f, 0.980f, 0.55f));
        bool open_flag = true;
        if (ImGui::Begin("3D 导航目标选择  [F4 / ZL+R3]", &open_flag, ImGuiWindowFlags_NoCollapse)) {
            if (s.picker_has_pin) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.88f, 1.0f, 1.0f));
                ImGui::Text("◆ 当前自定义导航：%s", s.picker_pin_name.c_str());
                ImGui::PopStyleColor();
                ImGui::SameLine();
                if (ImGui::SmallButton("[X] 恢复自动追踪任务")) {
                    if (pcb.clear) pcb.clear();
                    s.picker_has_pin = false;
                    s.picker_pin_name.clear();
                    s.want_picker_close.store(true);
                }
                ImGui::Separator();
            }
            ImGui::SetNextItemWidth(-1.f);
            ImGui::InputTextWithHint("##nav_filter", "输入关键词搜索地标 / 传送点 / NPC...", s.picker_filter, sizeof s.picker_filter);
            auto tab_btn = [&](const char* label, int idx) {
                const bool sel = (s.picker_tab == idx);
                if (sel) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.62f, 0.88f, 0.85f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.337f, 0.784f, 0.980f, 0.95f));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.08f, 0.16f, 0.24f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.14f, 0.28f, 0.42f, 0.85f));
                }
                if (ImGui::Button(label)) s.picker_tab = idx;
                ImGui::PopStyleColor(2);
            };
            tab_btn("全部", 0);
            ImGui::SameLine();
            tab_btn("地标 / 传送点", 1);
            ImGui::SameLine();
            tab_btn("NPC", 2);
            ImGui::SameLine();
            ImGui::TextDisabled("  [L / R 切换]");
            ImGui::Separator();

            std::string flt = s.picker_filter;
            for (char& ch : flt) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            const float footer_h = ImGui::GetFrameHeightWithSpacing() + 4.f;
            if (ImGui::BeginChild("##nav_list", ImVec2(0, -footer_h), false)) {
                int shown = 0;
                for (const auto& it : s.picker_items) {
                    if (s.picker_tab == 1 && it.group != 0) continue;
                    if (s.picker_tab == 2 && it.group != 1) continue;
                    if (!flt.empty()) {
                        std::string nl = it.name;
                        for (char& ch : nl) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                        if (nl.find(flt) == std::string::npos) continue;
                    }
                    ++shown;
                    ImGui::PushID(static_cast<int>(it.id & 0x7FFFFFFFu));
                    const char* badge = (it.group == 0) ? "◆ " : "● ";
                    const std::string row_label = std::string(badge) + it.name;
                    if (ImGui::Selectable(row_label.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
                        if (pcb.select) pcb.select(it.id);
                        s.want_picker_close.store(true);
                    }
                    if (!it.detail.empty()) {
                        const float dw = ImGui::CalcTextSize(it.detail.c_str()).x;
                        ImGui::SameLine(std::max(10.f, ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX() - dw - 8.f));
                        ImGui::TextDisabled("%s", it.detail.c_str());
                    }
                    ImGui::PopID();
                }
                if (shown == 0) ImGui::TextDisabled("当前地图没有匹配的地标或 NPC");
            }
            ImGui::EndChild();
            ImGui::Separator();
            ImGui::TextDisabled("[A/Enter] 导航   [L/R] 切换分类   [X] 清除自定义   [B/Esc/ZL+R3] 关闭");
        }
        ImGui::End();
        ImGui::PopStyleColor(7);
        if (!open_flag) s.want_picker_close.store(true);
    }
    s.ribbon_sink.active = false;
    s.ribbon_sink.verts.clear();
    if (hud_on && !kb_on) {
        std::function<void(const HudCanvas&)> cb;
        { std::lock_guard<std::mutex> hlk(s.hud_mu); cb = s.hud_cb; }
        if (cb) {
            HudCanvas canvas;
            canvas.width = f.extent.width;
            canvas.height = f.extent.height;
            canvas.draw_list = ImGui::GetForegroundDrawList();
            canvas.font = ImGui::GetFont();
            canvas.ribbon_sink = &s.ribbon_sink;
            cb(canvas);
        }
    }
    if (show_counter) s.draw_counter(f, now_s);
    if (note_on) s.draw_note(f);
    if (ptr_on) {   // a plain arrow, white with a dark outline, sized with the frame height
        const float k = std::max(1.f, static_cast<float>(f.extent.height) / 720.f);
        const float x = s.ptr_x.load(std::memory_order_relaxed), y = s.ptr_y.load(std::memory_order_relaxed);
        const ImVec2 pts[7] = {{0, 0}, {0, 16 * k}, {4 * k, 12.5f * k}, {7.5f * k, 19 * k}, {10 * k, 18 * k}, {6.5f * k, 11.5f * k}, {12 * k, 11.5f * k}};
        ImVec2 poly[7];
        for (int i = 0; i < 7; ++i) poly[i] = ImVec2(x + pts[i].x, y + pts[i].y);
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        dl->AddConvexPolyFilled(poly, 7, IM_COL32(255, 255, 255, 255));
        dl->AddPolyline(poly, 7, IM_COL32(0, 0, 0, 255), ImDrawFlags_Closed, 1.5f * k);
    }
    ImGui::Render();

    // one dynamic-rendering pass that draws the 3D nav ribbon (with scene depth occlusion) and ImGui draw data into view
    auto render_into = [&](VkImageView view, VkFormat target_fmt, VkAttachmentLoadOp load) {
        VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        att.imageView = view;
        att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att.loadOp = load;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, f.extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &att;
        vkCmdBeginRendering(f.cmd, &ri);
        if (s.nav_ribbon_ok && s.ribbon_sink.active && s.ribbon_sink.verts.size() >= 4) {
            extras::NavRibbonDepth rd;
            rd.view = f.scene_depth_view;
            rd.layout = f.scene_depth_layout;
            rd.flip_y = f.scene_depth_flip_y;
            rd.depth_mode = f.scene_depth_mode;
            s.nav_ribbon.record(f.cmd, target_fmt, f.extent, f.slot,
                                s.ribbon_sink.verts.data(), static_cast<uint32_t>(s.ribbon_sink.verts.size()),
                                s.ribbon_sink.params, rd);
        }
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), f.cmd);
        vkCmdEndRendering(f.cmd);
    };
    if (hdr_frame) {
        // HDR: the overlay goes into the cleared RGBA8 target, then the composite maps it onto the swapchain image (UI white = paper white)
        const extras::HdrState hs = extras::hdr_state();
        s.composite.begin_overlay(f.cmd);
        render_into(s.composite.target_view(), VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR);
        s.composite.composite(f.cmd, f.view, f.format, f.extent, f.hdr_output ? s.ctx.renderer->output_kind() : extras::OutputKind::Sdr8,
                              hs.paper_white_nits, hs.peak_nits);
    } else {
        render_into(f.view, f.format, VK_ATTACHMENT_LOAD_OP_LOAD);
    }
    s.drawn++;
    ImGui::SetCurrentContext(prev);
}

void Overlay::install_on_renderer() {
    impl_->ctx.renderer->set_overlay_hook([this](const FrameInfo& f) { draw(f); });
}

void Overlay::set_hud_callback(std::function<void(const HudCanvas&)> fn) {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->hud_mu);
    impl_->hud_cb = std::move(fn);
    impl_->has_hud_cb.store(static_cast<bool>(impl_->hud_cb), std::memory_order_relaxed);
}

void Overlay::set_picker(PickerCallbacks cb) {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->picker_mu);
    impl_->picker_cb = std::move(cb);
}

void Overlay::toggle_picker() {
    if (!impl_) return;
    Impl& s = *impl_;
    const bool now_open = !s.picker_open.load(std::memory_order_relaxed);
    s.picker_open.store(now_open, std::memory_order_relaxed);
    if (now_open) {
        s.picker_last_refresh_s = -1e9;
        s.picker_restore_relative = platform::has_window() && platform::relative_mouse();
        if (s.picker_restore_relative) platform::set_relative_mouse(false);
        if (platform::has_window()) platform::set_text_input(true);
    } else {
        if (platform::has_window()) platform::set_text_input(false);
        if (s.picker_restore_relative && platform::has_window()) platform::set_relative_mouse(true);
        s.picker_restore_relative = false;
        std::lock_guard<std::mutex> lk(s.mu);
        clear_imgui_input(s.imgui);
    }
    NXLOG_I(kTag, "nav picker %s", now_open ? "shown" : "hidden");
}

bool Overlay::picker_open() const {
    return impl_ && impl_->picker_open.load(std::memory_order_relaxed);
}

uint64_t Overlay::create_hud_texture_bc(uint32_t w, uint32_t h, const void* linear, size_t bytes,
                                        uint32_t vk_format, const char* label) {
    if (!impl_ || !impl_->backend_ready || !impl_->ctx.device || !linear || w < 4 || h < 4) return 0;
    const size_t expected = static_cast<size_t>((w + 3u) / 4u) * static_cast<size_t>((h + 3u) / 4u) * 16ull;
    if (bytes < expected) return 0;

    Impl& s = *impl_;
    Device& dev = *s.ctx.device;
    std::lock_guard<std::mutex> tlk(s.hud_tex_mu);

    if (!s.hud_tex_sampler) {
        VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        if (vkCreateSampler(dev.device(), &sci, nullptr, &s.hud_tex_sampler) != VK_SUCCESS) {
            return 0;
        }
    }

    gfx::ImageDesc id;
    id.format = static_cast<VkFormat>(vk_format);
    id.width = w;
    id.height = h;
    id.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    id.name = "hud_bc_atlas";
    gfx::Image img;
    if (dev.create_image(id, &img) != VK_SUCCESS) return 0;

    gfx::Buffer staging;
    if (dev.create_buffer(expected, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gfx::MemoryUse::Upload, &staging, "hud_bc_stage") != VK_SUCCESS) {
        dev.destroy_image(img);
        return 0;
    }
    if (!staging.mapped) {
        dev.destroy_buffer(staging);
        dev.destroy_image(img);
        return 0;
    }
    std::memcpy(staging.mapped, linear, expected);
    dev.flush_buffer(staging);

    dev.submit_once([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cb, &di);

        VkBufferImageCopy reg = {};
        reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reg.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cb, staging.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);

        b.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(cb, &di);
    });
    dev.destroy_buffer(staging);

    VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(s.hud_tex_sampler, img.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!ds) {
        dev.destroy_image(img);
        return 0;
    }
    s.hud_textures.push_back({img, ds});
    NXLOG_I(kTag, "uploaded HUD %s texture %ux%u (%zu KiB)", label, w, h, expected / 1024u);
    return reinterpret_cast<uint64_t>(ds);
}

uint64_t Overlay::create_hud_texture_bc3(uint32_t w, uint32_t h, const void* bc3_linear, size_t bytes) {
    return create_hud_texture_bc(w, h, bc3_linear, bytes, static_cast<uint32_t>(VK_FORMAT_BC3_UNORM_BLOCK), "BC3");
}

uint64_t Overlay::create_hud_texture_bc7(uint32_t w, uint32_t h, const void* bc7_linear, size_t bytes) {
    return create_hud_texture_bc(w, h, bc7_linear, bytes, static_cast<uint32_t>(VK_FORMAT_BC7_UNORM_BLOCK), "BC7");
}

}  // namespace nxrt::gfx
