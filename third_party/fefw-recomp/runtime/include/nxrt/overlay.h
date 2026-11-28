// SPDX-License-Identifier: GPL-3.0-or-later
// Dear ImGui overlay on the Vulkan path (S4b): the settings / developer menu of the PC port. Hidden by default; a key
// (F1, [gfx] overlay_key = any SDL scancode name) toggles it. ImGui is fed from the nxrt input queue (input.h), not from an
// ImGui platform backend, so it works the same with a window and headless. While it is visible and wants the mouse or the
// keyboard, handle_event() returns true and the game must not see the event. Showing the overlay releases the relative mouse
// mode and hiding it restores it.
// Changes made in the menu that touch the swapchain or the window (present mode, fullscreen) are applied by apply_pending(),
// which the frame loop calls before Renderer::begin_frame().
#pragma once
#include <functional>
#include <memory>
#include <string>

#include "nxrt/frame_limiter.h"
#include "nxrt/gfx.h"
#include "nxrt/input.h"

namespace nxrt::gfx {

// Background shader work (S24). The threads that compile pipeline parts publish how many jobs are waiting or running; while there are some
// the overlay shows a small note ("Preparing shaders ..."). Any thread. [nvn] compile_note = false keeps the note off.
void set_shader_backlog(int pending);
int shader_backlog();

// 3D ground navigation ribbon vertex & parameters (rendered by native Vulkan pass before ImGui, with depth occlusion).
struct Ribbon3DVertex {
    float x = 0.f, y = 0.f, z = 0.f;
    float u = 0.f, v = 0.f;
    float fade = 1.f;
};

struct Ribbon3DParams {
    float view_proj[16] = {};
    float color[4] = {0.22f, 0.85f, 1.0f, 0.88f};
    float time_sec = 0.f;
    float reveal_dist = 100.f;
    float occluded_alpha = 0.14f;
};

// Lightweight game-agnostic 2D foreground canvas backed by ImGui's foreground draw list (no ImGui header needed by callers,
// works transparently on both SDR and HDR swapchains, never captures input).
struct HudCanvas {
    uint32_t width = 0;
    uint32_t height = 0;
    void* draw_list = nullptr;
    void* font = nullptr;
    void* ribbon_sink = nullptr;

    void rect_filled(float x0, float y0, float x1, float y1, uint32_t rgba, float rounding = 0.f) const;
    void rect_filled_hgrad(float x0, float y0, float x1, float y1, uint32_t rgba_left, uint32_t rgba_right) const;
    void rect(float x0, float y0, float x1, float y1, uint32_t rgba, float rounding = 0.f, float thickness = 1.f) const;
    void line(float x0, float y0, float x1, float y1, uint32_t rgba, float thickness = 1.5f) const;
    void triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t rgba) const;
    void quad_filled(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3, uint32_t rgba) const;
    void quad(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3, uint32_t rgba, float thickness = 1.5f) const;
    void convex_poly_filled(const float* xy, int count, uint32_t rgba) const;
    void convex_poly_hgrad(const float* xy, int count, float x_left, float x_right, uint32_t rgba_left, uint32_t rgba_right) const;
    void polyline(const float* xy, int count, uint32_t rgba, bool closed = false, float thickness = 1.5f) const;
    void circle_filled(float cx, float cy, float r, uint32_t rgba, int segments = 0) const;
    void circle(float cx, float cy, float r, uint32_t rgba, float thickness = 1.5f, int segments = 0) const;
    void text(float x, float y, float size, uint32_t rgba, const char* str) const;
    void text_outlined(float x, float y, float size, uint32_t rgba, uint32_t outline_rgba, const char* str) const;
    void measure_text(float size, const char* str, float* out_w, float* out_h) const;
    void image(uint64_t tex_id, float x0, float y0, float x1, float y1,
               float u0 = 0.f, float v0 = 0.f, float u1 = 1.f, float v1 = 1.f,
               uint32_t tint_rgba = 0xFFFFFFFFu) const;
    void image_rounded(uint64_t tex_id, float x0, float y0, float x1, float y1,
                       float u0, float v0, float u1, float v1,
                       float rounding, uint32_t tint_rgba = 0xFFFFFFFFu) const;
    void set_ribbon_3d(const Ribbon3DVertex* verts, uint32_t count, const Ribbon3DParams& params) const;
};

// Lightweight game-agnostic searchable destination picker window (toggled by caller, e.g. F4).
struct PickerItem {
    uint64_t id = 0;
    uint8_t group = 0;     // 0 = Landmark / Skip-Travel, 1 = NPC
    float dist = 0.f;
    std::string name;
    std::string detail;
};

struct PickerCallbacks {
    std::function<void(std::vector<PickerItem>& out_items, bool& out_has_pin, std::string& out_pin_name)> query;
    std::function<void(uint64_t id)> select;
    std::function<void()> clear;
};

class Overlay {
public:
    struct Context {
        Device* device = nullptr;
        Renderer* renderer = nullptr;
        FrameLimiter* limiter = nullptr;     // optional: shows and sets the frame cap
        bool has_window = true;              // false: no window controls (headless)
    };
    static std::unique_ptr<Overlay> create(const Context& ctx, std::string* error = nullptr);
    ~Overlay();
    Overlay(const Overlay&) = delete;
    Overlay& operator=(const Overlay&) = delete;

    // Feed every input event here (before the game sees it). Returns true when the overlay consumed it.
    bool handle_event(const input::Event& e);

    void set_visible(bool v);
    bool visible() const;
    void toggle();
    int toggle_scancode() const;             // SDL scancode of the toggle key
    void set_toggle_scancode(int sc);
    // The on-screen keyboard (the software keyboard applet's question, services_api.h swkbdui): drawn over the game whenever a question is
    // open, menu or not (the menu is not drawn meanwhile). While it is up it takes the keyboard, the mouse and the pad (handle_event returns
    // true for them). Physical keys type into it (needs SDL text input: platform::set_text_input), Enter answers, Escape cancels; the mouse
    // clicks its keys; the pad walks the key grid (d-pad or left stick, with repeat): A types the key, B deletes, Y space, X shift, L/R
    // move the caret, + answers, - cancels. Nothing here depends on the menu being visible.
    bool keyboard_open() const;
    bool blocks_game() const;                // the menu is visible or the on-screen keyboard is open: the game gets no input
    void set_keyboard_abxy(bool by_label);   // how the face buttons are named in the keyboard's help line (play settings label_abxy)
    // The merged pad of this poll (Switch buttons after the player's mapping, left stick -32767..32767, up = +), before it is held back
    // from the game. Returns the buttons the game must not see: all while the keyboard is open, then those still held from the press
    // that closed it until they are released. Call it on every pad poll.
    uint64_t set_keyboard_pad(uint64_t buttons, int lx, int ly);
    bool wants_input() const;                // visible and ImGui wants the mouse or keyboard
    bool wants_mouse() const;                // visible and the pointer is over the menu or a widget is being dragged
    bool wants_keyboard() const;             // visible and a text field or navigation has the keyboard
    // (both reflect the last drawn frame: ImGui decides hover while it builds the frame)

    // Extra panels for later streams (settings, dev menu): fn runs inside an ImGui window titled `title` while visible.
    void add_panel(const std::string& title, std::function<void()> fn);
    // A game's own menu: when set, fn is called each visible frame in place of the Diagnostics window and the panels, between NewFrame
    // and Render on the presenting thread; it opens and closes its own windows. request_hide() closes the overlay from inside it.
    void set_menu(std::function<void()> fn);
    void request_hide();
    const Context& context() const;
    // The pad for the menu (Switch buttons after the player's mapping, left stick -32767..32767, up = +), on every pad poll after
    // set_keyboard_pad. The chord ([gfx] overlay_pad, default "minus+plus"; "none" turns it off) shows and hides the overlay. While it is
    // visible the pad drives ImGui's gamepad navigation: A activate, B back, the d-pad and the left stick move, L / R and ZL / ZR are the
    // shoulders (X, Y, + and - are not passed on). Returns the buttons the game must not see: all while visible, then those still held from
    // the press that closed it until they are released.
    uint64_t set_menu_pad(uint64_t buttons, int lx, int ly);

    // Applies menu choices that cannot be made in the middle of a frame. Call before Renderer::begin_frame().
    void apply_pending();
    // The same in two halves for a multi-threaded caller: the present mode must change on the thread that presents (before its begin_frame), the
    // window (fullscreen) on the thread that owns the SDL window.
    void apply_present_pending();
    void apply_window_pending();

    // Corner frame-rate counter, off by default. Safe to call from any thread; it takes effect on the next presented frame.
    struct CounterConfig {
        bool show = false;
        int corner = 0;                  // 0 top left, 1 top right, 2 bottom left, 3 bottom right
        int detail = 1;                  // 0 FPS; 1 FPS+ms; 2 compact full (1% low/max); 3 detailed telemetry HUD + graph
        bool in_screenshots = false;     // leave it in screenshots too (by default it is not drawn into them)
        bool lang_zh = true;             // UI language hint for the detailed HUD
    };
    void set_counter(const CounterConfig& c);

    // A drawn arrow at a window pixel (test aid, S28: a screenshot of the game's own frame does not contain the OS cursor). Any thread.
    void set_pointer(bool show, float x, float y, bool in_screenshots);

    // Optional per-frame foreground HUD callback (runs on the presenting thread inside the ImGui frame, even when the F1 menu is hidden).
    // Pass an empty std::function to clear. Thread-safe.
    void set_hud_callback(std::function<void(const HudCanvas&)> fn);

    // Upload a linear BC3 (DXT5) texture for foreground HudCanvas::image / image_rounded (owned until Overlay shutdown).
    // Returns an opaque non-zero ImTextureID handle, or 0 on failure.
    uint64_t create_hud_texture_bc3(uint32_t w, uint32_t h, const void* bc3_linear, size_t bytes);
    // Upload a linear BC7 (BPTC) texture. The menu face atlas (/menu/image/mnu_face.wilay) is stored this way, so the
    // DLC and Switch 2 Edition blades only have a portrait here.
    uint64_t create_hud_texture_bc7(uint32_t w, uint32_t h, const void* bc7_linear, size_t bytes);

    // Optional searchable destination picker window (toggled independently of F1, e.g. F4).
    void set_picker(PickerCallbacks cb);
    void toggle_picker();
    bool picker_open() const;

    // The Renderer overlay hook: builds and draws the menu on top of the frame (no-op while hidden, unless the counter is on).
    void draw(const FrameInfo& f);
    void install_on_renderer();              // renderer->set_overlay_hook(draw)

    uint64_t frames_drawn() const;

private:
    Overlay() = default;
    // Shared body of create_hud_texture_bc3/bc7: vk_format is a VkFormat (kept as uint32_t so this header stays free of vulkan).
    uint64_t create_hud_texture_bc(uint32_t w, uint32_t h, const void* linear, size_t bytes, uint32_t vk_format, const char* label);
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace nxrt::gfx
