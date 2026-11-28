// SPDX-License-Identifier: GPL-3.0-or-later
// The render-thread half of NVN2 DLSS (api_dlss.h): an EvaluateDlss command upscales the input colour region into the output region with the
// host upscaler of [dlss] upscaler, then copies the result into the output texture (a blit: any colour format, any resolution scale).
//   taau      the built-in temporal upscaler (extras/taau.h), default: jitter, motion vectors, depth, the bias mask and the reset are used
//   bilinear  the input at its unjittered position, no history (a reference, and what Eden showed before its tensor fix)
//   fsr3      AMD FSR 3.1 through the FidelityFX DLLs (extras/temporal_ffx.cpp), else taau
//   dlss      NVIDIA DLSS through Streamline (extras/temporal_dlss.cpp; Streamline is started before the Vulkan instance), else taau
//   auto      dlss when Streamline started, else fsr3 when its DLLs are there, else taau
// A network keeps one history (two rgba16f images at the output size, used in turn), as NGX keeps one per feature. The evaluation index does
// not select it: Fortune's Weave alternates 0 and 1 every frame (main 0x648DE0), the console's per-frame slots of evaluation parameters.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "api_dlss.h"
#include "extras/framegen.h"
#include "extras/hdr_nvn.h"
#include "nvn_framegen.h"
#include "extras/taau.h"
#include "extras/temporal.h"
#include "nvn_format.h"
#include "nvn_gpu_internal.h"
#include "nxrt/config.h"
#include "nxrt/gfx_tuning.h"
#include "nxrt/log.h"
#include "nxrt/nvn_backend.h"

namespace nxrt::nvn::be {
namespace {

using gpu::device;
using gpu::Frame;

constexpr const char* kTag = "nvn.dlss";

enum class Upscaler { Taau, Bilinear, Fsr3, Dlss, Auto };

struct Settings {
    Upscaler upscaler = Upscaler::Taau;
    int jitter_sign = 0;     // bit 0 flips x, bit 1 flips y (debug)
    int mv_sign = 0;         // the same for the motion vectors
    float max_weight = 12.0f;
    float sharpness = 0.0f;  // the SDK upscalers' sharpening
};

// [dlss] hot_file (NXRT_DLSS_HOT_FILE): "key=value" lines, re-read four times a second, so the upscaler and its tuning can be
// A/B'd while the game runs (the ini itself is read once, HANDOVER 90). Keys: upscaler, history_weight, jitter_sign, mv_sign,
// sharpness. A changed upscaler bumps a generation counter, and a network whose history came from another upscaler resets.
std::atomic<int> g_upscaler_gen{0};
static int g_upscaler_seen = -1;
const char* upscaler_name(Upscaler u);   // defined below settings()

std::mutex g_live_mu;
std::atomic<uint32_t> g_live_seq{0};
bool g_live_has_upscaler = false;
Upscaler g_live_upscaler = Upscaler::Taau;
std::string g_live_upscaler_str;
std::string g_live_fsr_version;
bool g_live_has_sharp = false;
float g_live_sharpness = 0.0f;

Upscaler parse_upscaler_str(std::string u) {
    for (char& c : u) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (u == "bilinear" || u == "spatial") return Upscaler::Bilinear;
    if (u == "fsr3" || u == "fsr") return Upscaler::Fsr3;
    if (u == "dlss") return Upscaler::Dlss;
    if (u == "auto") return Upscaler::Auto;
    return Upscaler::Taau;
}

const char* upscaler_key(Upscaler u) {
    switch (u) {
        case Upscaler::Bilinear: return "bilinear";
        case Upscaler::Fsr3: return "fsr3";
        case Upscaler::Dlss: return "dlss";
        case Upscaler::Auto: return "auto";
        default: return "taau";
    }
}

const Settings& settings() {
    static Settings s = [] {
        Settings r;
        std::string u = config_get_str("dlss", "upscaler", "taau");
        for (char& c : u) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        r.upscaler = parse_upscaler_str(u);
        if (u != "taau" && u != "bilinear" && u != "spatial" && u != "fsr3" && u != "fsr" && u != "dlss" && u != "auto")
            NXLOG_W(kTag, "[dlss] upscaler = %s is not known: taau", u.c_str());
        r.jitter_sign = static_cast<int>(config_get_int("dlss", "jitter_sign", 0));
        r.mv_sign = static_cast<int>(config_get_int("dlss", "mv_sign", 0));
        r.max_weight = static_cast<float>(std::clamp(config_get_double("dlss", "history_weight", 12.0), 2.0, 64.0));
        r.sharpness = static_cast<float>(std::clamp(config_get_double("dlss", "sharpness", 0.0), 0.0, 1.0));
        return r;
    }();
    static uint32_t live_seen = 0;
    const uint32_t seq = g_live_seq.load(std::memory_order_acquire);
    if (seq != live_seen) {
        live_seen = seq;
        std::lock_guard<std::mutex> lk(g_live_mu);
        if (g_live_has_upscaler) {
            s.upscaler = g_live_upscaler;
            g_upscaler_seen = static_cast<int>(g_live_upscaler);
        }
        if (g_live_has_sharp) s.sharpness = g_live_sharpness;
    }
    static const std::string hot = config_get_str("dlss", "hot_file", "");
    if (!hot.empty()) {
        static std::chrono::steady_clock::time_point last{};
        static std::string last_hot_content;
        const auto now = std::chrono::steady_clock::now();
        if (last.time_since_epoch().count() == 0 || now - last > std::chrono::milliseconds(250)) {
            last = now;
            std::ifstream in(hot);
            std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (body != last_hot_content) {
                last_hot_content = body;
                size_t pos = 0;
                while (pos < body.size()) {
                    const size_t nl = body.find('\n', pos);
                    std::string line = (nl == std::string::npos) ? body.substr(pos) : body.substr(pos, nl - pos);
                    pos = (nl == std::string::npos) ? body.size() : nl + 1;
                    const size_t eq = line.find('=');
                    if (eq == std::string::npos) continue;
                    std::string k = line.substr(0, eq), v = line.substr(eq + 1);
                    auto trim = [](std::string& x) {
                        while (!x.empty() && (x.back() == ' ' || x.back() == '\t' || x.back() == '\r')) x.pop_back();
                        while (!x.empty() && (x.front() == ' ' || x.front() == '\t')) x.erase(x.begin());
                    };
                    trim(k);
                    trim(v);
                    if (k.empty() || v.empty()) continue;
                    if (k == "upscaler") {
                        for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                        const Upscaler u = v == "bilinear" || v == "spatial" ? Upscaler::Bilinear
                                         : v == "fsr3" || v == "fsr" ? Upscaler::Fsr3
                                         : v == "dlss" ? Upscaler::Dlss
                                         : v == "auto" ? Upscaler::Auto : Upscaler::Taau;
                        if (static_cast<int>(u) != g_upscaler_seen) {
                            g_upscaler_seen = static_cast<int>(u);
                            s.upscaler = u;
                            ++g_upscaler_gen;
                            NXLOG_I(kTag, "upscaler -> %s (hot)", upscaler_name(u));
                        }
                    } else if (k == "history_weight") {
                        s.max_weight = static_cast<float>(std::clamp(std::atof(v.c_str()), 2.0, 64.0));
                    } else if (k == "jitter_sign") {
                        s.jitter_sign = std::atoi(v.c_str());
                    } else if (k == "mv_sign") {
                        s.mv_sign = std::atoi(v.c_str());
                    } else if (k == "sharpness") {
                        s.sharpness = static_cast<float>(std::clamp(std::atof(v.c_str()), 0.0, 1.0));
                    }
                }
            }
        }
    }
    return s;
}

const char* upscaler_name(Upscaler u) {
    switch (u) {
        case Upscaler::Taau: return "taau (built-in temporal)";
        case Upscaler::Bilinear: return "bilinear";
        case Upscaler::Fsr3: return "FSR 3";
        case Upscaler::Dlss: return "DLSS (Streamline)";
        default: return "auto";
    }
}

std::atomic<int> g_in_use{-1};   // the Upscaler the last evaluation ran (dlss_upscaler_in_use, the F1 menu); -1 = none yet

struct NetGpu;   // the per-network GPU objects (below)

// An image of ours, kept in GENERAL layout.
struct Img {
    gfx::Image img;
    bool init = false;    // left UNDEFINED
    std::string name;     // what ensure() made it for ("dlss sdk colour", ...), so it can be reused after a size change
};

// Images are kept per (name, size, format) instead of being destroyed when the game's render size changes. The game
// scales its 3D resolution continuously (HANDOVER 95.11: 960x540 .. 1280x720 in steps of a few dozen frames), and the
// old code retired and re-made these four images on every step: that stalled the GPU (wait-free retire + create) and
// handed the upscaler a brand new image every few frames while its context stayed alive -- the FSR 4.1.2 crash on
// amdvlk lands on a barrier a few milliseconds after such a step.
struct Cached {
    std::string name;
    VkExtent3D extent = {};
    VkFormat format = VK_FORMAT_UNDEFINED;
    gfx::Image img;
};
constexpr size_t kCacheMax = 20;   // the game's dynamic-resolution ladder, plus a couple of spares

struct History {
    Img img[2];
    uint32_t cur = 0;    // the image the next pass writes
    bool valid = false;  // the other one holds a picture
    VkExtent2D size = {};
    float prev_jitter[2] = {};
};

struct NetGpu {
    bool dead = false;   // the back end shut down: the images are gone
    History hist;
    Img sdk[4];          // the SDK upscalers' colour, depth, motion (input size) and output (output size)
    bool sdk_valid = false;   // the SDK upscaler's own history holds this network's last frame
    VkExtent2D sdk_in = {}, sdk_out = {};
    Img dump;            // [dlss] dump_every: the input region as rgba16f
    bool dump_pair = false;   // the next evaluation is dumped too
    uint64_t evals = 0, resets = 0;
    int gen = -1;             // the upscaler generation this network's history was made with ([dlss] hot_file)
    std::chrono::steady_clock::time_point last{};
    bool logged = false;
    std::vector<Cached> cache;   // HANDOVER 95.11: images of the sizes this network has used, kept for reuse
};

// An image that is no longer needed right now is kept for its size instead of being destroyed (HANDOVER 95.11).
void stash(NetGpu& n, Img& i) {
    if (!i.img) return;
    if (n.cache.size() >= kCacheMax) {
        device().destroy_image(n.cache.front().img);
        n.cache.erase(n.cache.begin());
    }
    Cached c;
    c.name = i.name;
    c.extent = i.img.extent;
    c.format = i.img.format;
    c.img = i.img;
    n.cache.push_back(std::move(c));
    i.img = gfx::Image();
    i.init = false;
}

std::mutex g_mu;
std::set<NetGpu*> g_nets;

void destroy_images(NetGpu& n) {
    for (Cached& c : n.cache)
        if (c.img) device().destroy_image(c.img);
    n.cache.clear();
    for (Img& i : n.hist.img)
        if (i.img) device().destroy_image(i.img);
    for (Img& i : n.sdk)
        if (i.img) device().destroy_image(i.img);
    if (n.dump.img) device().destroy_image(n.dump.img);
    n.dump = Img();
    n.hist = History();
    for (Img& i : n.sdk) i = Img();
}

// DlssNet::free_gpu (any thread): the images go once the GPU is done with them.
void free_net_gpu(void* p) {
    auto* n = static_cast<NetGpu*>(p);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (n->dead) {
            delete n;
            return;
        }
        g_nets.erase(n);
    }
    gpu::defer_destroy([n] {
        destroy_images(*n);
        delete n;
    });
}

NetGpu* net_gpu(DlssNet& net) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto* n = static_cast<NetGpu*>(net.gpu);
    if (n && n->dead) {   // from a back end that was shut down (tests): start again
        delete n;
        n = nullptr;
    }
    if (!n) {
        n = new NetGpu();
        net.gpu = n;
        net.free_gpu = &free_net_gpu;
        g_nets.insert(n);
    }
    return n;
}

void retire(Img& i) {
    if (!i.img) return;
    gfx::Image old = i.img;
    gpu::defer_destroy([old]() mutable { device().destroy_image(old); });
    i = Img();
}

bool ensure(NetGpu& n, Img& i, VkExtent2D e, VkFormat fmt, const char* name) {
    if (i.img && i.img.extent.width == e.width && i.img.extent.height == e.height && i.img.format == fmt) return true;
    for (size_t k = 0; k < n.cache.size(); ++k) {   // this size came back: take the image back
        Cached& c = n.cache[k];
        if (c.extent.width == e.width && c.extent.height == e.height && c.format == fmt && c.name == name) {
            stash(n, i);
            i.img = c.img;
            i.init = true;   // it was in GENERAL when it was stashed
            i.name = name;
            n.cache.erase(n.cache.begin() + static_cast<long>(k));
            return true;
        }
    }
    stash(n, i);
    gfx::ImageDesc d;
    d.format = fmt;
    d.width = e.width;
    d.height = e.height;
    d.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    d.name = name;
    std::string why;
    if (device().create_image(d, &i.img, &why) != VK_SUCCESS) {
        NXLOG_E(kTag, "%s %ux%u: image not made (%s)", name, e.width, e.height, why.c_str());
        i = Img();
        return false;
    }
    i.init = false;
    i.name = name;
    return true;
}

void to_general(Frame& f, Img& i) {
    if (i.init) return;
    gpu::image_barrier(f.cb, i.img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
    i.init = true;
}

bool blit_dst_ok(VkFormat fmt) {
    static std::map<VkFormat, bool> cache;
    auto it = cache.find(fmt);
    if (it != cache.end()) return it->second;
    VkFormatProperties fp = {};
    vkGetPhysicalDeviceFormatProperties(device().physical(), fmt, &fp);
    const bool ok = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
    cache[fmt] = ok;
    return ok;
}

// A texture region in NVN texels, with the level and layer of its view.
struct Region {
    int x = 0, y = 0, w = 0, h = 0;
    uint32_t level = 0, layer = 0;
    float sc = 1.0f;           // image texels per NVN texel
    int img_w = 0, img_h = 0;  // the image level's size
};

Region region_of(Texture* t, const TextureViewInfo& v, bool has_view, int bx, int by, int w, int h) {
    Region r;
    if (has_view) {
        if (v.has_levels) r.level = static_cast<uint32_t>(std::max(0, v.min_level));
        if (v.has_layers) r.layer = static_cast<uint32_t>(std::max(0, v.min_layer));
    }
    const VkExtent3D e = gpu::tex_extent(t, r.level);
    const VkExtent3D ie = gpu::tex_image_extent(t, r.level);
    r.sc = gpu::tex_scale(t);
    r.img_w = static_cast<int>(ie.width);
    r.img_h = static_cast<int>(ie.height);
    r.x = std::clamp(bx, 0, static_cast<int>(e.width));
    r.y = std::clamp(by, 0, static_cast<int>(e.height));
    r.w = std::min(w > 0 ? w : static_cast<int>(e.width) - r.x, static_cast<int>(e.width) - r.x);
    r.h = std::min(h > 0 ? h : static_cast<int>(e.height) - r.y, static_cast<int>(e.height) - r.y);
    return r;
}

VkImageView sampled_view(Frame& f, Texture* t, const TextureViewInfo& v, bool has_view, const Region& r) {
    gpu::tex_ensure_current(t, f);
    gpu::tex_transition(t, f, gpu::tex_sampled_layout(t));
    gpu::ViewRequest q;
    q.type = VK_IMAGE_VIEW_TYPE_2D;
    q.base_level = r.level;
    q.level_count = 1;
    q.base_layer = r.layer;
    q.layer_count = 1;
    q.depth_aspect = true;
    if (has_view && v.has_format) {
        const FormatInfo* fi = format_info(v.format);
        if (fi && fi->vk != VK_FORMAT_UNDEFINED && !gpu::tex_is_depth(t)) q.format = fi->vk;
    }
    return gpu::tex_sampled_view(t, q);
}

// Where an auxiliary input (depth, motion, mask) is read for each texel of `per` (the colour region, or the output region; sizes in image
// texels): the base of its region in image texels and its image texels per texel of `per`.
void aux_map(const Region& a, float per_img_w, float per_img_h, float base[2], float ratio[2]) {
    base[0] = std::floor(a.x * a.sc);
    base[1] = std::floor(a.y * a.sc);
    ratio[0] = std::max(0.0f, std::min(a.w * a.sc, static_cast<float>(a.img_w) - base[0]) / std::max(per_img_w, 1.0f));
    ratio[1] = std::max(0.0f, std::min(a.h * a.sc, static_cast<float>(a.img_h) - base[1]) / std::max(per_img_h, 1.0f));
}

const char* fmt_name(Texture* t) { return t && t->fmt ? t->fmt->name : "?"; }

void log_first(const DlssEval& e, const Region& c, const Region& o, Upscaler u) {
    NXLOG_I(kTag, "network %u: first evaluation, %s: colour %s %ux%u region %d,%d %dx%d -> output %s %ux%u region %d,%d %dx%d; depth %s, motion %s "
                  "(scale %g %g), mask %s, jitter %+.3f %+.3f, reset %d, index %d, pre-exposure %g",
            e.net->id, upscaler_name(u), fmt_name(e.color.get()), e.color->b.width, e.color->b.height, c.x, c.y, c.w, c.h, fmt_name(e.output.get()),
            e.output->b.width, e.output->b.height, o.x, o.y, o.w, o.h, e.depth ? fmt_name(e.depth.get()) : "none", e.motion ? fmt_name(e.motion.get()) : "none",
            static_cast<double>(e.p.mv_scale[0]), static_cast<double>(e.p.mv_scale[1]), e.mask ? fmt_name(e.mask.get()) : "none",
            static_cast<double>(e.p.jitter[0]), static_cast<double>(e.p.jitter[1]), e.p.reset, e.p.eval_index, static_cast<double>(e.p.pre_exposure));
}

// The result (GENERAL layout) into the output texture's region.
void blit_out(Frame& f, const gfx::Image& src, Texture* out, const Region& o) {
    const VkFormat fmt = gpu::tex_vk_format(out);
    if (!blit_dst_ok(fmt)) {
        static bool told = false;
        if (!told) {
            told = true;
            NXLOG_E(kTag, "output format %s (Vulkan %d) cannot be blitted to: the DLSS output stays as it was", fmt_name(out), static_cast<int>(fmt));
        }
        return;
    }
    gpu::tex_ensure_current(out, f);
    gpu::tex_transition(out, f, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    gpu::tex_mark_gpu_written(out);
    VkImageBlit b = {};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[1] = {static_cast<int32_t>(src.extent.width), static_cast<int32_t>(src.extent.height), 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, o.level, o.layer, 1};
    const int32_t x0 = std::min(static_cast<int32_t>(std::lround(o.x * o.sc)), o.img_w - 1), y0 = std::min(static_cast<int32_t>(std::lround(o.y * o.sc)), o.img_h - 1);
    b.dstOffsets[0] = {x0, y0, 0};
    b.dstOffsets[1] = {std::min(x0 + static_cast<int32_t>(src.extent.width), o.img_w), std::min(y0 + static_cast<int32_t>(src.extent.height), o.img_h), 1};
    vkCmdBlitImage(f.cb, src.image, VK_IMAGE_LAYOUT_GENERAL, gpu::tex_image(out), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_NEAREST);
    gpu::memory_barrier(f.cb);
}

float half_to_float(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : e == 31 ? 0.0f : std::ldexp(static_cast<float>(m | 0x400), static_cast<int>(e) - 25);
    return (h & 0x8000) ? -v : v;
}

// [dlss] dump_every = N: every N-th evaluation of a network and the one after it write, in the working directory, the input region and the
// result as PNG files (dlss_<network>_<evaluation>_in.png / _out.png, exposed so that the input's mean luma shows as 0.18, x / (1 + x), sRGB)
// and, for checking the jitter and motion vector conventions on the game's frames, the raw input (_in.bin: rgba16f), the raw motion vectors
// (_mv.bin: rg16f, when the texture is) and _frame.txt (sizes, jitter, motion vector scale). [dlss] dump_max pairs at most.
// Debugging: the game hands over pre-exposure HDR, too dark to see as it is.
void maybe_dump(Frame& f, NetGpu& n, const DlssEval& e, const Region& c, const Region& rm, const gfx::Image& result,
                Texture* final_tex, const Region& ro) {
    static const int64_t every = config_get_int("dlss", "dump_every", 0);
    // [dlss] dump_max = N (default 8): how many pairs a run may write. A screen that only appears later in a
    // session (the blade tuning screen of issue #96) needs a cap high enough to still be reached.
    static const int64_t max_dumps = config_get_int("dlss", "dump_max", 8);
    static int dumps = 0;
    const bool second = n.dump_pair;
    n.dump_pair = false;
    if (every <= 0 || dumps >= max_dumps || (!second && n.evals % static_cast<uint64_t>(every) != 0)) return;
    const VkExtent2D in{static_cast<uint32_t>(std::lround(c.w * c.sc)), static_cast<uint32_t>(std::lround(c.h * c.sc))};
    if (!ensure(n, n.dump, in, VK_FORMAT_R16G16B16A16_SFLOAT, "dlss dump")) return;
    Texture* color = e.color.get();
    Texture* motion = e.motion.get();
    const bool mv_ok = motion && gpu::tex_gpu(motion) && gpu::tex_vk_format(motion) == VK_FORMAT_R16G16_SFLOAT && rm.w > 0 && rm.h > 0;
    const VkExtent2D mv{mv_ok ? static_cast<uint32_t>(std::lround(rm.w * rm.sc)) : 0u, mv_ok ? static_cast<uint32_t>(std::lround(rm.h * rm.sc)) : 0u};
    gfx::Buffer bi, bo, bm, bf;
    const VkDeviceSize si = static_cast<VkDeviceSize>(in.width) * in.height * 8, so = static_cast<VkDeviceSize>(result.extent.width) * result.extent.height * 8;
    bool ok = device().create_buffer(si, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &bi, "dlss dump") == VK_SUCCESS &&
              device().create_buffer(so, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &bo, "dlss dump") == VK_SUCCESS &&
              // _final.png: the game's own output texture after blit_out. The result image can be correct while the picture on
              // screen is black (issue #96), and this is the one place that tells the two apart.
              device().create_buffer(so, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &bf, "dlss dump") == VK_SUCCESS;
    if (ok && mv_ok)
        ok = device().create_buffer(static_cast<VkDeviceSize>(mv.width) * mv.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &bm,
                                    "dlss dump") == VK_SUCCESS;
    if (!ok) {
        for (gfx::Buffer* b : {&bi, &bo, &bm})
            if (b->buffer) device().destroy_buffer(*b);
        return;
    }
    ++dumps;
    n.dump_pair = !second;
    to_general(f, n.dump);
    gpu::tex_transition(color, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkImageBlit b = {};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c.level, c.layer, 1};
    const int32_t x0 = static_cast<int32_t>(std::lround(c.x * c.sc)), y0 = static_cast<int32_t>(std::lround(c.y * c.sc));
    b.srcOffsets[0] = {x0, y0, 0};
    b.srcOffsets[1] = {x0 + static_cast<int32_t>(in.width), y0 + static_cast<int32_t>(in.height), 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstOffsets[1] = {static_cast<int32_t>(in.width), static_cast<int32_t>(in.height), 1};
    vkCmdBlitImage(f.cb, gpu::tex_image(color), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, n.dump.img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
    if (mv_ok) {
        gpu::tex_transition(motion, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy r = {};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, rm.level, rm.layer, 1};
        r.imageOffset = {static_cast<int32_t>(std::lround(rm.x * rm.sc)), static_cast<int32_t>(std::lround(rm.y * rm.sc)), 0};
        r.imageExtent = {mv.width, mv.height, 1};
        vkCmdCopyImageToBuffer(f.cb, gpu::tex_image(motion), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, bm.buffer, 1, &r);
    }
    gpu::memory_barrier(f.cb);
    VkBufferImageCopy r = {};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {in.width, in.height, 1};
    vkCmdCopyImageToBuffer(f.cb, n.dump.img.image, VK_IMAGE_LAYOUT_GENERAL, bi.buffer, 1, &r);
    r.imageExtent = {result.extent.width, result.extent.height, 1};
    vkCmdCopyImageToBuffer(f.cb, result.image, VK_IMAGE_LAYOUT_GENERAL, bo.buffer, 1, &r);
    // The game's output texture, read back through the same region blit_out wrote: black here with a good result image means
    // the blit (or the layout it assumes) is wrong, not the upscaler.
    bool final_ok = false;
    if (final_tex && gpu::tex_gpu(final_tex) && ro.w > 0 && ro.h > 0) {
        gpu::tex_ensure_current(final_tex, f);
        gpu::tex_transition(final_tex, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        const int32_t fx0 = static_cast<int32_t>(std::lround(ro.x * ro.sc)), fy0 = static_cast<int32_t>(std::lround(ro.y * ro.sc));
        VkImageBlit fb = {};
        fb.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, ro.level, ro.layer, 1};
        fb.srcOffsets[0] = {std::max(0, fx0), std::max(0, fy0), 0};
        fb.srcOffsets[1] = {std::min(fx0 + static_cast<int32_t>(result.extent.width), ro.img_w),
                            std::min(fy0 + static_cast<int32_t>(result.extent.height), ro.img_h), 1};
        fb.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        fb.dstOffsets[1] = {static_cast<int32_t>(result.extent.width), static_cast<int32_t>(result.extent.height), 1};
        if (fb.srcOffsets[1].x > fb.srcOffsets[0].x && fb.srcOffsets[1].y > fb.srcOffsets[0].y) {
            vkCmdBlitImage(f.cb, gpu::tex_image(final_tex), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, n.dump.img.image,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &fb, VK_FILTER_NEAREST);
            gpu::memory_barrier(f.cb);
            VkBufferImageCopy fr = {};
            fr.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            fr.imageExtent = {result.extent.width, result.extent.height, 1};
            vkCmdCopyImageToBuffer(f.cb, n.dump.img.image, VK_IMAGE_LAYOUT_GENERAL, bf.buffer, 1, &fr);
            final_ok = true;
        }
    }
    gpu::memory_barrier(f.cb);
    char base[64], info[256];
    std::snprintf(base, sizeof base, "dlss_%u_%llu", e.net->id, static_cast<unsigned long long>(n.evals));
    std::snprintf(info, sizeof info, "in %u %u\njitter %.9g %.9g\nmv_scale %.9g %.9g\nmv %u %u\nmv_low_res %d\nreset %d\n", in.width, in.height,
                  static_cast<double>(e.p.jitter[0]), static_cast<double>(e.p.jitter[1]), static_cast<double>(e.p.mv_scale[0]),
                  static_cast<double>(e.p.mv_scale[1]), mv.width, mv.height, (e.net->flags & kDlssFlagMvLowRes) ? 1 : 0, e.p.reset);
    const VkExtent2D out{result.extent.width, result.extent.height};
    f.defer([bi, bo, bm, bf, final_ok, in, out, mv, name = std::string(base), text = std::string(info)]() mutable {
        device().invalidate_buffer(bi);
        device().invalidate_buffer(bo);
        if (bm.buffer) device().invalidate_buffer(bm);
        auto write = [](const std::string& path, const void* data, size_t size) {
            if (FILE* fp = std::fopen(path.c_str(), "wb")) {
                std::fwrite(data, 1, size, fp);
                std::fclose(fp);
            }
        };
        write(name + "_in.bin", bi.mapped, static_cast<size_t>(in.width) * in.height * 8);
        if (bm.buffer) write(name + "_mv.bin", bm.mapped, static_cast<size_t>(mv.width) * mv.height * 4);
        write(name + "_frame.txt", text.data(), text.size());
        const uint16_t* a = static_cast<const uint16_t*>(bi.mapped);
        double sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(in.width) * in.height; ++i) {
            const double l = 0.25 * half_to_float(a[i * 4]) + 0.5 * half_to_float(a[i * 4 + 1]) + 0.25 * half_to_float(a[i * 4 + 2]);
            if (std::isfinite(l) && l > 0) sum += l;
        }
        const double mean = sum / std::max<size_t>(1, static_cast<size_t>(in.width) * in.height);
        const float k = mean > 1e-12 ? static_cast<float>(0.18 / mean) : 1.0f;
        auto encode = [k](const gfx::Buffer& buf, VkExtent2D ext, const std::string& path) {
            const uint16_t* src = static_cast<const uint16_t*>(buf.mapped);
            std::vector<uint8_t> px(static_cast<size_t>(ext.width) * ext.height * 4);
            for (size_t i = 0; i < static_cast<size_t>(ext.width) * ext.height; ++i) {
                for (int ch = 0; ch < 3; ++ch) {
                    float v = std::max(0.0f, half_to_float(src[i * 4 + ch]) * k);
                    v = v / (1.0f + v);
                    v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
                    px[i * 4 + ch] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
                }
                px[i * 4 + 3] = 255;
            }
            return gfx::save_png_rgba(path, px.data(), ext.width, ext.height);
        };
        const bool wi = encode(bi, in, name + "_in.png"), wo = encode(bo, out, name + "_out.png");
        const bool wf = final_ok && encode(bf, out, name + "_final.png");
        double weight = 0;   // the history's accumulated sample weight (taau): low where it was clipped away
        const uint16_t* o = static_cast<const uint16_t*>(bo.mapped);
        for (size_t i = 0; i < static_cast<size_t>(out.width) * out.height; ++i) weight += half_to_float(o[i * 4 + 3]);
        weight /= std::max<size_t>(1, static_cast<size_t>(out.width) * out.height);
        NXLOG_I(kTag, "dumped %s: input %ux%u %s, result %ux%u %s, motion %ux%u, exposure x%.4g (input mean luma %.4g), mean history weight %.2f",
                name.c_str(), in.width, in.height, wi ? "written" : "not written", out.width, out.height, wo ? "written" : "not written", mv.width, mv.height,
                static_cast<double>(k), mean, weight);
        for (gfx::Buffer* b : {&bi, &bo, &bm, &bf})
            if (b->buffer) device().destroy_buffer(*b);
        (void)wf;
    });
}

// The SDK upscalers through the TemporalFrame interface (temporal.h). False: nothing recorded.
bool run_sdk(Frame& f, Upscaler u, NetGpu& n, const DlssEval& e, const extras::TaauPrepFrame& prep, VkExtent2D in, VkExtent2D out, const float jitter[2],
             bool reset, float frame_ms) {
    extras::TemporalBackend* be = u == Upscaler::Fsr3 ? extras::temporal_ffx(&device()) : extras::temporal_dlss(&device(), e.net->method == 2);
    extras::Taau* t = extras::Taau::get(device());
    if (!be || !t) return false;
    nxrt::extras::note_upscaler_input_size(in.width, in.height);   // HANDOVER 95.17: dynres record scanner / pin
    if (!ensure(n, n.sdk[0], in, VK_FORMAT_R16G16B16A16_SFLOAT, "dlss sdk colour") || !ensure(n, n.sdk[1], in, VK_FORMAT_R32_SFLOAT, "dlss sdk depth") ||
        !ensure(n, n.sdk[2], in, VK_FORMAT_R16G16_SFLOAT, "dlss sdk motion") || !ensure(n, n.sdk[3], out, VK_FORMAT_R16G16B16A16_SFLOAT, "dlss sdk output"))
        return false;
    VkDescriptorSet set = f.alloc_set(t->prep_set_layout());
    if (!set) return false;
    gpu::memory_barrier(f.cb);
    for (Img& i : n.sdk) to_general(f, i);
    extras::TaauPrepFrame p = prep;
    p.out_color = n.sdk[0].img.view;
    p.out_depth = n.sdk[1].img.view;
    p.out_motion = n.sdk[2].img.view;
    t->record_prep(f.cb, set, p);
    gpu::memory_barrier(f.cb);
    auto as_input = [](const Img& i) {
        extras::TemporalImage o;
        o.image = i.img.image;
        o.view = i.img.view;
        o.format = i.img.format;
        o.width = i.img.extent.width;
        o.height = i.img.extent.height;
        o.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        return o;
    };
    extras::TemporalFrame tf;
    tf.color = as_input(n.sdk[0]);
    tf.depth = as_input(n.sdk[1]);
    tf.motion = as_input(n.sdk[2]);
    tf.output = as_input(n.sdk[3]);
    tf.render = in;
    tf.out = out;
    tf.jitter_px[0] = jitter[0];
    tf.jitter_px[1] = jitter[1];
    tf.mv_scale[0] = static_cast<float>(in.width);
    tf.mv_scale[1] = static_cast<float>(in.height);
    tf.reset = reset;
    tf.frame_ms = frame_ms;
    tf.sharpness = settings().sharpness;
    static const float kIdentity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (float* m : {tf.view_to_clip, tf.clip_to_view, tf.clip_to_prev_clip, tf.prev_clip_to_clip}) std::memcpy(m, kIdentity, sizeof kIdentity);
    if (!be->evaluate(f.cb, tf)) return false;
    gpu::memory_barrier(f.cb);
    // SDK upscalers (notably FSR 4.1.2, issue #96) may leave output.a = 0.0. XC2's DLSS output
    // texture is RGBA8 and screens like Blade Resonance composite it onto a black swapchain with
    // SRC_ALPHA, turning the 3D scene completely black while UI renders on top. Force alpha = 1.0
    // before blit_out().
    if (VkDescriptorSet aset = f.alloc_set(t->alpha_set_layout())) {
        t->record_opaque_alpha(f.cb, aset, n.sdk[3].img.view, out);
        gpu::memory_barrier(f.cb);
    }
    return true;
}

// [dlss] frame_gen: the frame's depth and motion (the SDK input images; made here when the built-in upscaler ran) go to frame generation.
void feed_framegen(Frame& f, NetGpu& n, const extras::TaauPrepFrame& prep, VkExtent2D in, const float jitter[2], bool reset, bool depth_inverted,
                   bool prepared) {
    if (!extras::framegen_requested()) return;
    if (!prepared) {
        extras::Taau* t = extras::Taau::get(device());
        if (!t || !ensure(n, n.sdk[0], in, VK_FORMAT_R16G16B16A16_SFLOAT, "dlss sdk colour") || !ensure(n, n.sdk[1], in, VK_FORMAT_R32_SFLOAT, "dlss sdk depth") ||
            !ensure(n, n.sdk[2], in, VK_FORMAT_R16G16_SFLOAT, "dlss sdk motion"))
            return;
        VkDescriptorSet set = f.alloc_set(t->prep_set_layout());
        if (!set) return;
        gpu::memory_barrier(f.cb);
        for (int i = 0; i < 3; ++i) to_general(f, n.sdk[i]);
        extras::TaauPrepFrame p = prep;
        p.out_color = n.sdk[0].img.view;
        p.out_depth = n.sdk[1].img.view;
        p.out_motion = n.sdk[2].img.view;
        t->record_prep(f.cb, set, p);
    }
    gpu::memory_barrier(f.cb);
    extras::FramegenInputs fi;
    fi.depth = n.sdk[1].img.image;
    fi.motion = n.sdk[2].img.image;
    fi.size = in;
    fi.jitter_px[0] = jitter[0];
    fi.jitter_px[1] = jitter[1];
    fi.reset = reset;
    fi.depth_inverted = depth_inverted;
    fi.flip_y = gpu::window_origin_mode() == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
    extras::framegen_inputs(&device(), f.cb, fi);
    gpu::fg_note_eval();
}

}  // namespace

void dlss_replay(gpu::Frame& f, Obj* o) {
    auto* e = static_cast<DlssEval*>(o);
    if (!e || !e->net || !e->color || !e->output) return;
    Texture* color = e->color.get();
    Texture* out = e->output.get();
    // [gfx] render_scale: the output is a render target in all but name (the game's post passes read it like one), but this evaluation
    // is often its first use, which made it unscaled: the upscaler then wrote 1080p into a 4K post chain. Make it at the attachment
    // scale (and bring an unscaled one up), so the upscaler writes the full output size.
    if (gpu::tex_gpu_attachment(out) && gpu::tex_scale_wanted(out) != gpu::tex_scale(out))
        gpu::tex_rescale(out, f, gpu::tex_scale_wanted(out), "the upscaler's output");
    if (extras::hdr_highlights_kept()) gpu::tex_promote_float(out, f, "the upscaler's output");   // the tone map's highlights above 1
    if (!gpu::tex_gpu(color) || !gpu::tex_gpu(out) || gpu::tex_is_depth(out)) {
        static bool told = false;
        if (!told) {
            told = true;
            NXLOG_W(kTag, "evaluation skipped: the input or output texture has no usable image (first)");
        }
        return;
    }
    DlssNet& net = *e->net;
    const DlssParamsData& p = e->p;
    const Settings& s = settings();
    const Region c = region_of(color, e->color_view, e->has_color_view, p.color_base[0], p.color_base[1], p.in_size[0], p.in_size[1]);
    const Region ro = region_of(out, e->output_view, e->has_output_view, p.out_base[0], p.out_base[1], e->out_w, e->out_h);
    if (c.w <= 0 || c.h <= 0 || ro.w <= 0 || ro.h <= 0) return;
    const VkExtent2D in_img{std::max(1u, static_cast<uint32_t>(std::lround(c.w * c.sc))), std::max(1u, static_cast<uint32_t>(std::lround(c.h * c.sc)))};
    const VkExtent2D out_img{std::max(1u, static_cast<uint32_t>(std::lround(ro.w * ro.sc))), std::max(1u, static_cast<uint32_t>(std::lround(ro.h * ro.sc)))};

    NetGpu& n = *net_gpu(net);
    History& h = n.hist;

    Upscaler u = s.upscaler;
    if (u == Upscaler::Auto) u = extras::temporal_dlss_started() ? Upscaler::Dlss : extras::temporal_ffx_available() ? Upscaler::Fsr3 : Upscaler::Taau;
    if (!n.logged) {
        n.logged = true;
        log_first(*e, c, ro, u);
    }

    // Automatic negative mip bias for temporal upscalers (DLSS / FSR / TAAU, issue #79):
    // AMD & NVIDIA guidelines use log2(render_w / output_w) - 1, quantized to 0.25 steps so dynamic resolution
    // does not churn the sampler cache. Applied in tuned_sampler() only to material samplers (game aniso > 1).
    {
        static const bool auto_mip_bias = config_get_bool("upscale", "auto_mip_bias", true);
        float want_bias = 0.0f;
        if (auto_mip_bias && u != Upscaler::Bilinear && in_img.width > 0 && out_img.width > 0) {
            const float raw = std::log2(static_cast<float>(in_img.width) / static_cast<float>(out_img.width)) - 1.0f;
            want_bias = std::clamp(std::round(raw * 4.0f) * 0.25f, -3.0f, 0.0f);
        }
        const float prev_bias = gfx::tuning().upscale_lod_bias.exchange(want_bias, std::memory_order_relaxed);
        if (prev_bias != want_bias) {
            NXLOG_I(kTag, "upscaler auto mip bias: %.2f (in %ux%u -> out %ux%u, auto_mip_bias=%d)",
                    static_cast<double>(want_bias), in_img.width, in_img.height, out_img.width, out_img.height,
                    auto_mip_bias ? 1 : 0);
        }
    }

    // inputs
    extras::TaauFrame tf;
    tf.color = sampled_view(f, color, e->color_view, e->has_color_view, c);
    tf.color_layout = gpu::tex_sampled_layout(color);
    if (!tf.color) return;
    tf.in_size[0] = static_cast<float>(in_img.width);
    tf.in_size[1] = static_cast<float>(in_img.height);
    tf.in_base[0] = std::floor(c.x * c.sc);
    tf.in_base[1] = std::floor(c.y * c.sc);
    const bool mv_low = (net.flags & kDlssFlagMvLowRes) != 0;
    const Region rd = e->depth ? region_of(e->depth.get(), e->depth_view, e->has_depth_view, p.depth_base[0], p.depth_base[1], c.w, c.h) : Region();
    const Region rm = e->motion ? region_of(e->motion.get(), e->motion_view, e->has_motion_view, p.mv_base[0], p.mv_base[1], mv_low ? c.w : ro.w,
                                            mv_low ? c.h : ro.h)
                                : Region();
    const Region rk = e->mask ? region_of(e->mask.get(), e->mask_view, e->has_mask_view, p.mask_base[0], p.mask_base[1], c.w, c.h) : Region();
    tf.depth = e->depth && gpu::tex_gpu(e->depth.get()) ? sampled_view(f, e->depth.get(), e->depth_view, e->has_depth_view, rd) : VK_NULL_HANDLE;
    tf.motion = e->motion && gpu::tex_gpu(e->motion.get()) ? sampled_view(f, e->motion.get(), e->motion_view, e->has_motion_view, rm) : VK_NULL_HANDLE;
    tf.mask = e->mask && gpu::tex_gpu(e->mask.get()) ? sampled_view(f, e->mask.get(), e->mask_view, e->has_mask_view, rk) : VK_NULL_HANDLE;
    if (tf.depth) {
        tf.depth_layout = gpu::tex_sampled_layout(e->depth.get());
        aux_map(rd, tf.in_size[0], tf.in_size[1], tf.depth_base, tf.depth_ratio);
    } else {   // no depth: the colour stands in (dilation then follows its red channel)
        tf.depth = tf.color;
        tf.depth_layout = tf.color_layout;
        std::copy(tf.in_base, tf.in_base + 2, tf.depth_base);
    }
    const float jx = p.jitter[0] * ((s.jitter_sign & 1) ? -1.0f : 1.0f), jy = p.jitter[1] * ((s.jitter_sign & 2) ? -1.0f : 1.0f);
    tf.jitter[0] = jx * c.sc;
    tf.jitter[1] = jy * c.sc;
    // motion: texel * the game's scale = pixels at the render (or output) size, previous minus current -> output UV
    const float mvx = (s.mv_sign & 1) ? -1.0f : 1.0f, mvy = (s.mv_sign & 2) ? -1.0f : 1.0f;
    const float mv_px_w = static_cast<float>(mv_low ? c.w : ro.w), mv_px_h = static_cast<float>(mv_low ? c.h : ro.h);
    if (tf.motion) {
        tf.motion_layout = gpu::tex_sampled_layout(e->motion.get());
        tf.mv_scale[0] = mvx * p.mv_scale[0] / mv_px_w;
        tf.mv_scale[1] = mvy * p.mv_scale[1] / mv_px_h;
        tf.mv_output_res = !mv_low;
        if (mv_low) aux_map(rm, tf.in_size[0], tf.in_size[1], tf.mv_base, tf.mv_ratio);
        else aux_map(rm, static_cast<float>(out_img.width), static_cast<float>(out_img.height), tf.mv_base, tf.mv_ratio);
    } else {   // no motion vectors: zero motion
        tf.motion = tf.color;
        tf.motion_layout = tf.color_layout;
    }
    if ((net.flags & kDlssFlagMvJittered) && h.valid) {   // the vectors carry the jitter difference: take it out
        tf.mv_offset[0] = (jx - h.prev_jitter[0]) / static_cast<float>(c.w);
        tf.mv_offset[1] = (jy - h.prev_jitter[1]) / static_cast<float>(c.h);
    }
    if (tf.mask) {
        tf.mask_layout = gpu::tex_sampled_layout(e->mask.get());
        aux_map(rk, tf.in_size[0], tf.in_size[1], tf.mask_base, tf.mask_ratio);
    }
    tf.depth_inverted = (net.flags & kDlssFlagDepthInverted) != 0;
    tf.max_weight = s.max_weight;

    const auto now = std::chrono::steady_clock::now();
    const float frame_ms = n.evals ? static_cast<float>(std::clamp(std::chrono::duration<double, std::milli>(now - n.last).count(), 1.0, 100.0)) : 16.7f;
    n.last = now;
    bool reset = p.reset != 0 || !h.valid || h.size.width != out_img.width || h.size.height != out_img.height;
    // the upscaler was swapped through [dlss] hot_file: the history came from another one, so it starts over
    if (n.gen != g_upscaler_gen.load()) {
        n.gen = g_upscaler_gen.load();
        reset = true;
        h.valid = false;
        n.sdk_valid = false;
    }

    // the SDK upscalers' inputs (also frame generation's)
    extras::TaauPrepFrame prep;
    prep.color = tf.color;
    prep.depth = tf.depth;
    prep.motion = tf.motion;
    prep.color_layout = tf.color_layout;
    prep.depth_layout = tf.depth_layout;
    prep.motion_layout = tf.motion_layout;
    std::copy(tf.in_size, tf.in_size + 2, prep.in_size);
    std::copy(tf.in_base, tf.in_base + 2, prep.in_base);
    std::copy(tf.depth_base, tf.depth_base + 2, prep.depth_base);
    std::copy(tf.depth_ratio, tf.depth_ratio + 2, prep.depth_ratio);
    if (mv_low) {   // render UV units for the SDK: the shader's scale is in output UV, which is the same fraction of the picture
        std::copy(tf.mv_scale, tf.mv_scale + 2, prep.mv_scale);
        std::copy(tf.mv_base, tf.mv_base + 2, prep.mv_base);
        std::copy(tf.mv_ratio, tf.mv_ratio + 2, prep.mv_ratio);
    } else if (tf.motion != tf.color) {
        std::copy(tf.mv_scale, tf.mv_scale + 2, prep.mv_scale);
        aux_map(rm, tf.in_size[0], tf.in_size[1], prep.mv_base, prep.mv_ratio);
    }
    std::copy(tf.mv_offset, tf.mv_offset + 2, prep.mv_offset);

    if (u == Upscaler::Fsr3 || u == Upscaler::Dlss) {
        const bool sdk_reset = p.reset != 0 || !n.sdk_valid || n.sdk_in.width != in_img.width || n.sdk_in.height != in_img.height ||
                               n.sdk_out.width != out_img.width || n.sdk_out.height != out_img.height;
        if (run_sdk(f, u, n, *e, prep, in_img, out_img, tf.jitter, sdk_reset, frame_ms)) {
            blit_out(f, n.sdk[3].img, out, ro);
            n.sdk_valid = true;
            n.sdk_in = in_img;
            n.sdk_out = out_img;
            n.resets += sdk_reset ? 1 : 0;
            h.valid = false;   // the built-in history is not kept meanwhile
            ++n.evals;
            g_in_use.store(static_cast<int>(u), std::memory_order_relaxed);
            maybe_dump(f, n, *e, c, rm, n.sdk[3].img, out, ro);
            feed_framegen(f, n, prep, in_img, tf.jitter, sdk_reset, tf.depth_inverted, true);
            return;
        }
        static bool told = false;
        if (!told) {
            told = true;
            NXLOG_W(kTag, "%s is not available (its DLLs, or Streamline not started): the built-in temporal upscaler is used", upscaler_name(u));
        }
    }

    // the built-in upscaler
    extras::Taau* t = extras::Taau::get(device());
    if (!t) return;
    if (!ensure(n, h.img[0], out_img, VK_FORMAT_R16G16B16A16_SFLOAT, "dlss history") ||
        !ensure(n, h.img[1], out_img, VK_FORMAT_R16G16B16A16_SFLOAT, "dlss history"))
        return;
    if (h.size.width != out_img.width || h.size.height != out_img.height) {
        h.size = out_img;
        h.valid = false;
        reset = true;
    }
    VkDescriptorSet set = f.alloc_set(t->set_layout());
    if (!set) return;
    gpu::memory_barrier(f.cb);
    to_general(f, h.img[0]);
    to_general(f, h.img[1]);
    Img& dst = h.img[h.cur];
    Img& src = h.img[h.cur ^ 1u];
    tf.history_in = src.img.view;
    tf.history_out = dst.img.view;
    tf.out = out_img;
    tf.reset = reset;
    tf.spatial = u == Upscaler::Bilinear;
    t->record(f.cb, set, tf);
    gpu::memory_barrier(f.cb);
    blit_out(f, dst.img, out, ro);
    h.cur ^= 1u;
    h.valid = !tf.spatial;
    n.sdk_valid = false;
    h.prev_jitter[0] = jx;
    h.prev_jitter[1] = jy;
    n.resets += reset ? 1 : 0;
    ++n.evals;
    const Upscaler active = tf.spatial ? Upscaler::Bilinear : Upscaler::Taau;
    g_in_use.store(static_cast<int>(active), std::memory_order_relaxed);
    gpu::counters().dispatches.fetch_add(1, std::memory_order_relaxed);
    maybe_dump(f, n, *e, c, rm, dst.img, out, ro);
    feed_framegen(f, n, prep, in_img, tf.jitter, reset, tf.depth_inverted, false);
    if (n.evals % 1800 == 0)
        NXLOG_I(kTag, "network %u: %llu evaluations, %llu history resets, %s %ux%u -> %ux%u", net.id, static_cast<unsigned long long>(n.evals),
                static_cast<unsigned long long>(n.resets), upscaler_name(active), in_img.width, in_img.height, out_img.width, out_img.height);
}

const char* dlss_upscaler_in_use() {
    const int u = g_in_use.load(std::memory_order_relaxed);
    return u < 0 ? nullptr : upscaler_name(static_cast<Upscaler>(u));
}

void dlss_replay_shutdown() {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (NetGpu* n : g_nets) {
            destroy_images(*n);
            n->dead = true;
        }
        g_nets.clear();
    }
    extras::Taau::shutdown();
}

void dlss_set_live_impl(const char* upscaler, const char* fsr_version, float sharpness) {
    bool bump_gen = false;
    std::string log_up;
    std::string log_fv;
    float log_sharp = -1.0f;
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        if (upscaler && *upscaler) {
            const Upscaler u = parse_upscaler_str(upscaler);
            const std::string key = upscaler_key(u);
            const Upscaler prev_u = g_live_has_upscaler ? g_live_upscaler : parse_upscaler_str(config_get_str("dlss", "upscaler", "taau"));
            if (prev_u != u) bump_gen = true;
            g_live_upscaler = u;
            g_live_upscaler_str = key;
            g_live_has_upscaler = true;
            log_up = key;
        }
        if (fsr_version && *fsr_version) {
            const std::string prev_fv = !g_live_fsr_version.empty() ? g_live_fsr_version : config_get_str("upscale", "fsr_version", "3.1");
            if (prev_fv != fsr_version) bump_gen = true;
            g_live_fsr_version = fsr_version;
            log_fv = fsr_version;
        }
        if (sharpness >= 0.0f) {
            g_live_sharpness = std::clamp(sharpness, 0.0f, 1.0f);
            g_live_has_sharp = true;
            log_sharp = g_live_sharpness;
        }
    }
    if (!log_fv.empty()) extras::set_ffx_version_live(log_fv.c_str());
    if (bump_gen) g_upscaler_gen.fetch_add(1, std::memory_order_release);
    g_live_seq.fetch_add(1, std::memory_order_release);
    NXLOG_I("nvn.dlss", "live menu switch: upscaler=%s fsr_version=%s sharpness=%.2f (reset=%d)",
            log_up.empty() ? "-" : log_up.c_str(), log_fv.empty() ? "-" : log_fv.c_str(),
            static_cast<double>(log_sharp), bump_gen ? 1 : 0);
}

DlssLiveInfo dlss_live_info_impl() {
    DlssLiveInfo info;
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        info.upscaler = g_live_has_upscaler ? g_live_upscaler_str : config_get_str("dlss", "upscaler", "taau");
        info.fsr_version = !g_live_fsr_version.empty() ? g_live_fsr_version : config_get_str("upscale", "fsr_version", "3.1");
        info.sharpness = g_live_has_sharp ? g_live_sharpness
                                          : static_cast<float>(std::clamp(config_get_double("dlss", "sharpness", 0.0), 0.0, 1.0));
    }
    info.fsr_available = extras::temporal_ffx_available();
    const int cur = g_in_use.load(std::memory_order_relaxed);
    if (cur == static_cast<int>(Upscaler::Fsr3)) {
        const std::string v = extras::temporal_ffx_active_version();
        info.active_name = v.empty() ? "FSR 3" : v;
    } else if (cur >= 0) {
        info.active_name = upscaler_name(static_cast<Upscaler>(cur));
    }
    return info;
}

}  // namespace nxrt::nvn::be

namespace nxrt::nvn {

void set_dlss_live(const char* upscaler, const char* fsr_version, float sharpness) {
    be::dlss_set_live_impl(upscaler, fsr_version, sharpness);
}

DlssLiveInfo dlss_live_info() {
    return be::dlss_live_info_impl();
}

}  // namespace nxrt::nvn
