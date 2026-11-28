// SPDX-License-Identifier: GPL-3.0-or-later
// The command list interpreter of the NVN back end (render thread). Walks recorded lists in order, keeps the GPU state the
// console keeps (it persists across lists and submissions), and turns it into Vulkan: dynamic-rendering passes opened lazily on the
// first draw or clear of a render target set, graphics pipelines keyed by the baked state, descriptor sets built from the
// translated shaders' binding tables (nvn_shaders.h), vertex and index data straight from the imported pool memory.
// Lessons applied from SuperMarioWonderRecomp docs/nvn/DEBUG.md and DEBUG-2.md (see work notes in docs/S4d.md): viewport flip by a
// negative-height viewport, a memory barrier after every pass, textures sampled by a draw are made current before the pass,
// handles never dropped by count, deferred bind arrays read at record time.
#include <atomic>
#include <algorithm>
#include <bit>
#include <deque>
#include <mutex>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <tuple>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <intrin.h>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/frame_limiter.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/gfx_tuning.h"
#include "nxrt/log.h"
#include "api_dlss.h"
#include "nvn_decoded.h"
#include "nvn_enums.h"
#include "nvn_hudless.h"
#include "nvn_format.h"
#include "nvn_mask_clear.h"
#include "nvn_gpu_internal.h"
#include "nvn_tess_gen.h"
#include "nxrt/lowlat.h"
#include "extras/hdr_live.h"
#include "extras/hdr_nvn.h"
#include "extras/fsr1_spatial.h"
#include "extras/spatial_upscaler.h"
#include "extras/temporal.h"
#include "nvn_pipeline.h"
#include "nvn_spirv_patch.h"
#include "nvn_shaders.h"
#include "nvn_texprep.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.replay";
constexpr int kMaxDepth = 8;

enum Stage { kVertex = 0, kFragment = 1, kGeometry = 2, kTessCtl = 3, kTessEval = 4, kCompute = 5 };

struct GState {
    const ShaderInfo* stage[6] = {};
    struct Bind { uint64_t addr = 0, size = 0; };
    Bind vtx[16], ubo[6][16], ssbo[6][16], xfb[4];
    uint64_t tex[6][64] = {};
    uint64_t img[6][16] = {};
    Ref<TexturePool> tpool;
    Ref<SamplerPool> spool;
    VertexAttribInfo attr[16];
    uint32_t attr_count = 0;
    VertexStreamInfo stream[16];
    uint32_t stream_count = 0;
    BlendInfo blend[8];
    ColorInfo color;
    ChannelMaskInfo chmask;
    DepthStencilInfo ds;
    PolygonInfo poly;
    MultisampleInfo ms;
    float vp[16][4] = {};
    uint32_t nvp = 0;
    int32_t sc[16][4] = {};
    uint32_t nsc = 0;
    float dr[16][2] = {};
    bool depth_bounds = false;
    float db_near = 0, db_far = 1;
    uint32_t stencil_ref[2] = {0, 0}, stencil_vmask[2] = {0xff, 0xff}, stencil_wmask[2] = {0xff, 0xff};
    float blend_color[4] = {};
    float line_width = 1, point_size = 1, alpha_ref = 0;
    float bias_factor = 0, bias_units = 0, bias_clamp = 0;
    uint32_t sample_mask = 0xffffffff;
    bool rasterizer_discard = false, depth_clamp = false, prim_restart = false, render_enable = true;
    uint32_t restart_index = 0;
    // tessellation: nvnCommandBufferSetPatchSize (control points per patch; GL default 3) and the fixed levels of
    // nvnCommandBufferSetInner/OuterTessellationLevels (used by NVN only when the program has no control shader)
    uint32_t patch_size = 3;
    float tess_outer[4] = {1, 1, 1, 1}, tess_inner[2] = {1, 1};
    // render targets
    uint32_t ncolor = 0;
    Ref<Texture> color_tex[8];
    TextureViewInfo color_view[8];
    bool color_has_view[8] = {};
    uint32_t color_level[8] = {}, color_layer[8] = {}, color_layers[8] = {};   // level, first layer, layer count each colour target is attached with (trace)
    Ref<Texture> depth_tex;
    TextureViewInfo depth_view;
    bool depth_has_view = false;
    int window_origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
    int depth_mode = NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
    // NVN2 push constants (Op::PushConstants): graphics slot n is constant bank 24 + n; compute slot n sits in c[0] at 0x10 + 0x100 * n
    uint8_t push_gfx[8][256] = {};
    uint8_t push_cmp[8 * 256] = {};
};

GState S;
UploadAlloc g_push_buf[8];          // this submission's copy of each graphics push constant slot (made when a draw binds it)
UploadAlloc g_cs_c0;                // the c[0] of the dispatch being recorded (NVN2 compute: work-group counts, push constants)

struct Target {
    Texture* tex = nullptr;
    uint32_t level = 0, layer = 0, layers = 1;
    VkFormat view_format = VK_FORMAT_UNDEFINED;
    bool operator==(const Target& o) const { return tex == o.tex && level == o.level && layer == o.layer && layers == o.layers && view_format == o.view_format; }
};

struct Pass {
    bool open = false;
    bool targets_clean = false; // true once P.color/P.depth match S (cleared on Op::SetRenderTargets / end_pass)
    bool int_outputs = false;   // int_outputs_wanted() when targets_clean was verified
    uint32_t ncolor = 0;
    Target color[8];
    Target depth;
    VkExtent2D extent = {0, 0};
    VkFormat color_fmt[8] = {};
    VkFormat depth_fmt = VK_FORMAT_UNDEFINED;
    bool has_stencil = false;
    bool depth_ro = false;      // the depth attachment is in DEPTH_STENCIL_READ_ONLY_OPTIMAL (it is also sampled by the draws of this pass)
    bool depth_general = false; // ... and the pass also writes its stencil: GENERAL, the only layout that is both (see depth_pass_layout)
    uint32_t color_general = 0; // bit i: colour attachment i is also a storage image of the draw, so it has to be GENERAL
    uint32_t color_feedback = 0; // bit i: colour attachment i is in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT (sampled in-place, zero copy)
    uint32_t samples = 1;
    float scale = 1.0f;         // resolution scale of the attachments ([gfx] render_scale): viewports and scissors are multiplied by it
};

Pass P;
uint16_t g_vb_dirty_mask = 0xFFFF; // bit k: vertex stream k changed since last draw (skip resolve_read on unchanged streams)
uint16_t g_ubo_dirty_mask[6] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF}; // bit k: stage UBO slot k changed since last set 0 bind/push (NS1 XC2_BIND_SKIP)
uint64_t g_ubo_bind_gen = 1;       // bumped on UBO binds; fb_memo ignores it for the layouts with !has_handle_in_ubo
bool g_pending_write_mem = false;  // one or more do_write_memory (vkCmdUpdateBuffer) writes awaiting narrow transfer_post_barrier
uint64_t g_wm_last_addr = 0, g_wm_last_size = 0;
// [nvn] pass_barriers = tracked (S38). A pass whose draws can only have written its attachments (no storage buffers or images, no stream-out)
// does not end with a barrier: its written attachments are remembered here, and the next command that could depend on them gets one. A
// layout change of such an image is that barrier for the image (narrow masks, set_narrow_barriers); a later pass that renders to it again in
// the same layout, and every command other than a pass or a layout change, get a full barrier first (end_pass).
struct PendingWrite { Ref<Texture> tex; VkImageLayout layout; };
std::vector<PendingWrite> g_pending;
bool g_tracked = false;         // this submission uses tracked barriers
bool g_pass_unsafe = false;     // a draw of the open pass may write memory other than its attachments
bool g_prog_writes = false;     // the bound program has storage buffer or storage image bindings
bool g_mirror_stale = false;    // a GPU write made pages of a pool copy stale while a draw was being set up: bind again at the next draw

uint32_t g_pass_draws = 0;      // draws recorded in the open pass (GPU pass profiler)
uint64_t g_pass_verts = 0;      // vertices (indices for indexed draws) times instances of those draws
Frame* F = nullptr;
inline void flush_write_mem_barrier() {
    if (g_pending_write_mem && F) {
        transfer_post_barrier(F->cb);
        g_pending_write_mem = false;
        g_wm_last_addr = g_wm_last_size = 0;
    }
}
// This submission's descriptor sets by content key: open addressing, emptied by moving the generation (an unordered_map's node per set
// and its lookups were ~1.7 % of the render thread; rt agent, notes/rt.md). Grows when half full.
struct SetCache {
    struct Slot { uint64_t key = 0, gen = 0; VkDescriptorSet ds = VK_NULL_HANDLE; };
    std::vector<Slot> slots = std::vector<Slot>(8192);
    uint64_t gen = 1;
    size_t used = 0;
    static size_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return static_cast<size_t>(k); }
    VkDescriptorSet find(uint64_t key) const {
        const size_t m = slots.size() - 1;
        for (size_t i = mix(key) & m;; i = (i + 1) & m) {
            const Slot& sl = slots[i];
            if (sl.gen != gen) return VK_NULL_HANDLE;
            if (sl.key == key) return sl.ds;
        }
    }
    void put(uint64_t key, VkDescriptorSet ds) {
        if ((used + 1) * 2 > slots.size()) {
            std::vector<Slot> old;
            old.swap(slots);
            slots.assign(old.size() * 2, Slot());
            used = 0;
            for (const Slot& sl : old) if (sl.gen == gen) put(sl.key, sl.ds);
        }
        const size_t m = slots.size() - 1;
        for (size_t i = mix(key) & m;; i = (i + 1) & m) {
            Slot& sl = slots[i];
            if (sl.gen != gen) { sl = {key, gen, ds}; ++used; return; }
            if (sl.key == key) { sl.ds = ds; return; }
        }
    }
    void clear() { ++gen; used = 0; }
};
SetCache g_set_cache;
VkPipeline g_bound_pipeline = VK_NULL_HANDLE;
VkCommandBuffer g_bound_ds_cb = VK_NULL_HANDLE;
VkPipelineLayout g_bound_ds_layout = VK_NULL_HANDLE;
VkDescriptorSet g_bound_ds[5] = {};
struct BoundVbSlot { VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize off = 0, size = 0, stride = 0; uint64_t nvn_addr = 0, nvn_size = 0; bool valid = false; };
VkCommandBuffer g_bound_vb_cb = VK_NULL_HANDLE;
BoundVbSlot g_bound_vb[16] = {};
uint32_t g_vb_safe_mask = 0;    // bit b set when bound VB slot b is already GPU-safe (VRAM mirror, SnapMap snapshot, or dummy)
VkCommandBuffer g_bound_ib_cb = VK_NULL_HANDLE;
VkBuffer g_bound_ib_buf = VK_NULL_HANDLE;
VkDeviceSize g_bound_ib_off = 0;
VkIndexType g_bound_ib_type = VK_INDEX_TYPE_MAX_ENUM;
inline void invalidate_bound_state() {
    g_bound_ds_cb = VK_NULL_HANDLE;
    g_bound_ds_layout = VK_NULL_HANDLE;
    for (VkDescriptorSet& d : g_bound_ds) d = VK_NULL_HANDLE;
    g_bound_vb_cb = VK_NULL_HANDLE;
    for (BoundVbSlot& v : g_bound_vb) v.valid = false;
    g_vb_safe_mask = 0;
    g_bound_ib_cb = VK_NULL_HANDLE;
    g_bound_ib_buf = VK_NULL_HANDLE;
    g_bound_ib_type = VK_INDEX_TYPE_MAX_ENUM;
    for (uint16_t& m : g_ubo_dirty_mask) m = 0xFFFF;
}
inline void bind_index_buffer_cached(VkBuffer buf, VkDeviceSize off, VkIndexType type) {
    if (g_bound_ib_cb == F->cb && g_bound_ib_buf == buf && g_bound_ib_off == off && g_bound_ib_type == type) return;
    vkCmdBindIndexBuffer(F->cb, buf, off, type);
    g_bound_ib_cb = F->cb;
    g_bound_ib_buf = buf;
    g_bound_ib_off = off;
    g_bound_ib_type = type;
}
UploadAlloc g_support;
UploadAlloc g_draw_support;     // the support buffer the next descriptor sets bind: g_support, or one with render scales (support_for_draw)
std::unordered_map<uint64_t, UploadAlloc> g_support_scaled;   // this submission's support buffers with render scales, by content
MaskClear g_mask;
bool g_mask_ready = false, g_mask_ok = false;

// ---- [nvn] earlyz: the console's early depth test, emulated with the pass's own stencil buffer (HANDOVER 42.22.2) ----------------
// A fragment shader that writes a storage image makes a driver run the depth test *after* it (late-Z): the image writes of the fragments
// the test rejects land, their colour writes do not. The light list's OR shaders (fs eb74f4cb and its siblings) count the lights they
// registered in a storage image (t51), so rejected fragments spend a tile's four slots while the list (the colour target) stays empty -
// the black rectangle (HANDOVER 42.21). The console tests depth before the shader and never sees it. amdvlk faults while it builds a
// pipeline with EarlyFragmentTests for these shaders (42.21.3, 42.22.1), so the order is emulated at the draw instead:
//   1. the same geometry runs once more, through a variant of its pipeline whose colour writes are all masked off and whose storage
//      images are bound to a scratch image: nothing of it reaches the draw's targets. Its depth test is the draw's own (still late), but
//      the stencil operations run *after* that test, so they see the real result. The game's stencil test and write mask stay, plus one
//      bit of our own: ops fail KEEP, depth fail ZERO, pass INVERT. INVERT sets our zero bit, and it clears exactly the bits the game's
//      ZERO would have cleared (for the EQUAL test the draw uses, those bits are all 1 where the fragment passes - earlyz_wants checks it);
//      KEEP on the fail path is what the game's ZERO amounts to there (a failed one-bit EQUAL means that bit is already 0) and it is
//      required, or a second face of the same light volume would clear the bit its first face just set.
//   2. the real draw runs with its depth test off and its stencil test EQUAL on that bit, so only the fragments the depth test would have
//      kept reach the shader: image writes and colour writes agree again, tile for tile with the console.
//   3. its stencil operations zero that bit again, so the next draw starts from a clean one.
// OFF BY DEFAULT: see HANDOVER 44. The gate itself works (the pass's registration mask ends up with three bits per tile instead of
// four, i.e. the freed slot is real), but the freed slot is then taken by the OR draw whose per-light block carries index 0, and the
// tiles that lose their light to it come out dark - at the wharf that is a mottled, flickering dark patch on the roof and the crates
// (mask popcount 3 everywhere vs 4, list tiles holding light 0 where the untouched path holds light 0x16). Until that index-0
// registration is understood, this must stay off: it makes the picture worse, not better.
// [nvn] earlyz = 1|0 (default 0), [nvn] earlyz_bit = 0..7 (default 4: t43's pass uses 0x20, 0x40 and 0x80 only) and
// [nvn] earlyz_fs = the fragment shaders it applies to (hash prefixes, default the eleven light-list shaders below).
// the eleven fragment shaders of the light list's OR pass (HANDOVER 42.12, 42.21): the default of [nvn] earlyz_fs
constexpr const char* kEarlyzDefaultFs = "eb74f4cb,077ceb59,d3d6d5bf,a6db222b,1eb6d5fe,238b7791,fe2136f8,3c39992a,b8843fe5,766314ee,a629f4ac";
bool g_earlyz_on = false;
uint8_t g_earlyz_bit = 0x10;
bool g_earlyz_pending = false;      // this draw runs the two draws (the caller issues both)
bool g_earlyz_scratch = false;      // build_set binds the scratch storage image instead of the game's
VkPipeline g_earlyz_pipe = VK_NULL_HANDLE;
GraphicsKey g_pipe_key;             // the key the bound pipeline was made from (the pre-pass uses it with the colour writes masked off)
bool g_pipe_key_valid = false;
uint64_t g_draw_uint_samplers = 0;  // fragment refl.bindings bitmask bound to colour integer textures in this draw
uint64_t g_earlyz_draws = 0;        // draws emulated this session (trace summary)

// [nvn] earlyz_mask (default 1, HANDOVER 45.2): derive the registration mask from the list after every light-list draw. The OR
// shaders register a light in a storage image (t51), and a fragment shader that writes a storage image makes the driver run the
// depth/stencil test after it, so the fragments the test rejects still write the mask while their colour is dropped: the mask fills
// with lights the list never got, bitCount reaches four and every later light of that tile is skipped - the black rectangle. The
// colour IS gated by the test, so the list holds what the console's list holds, and the mask can simply be derived from it again:
//   mask = OR over the four slots of (flag == 0xff) ? 1 << (index & 31) : 0
// which is exactly the inverse of the shader's imageStore. One compute dispatch per list draw (640x360, a few microseconds); the
// pass is closed and reopened around it.
Ref<Texture> g_earlyz_mask_tex;     // the storage image this draw registers in (the pass's mask)
bool g_earlyz_mask_ready = false;   // the current draw is a list draw: rebuild after it
uint64_t g_earlyz_mask_draws = 0;

// What changed since the last draw. Ops set bits (mark_dirty); a draw handles and clears the groups it needs. A new submission, a
// new pass and anything that binds other pipelines sets them all (a new command buffer has no dynamic state, a new pass has other
// formats and extents).
// kDescUbo: only uniform buffers changed; the draw treats it as kDesc unless the uniform-only rebind applies ([nvn] ubo_only_rebind).
enum : uint32_t {
    kPipe = 1, kDesc = 2, kVb = 4, kDynVp = 8, kDynRaster = 16, kDynDepth = 32, kDynStencil = 64, kDynBias = 128, kDynBlend = 256, kDescUbo = 512,
    kAllDirty = 0x3ff
};
uint32_t g_dirty = kAllDirty;
// The program layout and command buffer of the last draw that bound all its descriptor sets (bind_sets): a uniform-only rebind keeps
// those sets only while both are still the same.
const ProgLayout* g_sets_pl = nullptr;
VkCommandBuffer g_sets_cb = VK_NULL_HANDLE;
// [nvn] ubo_only_rebind (default FALSE since HANDOVER 71: it fired on 129 draws a frame at no measured gain). A draw whose
// only binding change since the last draw is a uniform buffer pushes set 0 again and
// keeps the texture and storage sets it has bound, instead of resolving every texture and building every set again.
bool ubo_only_rebind_on() {
    static const bool on = nxrt::config_get_bool("nvn", "ubo_only_rebind", false);
    return on;
}
// [nvn] lean_draws (default true): cheaper forms of work every draw repeats, with the same Vulkan state as a result: the held-result
// check visits only the buffer slots that are bound (not all 180), the colour-feedback test compares image identities without looking
// every image up again, a resolved binding is read in place instead of copied, and a pushed set 0 pushes only the uniform buffers that
// differ from what the same layout pushed last in this command buffer (the delta push is adapted from Hookshot's bind_descriptors, GPL-3).
bool g_lean = true;
bool lean_setting() {
    static const bool on = nxrt::config_get_bool("nvn", "lean_draws", true);
    return on;
}
inline bool lean_on() { return g_lean; }
// The bound slots of S.vtx / S.ubo / S.ssbo as bit masks (kept by the BindBuffers op and by apply buffered state, made again at every submission).
uint32_t g_bm_vtx = 0, g_bm_ubo[6] = {}, g_bm_ssbo[6] = {};
// What set 0 of the graphics bind point holds from the last push: valid for one command buffer and one pipeline layout, until
// something else binds graphics state (begin_draw then finds no bound pipeline).
struct PushedSet0 {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    struct W { uint32_t binding = 0, type = 0; VkBuffer buffer = VK_NULL_HANDLE; VkDeviceSize offset = 0, range = 0; };
    W w[32] = {};
    uint32_t count = 0;
};
PushedSet0 g_pushed0;
uint64_t g_lean_push_full = 0, g_lean_push_delta = 0, g_lean_push_none = 0, g_lean_push_writes = 0;
// A command may write [addr, addr + size) of pool memory: device-local copies of those pages are stale, and bindings that point at them are
// made again (vertex buffers and descriptor sets).
// The ranges of pool memory the GPU writes, with the serial of the submission that did (a ring). The replay records commands: a range written
// by a submission that has not completed yet does not hold its result in game memory, so the CPU must not read it for a decision that needs
// the data (the arguments of an indirect draw it converts, cpu_view). A range that fell out of the ring while its submission was still running
// makes every range unreadable until that submission is done.
struct GpuWrite { uint64_t lo, hi, serial; };
constexpr size_t kGpuWriteRing = 1024;
GpuWrite g_gw[kGpuWriteRing];
size_t g_gw_next = 0, g_gw_count = 0;
uint64_t g_gw_lost_serial = 0;      // the newest serial among ranges dropped from the ring
void note_gw(uint64_t addr, uint64_t size) {
    if (!size || !F) return;
    GpuWrite& last = g_gw[(g_gw_next + kGpuWriteRing - 1) % kGpuWriteRing];
    if (g_gw_count && last.lo == addr && last.hi == addr + size) { last.serial = F->serial; return; }
    GpuWrite& w = g_gw[g_gw_next];
    if (g_gw_count == kGpuWriteRing) g_gw_lost_serial = std::max(g_gw_lost_serial, w.serial);
    else ++g_gw_count;
    w = {addr, addr + size, F->serial};
    g_gw_next = (g_gw_next + 1) % kGpuWriteRing;
}
// True when no submission still in flight (the one being recorded included) may write [addr, addr + size).
bool gw_settled(uint64_t addr, uint64_t size) {
    const uint64_t done = poll_completed();
    if (g_gw_lost_serial > done) return false;
    for (size_t i = 0; i < g_gw_count; ++i)
        if (g_gw[i].serial > done && addr < g_gw[i].hi && g_gw[i].lo < addr + size) return false;
    return true;
}

void gpu_wrote(uint64_t addr, uint64_t size, int use) {
    note_gw(addr, size);
    if (note_gpu_write(addr, size, use)) {
        g_dirty |= kVb | kDesc;
        g_vb_dirty_mask = 0xFFFF;
        g_mirror_stale = true;
    }
}
const ProgLayout* g_pl = nullptr;
VkPipeline g_pipe = VK_NULL_HANDLE;
VkPrimitiveTopology g_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
VkPrimitiveTopology g_pipe_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
bool g_restart = false, g_restart_valid = false;
uint32_t g_key_attr_count = 0;
uint8_t g_key_binding[16] = {};
uint32_t g_key_inst_mask = 0;   // bit b set when current pipeline binding b is per-instance (divisor > 0)
// Occlusion counter (NVN_COUNTER_TYPE_SAMPLES_PASSED): ResetCounter starts a Vulkan occlusion query outside any pass; it stays active across
// the passes that follow (the console counts continuously); ReportCounter ends it and copies its 64-bit result into the report.
constexpr uint32_t kOccSlots = 256;
VkQueryPool g_occ_pool = VK_NULL_HANDLE;
bool g_occ_tried = false;
uint32_t g_occ_next = 0;
int32_t g_occ_active = -1, g_occ_last = -1;

bool occ_ready() {
    if (g_occ_pool) return true;
    if (g_occ_tried || !device().features().occlusion_query) return false;
    g_occ_tried = true;
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qi.queryCount = kOccSlots;
    if (vkCreateQueryPool(vkdev(), &qi, nullptr, &g_occ_pool) != VK_SUCCESS) g_occ_pool = VK_NULL_HANDLE;
    return g_occ_pool != VK_NULL_HANDLE;
}

uint8_t g_key_dummy_binding = 0;      // 1 + the binding that feeds the defaults of unset vertex attributes (0: none)
bool g_want_depth_ro = false;         // some texture this draw samples is the current depth attachment
bool g_depth_general = false;         // ... and the draw writes that attachment's stencil too, so it needs GENERAL
uint32_t g_color_general = 0;         // bit i: colour attachment i is also a storage image of this draw (image load/store to a render target)
uint32_t g_color_feedback = 0;        // bit i: colour attachment i is sampled in-place via VK_EXT_attachment_feedback_loop_layout
std::vector<Texture*> g_storage_bound;   // the storage images prepare_textures moved to GENERAL for this draw
std::vector<Texture*> g_storage_unscale; // of those, the ones that must be unscaled ([gfx] render_scale: written, or bound to compute)

// The layout the depth-stencil attachment has for a pass. A pass that only samples it can leave it read-only; one that
// samples it AND writes its stencil cannot, because Vulkan has no "sampled and writable" depth layout - the descriptor
// would say DEPTH_STENCIL_READ_ONLY_OPTIMAL while the attachment is DEPTH_STENCIL_ATTACHMENT_OPTIMAL
// (VUID-vkCmdDrawIndexed-imageLayout-00344). GENERAL is the one layout valid for both uses.
VkImageLayout depth_pass_layout(bool depth_ro, bool depth_general) {
    return depth_ro      ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
           : depth_general ? VK_IMAGE_LAYOUT_GENERAL
                           : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
}
bool g_vb_bound_in_cb = false;        // vkCmdBindVertexBuffers2 was called in this command buffer (dynamic strides need one)

// [nvn] trace_frame = N (NXRT_NVN_TRACE_FRAME): every draw, clear, copy and dispatch of the frame that is presented as number N is logged
// at Info ("trace ..." lines): programs, render targets, sampled textures, blend and depth state. For finding out what a frame does.
bool g_xfb_active = false;
bool g_pipe_xfb = false;             // the bound pipeline is a stream-out variant
bool g_xfb_began = false;            // vkCmdBeginTransformFeedbackEXT is in effect in the open pass
bool g_xfb_have_counter = false;     // a counter buffer holds the bytes written so far: the next begin continues after them
gfx::Buffer g_xfb_counter;
// One byte counter per feedback control buffer (NVN's Begin/End/Draw take its address): slot * 16 bytes into g_xfb_counter. vkCmdEndTransformFeedbackEXT
// writes the bytes of the section there; DrawTransformFeedback reads them (vkCmdDrawIndirectByteCountEXT). written: the slot holds a count.
constexpr uint32_t kXfbSlots = 256, kXfbSlotBytes = 16;
std::unordered_map<uint64_t, uint32_t> g_xfb_slot_of;
uint32_t g_xfb_slot_next = 0, g_xfb_cur = 0;
bool g_xfb_slot_written[kXfbSlots] = {};
// Windowed GPU counters for the primitive counters (13 = primitives generated, 14 = feedback primitives written): a ResetCounter whose window
// (up to the matching ReportCounter, in the same list) holds an indirect or feedback draw starts a query the GPU counts those draws with;
// the CPU counts the other draws exactly (count_primitives) but cannot know the primitives of a draw whose arguments the GPU holds.
// [0] pipeline statistics (input assembly primitives; a converted quad counts as its two triangles), [1] transform feedback stream 0 query.
struct CounterWindow { VkQueryPool pool = VK_NULL_HANDLE; bool tried = false; uint32_t next = 0; int32_t active = -1, last = -1; };
constexpr uint32_t kCounterSlots = 64;
CounterWindow g_cw[2];
int g_dbg_depth = 0;                  // debug labels open in this submission (PushDebugGroup / PopDebugGroup)
// Primitive counters (NVN_COUNTER_TYPE_PRIMITIVES_GENERATED = 13, TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 14): counted on the CPU by every
// draw (count_primitives), process-wide like the console's, zeroed by ResetCounter of their type, dropped at shutdown.
uint64_t g_prims_generated = 0, g_prims_xfb = 0;
// Conditional render (SetRenderEnableConditional): the console compares the 64-bit counter words of two reports, at the address and
// 16 bytes after it (yuzu Maxwell3D::ProcessQueryCondition: IfEqual / IfNotEqual), when the command runs. The replay runs in GPU
// order, so the counters it writes itself (ReportCounter other than samples passed, ReportValue) are known here: g_reports keeps
// them by address. kReportUnknown marks an address the GPU writes later (samples passed): a condition on it renders, as before. An
// address the replay never wrote (the game's own memory) is read from the pool.
constexpr uint64_t kReportUnknown = ~0ull;
std::unordered_map<uint64_t, uint64_t> g_reports;
nxrt::GaugeCounter g_gauge_pipeline_binds("nvn.cnt.pipeline_binds");   // graphics pipeline binds (cumulative)
// Render-thread per-submission counter batches flushed at end of submission (avoids ~9k LOCK XADD/frame in hot draw path).
uint64_t g_sub_draws = 0, g_sub_lists = 0;
int64_t g_sub_pipe_binds = 0;
inline void flush_sub_counters() {
    if (g_sub_draws) { counters().draws.fetch_add(g_sub_draws, std::memory_order_relaxed); g_sub_draws = 0; }
    if (g_sub_lists) { counters().lists_replayed.fetch_add(g_sub_lists, std::memory_order_relaxed); g_sub_lists = 0; }
    if (g_sub_pipe_binds) { g_gauge_pipeline_binds.add(g_sub_pipe_binds); g_sub_pipe_binds = 0; }
}
VkQueryPool g_xq_pool = VK_NULL_HANDLE;     // transform feedback stream queries (counters 13 and 14)
bool g_trace = false;
struct TraceSnap { int index; int draw; std::string name; std::string desc; };
std::vector<TraceSnap> g_trace_snaps;
std::vector<int> g_trace_snap_draws;   // [nvn] trace_snap: draw numbers after which the targets are copied
int g_trace_snap_rt = 0;               // [nvn] trace_snap_rt: snapshot every draw that renders into this texture id (single value, kept)
std::vector<int> g_trace_snap_rts;     // the same as a comma list ("514,435,516"), so one run captures several targets
std::string g_trace_snap_fmt;          // [nvn] trace_snap_fmt = "RG32UI": snapshot every draw whose colour target has that format name (the
                                       // texture ids of a frame are not stable from run to run, the format of the light list is)
bool g_trace_snap_read = false;
std::vector<std::string> g_trace_snap_fs;
bool g_trace_snap_sampled = false;           // [nvn] trace_snap_sampled = 1: a snapshot draw also copies the textures that draw samples
std::vector<Texture*> g_trace_sampled;       // textures sampled by the last traced draw
std::vector<Ref<Texture>> g_trace_targets;   // the colour and depth targets drawn to in the traced frame, in order of first use
int64_t g_trace_frame = -2;
std::vector<int64_t> g_trace_extra;   // more frames to trace
// [nvn] trace_after_fs = "67d504": the frame after the first one that draws with a fragment shader of this hash prefix is traced
// (a frame number is not stable from run to run: the load times differ).
std::string g_trigger_fs;

// Debug: [nvn] skip_fs_file = FILE (env NXRT_NVN_SKIP_FS_FILE). FILE holds fragment shader hash prefixes separated by blanks; draws whose fragment
// shader starts with one of them are not drawn. The file is read again twice a second, so it can change while the game runs.
std::vector<std::string> g_skip_fs;
int g_draw_idx = 0;   // draws begun in this frame
uint32_t g_dbg_frame = 0;   // [fs] tag_textures: the frame number the sampled-texture log uses (HANDOVER 87)
bool skip_fs_draw(const std::string& hash) {
    static const std::string file = nxrt::config_get_str("nvn", "skip_fs_file", "");
    if (file.empty()) return false;
    static std::chrono::steady_clock::time_point last{};
    // Round 45 (audit): this runs for every draw; the file only needs a look twice a second, so the clock is read on the frame's
    // first draw instead of on every one (g_draw_idx is the draw counter, ++g_draw_idx runs just before this call).
    const bool poll = g_draw_idx <= 1 || last.time_since_epoch().count() == 0;
    const auto now = poll ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (poll && (last.time_since_epoch().count() == 0 || now - last > std::chrono::milliseconds(500))) {
        last = now;
        g_skip_fs.clear();
        std::ifstream f(file);
        std::string w;
        while (f >> w) g_skip_fs.push_back(w);
    }
    for (const std::string& p : g_skip_fs) {
        if (!p.empty() && p[0] == '#') {   // "#198" or "#198-210": the draw numbers of the frame (the first draw is #1, as in the trace)
            const int a = std::atoi(p.c_str() + 1);
            const size_t dash = p.find('-');
            const int b = dash == std::string::npos ? a : std::atoi(p.c_str() + dash + 1);
            if (g_draw_idx >= a && g_draw_idx <= b) return true;
        } else if (hash.compare(0, p.size(), p) == 0) return true;
    }
    return false;
}
// [nvn] hot_state_file = FILE (NXRT_NVN_HOT_STATE_FILE): "name=value" lines, re-read four times a second, so a state A/B runs
// without a restart - the same idea as skip_fs_file above. Only the switches named here are live; everything else is read once at
// startup. Handy lines (HANDOVER 90, the front-face A/B of HANDOVER 03 section 26):
//
//     front_face=0     0 = flip with the lower-left origin (the old, inverted choice), 1 = flip with the flipped viewport,
//                      2 = never flip, 3 = always flip        ([nvn] front_face_mode)
int hot_int(const char* name, int def) {
    static const std::string file = nxrt::config_get_str("nvn", "hot_state_file", "");
    if (file.empty()) return def;
    static std::chrono::steady_clock::time_point last{};
    static std::unordered_map<std::string, int> vals;
    static uint64_t epoch = 0;                      // bumps with every re-read of the file
    const auto now = std::chrono::steady_clock::now();
    if (last.time_since_epoch().count() == 0 || now - last > std::chrono::milliseconds(250)) {
        last = now;
        ++epoch;
        vals.clear();
        std::ifstream f(file);
        std::string line;
        while (std::getline(f, line)) {
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            while (!k.empty() && (k.back() == ' ' || k.back() == '\t' || k.back() == '\r')) k.pop_back();
            while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
            if (!k.empty() && !v.empty()) vals[k] = std::atoi(v.c_str());
        }
    }
    // Round 45 (audit): this runs twice for every draw, and the map lookup built a std::string temporary and hashed it each time.
    // Callers pass string literals, so the name's address identifies the caller: remember the answer per literal until the next re-read.
    struct Slot { const char* name = nullptr; int value = 0; uint64_t epoch = 0; };
    static Slot slots[8];
    Slot& s = slots[(reinterpret_cast<uintptr_t>(name) >> 4) & 7u];
    if (s.name == name && s.epoch == epoch) return s.value;
    const auto it = vals.find(name);
    const int v = it == vals.end() ? def : it->second;
    s.name = name;
    s.value = v;
    s.epoch = epoch ? epoch : 1;                    // 0 = "never filled": a file that never re-reads still gets one lookup
    return v;
}

bool g_trigger_read = false, g_trigger_seen = false, g_trigger_done = false;
int64_t g_trigger_present = -1;
int g_trace_n = 0;
std::atomic<bool> g_trace_req{false};   // backend_trace_next_frame(): trace the first whole frame after this flag is seen
std::atomic<int64_t> g_trace_span{0};    // backend_trace_frames(n): trace this many consecutive frames (no RT dumps unless asked)
#define TRACE(...) do { if (g_trace) NXLOG_I("nvn.trace", __VA_ARGS__); } while (0)

// [nvn] trace_compact = 0|1|2 (NXRT_NVN_TRACE_COMPACT, default 0 = auto): what a traced draw prints.
//   auto (0): the long form (uniform blocks, storage buffers, vertex streams, sampled textures) up to 600 draws, then the short
//             form for the rest of the frame. A 3,300-draw frame is otherwise 54 MB of log that no one reads.
//   1: always short - the draw line, its depth/blend/state lines and the index count of an indexed draw.
//   2: always long (the old behaviour).
bool trace_verbose() {
    static const int mode = static_cast<int>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "trace_compact", 0), 0, 2));
    if (mode == 1) return false;
    if (mode == 2) return true;
    return g_trace_n <= 600;
}

// One line per traced draw, kept for the end-of-frame summary: which blended draws ran without an effective depth test. Without
// it the answer to "who paints over the scene" is a manual grep over 3,300 draw blocks.
struct TraceDrawRec {
    int n = 0;
    int rt = 0, rt_count = 1;
    std::string fs;
    int blend = 0;
    int src_rgb = 0, dst_rgb = 0, eq_rgb = 0, src_a = 0, dst_a = 0, eq_a = 0;
    bool test = false, write = false;
    int func = 0;
};
std::vector<TraceDrawRec> g_trace_recs;

// [nvn] draw_occ = 1 (diagnostic): every draw of a traced frame is wrapped in an occlusion query, and the per-draw sample counts are logged
// after the frame ("occ #<draw> <samples> samples"). A draw that the game submitted but whose fragments the depth, stencil or alpha test
// all killed shows 0 samples, which is what separates "the game never submitted this piece" from "we drew it and lost it" - the question a
// hole in the frame leaves open (HANDOVER 18.7/19). Costs one query per draw for the traced frame only.
constexpr uint32_t kDrawOccMax = 8192;
VkQueryPool g_docc_pool = VK_NULL_HANDLE;
uint32_t g_docc_used = 0;
std::vector<std::pair<int, uint32_t>> g_docc_list;   // (draw number, query index) in submission order
bool g_docc_on = false, g_docc_read = false;

void draw_occ_read_config() {
    if (g_docc_read) return;
    g_docc_read = true;
    g_docc_on = nxrt::config_get_int("nvn", "draw_occ", 0) != 0;
}
void draw_occ_frame_begin(VkCommandBuffer cb) {
    draw_occ_read_config();
    if (!g_docc_on || !g_trace) return;
    if (g_docc_pool == VK_NULL_HANDLE) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_OCCLUSION;
        qi.queryCount = kDrawOccMax;
        if (vkCreateQueryPool(vkdev(), &qi, nullptr, &g_docc_pool) != VK_SUCCESS) { g_docc_pool = VK_NULL_HANDLE; g_docc_on = false; return; }
        NXLOG_I(kTag, "draw_occ: %u occlusion queries for the traced frame", kDrawOccMax);
    }
    g_docc_used = 0;
    g_docc_list.clear();
    vkCmdResetQueryPool(cb, g_docc_pool, 0, kDrawOccMax);   // outside every render pass: the start of the frame's command buffer
    NXLOG_I(kTag, "draw_occ: query window reset on command buffer %p", static_cast<void*>(cb));
}
bool draw_occ_begin(uint32_t* q) {
    if (!g_docc_on || !g_trace || !F || g_docc_pool == VK_NULL_HANDLE || g_docc_used >= kDrawOccMax) return false;
    *q = g_docc_used++;
    vkCmdBeginQuery(F->cb, g_docc_pool, *q, device().features().occlusion_query_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
    return true;
}
void draw_occ_end(uint32_t q) {
    vkCmdEndQuery(F->cb, g_docc_pool, q);
    static bool told = false;
    if (!told) { told = true; NXLOG_I(kTag, "draw_occ: queries recorded on command buffer %p", static_cast<void*>(F->cb)); }
    g_docc_list.push_back({g_trace_n, q});
}
struct DrawOccScope {
    bool on;
    uint32_t q = 0;
    DrawOccScope() : on(draw_occ_begin(&q)) {}
    ~DrawOccScope() { if (on) draw_occ_end(q); }
};
void draw_occ_report() {
    if (!g_docc_on || !g_docc_pool || g_docc_list.empty()) return;
    const uint32_t n = g_docc_used;
    std::vector<uint64_t> res(n, 0);
    // The counts become available once the submission that holds the queries has run. That is normally the present's own submission;
    // if it is a frame behind (submit batching), the report simply waits for the next present instead of stalling the render thread.
    finish();   // a diagnostic run: wait for the submission so the query results are there (no later present reports them)
    const VkResult r = vkGetQueryPoolResults(vkdev(), g_docc_pool, 0, n, n * sizeof(uint64_t), res.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    static int waits = 0;
    if (r == VK_NOT_READY) {
        if (++waits < 240) return;   // about 4 s of presents
        NXLOG_W(kTag, "draw_occ: no query results after %d presents: %u draws had %u queries", waits, static_cast<unsigned>(g_docc_list.size()), n);
        g_docc_list.clear();
        waits = 0;
        return;
    }
    waits = 0;
    if (r != VK_SUCCESS) {
        NXLOG_W(kTag, "draw_occ: vkGetQueryPoolResults failed (%d)", static_cast<int>(r));
        g_docc_list.clear();
        return;
    }
    for (const auto& e : g_docc_list) NXLOG_I("nvn.trace", "occ #%d %llu samples", e.first, static_cast<unsigned long long>(e.second < n ? res[e.second] : 0));
    NXLOG_I(kTag, "draw_occ: %u draws measured in the traced frame", static_cast<unsigned>(g_docc_list.size()));
    g_docc_list.clear();
}

// [nvn] barrier_keeps_pass = 1 (default 0): an NVN barrier is a memory dependency, and XC2 issues one around most of its
// skinning draws (a draw that writes the blended bone matrices the character draws read). Ending the render pass for each
// of them costs a pass break apiece; nxnative's XC2 port keeps them inside the pass ("104 a frame take a barrier inside
// the pass, none ends it", docs/reference/rendering.md "Ordering barriers before a draw that writes"). Vulkan allows a
// memory barrier inside a render pass as long as it involves no layout transition, which memory_barrier() never does.
// Measured with [nvn] replay_stats; the default keeps the old behaviour until a run says it is safe.
bool barrier_keeps_pass() {
    static const bool on = nxrt::config_get_bool("nvn", "barrier_keeps_pass", false);
    return on;
}

// ---- [nvn] replay_stats = N (debug, default 0 = off): the replay's cost per draw (notes/rt.md) ---------------------------------
// The phases of begin_draw, the draw call itself, out_before_op, dispatches and the whole replay are timed with the TSC; every N presents
// one log line (nvn.rstats) gives the averages per frame and per draw, with the counts behind them. Render thread only.
enum RsZone : int { kRsHead, kRsPrep, kRsFb, kRsPass, kRsPipe, kRsDyn, kRsSets, kRsVb, kRsCall, kRsOut, kRsDispatch, kRsAll, kRsAlloc,
                   kRsKey, kRsLookup, kRsBind,   // the three parts of the pipe zone (nested timers, they do not disturb the marks)
                   kRsPrepAttach, kRsPrepTex, kRsPrepUpscale,   // the three parts of the prep zone: attachments, textures, upscale
                   kRsN };
struct RStats {
    uint64_t cyc[kRsN] = {};
    uint64_t draws = 0, draws_desc = 0, draws_pipe = 0, draws_ubo = 0, sets = 0, set_hits = 0, set_writes = 0, res_tex = 0, res_img = 0, res_memo = 0;
    uint64_t binds = 0, binds_same = 0, layouts = 0, keys = 0, dispatches = 0;
    uint64_t miss_first = 0, miss_uboofs = 0, miss_ubo = 0, miss_tex = 0, miss_other = 0;   // set cache misses by what differs from the last set of that layout
    uint64_t culled = 0;   // draws whose scissor leaves no pixel of the target (census: the game culls, so this should be ~0)
};
RStats g_rs;
uint64_t g_rs_t = 0;   // TSC at the last phase mark
// [nvn] fb_memo (defined with feedback_prepare, below): the report line prints what the memo answered.
extern uint64_t g_fb_memo_hits, g_fb_memo_bad;
int fb_memo_mode();
// [nvn] rt_check = 1 (debug, default 0): every memo answer of the rt changes (resolve_memo, out_memo; pool_cache in nvn_objects.cpp, the
// view and sampler caches in nvn_gpu_texture.cpp) is compared with the answer computed without it. A mismatch is logged (nvn.rtcheck,
// first 20); with replay_stats the stats line counts the checks. No mismatch = the same Vulkan commands as without the memos.
bool rt_check() {
    static const bool on = nxrt::config_get_bool("nvn", "rt_check", false);
    return on;
}
uint64_t g_chk_n = 0, g_chk_bad = 0;
void chk(bool ok, const char* what) {
    ++g_chk_n;
    if (!ok && g_chk_bad++ < 20) NXLOG_W("nvn.rtcheck", "mismatch: %s", what);
}
int64_t rs_every() {
    static const int64_t n = nxrt::config_get_int("nvn", "replay_stats", 0);
    return n;
}
// 1-in-16 frame sampling for replay_stats: eliminates 94% of __rdtsc() overhead while reporting exact per-frame averages
bool g_rs_active = false;
uint64_t g_rs_sampled_frames = 0;
inline bool rs_on() { return g_rs_active; }
inline void rs_start() { if (rs_on()) g_rs_t = __rdtsc(); }
inline void rs_mark(int zone) {
    if (!rs_on()) return;
    const uint64_t t = __rdtsc();
    g_rs.cyc[zone] += t - g_rs_t;
    g_rs_t = t;
}
struct RsScope {
    int zone; uint64_t t0;
    explicit RsScope(int z) : zone(z), t0(rs_on() ? __rdtsc() : 0) {}
    ~RsScope() { if (t0) g_rs.cyc[zone] += __rdtsc() - t0; }
};
void rs_report() {
    using clk = std::chrono::steady_clock;
    static uint64_t last_p = 0, last_tsc = 0;
    static clk::time_point last_t{};
    const int64_t every = rs_every();
    if (every <= 0) { g_rs_active = false; return; }
    const uint64_t p = counters().presents.load(std::memory_order_relaxed);
    const bool was_active = g_rs_active;
    if (was_active) ++g_rs_sampled_frames;
    g_rs_active = ((p & 15u) == 0);
    if (!last_tsc) { last_p = p; last_tsc = __rdtsc(); last_t = clk::now(); g_rs = RStats(); g_rs_sampled_frames = 0; return; }
    if (p < last_p + static_cast<uint64_t>(every)) return;
    const uint64_t tsc = __rdtsc();
    const clk::time_point now = clk::now();
    const double sec = std::chrono::duration<double>(now - last_t).count();
    const double us_per_cyc = sec > 0 ? sec * 1e6 / static_cast<double>(tsc - last_tsc) : 0.0;
    const double all_frames = static_cast<double>(std::max<uint64_t>(1, p - last_p));
    const double frames = static_cast<double>(std::max<uint64_t>(1, g_rs_sampled_frames));
    const RStats& s = g_rs;
    auto fms = [&](int z) { return static_cast<double>(s.cyc[z]) * us_per_cyc / 1000.0 / frames; };     // ms per frame
    const double draws_pf = static_cast<double>(s.draws) / all_frames;
    auto dus = [&](int z) { return draws_pf > 0.0 ? fms(z) * 1000.0 / draws_pf : 0.0; };               // us per draw
    double setup = 0;
    for (int z = kRsHead; z <= kRsVb; ++z) setup += fms(z);
    const double pf = 1.0 / all_frames;
    NXLOG_I("nvn.rstats", "%.0f frames: replay %.2f ms/frame = draw setup %.2f + call %.2f + out %.2f + dispatch %.2f + rest %.2f | %.0f draws/frame "
            "(%.0f desc, %.0f pipe, %.0f ubo-only, %.0f layouts, %.0f keys), %.0f dispatches, %.0f culled | per draw us: head %.2f prep %.2f fb %.2f pass %.2f pipe %.2f dyn %.2f "
            "sets %.2f (alloc+update %.2f) vb %.2f call %.2f out %.2f | sets/frame %.0f (%.1f%% cached, %.0f writes) | resolves/frame tex %.0f img %.0f (memo %.0f) | "
            "binds/frame %.0f (%.1f%% same) | set misses/frame: new layout or submission %.0f, only uniform offsets %.0f, only uniform buffers %.0f, "
            "textures %.0f, other %.0f | rt_check %llu checked, %llu mismatches",
            frames, fms(kRsAll), setup, fms(kRsCall), fms(kRsOut), fms(kRsDispatch),
            fms(kRsAll) - setup - fms(kRsCall) - fms(kRsOut) - fms(kRsDispatch), s.draws * pf, s.draws_desc * pf, s.draws_pipe * pf, s.draws_ubo * pf, s.layouts * pf,
            s.keys * pf, s.dispatches * pf, s.culled * pf, dus(kRsHead), dus(kRsPrep), dus(kRsFb), dus(kRsPass), dus(kRsPipe), dus(kRsDyn), dus(kRsSets), dus(kRsAlloc), dus(kRsVb),
            dus(kRsCall), dus(kRsOut), s.sets * pf, s.sets ? 100.0 * static_cast<double>(s.set_hits) / static_cast<double>(s.sets) : 0.0,
            s.set_writes * pf, s.res_tex * pf, s.res_img * pf, s.res_memo * pf, s.binds * pf,
            s.binds ? 100.0 * static_cast<double>(s.binds_same) / static_cast<double>(s.binds) : 0.0, s.miss_first * pf, s.miss_uboofs * pf,
            s.miss_ubo * pf, s.miss_tex * pf, s.miss_other * pf, static_cast<unsigned long long>(g_chk_n), static_cast<unsigned long long>(g_chk_bad));
    // [nvn] push_uniforms: what the set 0 pushes cost (writes pushed a frame, and how many of them changed the set)
    NXLOG_I("nvn.rstats", "set 0 pushes/frame: full %.0f delta %.0f none %.0f, writes %.0f (lean_draws %s, push_uniforms %s, ubo_only_rebind %s)",
            static_cast<double>(g_lean_push_full) * pf, static_cast<double>(g_lean_push_delta) * pf, static_cast<double>(g_lean_push_none) * pf,
            static_cast<double>(g_lean_push_writes) * pf, lean_on() ? "on" : "off", push_uniforms_on() ? "on" : "off", ubo_only_rebind_on() ? "on" : "off");
    // What the prep zone is made of: the attachment/stencil decisions, prepare_textures, and the upscale hookup.
    NXLOG_I("nvn.rstats", "prep split: attach %.2f, textures %.2f, upscale %.2f us/draw (prep zone %.2f)",
            dus(kRsPrepAttach), dus(kRsPrepTex), dus(kRsPrepUpscale), dus(kRsPrep));
    // [nvn] replay_stats (round 45): what the pipe zone is made of - the key build, the pipeline lookup and the bind.
    NXLOG_I("nvn.rstats", "pipe split: key %.2f lookup %.2f bind %.2f us/draw (pipe zone %.2f; %.0f keys/frame)",
            dus(kRsKey), dus(kRsLookup), dus(kRsBind), dus(kRsPipe), s.keys * pf);
    // [nvn] fb_memo: how often the feedback scan was answered from the memo, and whether the self-check ever disagreed.
    static uint64_t fb_hits_last = 0;
    NXLOG_I("nvn.rstats", "fb memo: %.0f hits/frame of %.0f draws, mode %d, %llu mismatches since start",
            (g_fb_memo_hits - fb_hits_last) * pf, s.draws * pf, fb_memo_mode(), static_cast<unsigned long long>(g_fb_memo_bad));
    fb_hits_last = g_fb_memo_hits;
    g_lean_push_full = g_lean_push_delta = g_lean_push_none = g_lean_push_writes = 0;
    last_p = p;
    last_tsc = tsc;
    last_t = now;
    g_rs = RStats();
    g_rs_sampled_frames = 0;
}

const char* tex_desc(const Texture* t, char* buf, size_t n) {
    if (!t) { std::snprintf(buf, n, "-"); return buf; }
    // The guest address goes last: two texture objects over the same storage (NVN makes a sampled view and an attachment view
    // for the same map) then read as the same @... and an alias is visible in one glance (HANDOVER 29; t124/t125).
    std::snprintf(buf, n, "t%u:%s %dx%dx%d L%u/%u%s%s @%p", t->id, t->fmt ? t->fmt->name : "?", t->b.width, t->b.height, t->b.depth, t->levels, t->layers,
                  t->b.samples > 1 ? " ms" : "", t->b.target == NVN_TEXTURE_TARGET_3D ? " 3D" : "", static_cast<const void*>(t->mem));
    return buf;
}

// Percentage of the texture's storage (in the game's memory, first 16 MiB) that holds a non-zero 8-byte word; -1 without storage. Trace only.
int storage_nonzero_pct(const Texture* t) {
    if (!t || !t->mem || !t->storage_size) return -1;
    const uint64_t n = std::min<uint64_t>(t->storage_size, 16ull << 20) / 8;
    if (!n) return -1;
    uint64_t nz = 0;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t w;
        std::memcpy(&w, t->mem + i * 8, 8);
        nz += w != 0;
    }
    return static_cast<int>(nz * 100 / n);
}

void log_once(const char* key, const char* msg) {
    static std::unordered_map<std::string, bool> seen;
    if (seen.emplace(key, true).second) NXLOG_W(kTag, "%s", msg);
}

bool first_time_key(const std::string& key) {   // render thread only
    static std::unordered_map<std::string, bool> seen;
    return seen.size() < 4096 && seen.emplace(key, true).second;
}

// ---- helpers ----------------------------------------------------------------------------------------------------------------
VkFormat scaled_for(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UINT: return VK_FORMAT_R8_USCALED;
        case VK_FORMAT_R8G8_UINT: return VK_FORMAT_R8G8_USCALED;
        case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8_USCALED;
        case VK_FORMAT_R8G8B8A8_UINT: return VK_FORMAT_R8G8B8A8_USCALED;
        case VK_FORMAT_R16_UINT: return VK_FORMAT_R16_USCALED;
        case VK_FORMAT_R16G16_UINT: return VK_FORMAT_R16G16_USCALED;
        case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16_USCALED;
        case VK_FORMAT_R16G16B16A16_UINT: return VK_FORMAT_R16G16B16A16_USCALED;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: return VK_FORMAT_A2B10G10R10_USCALED_PACK32;
        case VK_FORMAT_R8_SINT: return VK_FORMAT_R8_SSCALED;
        case VK_FORMAT_R8G8_SINT: return VK_FORMAT_R8G8_SSCALED;
        case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8_SSCALED;
        case VK_FORMAT_R8G8B8A8_SINT: return VK_FORMAT_R8G8B8A8_SSCALED;
        case VK_FORMAT_R16_SINT: return VK_FORMAT_R16_SSCALED;
        case VK_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SSCALED;
        case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16_SSCALED;
        case VK_FORMAT_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SSCALED;
        case VK_FORMAT_A2B10G10R10_SINT_PACK32: return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
        default: return f;
    }
}

bool is_int_format(VkFormat f) { return scaled_for(f) != f; }

VkImageViewType view_type_for(const DescriptorBinding& b) {
    const ImageShape& s = b.image;
    switch (s.dim) {
        case 0: return s.arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        case 1: return s.arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        case 2: return VK_IMAGE_VIEW_TYPE_3D;
        case 3: return s.arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}

// ---- passes ----------------------------------------------------------------------------------------------------------------
// Ends transform feedback in the open pass, remembering the bytes written in the counter buffer (a pass break must not restart the buffer).
void xfb_end_in_pass() {
    if (!g_xfb_began) return;
    VkBuffer cbuf = g_xfb_counter.buffer;
    const VkDeviceSize off = static_cast<VkDeviceSize>(g_xfb_cur) * kXfbSlotBytes;
    vkCmdEndTransformFeedbackEXT(F->cb, 0, 1, &cbuf, &off);
    g_xfb_began = false;
    g_xfb_have_counter = true;
    g_xfb_slot_written[g_xfb_cur] = true;
}

// GPU pass profiler: closes the interval since the previous stamp under this label (no-op unless [prof] gpu_passes).
void gstamp(const char* label) {
    if (gpass_on()) gpass_stamp(*F, label, 0);
}

// Two texture objects over one image (as shares_image further down, which this code cannot see yet).
bool same_image(Texture* a, Texture* b) { return a == b || (a && b && a->gpu && a->gpu == b->gpu); }

// to_pass: the caller only starts another pass or changes image layouts next (ensure_pass, prepare_textures); with tracked barriers the pass
// then ends without a barrier when it only wrote its attachments. Any other caller gets a full barrier, also for passes ended earlier
// without one.
void split_maybe();   // [nvn] split_draws, defined after end_pass

bool end_pass(bool to_pass = false) {
    if (!P.open) {
        if (!to_pass) {
            if (!g_pending.empty()) {
                memory_barrier(F->cb);
                g_pending.clear();
                g_pending_write_mem = false;
                g_wm_last_addr = g_wm_last_size = 0;
                return true;
            }
            flush_write_mem_barrier();
        }
        return false;
    }
    xfb_end_in_pass();
    vkCmdEndRendering(F->cb);
    P.open = false;
    P.targets_clean = false;
    if (gpass_on()) {
        char b[112];
        std::snprintf(b, sizeof b, "pass %ux%u c%u:%d d%d%s", P.extent.width, P.extent.height, P.ncolor, static_cast<int>(P.color_fmt[0]), static_cast<int>(P.depth_fmt),
                      P.depth_ro ? " ro" : "");
        gpass_stamp(*F, b, g_pass_draws);
        char vb[128];
        std::snprintf(vb, sizeof vb, "gv:%s", b);
        ft::mark(vb, static_cast<uint32_t>(std::min<uint64_t>(g_pass_verts, 0xffffffffull)));
    }
    bool barrier_emitted = false;
    if (to_pass && g_tracked && !g_pass_unsafe) {
        auto remember = [](Texture* t, VkImageLayout l) {
            if (!t || !tex_gpu(t)) return;
            for (const PendingWrite& w : g_pending) if (w.tex.get() == t) return;
            g_pending.push_back({Ref<Texture>::share(t), l});
        };
        for (uint32_t i = 0; i < P.ncolor; ++i)
            remember(P.color[i].tex, (P.color_general >> i & 1u)
                                         ? VK_IMAGE_LAYOUT_GENERAL
                                         : ((P.color_feedback >> i & 1u) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                         : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
        if (!P.depth_ro) remember(P.depth.tex, depth_pass_layout(P.depth_ro, P.depth_general));
    } else {
        memory_barrier(F->cb);
        g_pending.clear();
        g_pending_write_mem = false;
        g_wm_last_addr = g_wm_last_size = 0;
        barrier_emitted = true;
    }
    g_pass_unsafe = false;
    g_bound_pipeline = VK_NULL_HANDLE;
    invalidate_bound_state();
    g_dirty = kAllDirty;
    g_vb_dirty_mask = 0xFFFF;
    split_maybe();
    return barrier_emitted;
}

// [nvn] split_draws = N (0 = off): after N draws, at this pass boundary, the commands recorded so far go to the GPU (nvn_gpu.cpp
// split_flush) and the recording continues in a new command buffer, so the GPU works on the frame's early passes while the replay records
// the rest (NS1's split_draws; the present and the game's fences still see one submission, signalled by its last part).
uint64_t g_split_since = 0;
std::chrono::steady_clock::time_point g_split_t0{};
void split_maybe() {
    static const int64_t n = nxrt::config_get_int("nvn", "split_draws", 0);
    static const bool force_split = nxrt::config_get_bool("nvn", "force_split_draws", false);
    static const int64_t min_us = std::max<int64_t>(0, nxrt::config_get_int("nvn", "split_us", 0));
    if (n <= 0 || (!force_split && mirror_capture_on()) || !F || g_xfb_active || g_split_since < static_cast<uint64_t>(n)) return;
    if (min_us > 0) {
        const auto now = std::chrono::steady_clock::now();
        if (g_split_t0.time_since_epoch().count() != 0 &&
            std::chrono::duration_cast<std::chrono::microseconds>(now - g_split_t0).count() < min_us)
            return;
        g_split_t0 = now;
    }
    if (!split_flush(*F)) return;
    g_split_since = 0;
    invalidate_bound_state();
    g_dirty = kAllDirty;      // a fresh command buffer: no bound pipeline, no descriptor sets, no dynamic state
    g_vb_dirty_mask = 0xFFFF;
    g_pending.clear();        // the barrier that starts the new part covers these writes
    g_pending_write_mem = false;
    g_wm_last_addr = g_wm_last_size = 0;
}

// A pass is about to use `t` as an attachment in `layout`. An earlier pass that ended without a barrier wrote it, and no layout change
// (which would carry the dependency) happens in between: the two passes need a barrier.
bool pending_conflict(Texture* t, VkImageLayout layout) {
    if (!t || g_pending.empty()) return false;
    for (const PendingWrite& w : g_pending)
        if (same_image(w.tex.get(), t) && tex_layout(t) == w.layout && layout == w.layout) return true;
    return false;
}

// The draw's colour state asks for a logic op. Vulkan only defines logic operations on integer attachments, so an integer target is
// rendered through its own format then and the fragment shader is patched to write that integer type bit-exactly (HANDOVER 35).
bool logic_op_active() { return S.color.logic_op != 0 && S.color.logic_op != NVN_LOGIC_OP_COPY && device().features().logic_op; }
bool int_outputs_wanted() { return logic_op_active() && S.stage[kFragment] && fs_int_outputs_available(S.stage[kFragment]); }

Target make_target(Texture* t, const TextureViewInfo& v, bool has_view) {
    Target tg;
    tg.tex = t;
    if (has_view) {
        if (v.has_levels) tg.level = static_cast<uint32_t>(v.min_level);
        if (v.has_layers) {
            tg.layer = static_cast<uint32_t>(v.min_layer);
            tg.layers = std::max(1, v.num_layers);
        }
        else if (t && t->layers > 1 && t->b.target != NVN_TEXTURE_TARGET_3D) tg.layers = t->layers;   // a view without layers
        if (v.has_format) {
            const FormatInfo* fi = format_info(v.format);
            if (fi && fi->vk != VK_FORMAT_UNDEFINED) tg.view_format = fi->vk;
        }
    } else if (t && t->layers > 1 && t->b.target != NVN_TEXTURE_TARGET_3D) {
        // no view: an array or cube map target is attached with all its layers (layered rendering: a geometry shader picks the layer with
        // gl_Layer; without it the shaders draw to layer 0)
        tg.layers = t->layers;
    }
    // the resolved attachment format is part of the target's identity, so a pass is restarted when it changes
    if (tg.tex && int_outputs_wanted()) tg.view_format = tex_attachment_format(tg.tex, tg.view_format, true);
    return tg;
}

bool targets_match() {
    if (P.ncolor != S.ncolor) return false;
    for (uint32_t i = 0; i < S.ncolor; ++i)
        if (!(P.color[i] == make_target(S.color_tex[i].get(), S.color_view[i], S.color_has_view[i]))) return false;
    return P.depth == make_target(S.depth_tex.get(), S.depth_view, S.depth_has_view);
}

// The current render targets get their images (scaled when they qualify, tex_gpu_attachment). With `fix` (no pass open), as Ryujinx does
// when it binds render targets: a target whose image was made unscaled (first used another way) is scaled now, and the targets of one
// pass are brought to one scale (the lowest when one of them cannot be scaled). Returns the scale of the targets.
float attach_targets(bool fix) {
    Texture* ts[9];
    uint32_t n = 0;
    for (uint32_t i = 0; i < S.ncolor && i < 8; ++i) if (S.color_tex[i]) ts[n++] = S.color_tex[i].get();
    if (S.depth_tex) ts[n++] = S.depth_tex.get();
    float lo = 0.0f, hi = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        if (!tex_gpu_attachment(ts[i])) continue;
        if (fix) {
            const float want = tex_scale_wanted(ts[i]);
            if (want != tex_scale(ts[i])) tex_rescale(ts[i], *F, want, "used as a render target");
            if (ts[i]->is_display && extras::hdr_highlights_kept()) tex_promote_float(ts[i], *F, "the window");   // the game's highlights above 1
        }
        const float sc = tex_scale(ts[i]);
        lo = lo == 0.0f ? sc : std::min(lo, sc);
        hi = std::max(hi, sc);
    }
    if (lo == 0.0f) return 1.0f;
    if (lo != hi && fix) {
        for (uint32_t i = 0; i < n; ++i)
            if (tex_gpu(ts[i]) && tex_scale(ts[i]) != lo) tex_rescale(ts[i], *F, lo, "a render target of the same pass cannot be scaled");
    }
    return lo;
}

// The area the viewports cover, at most the device's framebuffer size.
VkExtent2D viewport_extent() {
    float w = 0.0f, h = 0.0f;
    for (uint32_t i = 0; i < S.nvp; ++i) {
        w = std::max(w, S.vp[i][0] + std::fabs(S.vp[i][2]));
        h = std::max(h, S.vp[i][1] + std::fabs(S.vp[i][3]));
    }
    const VkPhysicalDeviceLimits& lim = device().properties().limits;
    auto px = [](float v, uint32_t max) { return v > 0.0f ? static_cast<uint32_t>(std::min(std::ceil(v), static_cast<float>(max))) : 0u; };
    return {px(w, lim.maxFramebufferWidth), px(h, lim.maxFramebufferHeight)};
}

// Starts a pass on the current render targets if none is open for them. False when there is nothing to render to. `writes`: the program
// writes images or buffers, so a draw without render targets still has an effect; it gets a pass without attachments over its viewports
// (Fortune's Weave's SSAO deinterleaves depth and stores its occlusion into 2D arrays from such draws).
bool ensure_pass(bool depth_ro = false, bool writes = false, bool depth_general = false, uint32_t color_general = 0, uint32_t color_feedback = 0) {
    const bool want_int_out = int_outputs_wanted();
    if (P.open && P.targets_clean && P.int_outputs == want_int_out && P.depth_ro == depth_ro && P.depth_general == depth_general &&
        P.color_general == color_general && (P.color_feedback & color_feedback) == color_feedback &&
        (!writes || S.ncolor || S.depth_tex))
        return true;
    const VkExtent2D bare = writes && !S.ncolor && !S.depth_tex ? viewport_extent() : VkExtent2D{0, 0};
    const bool matched = P.open && targets_match();
    if (matched && P.depth_ro == depth_ro && P.depth_general == depth_general && P.color_general == color_general &&
        (P.color_feedback & color_feedback) == color_feedback &&
        bare.width <= P.extent.width && bare.height <= P.extent.height) {
        P.targets_clean = true;
        P.int_outputs = want_int_out;
        return true;
    }
    const uint32_t cf = matched ? (P.color_feedback | color_feedback) : color_feedback;
    end_pass(true);
    flush_write_mem_barrier();
    if (!g_pending.empty()) {
        bool conflict = false;
        for (uint32_t i = 0; i < S.ncolor && !conflict; ++i) {
            const VkImageLayout want_cl = (color_general >> i & 1u) ? VK_IMAGE_LAYOUT_GENERAL
                : ((cf >> i & 1u) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            conflict = pending_conflict(S.color_tex[i].get(), want_cl);
        }
        if (!conflict && S.depth_tex)
            conflict = pending_conflict(S.depth_tex.get(), depth_pass_layout(depth_ro, depth_general));
        if (conflict) {
            memory_barrier(F->cb);
            g_pending.clear();
        }
    }
    if (gpass_on()) gpass_stamp(*F, "gap", 0);      // copies, dispatches, barriers and layout changes since the previous pass
    g_pass_draws = 0;
    g_pass_verts = 0;
    const float pass_scale = attach_targets(true);
    P = Pass();
    P.depth_ro = depth_ro;
    P.depth_general = depth_general;
    P.color_general = color_general;
    P.color_feedback = cf;
    P.scale = pass_scale;
    VkRenderingAttachmentInfo colors[8] = {};
    VkRenderingAttachmentInfo depth = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingAttachmentInfo stencil = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    bool any = false;
    VkExtent2D extent = {UINT32_MAX, UINT32_MAX};
    P.ncolor = S.ncolor;
    for (uint32_t i = 0; i < S.ncolor; ++i) {
        colors[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        Texture* t = S.color_tex[i].get();
        P.color[i] = make_target(t, S.color_view[i], S.color_has_view[i]);
        if (!t || !tex_gpu(t)) continue;
        const bool fresh = tex_layout(t) == VK_IMAGE_LAYOUT_UNDEFINED;
        // a render target that the same draw writes through a storage image has to be GENERAL: the descriptor says
        // GENERAL (prepare_textures moves storage images there), so the attachment cannot be the colour-attachment
        // layout (VUID-vkCmdDrawIndexed-imageLayout-00344). A colour attachment sampled in-place via
        // VK_EXT_attachment_feedback_loop_layout uses ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT.
        const VkImageLayout cl = (color_general >> i & 1u) ? VK_IMAGE_LAYOUT_GENERAL
            : ((cf >> i & 1u) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        tex_transition(t, *F, cl);
        tex_mark_gpu_written(t);
        VkImageView v = tex_attachment_view(t, P.color[i].level, P.color[i].layer, P.color[i].layers, P.color[i].view_format, int_outputs_wanted());
        if (!v) continue;
        colors[i].imageView = v;
        colors[i].imageLayout = cl;
        colors[i].loadOp = fresh ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        // Diagnostic (HANDOVER 31): a pass that starts on a target whose layout went back to UNDEFINED wipes it. The probe
        // list t50 is built by the probe passes and then lost before the lighting draw; this says whether the cause is here.
        if (g_trace && colors[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
            TRACE("    pass begin: CLEAR color %u t%u (gpu %d, draws since the last pass %d)", i, t->id, t->gpu ? 1 : 0, g_pass_draws);
        if (g_trace)
            TRACE("    pass begin: color %u t%u load %s fresh %d fmt %u level %u layer %u draws since the last pass %d", i, t->id,
                  fresh ? "CLEAR" : "LOAD", fresh ? 1 : 0, static_cast<unsigned>(P.color[i].view_format), P.color[i].level, P.color[i].layer, g_pass_draws);
        colors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        P.color_fmt[i] = tex_attachment_format(t, P.color[i].view_format, int_outputs_wanted());
        if (g_trace)
            TRACE("    pass color %u t%u requested %u resolved %u (image %u) mutable %d", i, t->id, static_cast<unsigned>(P.color[i].view_format),
                  static_cast<unsigned>(P.color_fmt[i]), static_cast<unsigned>(tex_vk_format(t)), t->gpu ? 1 : 0);
        const VkExtent3D e = tex_image_extent(t, P.color[i].level);
        extent.width = std::min(extent.width, e.width);
        extent.height = std::min(extent.height, e.height);
        any = true;
    }
    Texture* dt = S.depth_tex.get();
    P.depth = make_target(dt, S.depth_view, S.depth_has_view);
    bool have_depth = false;
    if (dt && tex_gpu(dt)) {
        const bool fresh = tex_layout(dt) == VK_IMAGE_LAYOUT_UNDEFINED;
        const VkImageLayout dlayout = depth_pass_layout(depth_ro, depth_general);
        tex_transition(dt, *F, dlayout);
        if (!depth_ro) tex_mark_gpu_written(dt);
        VkImageView v = tex_attachment_view(dt, P.depth.level, P.depth.layer, P.depth.layers, P.depth.view_format);
        if (v) {
            depth.imageView = stencil.imageView = v;
            depth.imageLayout = stencil.imageLayout = dlayout;
            depth.loadOp = stencil.loadOp = (fresh && !depth_ro) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.storeOp = stencil.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.clearValue.depthStencil = {1.0f, 0};
            stencil.clearValue.depthStencil = {1.0f, 0};
            P.depth_fmt = tex_vk_format(dt);
            P.has_stencil = (tex_aspect(dt) & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
            const VkExtent3D e = tex_image_extent(dt, P.depth.level);
            extent.width = std::min(extent.width, e.width);
            extent.height = std::min(extent.height, e.height);
            have_depth = true;
            any = true;
        }
    }
    if (!any && bare.width && bare.height) {
        extent = bare;
        any = true;
    }
    if (!any && g_xfb_active && device().features().transform_feedback) {   // a stream-out pass without attachments
        extent = {1, 1};
        any = true;
    }
    if (!any || extent.width == UINT32_MAX) return false;
    P.extent = extent;
    P.samples = std::max<uint32_t>(1, S.color_tex[0] ? static_cast<uint32_t>(S.color_tex[0]->b.samples) : 1);
    if (P.samples == 0) P.samples = 1;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    uint32_t layer_count = UINT32_MAX;     // layered rendering: the layers every attachment has in common
    for (uint32_t i = 0; i < S.ncolor; ++i)
        if (colors[i].imageView) layer_count = std::min(layer_count, std::max<uint32_t>(1, P.color[i].layers));
    if (have_depth) layer_count = std::min(layer_count, std::max<uint32_t>(1, P.depth.layers));
    ri.layerCount = layer_count == UINT32_MAX ? 1 : layer_count;
    uint32_t last = 0;
    for (uint32_t i = 0; i < S.ncolor; ++i)
        if (colors[i].imageView) last = i + 1;
    ri.colorAttachmentCount = S.ncolor;
    ri.pColorAttachments = colors;
    if (have_depth) {
        ri.pDepthAttachment = &depth;
        if (P.has_stencil) ri.pStencilAttachment = &stencil;
    }
    (void)last;
    gstamp("pass setup");
    {
        char cp[96];
        std::snprintf(cp, sizeof cp, "pass %ux%u c%u:%u d:%u", extent.width, extent.height,
                      S.ncolor, S.ncolor ? static_cast<unsigned>(P.color_fmt[0]) : 0u,
                      have_depth ? static_cast<unsigned>(P.depth_fmt) : 0u);
        device().cmd_checkpoint(F->cb, cp);
    }
    vkCmdBeginRendering(F->cb, &ri);
    P.open = true;
    P.targets_clean = true;
    P.int_outputs = want_int_out;
    counters().passes.fetch_add(1, std::memory_order_relaxed);
    g_bound_pipeline = VK_NULL_HANDLE;
    return true;
}

// ---- viewport conversion -------------------------------------------------------------------------------------------------
VkRect2D scissor_rect(const int32_t* r) {
    int32_t x = r[0], y = r[1], w = r[2], h = r[3];
    if (P.scale != 1.0f) {     // the edges scale (a rectangle covering the target still covers it)
        const float sc = P.scale;
        const int32_t x1 = static_cast<int32_t>(std::ceil((static_cast<float>(x) + static_cast<float>(w)) * sc - 0.001f));
        const int32_t y1 = static_cast<int32_t>(std::ceil((static_cast<float>(y) + static_cast<float>(h)) * sc - 0.001f));
        x = static_cast<int32_t>(std::floor(static_cast<float>(x) * sc + 0.001f));
        y = static_cast<int32_t>(std::floor(static_cast<float>(y) * sc + 0.001f));
        w = x1 - x;
        h = y1 - y;
    }
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    w = std::max(0, std::min<int32_t>(w, static_cast<int32_t>(P.extent.width) - x));
    h = std::max(0, std::min<int32_t>(h, static_cast<int32_t>(P.extent.height) - y));
    return {{x, y}, {static_cast<uint32_t>(w), static_cast<uint32_t>(h)}};
}

void set_viewports_and_scissors() {
    VkCommandBuffer cb = F->cb;
    VkViewport vp[16];
    VkRect2D sc[16];
    const uint32_t nvp = S.nvp ? S.nvp : 1;
    for (uint32_t i = 0; i < nvp; ++i) {
        float x = S.vp[i][0], y = S.vp[i][1], w = S.vp[i][2], h = S.vp[i][3];
        if (!S.nvp) { x = 0; y = 0; w = static_cast<float>(P.extent.width); h = static_cast<float>(P.extent.height); }
        else if (P.scale != 1.0f) { x *= P.scale; y *= P.scale; w *= P.scale; h *= P.scale; }
        // Upper-left origin: window y is the image row from the top and NDC +y is the top edge of the rectangle (negative height).
        // Lower-left origin (the GL convention): window y 0 is memory row 0 and NDC -y is the first row, so no flip.
        const bool lower_left = S.window_origin == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
        // HANDOVER 90: Eden-NS2's docs/DriverBugs.md lists "reversed viewport depth range does not work as intended" for AMD (and
    // Intel/Qualcomm/Apple). Log the range the game asks for, so a reversed one (min > max) can be seen instead of guessed.
    if (first_time_key("depth range")) NXLOG_I(kTag, "depth range: viewport 0 asks min %g max %g (window origin %d, reversed = %s)", S.dr[i][0], S.dr[i][1],
                                                static_cast<int>(S.window_origin), S.dr[i][0] > S.dr[i][1] ? "yes" : "no");
    vp[i] = lower_left ? VkViewport{x, y, w, h, S.dr[i][0], S.dr[i][1] == 0.0f && S.dr[i][0] == 0.0f ? 1.0f : S.dr[i][1]}
                           : VkViewport{x, y + h, w, -h, S.dr[i][0], S.dr[i][1] == 0.0f && S.dr[i][0] == 0.0f ? 1.0f : S.dr[i][1]};
        if (i < S.nsc) sc[i] = scissor_rect(S.sc[i]);
        else sc[i] = {{0, 0}, P.extent};
    }
    vkCmdSetViewportWithCount(cb, nvp, vp);
    vkCmdSetScissorWithCount(cb, nvp, sc);
}

// ---- descriptor sets ---------------------------------------------------------------------------------------------------------
struct TexResolved {
    Texture* tex = nullptr;
    TextureViewInfo view;
    bool has_view = false;
    const SamplerInfo* smp_ptr = nullptr;
    VkSampler vk_sampler = VK_NULL_HANDLE;
    bool has_sampler = false;
};

// Render-thread lock-free snapshots of the active TexturePool and SamplerPool:
// Synced incrementally via changes[] only when pool->version_atomic changes (0 times during 99.9% of a frame),
// eliminating ~10,000 std::shared_lock acquisitions, 80-byte SamplerInfo copies, and sampler_for() map lookups per frame.
struct TpoolSnap {
    Ref<TexturePool> pool;
    uint64_t version = ~0ull;
    struct Entry {
        Ref<Texture> tex;
        TextureViewInfo view;
        bool has_view = false;
        bool used = false;
    };
    std::vector<Entry> entries;
    void clear() { pool = Ref<TexturePool>(); version = ~0ull; entries.clear(); }
    void sync(TexturePool* tp) {
        if (!tp) return;
        const uint64_t v = tp->version_atomic.load(std::memory_order_acquire);
        if (pool.get() == tp && version == v) return;
        std::shared_lock<std::shared_mutex> lk(tp->lock);
        const uint64_t nv = tp->version;
        const size_t n = tp->entries.size();
        const bool partial = (pool.get() == tp && entries.size() == n && version <= nv && (nv - version) <= tp->changes.size());
        if (partial) {
            const size_t delta = static_cast<size_t>(nv - version);
            const size_t start = tp->changes.size() - delta;
            for (size_t i = start; i < tp->changes.size(); ++i) {
                const uint32_t id = tp->changes[i];
                if (id < n) {
                    const TexturePool::Entry& src = tp->entries[id];
                    entries[id].tex = Ref<Texture>::share(src.tex.get());
                    entries[id].view = src.view;
                    entries[id].has_view = src.has_view;
                    entries[id].used = src.used;
                }
            }
        } else {
            pool = Ref<TexturePool>::share(tp);
            entries.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const TexturePool::Entry& src = tp->entries[i];
                entries[i].tex = Ref<Texture>::share(src.tex.get());
                entries[i].view = src.view;
                entries[i].has_view = src.has_view;
                entries[i].used = src.used;
            }
        }
        version = nv;
    }
};
struct SpoolSnap {
    Ref<SamplerPool> pool;
    uint64_t version = ~0ull;
    int aniso = -1;
    float bias = 1e9f;
    struct Entry {
        SamplerInfo smp;
        VkSampler vk = VK_NULL_HANDLE;
        bool used = false;
    };
    std::vector<Entry> entries;
    void clear() { pool = Ref<SamplerPool>(); version = ~0ull; entries.clear(); }
    void sync(SamplerPool* sp) {
        if (!sp) return;
        const uint64_t v = sp->version_atomic.load(std::memory_order_acquire);
        const int a = nxrt::gfx::tuning().anisotropy.load(std::memory_order_relaxed);
        const float b = nxrt::gfx::tuning().lod_bias.load(std::memory_order_relaxed);
        if (pool.get() == sp && version == v && aniso == a && bias == b) return;
        if (pool.get() == sp && version == v) {
            for (Entry& e : entries) e.vk = VK_NULL_HANDLE;
            aniso = a;
            bias = b;
            return;
        }
        std::shared_lock<std::shared_mutex> lk(sp->lock);
        const uint64_t nv = sp->version;
        const size_t n = sp->entries.size();
        const bool partial = (pool.get() == sp && entries.size() == n && version <= nv && (nv - version) <= sp->changes.size());
        if (partial) {
            const size_t delta = static_cast<size_t>(nv - version);
            const size_t start = sp->changes.size() - delta;
            for (size_t i = start; i < sp->changes.size(); ++i) {
                const uint32_t id = sp->changes[i];
                if (id < n) {
                    entries[id].smp = sp->entries[id];
                    entries[id].used = sp->used[id] != 0;
                    entries[id].vk = VK_NULL_HANDLE;
                }
            }
            if (aniso != a || bias != b) {
                for (Entry& e : entries) e.vk = VK_NULL_HANDLE;
            }
        } else {
            pool = Ref<SamplerPool>::share(sp);
            entries.resize(n);
            for (size_t i = 0; i < n; ++i) {
                entries[i].smp = sp->entries[i];
                entries[i].used = sp->used[i] != 0;
                entries[i].vk = VK_NULL_HANDLE;
            }
        }
        version = nv;
        aniso = a;
        bias = b;
    }
    VkSampler get_vk(uint32_t sid) {
        Entry& e = entries[sid];
        if (!e.vk) e.vk = sampler_for(e.smp);
        return e.vk;
    }
};
TpoolSnap g_tpool_snap;
SpoolSnap g_spool_snap;

// The scale nxrt/gfx_tuning.h asks for on uniform block `idx` of this binding's shader (1 = none): only the chosen fragment shader's block.
float ufs_scale_for(const BindingDesc& d, int idx) {
    const nxrt::gfx::Tuning& t = nxrt::gfx::tuning();
    const uint64_t want = t.ufs_hash.load(std::memory_order_relaxed);
    if (!want || d.nvn_stage != kFragment || idx != t.ufs_block.load(std::memory_order_relaxed)) return 1.0f;
    const float sc = t.ufs_scale.load(std::memory_order_relaxed);
    if (sc == 1.0f || !d.shader || std::strtoull(d.shader->hash.c_str(), nullptr, 16) != want) return 1.0f;
    return sc;
}

uint32_t read_handle_word(int stage, int bank, int word) {
    // a texture handle the game wrote into uniform block (bank - 3)
    const int ub = bank - 3;
    if (ub < 0 || ub >= 16) return 0;
    const GState::Bind& b = S.ubo[stage][ub];
    if (!b.addr || static_cast<uint64_t>(word) * 4 + 4 > b.size) return 0;
    uint32_t v = 0;
    std::memcpy(&v, host_ptr(b.addr) + static_cast<size_t>(word) * 4, 4);
    return v;
}

// [nvn] resolve_memo (default on): within one draw or dispatch each texture and image binding is resolved once. samples_depth_target,
// vol3d_prepare, prepare_textures, feedback_prepare, support_for_draw and build_set each resolved every binding again, each time under the
// pool locks (notes/rt.md). Nothing between those calls changes what a binding resolves to: no op runs inside a draw's setup.
bool memo_on() {
    static const bool on = nxrt::config_get_bool("nvn", "resolve_memo", true);
    return on;
}
struct ResMemo { uint64_t te = 0, ie = 0; TexResolved t, i; };
std::vector<ResMemo> g_memo;
const BindingDesc* g_memo_base = nullptr;   // the bindings of the draw or dispatch being set up (null: no memo)
size_t g_memo_n = 0;
uint64_t g_memo_epoch = 0;
struct MemoScope {
    const BindingDesc* base; size_t n;
    explicit MemoScope(const ProgLayout& pl) : base(g_memo_base), n(g_memo_n) {
        g_tpool_snap.sync(S.tpool.get());
        g_spool_snap.sync(S.spool.get());
        if (!memo_on()) return;
        g_memo_base = pl.bindings.data();
        g_memo_n = pl.bindings.size();
        if (g_memo.size() < g_memo_n) g_memo.resize(g_memo_n);
        ++g_memo_epoch;
    }
    ~MemoScope() {
        if (!memo_on()) return;
        g_memo_base = base;
        g_memo_n = n;
        ++g_memo_epoch;    // an outer scope (none today) resolves again
    }
};
ResMemo* memo_for(const BindingDesc& d) {
    const BindingDesc* p = &d;
    if (!g_memo_base || p < g_memo_base || p >= g_memo_base + g_memo_n) return nullptr;
    return &g_memo[static_cast<size_t>(p - g_memo_base)];
}

TexResolved resolve_texture_raw(const BindingDesc& d);
TexResolved resolve_image_raw(const BindingDesc& d);
bool same_resolved(const TexResolved& a, const TexResolved& b) {
    if (a.tex != b.tex || a.has_view != b.has_view || a.has_sampler != b.has_sampler) return false;
    if (a.has_sampler && a.smp_ptr != b.smp_ptr && (!a.smp_ptr || !b.smp_ptr || !(*a.smp_ptr == *b.smp_ptr))) return false;
    if (!a.has_view) return true;
    const TextureViewInfo& x = a.view; const TextureViewInfo& y = b.view;
    return x.has_levels == y.has_levels && x.has_layers == y.has_layers && x.has_format == y.has_format && x.has_swizzle == y.has_swizzle &&
           x.has_depth_stencil_mode == y.has_depth_stencil_mode && x.has_target == y.has_target && x.min_level == y.min_level &&
           x.num_levels == y.num_levels && x.min_layer == y.min_layer && x.num_layers == y.num_layers && x.format == y.format &&
           std::memcmp(x.swizzle, y.swizzle, sizeof x.swizzle) == 0 && x.depth_stencil_mode == y.depth_stencil_mode && x.target == y.target;
}
TexResolved resolve_texture(const BindingDesc& d) {
    ResMemo* m = memo_for(d);
    if (!m) return resolve_texture_raw(d);
    if (m->te != g_memo_epoch) { m->t = resolve_texture_raw(d); m->te = g_memo_epoch; }
    else {
        ++g_rs.res_memo;
        if (rt_check()) chk(same_resolved(m->t, resolve_texture_raw(d)), "resolve_texture memo");
    }
    return m->t;
}
TexResolved resolve_image(const BindingDesc& d) {
    ResMemo* m = memo_for(d);
    if (!m) return resolve_image_raw(d);
    if (m->ie != g_memo_epoch) { m->i = resolve_image_raw(d); m->ie = g_memo_epoch; }
    else {
        ++g_rs.res_memo;
        if (rt_check()) chk(same_resolved(m->i, resolve_image_raw(d)), "resolve_image memo");
    }
    return m->i;
}
// [nvn] lean_draws: the memo's own entry instead of a copy of it (144 bytes, five times a binding a draw). The reference holds until the
// next MemoScope begins or ends; without a memo entry (or with the setting off) the answer is in `tmp`.
const TexResolved& resolve_texture_ref(const BindingDesc& d, TexResolved& tmp) {
    ResMemo* m = lean_on() ? memo_for(d) : nullptr;
    if (!m) { tmp = resolve_texture(d); return tmp; }
    if (m->te != g_memo_epoch) { m->t = resolve_texture_raw(d); m->te = g_memo_epoch; }
    else {
        ++g_rs.res_memo;
        if (rt_check()) chk(same_resolved(m->t, resolve_texture_raw(d)), "resolve_texture memo");
    }
    return m->t;
}
const TexResolved& resolve_image_ref(const BindingDesc& d, TexResolved& tmp) {
    ResMemo* m = lean_on() ? memo_for(d) : nullptr;
    if (!m) { tmp = resolve_image(d); return tmp; }
    if (m->ie != g_memo_epoch) { m->i = resolve_image_raw(d); m->ie = g_memo_epoch; }
    else {
        ++g_rs.res_memo;
        if (rt_check()) chk(same_resolved(m->i, resolve_image_raw(d)), "resolve_image memo");
    }
    return m->i;
}

TexResolved resolve_texture_raw(const BindingDesc& d) {
    ++g_rs.res_tex;
    TexResolved r;
    const DescriptorBinding& b = d.shader->refl.bindings[d.index];
    uint64_t handle = 0;
    if (b.nvn.source == NvnSource::SamplerBinding) {
        const int idx = b.nvn.index;
        if (idx >= 0 && idx < 64) handle = S.tex[d.nvn_stage][idx];
    } else if (b.nvn.source == NvnSource::HandleInUniform) {
        handle = read_handle_word(d.nvn_stage, b.nvn.bank, b.nvn.word);
    } else if (b.nvn.source == NvnSource::ImageBinding) {     // NVN2 compute: the bound image read through its handle (sampler id 0)
        const int idx = b.nvn.index;
        if (idx >= 0 && idx < 16) handle = S.img[d.nvn_stage][idx];
    }
    const uint32_t tid = handle_texture_id(handle), sid = handle_sampler_id(handle);
    if (S.tpool) {
        if (g_tpool_snap.pool.get() != S.tpool.get()) g_tpool_snap.sync(S.tpool.get());
        if (tid < g_tpool_snap.entries.size()) {
            const TpoolSnap::Entry& e = g_tpool_snap.entries[tid];
            if (e.used && e.tex) {
                r.tex = e.tex.get();
                r.view = e.view;
                r.has_view = e.has_view;
            }
        }
    }
    if (S.spool) {
        if (g_spool_snap.pool.get() != S.spool.get()) g_spool_snap.sync(S.spool.get());
        if (sid < g_spool_snap.entries.size() && g_spool_snap.entries[sid].used) {
            r.smp_ptr = &g_spool_snap.entries[sid].smp;
            r.vk_sampler = g_spool_snap.get_vk(sid);
            r.has_sampler = true;
        }
    }
    return r;
}

ViewRequest make_request(const DescriptorBinding& b, const TexResolved& r) {
    ViewRequest q;
    q.type = view_type_for(b);
    q.depth_compare = b.image.depth;
    const Texture& t = *r.tex;
    int sw[4] = {t.b.swizzle[0], t.b.swizzle[1], t.b.swizzle[2], t.b.swizzle[3]};
    int dsmode = t.b.depth_stencil_mode;
    if (r.has_view) {
        const TextureViewInfo& v = r.view;
        if (v.has_levels) { q.base_level = static_cast<uint32_t>(v.min_level); q.level_count = static_cast<uint32_t>(v.num_levels); }
        if (v.has_layers) { q.base_layer = static_cast<uint32_t>(v.min_layer); q.layer_count = static_cast<uint32_t>(v.num_layers); }
        if (v.has_format) {
            const FormatInfo* fi = format_info(v.format);
            if (fi) q.format = fi->vk;
        }
        if (v.has_swizzle) for (int i = 0; i < 4; ++i) sw[i] = v.swizzle[i];
        if (v.has_depth_stencil_mode) dsmode = v.depth_stencil_mode;
    }
    q.swizzle.r = vk_swizzle(sw[0]);
    q.swizzle.g = vk_swizzle(sw[1]);
    q.swizzle.b = vk_swizzle(sw[2]);
    q.swizzle.a = vk_swizzle(sw[3]);
    q.depth_aspect = dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
    switch (b.image.component) {
        case ScalarType::Float: q.component = 1; break;
        case ScalarType::Sint: q.component = 2; break;
        case ScalarType::Uint: q.component = 3; break;
        default: break;
    }
    return q;
}

// The image a draw reads or writes through an image binding (nvnCommandBufferBindImage): the handle is a texture pool id.
TexResolved resolve_image_raw(const BindingDesc& d) {
    ++g_rs.res_img;
    TexResolved r;
    const DescriptorBinding& b = d.shader->refl.bindings[d.index];
    uint64_t handle = 0;
    if (b.nvn.source == NvnSource::ImageBinding) {
        const int idx = b.nvn.index;
        if (idx >= 0 && idx < 16) handle = S.img[d.nvn_stage][idx];
    } else if (b.nvn.source == NvnSource::HandleInUniform) {
        handle = read_handle_word(d.nvn_stage, b.nvn.bank, b.nvn.word);
    }
    const uint32_t tid = handle_texture_id(handle);
    if (S.tpool) {
        if (g_tpool_snap.pool.get() != S.tpool.get()) g_tpool_snap.sync(S.tpool.get());
        if (tid < g_tpool_snap.entries.size()) {
            const TpoolSnap::Entry& e = g_tpool_snap.entries[tid];
            if (e.used && e.tex) {
                r.tex = e.tex.get();
                r.view = e.view;
                r.has_view = e.has_view;
            }
        }
    }
    return r;
}

// True when two texture objects are the same image: the object itself, or another object over the same storage (S4f: objects over one piece of
// memory share one image, so the one the pass renders to and the one a shader samples can be two objects).
bool shares_image(Texture* a, const Texture* b) {
    if (!a || !b) return false;
    if (a == b) return true;
    if (!tex_gpu(a)) return false;
    return a->gpu != nullptr && a->gpu == b->gpu;
}

// True when a texture the program samples is the depth attachment of the current target set: the draw then runs in a pass whose depth
// attachment is read-only (DEPTH_STENCIL_READ_ONLY_OPTIMAL serves as attachment and as sampled layout).
bool samples_depth_target(const ProgLayout& pl) {
    if (!S.depth_tex || pl.sampler_binding_idx.empty()) return false;
    for (uint32_t bi : pl.sampler_binding_idx) {
        const BindingDesc& d = pl.bindings[bi];
        TexResolved tmp;
        if (shares_image(resolve_texture_ref(d, tmp).tex, S.depth_tex.get())) return true;
    }
    return false;
}

// A colour attachment cannot be sampled in the pass that renders to it (no feedback loops): the binding gets the dummy view.
bool is_color_feedback(Texture* t) {
    if (!t) return false;
    if (lean_on()) {
        // [nvn] lean_draws: a texture that has its image shares an attachment's when the two image objects are one (what shares_image
        // finds after looking the texture's image up again for every attachment). Without an image: the full way, which makes it.
        if (tex_gpu(t)) {
            for (uint32_t i = 0; i < S.ncolor; ++i) {
                Texture* a = S.color_tex[i].get();
                if (!a) continue;
                if (!a->gpu) tex_gpu_attachment(a);
                if (a == t || a->gpu == t->gpu) return true;
            }
            return false;
        }
    }
    for (uint32_t i = 0; i < S.ncolor; ++i) {
        Texture* a = S.color_tex[i].get();
        if (a && !a->gpu) tex_gpu_attachment(a);   // an attachment gets its image (or the one it shares) here, not later in ensure_pass: the check runs before the pass begins (S4g)
        if (shares_image(t, a)) return true;
    }
    return false;
}

// A draw that samples its own colour attachment (the bloom composite of some frames reads the tone-mapped picture while it writes it back, one
// pixel at a time): the console allows that, Vulkan has no such loop. The draw samples a copy of the image made just before it, which holds the
// same values a pointwise read sees on the hardware (and the right ones when the shader reads another mip level of the attachment).
struct FbClone {
    Ref<Texture> tex;
    uint64_t last = 0;                                     // serial of the last submission that used the copy
};
std::unordered_map<Texture*, FbClone> g_fb_clone;          // attachment -> its copy (same shape, no storage in game memory)
std::unordered_map<Texture*, Texture*> g_fb_use;           // attachment -> the copy the current draw samples
std::unordered_set<Texture*> g_fb_valid;                   // attachments whose FbClone snapshot is reusable across consecutive barrier-free feedback draws
bool g_fb_seen = false;                                    // the last check found a feedback binding: check every draw until it goes away
// [nvn] fb_memo (round 45): the scan below is the replay's largest remaining cost (~2.2 us/draw, 31% of setup) because the engine
// rebinds the program between draws, which sets kDesc and therefore runs the scan again for the same layout and the same bindings.
// xc2-recomp 0.1.13 does the same scan once per (shader, texture bindings, pass) in runtime/vk_exec_draw.c feedback_scan (1310).
// g_bind_gen is the "the bindings this layout samples have not changed since" counter; the memo caches the NEGATIVE answer only:
// a draw that does need a copy still runs the full path (the copy is per draw).
//   [nvn] fb_memo: 0 = off, 1 = on, 2 = check (recompute on a hit, report a mismatch; xc2-recomp calls this XC2_TEX_MEMO=2)
uint64_t g_bind_gen = 1;
int fb_memo_mode() {
    // Default 1 since 2026-11-21: the scan costs 7.4 us per draw in the DLC's rainforest (11 ms a frame at 1543 draws) and
    // the memo cuts it to 4.2; mode 1 recomputes the first 20,000 hits and then one in 256, so a wrong answer shows up in
    // the log instead of silently binding the wrong copy.
    static const int mode = static_cast<int>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "fb_memo", 1), 0, 2));
    return mode;
}
struct FbMemoKey {
    const ProgLayout* pl = nullptr;
    void* cb = nullptr;
    uint64_t gen = 0;
    uint32_t ncolor = 0;
    const Texture* color[8] = {};
};
FbMemoKey g_fb_key;                       // the key of the last scan
bool g_fb_key_any = false;                // what that scan answered
uint64_t g_fb_memo_hits = 0, g_fb_memo_bad = 0;
std::atomic<int64_t> g_fb_clone_n{0};                      // g_fb_clone's size, mirrored for the gauge (the map itself is render thread only)

Texture* feedback_clone(Texture* t) {
    FbClone& fe = g_fb_clone[t];
    fe.last = F ? F->serial : 0;
    Ref<Texture>& c = fe.tex;
    g_fb_clone_n.store(static_cast<int64_t>(g_fb_clone.size()), std::memory_order_relaxed);
    if (c && (c->b.width != t->b.width || c->b.height != t->b.height || c->b.depth != t->b.depth || c->fmt != t->fmt || c->levels != t->levels || c->layers != t->layers ||
              (c->gpu && tex_scale(c.get()) != tex_scale(t)))) {
        c = Ref<Texture>();                                 // the game made another texture at this address
        g_fb_valid.erase(t);
    }
    if (!c) {
        Texture* n = new Texture();
        n->b = t->b;
        n->fmt = t->fmt;
        n->levels = t->levels;
        n->layers = t->layers;
        n->block_linear = t->block_linear;
        n->cpu_dirty.store(false);
        c = Ref<Texture>(n);
        static uint32_t next_id = 0x7f000000;
        n->id = next_id++;
    }
    return tex_gpu_scaled(c.get(), tex_scale(t)) ? c.get() : nullptr;
}

void feedback_prepare(const ProgLayout& pl) {
    g_fb_use.clear();
    g_color_feedback = 0;
    if (!S.ncolor || !pl.has_sampler) {   // no colour attachment, or nothing sampled: no binding can be a feedback one
        g_fb_seen = false;
        return;
    }
    // [nvn] fb_memo: same layout, same bindings, same pass, same targets => the last scan's answer still holds. Only the negative
    // answer is reused (a draw that samples its own target needs its own copy, made below).
    FbMemoKey k;
    std::memset(&k, 0, sizeof k);
    k.pl = &pl;
    k.cb = F->cb;
    k.gen = pl.has_handle_in_ubo ? (g_bind_gen + g_ubo_bind_gen) : g_bind_gen;
    k.ncolor = S.ncolor;
    for (uint32_t i = 0; i < S.ncolor && i < 8; ++i) k.color[i] = S.color_tex[i].get();
    const int memo = fb_memo_mode();
    const bool same = memo != 0 && g_fb_key.pl != nullptr && std::memcmp(&k, &g_fb_key, sizeof k) == 0;
    if (same) ++g_fb_memo_hits;
    // In mode 1 the first 20,000 hits and then one in 256 are recomputed anyway: the same self-check as mode 2, sampled, so a wrong
    // "no feedback" shows up in the log without paying for the scan on every draw (a run makes millions of hits).
    const bool recheck = same && (memo == 2 || (memo == 1 && (g_fb_memo_hits <= 20000 || (g_fb_memo_hits & 255u) == 0)));
    if (same && memo == 1 && !g_fb_key_any && !recheck) {
        g_fb_seen = false;
        return;
    }
    if (pl.sampler_binding_idx.empty()) {
        g_fb_seen = false;
        return;
    }
    static const bool fb_debug = nxrt::config_get_bool("nvn", "fb_debug", false);
    static const std::string fb_debug_hash = nxrt::config_get_str("nvn", "fb_debug_hash", "");
    Texture* rt_tex[8] = {};
    void* rt_gpu[8] = {};
    const uint32_t nrt = std::min<uint32_t>(S.ncolor, 8u);
    for (uint32_t i = 0; i < nrt; ++i) {
        Texture* a = S.color_tex[i].get();
        rt_tex[i] = a;
        if (a) {
            if (!a->gpu) tex_gpu_attachment(a);
            rt_gpu[i] = a->gpu;
        }
    }
    bool any = false;
    for (uint32_t bi : pl.sampler_binding_idx) {
        const BindingDesc& d = pl.bindings[bi];
        TexResolved tmp;
        const TexResolved& r = resolve_texture_ref(d, tmp);
        bool matches_rt = false;
        if (r.tex) {
            void* tg = r.tex->gpu;
            if (tg) {
                for (uint32_t i = 0; i < nrt; ++i) {
                    if (rt_tex[i] == r.tex || (rt_gpu[i] && rt_gpu[i] == tg)) { matches_rt = true; break; }
                }
            } else if (is_color_feedback(r.tex)) {
                matches_rt = true;
            }
        }
        if (fb_debug && S.stage[kFragment]) {
            const bool feedback = matches_rt && tex_gpu(r.tex);
            const bool wanted = fb_debug_hash.empty() ? feedback : S.stage[kFragment]->hash.rfind(fb_debug_hash, 0) == 0;
            static std::atomic<uint32_t> fb_logged{0};
            if (wanted && (fb_debug_hash.size() || fb_logged.fetch_add(1) < 500)) {
                NXLOG_I(kTag, "fb_debug: fs %s binding set%u b%u tex t%s gpu %d feedback %d samples %u used %d ncolor %u",
                        S.stage[kFragment]->hash.c_str(), d.set, d.binding, r.tex ? std::to_string(r.tex->id).c_str() : "-",
                        r.tex ? (tex_gpu(r.tex) ? 1 : 0) : -1, feedback ? 1 : 0,
                        r.tex ? r.tex->b.samples : 0u, r.tex ? (g_fb_use.count(r.tex) ? 1 : 0) : -1, S.ncolor);
            }
        }
        if (!matches_rt || !tex_gpu(r.tex) || r.tex->b.samples > 1 || g_fb_use.count(r.tex)) continue;
        any = true;
        // Hardware zero-copy feedback loop (VK_EXT_attachment_feedback_loop_layout): sample the colour attachment
        // directly in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT instead of breaking the pass and copying full-screen images.
        // Matches Switch NVN barrier(0x14) (ORDER_FRAGMENTS_TILED | INVALIDATE_TEXTURE) semantics with zero copies.
        static const bool fb_reuse = nxrt::config_get_bool("nvn", "fb_reuse", true);
        if (fb_reuse && device().features().attachment_feedback_loop_layout && tex_has_feedback_loop(r.tex)) {
            for (uint32_t i = 0; i < S.ncolor; ++i) {
                if (shares_image(r.tex, S.color_tex[i].get())) g_color_feedback |= (1u << i);
            }
            g_fb_use[r.tex] = r.tex;
            {
                static std::set<std::string> told_hw;
                const char* fsh = S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-";
                if (told_hw.insert(fsh).second)
                    NXLOG_I(kTag, "feedback loop (hw): fs %s samples its own colour attachment %u (%s %dx%d) in-place (zero copy)", fsh, r.tex->id,
                            r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width, r.tex->b.height);
            }
            continue;
        }
        Texture* c = feedback_clone(r.tex);
        if (!c) continue;
        // [nvn] fb_reuse (default 1): consecutive self-sampling draws (e.g. 96 ground decals sampling G-buffer t286/t288)
        // in the same render-target set without an intervening barrier, clear, copy, dispatch or non-feedback write reuse
        // the snapshot taken before the first draw — matching console texture-cache semantics and avoiding ~96 full-screen
        // 1440p copies and 5-MRT pass breaks per frame (~30 ms GPU saved in Uraya).
        if (fb_reuse && g_fb_valid.count(r.tex)) {
            g_fb_use[r.tex] = c;
            continue;
        }
        end_pass();
        tex_ensure_current(r.tex, *F);                                    // storage the game wrote since the image was made reaches the copy too
        tex_transition(r.tex, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);   // the next pass moves it back to the attachment layout and loads it
        tex_transition(c, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        std::vector<VkImageCopy> regions;
        for (uint32_t lv = 0; lv < tex_levels(r.tex); ++lv) {
            const VkExtent3D e = tex_image_extent(r.tex, lv);
            VkImageCopy rc = {};
            rc.srcSubresource = {tex_aspect(r.tex), lv, 0, tex_layers(r.tex)};
            rc.dstSubresource = {tex_aspect(c), lv, 0, tex_layers(c)};
            rc.extent = e;
            regions.push_back(rc);
        }
        vkCmdCopyImage(F->cb, tex_image(r.tex), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(c), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(regions.size()), regions.data());
        tex_mark_gpu_written(c);
        tex_transition(c, *F, tex_sampled_layout(c));
        g_fb_valid.insert(r.tex);
        g_fb_use[r.tex] = c;
        {   // one line per fragment shader that samples its own colour attachment (HANDOVER 51: which draws the copy covers)
            static std::set<std::string> told;
            const char* fsh = S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-";
            if (told.insert(fsh).second)
                NXLOG_I(kTag, "feedback copy: fs %s samples its own colour attachment %u (%s %dx%d): the copy is bound for this draw", fsh, r.tex->id,
                        r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width, r.tex->b.height);
        }
        counters().copies.fetch_add(1, std::memory_order_relaxed);
        gstamp("feedback copy");
    }
    // [nvn] fb_memo = 2, or the sampled check of mode 1: a hit was recomputed - the two answers must agree (xc2-recomp's
    // XC2_TEX_MEMO=2 does the same check).
    if (recheck && g_fb_key_any != any && g_fb_memo_bad++ < 8)
        NXLOG_W(kTag, "fb_memo: the memo said %d, a fresh scan says %d (layout %p, %u colour targets, %u sampler bindings)",
                g_fb_key_any ? 1 : 0, any ? 1 : 0, static_cast<const void*>(&pl), S.ncolor, pl.has_sampler ? 1u : 0u);
    if (memo) {
        g_fb_key = k;
        g_fb_key_any = any;
    }
    g_fb_seen = any;
}

void flush_compute_out(int why = kOutGpuReader);       // compute shadows (below): results of direct compute bindings go back to the pool
void compute_out_guard(uint64_t addr, uint64_t size, int why = kOutGpuReader);
// [addr, +size) of game memory a texture upload reads: held compute results there go back first (compute_out_lazy; else all of them).
void out_texture_read(Texture* t) {
    if (!out_lazy()) { flush_compute_out(kOutTexture); return; }
    if (t && t->mem && t->cpu_dirty.load()) compute_out_guard(reinterpret_cast<uint64_t>(t->mem), t->storage_size ? t->storage_size : 4096, kOutTexture);
}

// A sampler3D reading a texture whose storage is 2D (a 2D texture the game left bound to the slot, or a 3D texture view of 2D storage): the
// console samples it as a volume one slice deep. Vulkan cannot view a 2D image as 3D, so the draw samples a 3D copy (w x h x 1, every level) of the
// texture's layer, made before the pass and refreshed once per submission (and whenever the CPU wrote the texture).
struct Vol3dCopy {
    Ref<Texture> src;                                      // keeps the key alive while the copy exists
    Ref<Texture> tex;
    uint32_t layer = 0;
    uint64_t copied = UINT64_MAX;                          // serial of the submission of the last copy
    uint64_t last = 0;                                     // serial of the last submission that used it
};
std::unordered_map<Texture*, Vol3dCopy> g_vol3d;           // 2D texture -> its 3D copy
std::unordered_map<Texture*, Texture*> g_vol3d_use;        // 2D texture -> the copy the current draw samples

bool wants_vol3d(const DescriptorBinding& b, const TexResolved& r) {
    if (!r.tex || view_type_for(b) != VK_IMAGE_VIEW_TYPE_3D) return false;
    const int target = r.tex->b.target;
    return target != NVN_TEXTURE_TARGET_3D && target != NVN_TEXTURE_TARGET_BUFFER && !target_is_multisample(target) && r.tex->b.samples <= 1 &&
           !tex_is_depth(r.tex);
}

void vol3d_prepare(const ProgLayout& pl) {
    if (!g_vol3d_use.empty()) g_vol3d_use.clear();
    if (!pl.has_sampler_3d) return;
    for (uint32_t bi : pl.sampler_binding_idx) {
        const BindingDesc& d = pl.bindings[bi];
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.image.dim != 2) continue;
        TexResolved tmp;
        const TexResolved& r = resolve_texture_ref(d, tmp);   // nothing below begins a MemoScope, so the entry stays
        if (!wants_vol3d(b, r) || g_vol3d_use.count(r.tex) || !tex_gpu(r.tex)) continue;
        const uint32_t layer = r.has_view && r.view.has_layers ? static_cast<uint32_t>(r.view.min_layer) : 0;
        if (layer >= tex_layers(r.tex)) continue;
        Vol3dCopy& e = g_vol3d[r.tex];
        if (e.src.get() != r.tex) e.src = Ref<Texture>::share(r.tex);
        e.last = F ? F->serial : 0;
        Ref<Texture>& c = e.tex;
        if (c && (c->b.width != r.tex->b.width || c->b.height != r.tex->b.height || c->fmt != r.tex->fmt || c->levels != r.tex->levels || e.layer != layer ||
                  (c->gpu && tex_scale(c.get()) != tex_scale(r.tex))))
            c = Ref<Texture>();
        if (!c) {
            Texture* n = new Texture();
            n->b = r.tex->b;
            n->b.target = NVN_TEXTURE_TARGET_3D;
            n->b.depth = 1;
            n->fmt = r.tex->fmt;
            n->levels = r.tex->levels;
            n->layers = 1;
            n->block_linear = r.tex->block_linear;
            n->cpu_dirty.store(false);
            static uint32_t next_id = 0x7e000000;
            n->id = next_id++;
            c = Ref<Texture>(n);
            e.layer = layer;
            e.copied = UINT64_MAX;
        }
        if (!tex_gpu_scaled(c.get(), tex_scale(r.tex))) continue;
        const uint64_t serial = F ? F->serial : 0;
        if (e.copied != serial || r.tex->cpu_dirty.load()) {
            out_texture_read(r.tex);
            end_pass();
            tex_ensure_current(r.tex, *F);
            tex_transition(r.tex, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            tex_transition(c.get(), *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            std::vector<VkImageCopy> regions;
            for (uint32_t lv = 0; lv < tex_levels(r.tex); ++lv) {
                VkImageCopy rc = {};
                rc.srcSubresource = {tex_aspect(r.tex), lv, layer, 1};
                rc.dstSubresource = {tex_aspect(c.get()), lv, 0, 1};
                rc.extent = tex_image_extent(r.tex, lv);
                rc.extent.depth = 1;
                regions.push_back(rc);
            }
            vkCmdCopyImage(F->cb, tex_image(r.tex), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(c.get()), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(regions.size()), regions.data());
            tex_mark_gpu_written(c.get());
            tex_transition(c.get(), *F, tex_sampled_layout(c.get()));
            tex_transition(r.tex, *F, tex_sampled_layout(r.tex));   // the same draw may sample it as 2D too
            e.copied = serial;
            counters().copies.fetch_add(1, std::memory_order_relaxed);
            gstamp("3D copy of a 2D texture");
            if (first_time_key("vol3d " + std::to_string(r.tex->id)))
                NXLOG_I(kTag, "texture %u (%s %dx%d, target %d) is sampled as 3D by %s binding %s: it gets a 3D copy one slice deep", r.tex->id,
                        r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width, r.tex->b.height, r.tex->b.target, d.shader->hash.c_str(), b.name.c_str());
        }
        g_vol3d_use[r.tex] = c.get();
    }
}

// Makes the textures a draw samples current (uploaded, in their sampled layout) and the storage images it binds writable (GENERAL).
// Ends the pass when it has to touch images. The depth attachment itself is left to the pass.
void prepare_textures(const ProgLayout& pl) {
    vol3d_prepare(pl);
    // kept from call to call (render thread; never re-entered): three heap allocations a draw otherwise
    static std::vector<Texture*> need, need_storage;
    need.clear();
    need_storage.clear();
    g_storage_bound.clear();
    g_storage_unscale.clear();
    bool written_gfx = false;   // a draw (not a dispatch) writes a storage image, which stays unscaled
    uint64_t uint_samplers = 0;
    static const bool uint_tex_patch = nxrt::config_get_bool("nvn", "uint_tex_patch", true);
    static const bool ui_probe = nxrt::config_get_bool("fs", "tag_textures", false);
    for (uint32_t bi : pl.prep_tex_idx) {
        const BindingDesc& d = pl.bindings[bi];
        TexResolved tmp;
        if (d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            const TexResolved& r = resolve_image_ref(d, tmp);
            if (!r.tex || !tex_gpu(r.tex) || !tex_has_storage(r.tex)) continue;
            g_storage_bound.push_back(r.tex);
            // [gfx] render_scale: an image a fragment shader only reads (imageLoad at its own pixel, XC2 NS2's decals reading the G-buffer
            // normals) keeps its scale: the translator scales imageLoad coordinates like texelFetch (TexelFetchScale), and support_for_draw
            // hands it the scale. Fragment shaders whose 2D storage-image atomics were patched to raw ivec2(gl_FragCoord.xy)
            // (spirv_scale_frag_image_atomic, issue #94) also keep their scale when matched 1:1 with the pass's render target.
            // Other written images and compute bindings still go back to 1.
            bool keep = d.nvn_stage == kFragment && !d.shader->refl.bindings[d.index].written;
            if (!keep && d.nvn_stage == kFragment && render_scale() != 1.0f) {
                static std::unordered_map<const ShaderInfo*, bool> s_atomic_scaled;
                auto it = s_atomic_scaled.find(d.shader);
                if (it == s_atomic_scaled.end())
                    it = s_atomic_scaled.emplace(d.shader, spirv_can_scale_frag_image_atomic(d.shader->spirv)).first;
                if (it->second) {
                    Texture* rt = S.ncolor && S.color_tex[0] ? S.color_tex[0].get() : S.depth_tex.get();
                    if (rt && r.tex->b.width == rt->b.width && r.tex->b.height == rt->b.height && tex_scale(r.tex) == tex_scale(rt)) {
                        keep = true;
                    }
                }
            }
            if (r.tex->cpu_dirty.load() || tex_layout(r.tex) != VK_IMAGE_LAYOUT_GENERAL || (tex_scale(r.tex) != 1.0f && !keep)) need_storage.push_back(r.tex);
            if (!keep) g_storage_unscale.push_back(r.tex);
            if (!keep && d.nvn_stage != kCompute) written_gfx = true;
            continue;
        }
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        const TexResolved& r = resolve_texture_ref(d, tmp);
        if (!r.tex) continue;
        // [fs] tag_textures (HANDOVER 87): log the tagged resource this draw samples, once per texture per frame, so a UI element that
        // shows up in the frame can be named. Untagged textures (the game's own render targets, the effects) stay silent.
        if (ui_probe && r.tex->dbg_frame != g_dbg_frame) {
            r.tex->dbg_frame = g_dbg_frame;
            static const uint32_t from = static_cast<uint32_t>(config_get_int("nvn", "ui_probe_from", 0));
            static std::atomic<int> shown{0};
            if (g_dbg_frame >= from && shown.fetch_add(1) < 20000)
                NXLOG_I("nvn.ui", "frame %u: sampled texture %u%s%s", g_dbg_frame, r.tex->id,
                        r.tex->dbg_file[0] ? " from " : "", r.tex->dbg_file);
        }
        if (!tex_gpu(r.tex)) continue;
        const bool depth_owned = shares_image(r.tex, S.depth_tex.get());
        const bool fb_owned = d.nvn_stage != kCompute && is_color_feedback(r.tex);
        if (uint_tex_patch && d.nvn_stage == kFragment && d.index >= 0 && d.index < 64 &&
            b.image.component == ScalarType::Float && !b.image.depth) {
            bool want_uint = false;
            if (!tex_is_depth(r.tex)) {
                const VkFormat tf = tex_vk_format(r.tex);
                want_uint = vk_format_is_integer(tf) && !vk_format_is_signed_int(tf);
            } else if (!fb_owned && tex_aspect(r.tex) == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
                // Depth-stencil STENCIL plane sampled through a float sampler (#4, #39, #44, #54): rewrite to usampler at runtime
                // (matching patch_ds_samplers.py) so NVIDIA (where ds_stencil_bits defaults to 0 and float samplers read DEPTH32F,
                // causing iso-depth stripes in a299a459/292faa92) and AMD both read S8_UINT legally.
                int dsmode = r.tex->b.depth_stencil_mode;
                if (r.has_view && r.view.has_depth_stencil_mode) dsmode = r.view.depth_stencil_mode;
                static const std::string ds_depth_list = nxrt::config_get_str("nvn", "ds_depth_shaders", "");
                const bool in_ds_depth = !ds_depth_list.empty() && ds_depth_list.find(d.shader->hash) != std::string::npos;
                want_uint = (dsmode != NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH) && !in_ds_depth;
            }
            if (want_uint) {
                uint_samplers |= 1ull << d.index;
                if (first_time_key("uint-tex-patch " + d.shader->hash + " " + std::to_string(b.set) + " " + std::to_string(b.binding))) {
                    NXLOG_I(kTag, "uint_tex_patch: fs %s s%ub%u (%s) samples %s texture %u (%s) as usampler",
                            d.shader->hash.c_str(), b.set, b.binding, b.name.c_str(),
                            tex_is_depth(r.tex) ? "stencil plane of" : "integer",
                            r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?");
                }
            }
        }
        // owned by the pass; a compute dispatch runs outside every pass, so the graphics attachments do not own what it samples
        if (depth_owned || fb_owned) continue;
        if (r.tex->cpu_dirty.load() || tex_layout(r.tex) != tex_sampled_layout(r.tex)) {
            need.push_back(r.tex);
            if (r.tex->cpu_dirty.load() && ft::on()) ft::mark(b.nvn.source == NvnSource::SamplerBinding ? "tx.dirty_bind" : "tx.dirty_uniform", r.tex->id);
        }
    }
    g_draw_uint_samplers = uint_samplers;
    if (pl.has_set[kPoolSet]) {
        // a shader indexing the pools may sample any registered texture: those not ready, except what the pass renders to (their pool
        // elements hold the dummy meanwhile, nvn_pipeline.cpp)
        const bool compute = std::any_of(pl.bindings.begin(), pl.bindings.end(), [](const BindingDesc& d) { return d.nvn_stage == kCompute; });
        const size_t from = need.size();
        pool_textures_to_prepare(S.tpool.get(), need);
        need.erase(std::remove_if(need.begin() + static_cast<std::ptrdiff_t>(from), need.end(),
                                  [&](Texture* t) { return shares_image(t, S.depth_tex.get()) || (!compute && is_color_feedback(t)); }),
                   need.end());
    }
    // an image this op also writes (an NVN2 compute shader sampling its own image) stays in GENERAL, the layout both descriptors name
    need.erase(std::remove_if(need.begin(), need.end(),
                              [&](Texture* t) { return std::any_of(g_storage_bound.begin(), g_storage_bound.end(), [&](Texture* s) { return shares_image(t, s); }); }),
               need.end());
    // [gfx] render_scale: a draw that writes an unscaled storage image writes it at its own pixel (imageStore coordinates are not scaled),
    // so its render targets go back to 1 as well and the two stay texel for texel. XC2 NS2's light list: the list (RG32UI, a colour
    // target) and the registration mask (R32UI, a storage image) are rebuilt into each other (earlyz_mask) and read together; a list at
    // 2x beside a mask at 1x put every other tile's lights in the wrong place (a dot grid over the shadows).
    if (written_gfx && render_scale() != 1.0f) {
        Texture* ts[9];
        uint32_t n = 0;
        for (uint32_t i = 0; i < S.ncolor && i < 8; ++i) if (S.color_tex[i]) ts[n++] = S.color_tex[i].get();
        if (S.depth_tex) ts[n++] = S.depth_tex.get();
        bool any = false;
        for (uint32_t i = 0; i < n; ++i) any |= tex_gpu(ts[i]) && tex_scale(ts[i]) != 1.0f;
        if (any) {
            end_pass(true);
            for (uint32_t i = 0; i < n; ++i)
                if (tex_gpu(ts[i]) && tex_scale(ts[i]) != 1.0f) tex_rescale(ts[i], *F, 1.0f, "a render target of a draw that writes an unscaled storage image");
        }
    }
    if (need.empty() && need_storage.empty()) return;
    {   // [nvn] tex_prologue (default on): a texture no command has touched yet (its image is still UNDEFINED) gets its first upload in the
        // submission's prologue, a command buffer that runs before everything recorded for it, so the pass need not end. An area load drew
        // ~400 new textures in one frame, each ending and restarting the G-buffer pass (~90 ms of GPU time for that frame).
        static const bool pro_on = config_get_bool("nvn", "tex_prologue", true);
        if (need.size() >= 32) {   // diagnostic: what a big upload list holds (layouts, dirty, uploaded) - once per size class
            static int told = 0;
            if (told < 12) {
                ++told;
                std::map<int, int> lay;
                int dirty = 0, ov = 0;
                for (Texture* t : need) { lay[static_cast<int>(tex_layout(t))]++; dirty += t->cpu_dirty.load() ? 1 : 0; ov += t->overflow ? 1 : 0; }
                std::string sl;
                for (auto& kv : lay) sl += " " + std::to_string(kv.first) + ":" + std::to_string(kv.second);
                NXLOG_I(kTag, "tex_prologue diag: %zu textures to prepare at one draw (pool set %d): layouts%s, cpu_dirty %d, overflow %d", need.size(),
                        pl.has_set[kPoolSet] ? 1 : 0, sl.c_str(), dirty, ov);
            }
        }
        if (pro_on) {
            static uint64_t n_pro = 0;
            auto first_use = [](Texture* t) { return tex_layout(t) == VK_IMAGE_LAYOUT_UNDEFINED && !t->overflow; };
            if (std::any_of(need.begin(), need.end(), first_use)) {
                if (VkCommandBuffer pc = F->prologue()) {
                    Frame pf = *F;
                    pf.cb = pc;
                    for (Texture* t : need) {
                        if (!first_use(t)) continue;
                        out_texture_read(t);
                        tex_ensure_current(t, pf);
                        tex_transition(t, pf, tex_sampled_layout(t));
                        ++n_pro;
                    }
                    need.erase(std::remove_if(need.begin(), need.end(), [](Texture* t) { return tex_layout(t) == tex_sampled_layout(t) && !t->cpu_dirty.load(); }),
                               need.end());
                    if ((n_pro & 1023) == 1) NXLOG_I(kTag, "tex_prologue: %llu first uploads recorded in a submission prologue so far", (unsigned long long)n_pro);
                }
            }
            if (need.empty() && need_storage.empty()) return;
        }
    }
    for (Texture* t : need) out_texture_read(t);           // an upload reads game memory: compute results held there go back first
    for (Texture* t : need_storage) out_texture_read(t);
    end_pass(true);     // layout changes and uploads follow: each carries the dependency for its own image
    counters().pass_breaks.fetch_add(1, std::memory_order_relaxed);
    for (Texture* t : need) {
        tex_ensure_current(t, *F);
        tex_transition(t, *F, tex_sampled_layout(t));
    }
    for (Texture* t : need_storage) {
        if (tex_scale(t) != 1.0f && std::find(g_storage_unscale.begin(), g_storage_unscale.end(), t) != g_storage_unscale.end())
            tex_rescale(t, *F, 1.0f, "bound as a storage image");   // imageStore coordinates are not scaled (Ryujinx: ResScaleUnsupported)
        if (t->is_display && extras::hdr_highlights_kept()) tex_promote_float(t, *F, "the window");
        tex_ensure_current(t, *F);
        tex_transition(t, *F, VK_IMAGE_LAYOUT_GENERAL);
        tex_mark_gpu_written(t);
    }
    gstamp("uploads/layouts");
}

// SPIR-V ImageFormat of a storage image -> VkFormat (UNDEFINED: Unknown or unmapped).
VkFormat image_format_vk(uint32_t f) {
    switch (f) {
        case 1: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case 2: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case 3: return VK_FORMAT_R32_SFLOAT;
        case 4: return VK_FORMAT_R8G8B8A8_UNORM;
        case 5: return VK_FORMAT_R8G8B8A8_SNORM;
        case 6: return VK_FORMAT_R32G32_SFLOAT;
        case 7: return VK_FORMAT_R16G16_SFLOAT;
        case 8: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case 9: return VK_FORMAT_R16_SFLOAT;
        case 10: return VK_FORMAT_R16G16B16A16_UNORM;
        case 11: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case 12: return VK_FORMAT_R16G16_UNORM;
        case 13: return VK_FORMAT_R8G8_UNORM;
        case 14: return VK_FORMAT_R16_UNORM;
        case 15: return VK_FORMAT_R8_UNORM;
        case 16: return VK_FORMAT_R16G16B16A16_SNORM;
        case 17: return VK_FORMAT_R16G16_SNORM;
        case 18: return VK_FORMAT_R8G8_SNORM;
        case 19: return VK_FORMAT_R16_SNORM;
        case 20: return VK_FORMAT_R8_SNORM;
        case 21: return VK_FORMAT_R32G32B32A32_SINT;
        case 22: return VK_FORMAT_R16G16B16A16_SINT;
        case 23: return VK_FORMAT_R8G8B8A8_SINT;
        case 24: return VK_FORMAT_R32_SINT;
        case 25: return VK_FORMAT_R32G32_SINT;
        case 26: return VK_FORMAT_R16G16_SINT;
        case 27: return VK_FORMAT_R8G8_SINT;
        case 28: return VK_FORMAT_R16_SINT;
        case 29: return VK_FORMAT_R8_SINT;
        case 30: return VK_FORMAT_R32G32B32A32_UINT;
        case 31: return VK_FORMAT_R16G16B16A16_UINT;
        case 32: return VK_FORMAT_R8G8B8A8_UINT;
        case 33: return VK_FORMAT_R32_UINT;
        case 34: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case 35: return VK_FORMAT_R32G32_UINT;
        case 36: return VK_FORMAT_R16G16_UINT;
        case 37: return VK_FORMAT_R8G8_UINT;
        case 38: return VK_FORMAT_R16_UINT;
        case 39: return VK_FORMAT_R8_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

int storage_slot_for(int nvn_stage, int c0_offset) {
    // c[0] storage buffer descriptors (byte offsets, as the translator reports them): 0x110 + 0x100 * hardware stage + 0x10 * binding
    // with hardware stage 0 vertex, 3 geometry, 4 fragment; compute: 0x310 + 0x10 * binding. Checked against the translated shaders of the
    // game (vertex 0x110, geometry 0x410, fragment 0x510, compute 0x310 and 0x320).
    int base = 0;
    switch (nvn_stage) {
        case kVertex: base = 0x110; break;
        case kTessCtl: base = 0x210; break;     // tessellation: Ryujinx's hardware stage order (1, 2), not checked on a game shader
        case kTessEval: base = 0x310; break;
        case kGeometry: base = 0x410; break;
        case kFragment: base = 0x510; break;
        case kCompute: base = 0x310; break;
        default: return 0;
    }
    if (c0_offset < base) return 0;
    return (c0_offset - base) / 0x10;
}

// ---- compute shadows: storage and texel buffers of a dispatch in device-local memory ---------------------------------------------
// Pools are the game's own memory imported into Vulkan, so a shader reaches storage buffers and texel buffers over PCIe. Compute shaders
// that work on such buffers (the exposure histogram's atomics, the 4 MiB simulation buffers) took 9 ms each on an RTX 3080. [nvn]
// compute_shadows (default on): each storage or texel buffer range a dispatch binds (up to 16 MiB) is bound from device-local memory.
//  * Direct ([nvn] compute_direct, default on): the range is in a pool with a device-local copy (nvn_gpu_pool.cpp, the pool mirror). The
//    dispatch binds the copy itself. Pages the copy does not hold (the CPU wrote them since the GPU last used them) are copied in from the
//    pool first (mirror_fill), so only the CPU's writes cross PCIe. A range the shader may write is written in the copy; it goes back to the
//    pool once, when something other than a compute binding of the same range could look at it: before any command other than a dispatch,
//    a state change or a barrier, before a dispatch reads it another way (uniform block, indirect arguments, another range on its pages,
//    a texture upload), and at the end of the submission, before the serial the CPU waits for can signal. A run of dispatches over the
//    same buffers thus copies each result out once instead of once per dispatch.
//  * Shadow (otherwise: no copy for the pool, hot or banned pages, a range over two chunks): the range is copied into a buffer of its own
//    before the dispatch and copied back after it when the shader can write it (and into the pool copy, when there is one).
// Game memory holds every result by the end of the submission, in queue order, as before. Ranges that overlap another range of the same
// dispatch stay in the pool.
struct Shadow { gfx::Buffer buf; VkDeviceSize size = 0; uint64_t last = 0; };
std::unordered_map<uint64_t, Shadow> g_shadows;            // by GPU address and size
struct ShadowUse { uint64_t addr; VkDeviceSize size; BufRef host; VkBuffer buf; VkDeviceSize off; bool writes; bool direct; MirrorSpan mirror; };
std::vector<ShadowUse> g_shadow_uses;                       // this dispatch
// Results of direct bindings still only in the pool copy: nvn_gpu_pool.cpp (out_add ... out_end_submit, "compute out").
bool g_shadowing = false;
bool g_sh_barrier = false, g_sh_copied = false;             // this dispatch: the barrier before the copies in is recorded / some copy in is
uint64_t g_sh_uses = 0, g_sh_from_mirror = 0, g_sh_bytes_mirror = 0, g_sh_bytes_pool = 0, g_sh_bytes_out = 0, g_sh_no_span = 0;   // summary line
uint64_t g_sh_direct = 0, g_sh_direct_filled = 0, g_sh_bytes_fill = 0, g_sh_bytes_dout = 0, g_sh_merged = 0, g_sh_flushes = 0;
constexpr VkDeviceSize kShadowMax = 16ull << 20;

bool compute_shadows_on() {
    static const bool on = nxrt::config_get_bool("nvn", "compute_shadows", true);
    return on;
}
bool compute_direct_on() {
    static const bool on = nxrt::config_get_bool("nvn", "compute_direct", true);
    return on;
}

// Every result still only in the pool copy goes back to the pool (in queue order, before what is recorded next).
void flush_compute_out(int why) {
    if (!out_any()) return;
    end_pass();
    if (out_flush(F->cb, 0, UINT64_MAX, why)) {
        ++g_sh_flushes;
        gstamp("compute out");
    }
}

// [addr, addr + size) is about to be read or written some other way than a direct binding of exactly a held range: the results sharing a
// page with it go back first (a page granular test: the pool copy is filled and tracked by whole pages).
void compute_out_guard(uint64_t addr, uint64_t size, int why) {
    if (!size || !out_overlaps(addr, size)) return;
    end_pass();
    if (out_flush(F->cb, addr, size, why)) {
        ++g_sh_flushes;
        gstamp("compute out");
    }
}

// Ops after which the pending results may stay in the pool copy: state that touches no memory, barriers (the copies carry their own),
// dispatches (they check their own ranges) and calls of other lists (their ops are checked one by one). Anything else may read the pool.
bool compute_out_may_wait(Op op) {
    if (op == Op::Dispatch || op == Op::DispatchIndirect || op == Op::Barrier || op == Op::CallCommands || op == Op::PushDebugGroup || op == Op::PopDebugGroup ||
        op == Op::PushConstants) return true;
    return op < Op::DrawArrays && op != Op::SetRenderEnable && op != Op::SetRenderTargets;
}

// A buffer a draw reads: results it shares a page with go back first, unless the draw binds the pool copy for it (vertex, index and aligned
// uniform buffers whose pages are all current there, resolve_read).
void out_draw_read(uint64_t addr, uint64_t size, bool copy_ok, int why) {
    if (!addr) return;
    if (!size) size = 65536;
    if (!out_overlaps(addr, size)) return;
    if (copy_ok && out_servable(addr, size)) return;
    compute_out_guard(addr, size, why);
}

// The buffer for [addr, +size) that a command reads through a plain buffer binding (storage buffers, indirect arguments, copy sources): the
// pool copy when it holds a compute result there and is current for the whole range, else the pool, after the results there went back
// (flush: false inside a pass, where out_before_op has done that already).
BufRef held_or_pool(uint64_t addr, uint64_t size, int why, bool flush) {
    if (size && out_overlaps(addr, size)) {
        if (out_lazy() && out_servable(addr, size)) {
            const MirrorSpan ms = mirror_span(addr, size);
            BufRef r = resolve_buffer(addr, size);
            if (ms.buf && r.ok) {
                r.buffer = ms.buf;
                r.offset = ms.offset;
                r.avail = ms.avail;
                return r;
            }
        }
        if (flush) compute_out_guard(addr, size, why);
    }
    return resolve_buffer(addr, size);
}

// out_draw_bindings' memo, one slot per binding it checks: the range last found clear of held results, and out_gen() then. The same
// range at the same generation is clear again (out_overlaps depends on nothing else). The checks were ~7 % of the render thread: ~180
// slots a draw, most of them bound to the same range as at the draw before (notes/rt.md). [nvn] out_memo (default on).
bool out_memo_on() {
    static const bool on = nxrt::config_get_bool("nvn", "out_memo", true);
    return on;
}
struct OutSlot { uint64_t addr = 0, size = 0, gen = ~0ull; };
OutSlot g_out_slot[16 + kCompute * 32 + 4];
void out_draw_read_slot(OutSlot& m, const GState::Bind& b, bool copy_ok, int why, uint64_t gen) {
    if (!b.addr) return;
    if (m.gen != ~0ull && m.addr == b.addr && m.size == b.size &&
        (m.gen == gen || out_clear_since(b.addr, b.size ? b.size : 65536, m.gen))) {   // clear then, and nothing held there was added since
        if (rt_check()) chk(!out_overlaps(b.addr, b.size ? b.size : 65536), "out memo");
        m.gen = gen;
        return;
    }
    if (!out_overlaps(b.addr, b.size ? b.size : 65536)) {
        m.addr = b.addr;
        m.size = b.size;
        m.gen = gen;
        return;
    }
    m.gen = ~0ull;
    out_draw_read(b.addr, b.size, copy_ok, why);
}

// The reads of a draw's bound buffers (out_before_op).
void out_draw_bindings() {
    if (out_memo_on()) {
        const uint64_t gen = out_gen();
        const VkDeviceSize ua = device().properties().limits.minUniformBufferOffsetAlignment;
        OutSlot* m = g_out_slot;
        if (lean_on()) {   // only the bound slots, by the masks the BindBuffers op keeps: an unbound slot does nothing below either
            if (rt_check()) {
                uint32_t v = 0;
                for (uint32_t k = 0; k < 16; ++k) v |= S.vtx[k].addr ? 1u << k : 0u;
                bool ok = v == g_bm_vtx;
                for (int st = 0; st < kCompute; ++st)
                    for (uint32_t k = 0; k < 16; ++k)
                        ok = ok && !!S.ubo[st][k].addr == !!(g_bm_ubo[st] >> k & 1) && !!S.ssbo[st][k].addr == !!(g_bm_ssbo[st] >> k & 1);
                chk(ok, "bound-slot masks");
            }
            for (uint32_t bm = g_bm_vtx; bm; bm &= bm - 1) {
                const uint32_t k = static_cast<uint32_t>(std::countr_zero(bm));
                out_draw_read_slot(m[k], S.vtx[k], true, kOutDrawBuffer, gen);
            }
            m += 16;
            for (int st = 0; st < kCompute; ++st, m += 32) {
                if (!S.stage[st]) continue;
                for (uint32_t bm = g_bm_ubo[st]; bm; bm &= bm - 1) {
                    const uint32_t k = static_cast<uint32_t>(std::countr_zero(bm));
                    out_draw_read_slot(m[k], S.ubo[st][k], ua && (S.ubo[st][k].addr % ua) == 0, kOutDrawBuffer, gen);
                }
                for (uint32_t bm = g_bm_ssbo[st]; bm; bm &= bm - 1) {
                    const uint32_t k = static_cast<uint32_t>(std::countr_zero(bm));
                    out_draw_read_slot(m[16 + k], S.ssbo[st][k], true, kOutStorageXfb, gen);
                }
            }
            for (const GState::Bind& b : S.xfb) out_draw_read_slot(*m++, b, false, kOutStorageXfb, gen);
            return;
        }
        for (const GState::Bind& b : S.vtx) out_draw_read_slot(*m++, b, true, kOutDrawBuffer, gen);
        for (int st = 0; st < kCompute; ++st) {
            if (!S.stage[st]) { m += 32; continue; }
            for (const GState::Bind& b : S.ubo[st]) out_draw_read_slot(*m++, b, ua && (b.addr % ua) == 0, kOutDrawBuffer, gen);
            for (const GState::Bind& b : S.ssbo[st]) out_draw_read_slot(*m++, b, true, kOutStorageXfb, gen);
        }
        for (const GState::Bind& b : S.xfb) out_draw_read_slot(*m++, b, false, kOutStorageXfb, gen);
        return;
    }
    for (const GState::Bind& b : S.vtx) out_draw_read(b.addr, b.size, true, kOutDrawBuffer);
    const VkDeviceSize ualign = device().properties().limits.minUniformBufferOffsetAlignment;
    for (int st = 0; st < kCompute; ++st) {
        if (!S.stage[st]) continue;
        for (const GState::Bind& b : S.ubo[st]) out_draw_read(b.addr, b.size, ualign && (b.addr % ualign) == 0, kOutDrawBuffer);
        for (const GState::Bind& b : S.ssbo[st]) out_draw_read(b.addr, b.size, true, kOutStorageXfb);   // bound from the copy (build_set)
    }
    for (const GState::Bind& b : S.xfb) out_draw_read(b.addr, b.size, false, kOutStorageXfb);
}

const uint8_t* cpu_view(uint64_t addr, uint64_t size);   // below

// The bytes from index_address an indexed indirect draw reads: max over its draws of (first index + count) * index size, from the
// arguments (and draw count) when the CPU can read them (not a held result, not written by a submission that has not completed).
// Otherwise the GPU writes them too and nobody knows the count here: everything from index_address to the end of its pool.
uint64_t indirect_index_span(const P_DrawIndirect& p) {
    const uint32_t isz0 = index_size_bytes(static_cast<int>(p.type));
    const uint64_t isz = isz0 ? isz0 : 4;
    const uint64_t stride = p.stride ? static_cast<uint64_t>(p.stride) : 20;
    const auto whole = [&] { const BufRef b = resolve_buffer(p.index_address, 1); return b.ok && b.avail ? static_cast<uint64_t>(b.avail) : uint64_t{1} << 20; };
    uint64_t n = 1;
    if (p.multi && p.count_address) {
        const uint8_t* c = out_overlaps(p.count_address, 4) ? nullptr : cpu_view(p.count_address, 4);
        if (!c) return whole();
        uint32_t v = 0;
        std::memcpy(&v, c, 4);
        n = std::min<uint64_t>(v, p.max_count);
    }
    if (!n) return 0;
    const uint64_t args = stride * (n - 1) + 20;
    const uint8_t* a = out_overlaps(p.indirect_address, args) ? nullptr : cpu_view(p.indirect_address, args);
    if (!a) return whole();
    uint64_t end = 0;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t v[3];
        std::memcpy(v, a + i * stride, sizeof v);   // count, instance count, first index
        if (v[0] && v[1]) end = std::max(end, (static_cast<uint64_t>(v[2]) + v[0]) * isz);
    }
    return end;
}

// [nvn] compute_out_lazy: before op, the results it may read or overwrite through the pool go back (instead of all of them before any
// command that is not a dispatch). Ops that only touch images, state and syncs leave them in the copy; a fence's serial is the
// submission's, and out_end_submit decides at its end.
void out_before_op(const RecordReader& r) {
    if (!out_any()) return;
    switch (r.op()) {
        case Op::Dispatch: case Op::DispatchIndirect: case Op::Barrier: case Op::CallCommands: case Op::PushDebugGroup: case Op::PopDebugGroup:
        case Op::ClearColor: case Op::ClearDepthStencil: case Op::ClearTexture: case Op::CopyTextureToTexture: case Op::Downsample:
        case Op::DrawTexture: case Op::DiscardColor: case Op::DiscardDepthStencil: case Op::ResolveDepth: case Op::FenceSync: case Op::WaitSync:
        case Op::ResetCounter: case Op::WaitEvent: case Op::SetRenderEnable: case Op::EvaluateDlss:
        case Op::PushConstants:                                        // the Deferred form reads its data through cpu_view (never a held result)
            return;
        case Op::CopyBufferToTexture: case Op::CopyTextureToBuffer:    // do_copy_buffer_texture: its own range (it knows the size)
            return;
        case Op::DrawArrays:
        case Op::DrawElements: {
            out_draw_bindings();
            if (r.op() == Op::DrawElements) {
                const auto& p = r.payload<P_DrawElements>();
                const uint32_t isz = index_size_bytes(static_cast<int>(p.type));
                out_draw_read(p.index_address, static_cast<uint64_t>(p.count) * (isz ? isz : 4), true, kOutDrawBuffer);
            }
            return;
        }
        case Op::DrawIndirect: {
            out_draw_bindings();
            const auto& p = r.payload<P_DrawIndirect>();
            const bool indexed = p.type != 0xffffffffu;
            const uint64_t cmd = indexed ? 20 : 16;
            const uint64_t stride = p.stride ? static_cast<uint64_t>(p.stride) : cmd;
            const uint64_t n = p.multi && p.count_address ? std::max<uint64_t>(1, p.max_count) : 1;
            out_draw_read(p.indirect_address, stride * (n - 1) + cmd, true, kOutIndirect);   // do_draw_indirect binds the copy for them
            if (p.multi && p.count_address) out_draw_read(p.count_address, 4, true, kOutIndirect);
            if (indexed) compute_out_guard(p.index_address, indirect_index_span(p), kOutIndirect);   // the draw binds the pool for its indices
            return;
        }
        case Op::ClearBuffer: {
            const auto& p = r.payload<P_ClearBuffer>();
            compute_out_guard(p.address, p.size, kOutBufferWrite);
            return;
        }
        case Op::CopyBufferToBuffer: {
            const auto& p = r.payload<P_CopyBufferToBuffer>();
            out_draw_read(p.src, p.size, true, kOutBufferRead);       // the copy reads the pool copy (held_or_pool)
            compute_out_guard(p.dst, p.size, kOutBufferWrite);
            return;
        }
        case Op::UpdateUniformBuffer: {
            const auto& p = r.payload<P_UpdateUniformBuffer>();
            compute_out_guard(p.address + static_cast<uint64_t>(p.offset), p.size, kOutBufferWrite);
            return;
        }
        case Op::ReportCounter: compute_out_guard(r.payload<P_Counter>().address, 32, kOutBufferWrite); return;
        case Op::ReportValue: compute_out_guard(r.payload<P_ReportValue>().address, 32, kOutBufferWrite); return;
        case Op::SignalEvent: {
            const auto& p = r.payload<P_Event>();
            if (p.event && p.event->pool && p.event->pool->base)
                compute_out_guard(p.event->pool->gpu_base + static_cast<uint64_t>(p.event->offset), 4, kOutEvent);
            return;
        }
        case Op::DrawXfb: flush_compute_out(kOutIndirect); return;
        default:
            if (r.op() < Op::DrawArrays) return;    // state
            flush_compute_out(kOutOther);           // feedback begin/end and the rest: everything first
            return;
    }
}

void sh_barrier_once() {
    if (g_sh_barrier) return;
    memory_barrier(F->cb);       // earlier writes to these ranges (any command) before the copies read them
    g_sh_barrier = true;
}

// The buffer a dispatch binds for [addr, addr + size): the pool copy, the shadow (its copy-in recorded now), or `host` when it cannot have
// either.
BufRef shadow_of(uint64_t addr, VkDeviceSize size, const BufRef& host, bool writes) {
    if (!g_shadowing || !host.ok || size == 0) return host;
    for (ShadowUse& u : g_shadow_uses) {
        if (u.addr == addr && u.size == size) {           // the same range bound twice: one binding
            u.writes |= writes;
            BufRef r = host; r.buffer = u.buf; r.offset = u.off; r.avail = size;
            return r;
        }
        if (addr < u.addr + u.size && u.addr < addr + size) {   // overlapping ranges stay coherent in the pool
            compute_out_guard(addr, size, kOutDispatch);
            return host;
        }
    }
    // a result an earlier dispatch left in the pool copy: this dispatch binds it there again (the pages are not valid for the tracker
    // until the result goes back, so mirror_span would not say so)
    {
        VkBuffer ob = VK_NULL_HANDLE;
        VkDeviceSize oo = 0;
        if (out_exact(addr, size, &ob, &oo)) {
            ++g_sh_uses;
            ++g_sh_direct;
            g_shadow_uses.push_back({addr, size, host, ob, oo, writes, true, MirrorSpan{ob, oo, true}});
            BufRef r = host; r.buffer = ob; r.offset = oo; r.avail = size;
            return r;
        }
    }
    if (!out_lazy()) compute_out_guard(addr, size, kOutDispatch);
    if (size > kShadowMax) { compute_out_guard(addr, size, kOutDispatch); return host; }
    const MirrorSpan ms = mirror_span(addr, size);    // pages holding a compute result count as current there (compute_out_lazy)
    if (ms.buf && compute_direct_on()) {
        bool ok = ms.valid;
        if (!ok) {
            sh_barrier_once();
            uint64_t filled = 0;
            ok = mirror_fill(F->cb, addr, size, &filled);
            if (filled) { g_sh_copied = true; g_sh_bytes_fill += filled; }
            if (ok) ++g_sh_direct_filled;
        }
        if (ok) {
            ++g_sh_uses;
            ++g_sh_direct;
            g_shadow_uses.push_back({addr, size, host, ms.buf, ms.offset, writes, true, ms});
            BufRef r = host; r.buffer = ms.buf; r.offset = ms.offset; r.avail = size;
            return r;
        }
    }
    compute_out_guard(addr, size, kOutDispatch);      // the shadow reads the pool: held results there go back first
    const uint64_t key = addr ^ (static_cast<uint64_t>(size) << 40) ^ (static_cast<uint64_t>(size) >> 24);
    Shadow& sh = g_shadows[key];
    if (!sh.buf || sh.size != size) {
        if (sh.buf) { gfx::Buffer old = sh.buf; forget_buffer_views(old.buffer); defer_destroy([old]() mutable { device().destroy_buffer(old); }); sh.buf = gfx::Buffer{}; }
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (device().create_buffer(size, usage, gfx::MemoryUse::GpuOnly, &sh.buf, "nvn compute shadow") != VK_SUCCESS) {
            sh.buf = gfx::Buffer{};
            g_shadows.erase(key);
            return host;
        }
        sh.size = size;
    }
    sh.last = F->serial;
    sh_barrier_once();
    // The device-local pool copy, when it holds the range, is the cheap source (video memory to video memory instead of over PCIe); after a
    // dispatch that writes the range, the result goes to both, so the next dispatch finds the copy valid again (mirror_span_written).
    ++g_sh_uses;
    if (!ms.buf) ++g_sh_no_span;
    if (ms.valid) { ++g_sh_from_mirror; g_sh_bytes_mirror += size; } else g_sh_bytes_pool += size;
    VkBufferCopy c = {ms.valid ? ms.offset : host.offset, 0, size};
    vkCmdCopyBuffer(F->cb, ms.valid ? ms.buf : host.buffer, sh.buf.buffer, 1, &c);
    g_sh_copied = true;
    g_shadow_uses.push_back({addr, size, host, sh.buf.buffer, 0, writes, false, ms});
    BufRef r = host;
    r.buffer = sh.buf.buffer;
    r.offset = 0;
    r.avail = size;
    return r;
}

// After the dispatch: the written shadows go back to the pool; written direct ranges wait in g_out_pending. Shadows unused for a while
// are freed.
void shadows_after_dispatch() {
    bool any = false;
    for (const ShadowUse& u : g_shadow_uses) {
        if (!u.writes) continue;
        if (u.direct) {
            if (out_add(u.addr, u.size, u.buf, u.off, u.host)) ++g_sh_merged;   // written again before it went back: one copy out for both
            continue;
        }
        if (!any) { memory_barrier(F->cb); any = true; }
        VkBufferCopy c = {0, u.host.offset, u.size};
        vkCmdCopyBuffer(F->cb, u.buf, u.host.buffer, 1, &c);
        g_sh_bytes_out += u.size;
        if (u.mirror.buf) {
            VkBufferCopy mc = {0, u.mirror.offset, u.size};
            vkCmdCopyBuffer(F->cb, u.buf, u.mirror.buf, 1, &mc);
            mirror_span_written(u.addr, u.size);
        }
    }
    if (any) gstamp("shadow out");
    g_shadow_uses.clear();
    g_shadowing = false;
    static uint64_t last_sweep = 0;
    if (F->serial - last_sweep > 600) {
        last_sweep = F->serial;
        NXLOG_I("nvn.replay", "compute shadows: %llu bound (%llu from the pool copy, %llu with no copy), in from the copy %llu MiB, in from the pool %llu MiB, out %llu MiB; "
                "direct to the pool copy %llu (%llu after filling pages, in %llu MiB, out %llu MiB in %llu flushes, %llu writes merged)",
                (unsigned long long)g_sh_uses, (unsigned long long)g_sh_from_mirror, (unsigned long long)g_sh_no_span, (unsigned long long)(g_sh_bytes_mirror >> 20),
                (unsigned long long)(g_sh_bytes_pool >> 20), (unsigned long long)(g_sh_bytes_out >> 20), (unsigned long long)g_sh_direct,
                (unsigned long long)g_sh_direct_filled, (unsigned long long)(g_sh_bytes_fill >> 20), (unsigned long long)(out_bytes_total() >> 20),
                (unsigned long long)g_sh_flushes, (unsigned long long)g_sh_merged);
        for (auto it = g_shadows.begin(); it != g_shadows.end();) {
            if (F->serial - it->second.last > 600) {
                gfx::Buffer old = it->second.buf;
                forget_buffer_views(old.buffer);
                defer_destroy([old]() mutable { device().destroy_buffer(old); });
                it = g_shadows.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// ---- [nvn] earlyz: the scratch target of the depth pre-pass (HANDOVER 42.22.2) --------------------------------------------------
// The pre-pass runs the draw's own fragment shader (through a pipeline variant with every colour write masked off), and that shader
// registers the light in the storage image t51. The pre-pass binds this image there instead, so neither the colour target nor the mask
// of the real draw sees anything. One image per format, grown on demand: the shader's tile coordinates come from gl_FragCoord, so it
// only has to be at least as large as the target.
struct EarlyzScratchImage {
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    gfx::Image img;
    uint32_t w = 0, h = 0;
};
EarlyzScratchImage g_earlyz_scratch_img;
VkImageView earlyz_scratch_view(VkFormat fmt, uint32_t w, uint32_t h) {
    if (fmt == VK_FORMAT_UNDEFINED || !w || !h) return VK_NULL_HANDLE;
    if (g_earlyz_scratch_img.img && g_earlyz_scratch_img.fmt == fmt && g_earlyz_scratch_img.w >= w && g_earlyz_scratch_img.h >= h)
        return g_earlyz_scratch_img.img.view;
    if (g_earlyz_scratch_img.img) {
        gfx::Image old = g_earlyz_scratch_img.img;
        defer_destroy([old]() mutable { device().destroy_image(old); });
        g_earlyz_scratch_img = EarlyzScratchImage{};
    }
    gfx::ImageDesc d;
    d.format = fmt;
    d.width = std::max(w, 1024u);
    d.height = std::max(h, 1024u);
    d.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    d.name = "nvn earlyz scratch";
    std::string why;
    if (device().create_image(d, &g_earlyz_scratch_img.img, &why) != VK_SUCCESS) {
        NXLOG_W(kTag, "earlyz: no scratch storage image (format %d, %ux%u), the depth pre-pass stays off: %s", static_cast<int>(fmt), d.width, d.height, why.c_str());
        g_earlyz_scratch_img = EarlyzScratchImage{};
        return VK_NULL_HANDLE;
    }
    g_earlyz_scratch_img.fmt = fmt;
    g_earlyz_scratch_img.w = d.width;
    g_earlyz_scratch_img.h = d.height;
    return g_earlyz_scratch_img.img.view;
}

Texture* g_su_tex = nullptr;                 // scene upscaler override (S14), see scene_upscale_for_draw
VkImageView g_su_view = VK_NULL_HANDLE;
Ref<Texture> g_tsr_depth;                   // temporal upscaling: the depth target of the G-buffer draw that gave this frame's camera
Ref<Texture> g_scene_depth;                 // latest 3D G-buffer depth target for read-only overlay occlusion (nav ribbon)
int g_scene_depth_mode = 0;                 // 0=fwd [0,1], 1=rev [0,1], 2=fwd [-1,1], 3=rev [-1,1]
uint64_t g_tsr_frame = 0;                    // temporal upscaling: tone-map draws so far (the jitter phase of the frame being drawn)
// Per-submission cache for vertex UBO 0 temporal jittering: avoids re-running 4x4 Gauss-Jordan parse_camera() + F->upload() + memcpy
// thousands of times per frame on the same camera block (and keeps its upload offset stable so SetCache hits on draws 2..N).
struct TsrUboCacheEntry {
    uint64_t addr = 0;
    VkDeviceSize n = 0;
    uint32_t w = 0, h = 0;
    uint64_t tsr_frame = 0;
    bool jittered = false;
    UploadAlloc u;
};
constexpr uint32_t kTsrCacheSlots = 16;
TsrUboCacheEntry g_tsr_cache[kTsrCacheSlots];
uint32_t g_tsr_cache_n = 0;
uint32_t g_tsr_cache_next = 0;
inline void tsr_cache_clear() { g_tsr_cache_n = 0; g_tsr_cache_next = 0; }
// Temporal upscaling: a copy of a vertex slot 0 camera block with this frame's jitter added (extras::temporal_jitter_block). False: left alone.
bool tsr_jitter(uint32_t* words, size_t n_words) {
    Texture* d = S.depth_tex.get();
    if (!d || !tex_gpu(d)) return false;
    const VkExtent3D e = tex_image_extent(d, 0);
    return extras::temporal_jitter_block(words, n_words, e.width, e.height, g_tsr_frame);
}

// build_set's vectors keep their capacity from call to call (allocated and freed for every set of every draw, they were ~10 % of the render
// thread in Dagsion, notes/perf.md); a call made while another one builds takes the next scratch.
struct SetScratch {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> binfo;
    std::vector<VkDescriptorImageInfo> iinfo;
    std::vector<VkBufferView> tinfo;
};
thread_local std::vector<std::unique_ptr<SetScratch>> t_set_scratch;
thread_local size_t t_set_depth = 0;

// Set 0 of a ProgLayout::push_set0 program: build_set writes directly into a fixed stack struct (zero heap alloc, zero vector copy).
struct PushSet0 {
    VkWriteDescriptorSet writes[32];
    VkDescriptorBufferInfo binfo[32];
    uint32_t count = 0;
    bool ready = false;
};

VkDescriptorSet build_set(const ProgLayout& pl, uint32_t set, PushSet0* out_push0 = nullptr) {
    const std::vector<uint32_t>& sidx = pl.set_binding_idx[set];
    const size_t nb = sidx.size();
    const bool use_push0 = (set == 0 && pl.push_set0 && out_push0 != nullptr && nb <= 32);
    const bool use_tpl = (!use_push0 && pl.templates[set] != VK_NULL_HANDLE && nb <= 64 && !(set == 0 && pl.push_set0) && vkUpdateDescriptorSetWithTemplate != nullptr);
    DescTemplateSlot tpl_slots[64];
    size_t tpl_idx = 0;
    if (use_push0) { out_push0->count = 0; out_push0->ready = false; }

    std::vector<VkWriteDescriptorSet>* p_writes = nullptr;
    std::vector<VkDescriptorBufferInfo>* p_binfo = nullptr;
    std::vector<VkDescriptorImageInfo>* p_iinfo = nullptr;
    std::vector<VkBufferView>* p_tinfo = nullptr;
    struct Depth {
        bool active;
        Depth(bool a) : active(a) { if (active) ++t_set_depth; }
        ~Depth() { if (active) --t_set_depth; }
    } depth(!use_tpl && !use_push0);
    if (!use_tpl && !use_push0) {
        if (t_set_scratch.size() <= t_set_depth - 1) t_set_scratch.push_back(std::make_unique<SetScratch>());
        SetScratch& sc = *t_set_scratch[t_set_depth - 1];
        p_writes = &sc.writes;
        p_binfo = &sc.binfo;
        p_iinfo = &sc.iinfo;
        p_tinfo = &sc.tinfo;
        p_writes->clear();
        p_binfo->clear();
        p_iinfo->clear();
        p_tinfo->clear();
        p_binfo->reserve(nb);
        p_iinfo->reserve(nb);
        p_tinfo->reserve(nb);
    }
    uint64_t key = 1469598103934665603ull + set;
    auto mix = [&](uint64_t v) { key = (key ^ v) * 1099511628211ull; };
    if (!use_push0) mix(reinterpret_cast<uint64_t>(pl.layout));
    for (uint32_t bi_idx : sidx) {
        const BindingDesc& d = pl.bindings[bi_idx];
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        if (!use_tpl && !use_push0) {
            w.dstBinding = d.binding;
            w.descriptorCount = 1;
            w.descriptorType = d.type;
        }
        if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            VkDescriptorBufferInfo bi = {dummy_buffer(), 0, 65536};
            switch (b.nvn.source) {
                case NvnSource::SupportBuffer:
                    bi = g_draw_support ? VkDescriptorBufferInfo{g_draw_support.buffer, g_draw_support.offset, sizeof(SupportBuffer)}
                                        : VkDescriptorBufferInfo{g_support.buffer, g_support.offset, sizeof(SupportBuffer)};
                    break;
                case NvnSource::DriverConstants: {
                    if (d.nvn_stage == kCompute && g_cs_c0) {        // NVN2 compute: this dispatch's own (do_dispatch)
                        bi = {g_cs_c0.buffer, g_cs_c0.offset, g_cs_c0.size};
                        break;
                    }
                    VkDeviceSize range = 0;
                    VkBuffer buf = driver_constants_buffer(&range);
                    if (buf) bi = {buf, 0, range};
                    break;
                }
                case NvnSource::PushConstants: {                     // graphics push constant slot n (Op::PushConstants)
                    const int slot = b.nvn.index;
                    if (slot < 0 || slot >= 8) break;
                    UploadAlloc& u = g_push_buf[slot];
                    if (!u) {
                        u = F->upload(256, std::max<VkDeviceSize>(device().properties().limits.minUniformBufferOffsetAlignment, 256));
                        if (u) std::memcpy(u.ptr, S.push_gfx[slot], 256);
                    }
                    if (u) bi = {u.buffer, u.offset, 256};
                    break;
                }
                case NvnSource::ConstantData: {
                    VkDeviceSize range = 0;
                    VkBuffer buf = constant_data_buffer(d.shader, &range);
                    if (buf) bi = {buf, 0, range};
                    break;
                }
                case NvnSource::UniformBlock: {
                    const int idx = b.nvn.index;
                    if (idx >= 0 && idx < 16 && S.ubo[d.nvn_stage][idx].addr) {
                        const GState::Bind& bind = S.ubo[d.nvn_stage][idx];
                        if (d.nvn_stage == kCompute || !out_lazy()) {                   // draws: checked before the op (out_before_op)
                            const VkDeviceSize ua = device().properties().limits.minUniformBufferOffsetAlignment;
                            out_draw_read(bind.addr, bind.size ? bind.size : 65536, ua && (bind.addr % ua) == 0, kOutDrawBuffer);   // resolve_read serves the copy
                        }
                        BufRef br = resolve_read(bind.addr, bind.size, kReadUniform);
                        const VkDeviceSize align = device().properties().limits.minUniformBufferOffsetAlignment;
                        const float ufs = ufs_scale_for(d, idx);
                        const bool tsr = d.nvn_stage == kVertex && idx == 0 && bind.size >= 272 && extras::temporal_wanted() && S.depth_tex;
                        // The console's constants are one flat buffer: a shader reads past the size the game declared for a block
                        // without noticing. XC2's shadow caster takes the fourth row of its 4x4 light matrix from a block declared as
                        // 48 bytes (three rows), and a binding of exactly 48 makes robustBufferAccess return zeros for that row:
                        // gl_Position.w becomes 0, the caster collapses, the cascade maps stay empty and the sun leaks through the
                        // roof (HANDOVER 28.9). Bind the block plus one matrix row of slack, taken from the same pool bytes.
                        constexpr VkDeviceSize kBlockSlack = 64;
                        const VkDeviceSize want = std::min<VkDeviceSize>((bind.size ? bind.size : 65536) + kBlockSlack, 65536);
                        bool bound_tsr = false;
                        if (br.ok && tsr && ufs == 1.0f) {
                            Texture* dtex = S.depth_tex.get();
                            const VkExtent3D de = (dtex && tex_gpu(dtex)) ? tex_image_extent(dtex, 0) : VkExtent3D{0, 0, 0};
                            const VkDeviceSize n = std::min<VkDeviceSize>({want, br.avail, 65536});
                            if (de.width >= 16 && de.height >= 16 && n >= 272) {
                                int hit = -1;
                                for (uint32_t ci = 0; ci < g_tsr_cache_n; ++ci) {
                                    const TsrUboCacheEntry& ce = g_tsr_cache[ci];
                                    if (ce.addr == bind.addr && ce.n == n && ce.w == de.width && ce.h == de.height && ce.tsr_frame == g_tsr_frame) {
                                        hit = static_cast<int>(ci);
                                        break;
                                    }
                                }
                                if (hit >= 0) {
                                    if (g_tsr_cache[hit].jittered) {
                                        bi = {g_tsr_cache[hit].u.buffer, g_tsr_cache[hit].u.offset, n};
                                        bound_tsr = true;
                                    }
                                } else {
                                    const uint8_t* src = host_ptr(bind.addr);
                                    if (src) {
                                        uint32_t hdr[68];
                                        std::memcpy(hdr, src, sizeof hdr);
                                        const bool did_jit = extras::temporal_jitter_block(hdr, 68, de.width, de.height, g_tsr_frame);
                                        UploadAlloc u;
                                        if (did_jit) {
                                            u = F->upload(n, std::max<VkDeviceSize>(align, 256));
                                            if (u) {
                                                std::memcpy(u.ptr, hdr, sizeof hdr);
                                                if (n > sizeof hdr) std::memcpy(u.ptr + sizeof hdr, src + sizeof hdr, static_cast<size_t>(n - sizeof hdr));
                                                bi = {u.buffer, u.offset, n};
                                                bound_tsr = true;
                                            }
                                        }
                                        if (!did_jit || bound_tsr) {
                                            const uint32_t slot = g_tsr_cache_n < kTsrCacheSlots ? g_tsr_cache_n++ : (g_tsr_cache_next++ & (kTsrCacheSlots - 1));
                                            g_tsr_cache[slot] = {bind.addr, n, de.width, de.height, g_tsr_frame, did_jit && bound_tsr, u};
                                        }
                                    }
                                }
                            }
                        }
                        if (bound_tsr) {
                            // already bound from the cached jittered upload copy
                        } else if (br.ok && (ufs != 1.0f || tsr)) {
                            // the game's scale on some floats of this block (nxrt/gfx_tuning.h): bound from a patched copy in the upload ring
                            const VkDeviceSize n = std::min<VkDeviceSize>({want, br.avail, 65536});
                            UploadAlloc u = F->upload(n, std::max<VkDeviceSize>(align, 256));
                            if (u) {
                                std::memcpy(u.ptr, host_ptr(bind.addr), n);
                                if (tsr && n >= 272) tsr_jitter(reinterpret_cast<uint32_t*>(u.ptr), static_cast<size_t>(n / 4));   // temporal upscaling
                                const uint32_t mask = ufs != 1.0f ? nxrt::gfx::tuning().ufs_mask.load(std::memory_order_relaxed) : 0u;
                                for (uint32_t i = 0; i < 32; ++i) {
                                    if (!(mask & (1u << i)) || (i + 1) * 4 > n) continue;
                                    float v;
                                    std::memcpy(&v, static_cast<uint8_t*>(u.ptr) + i * 4, 4);
                                    v *= ufs;
                                    std::memcpy(static_cast<uint8_t*>(u.ptr) + i * 4, &v, 4);
                                }
                                bi = {u.buffer, u.offset, n};
                            }
                        } else if (br.ok && (br.offset % align) == 0) {
                            bi = {br.buffer, br.offset, std::min<VkDeviceSize>({want, br.avail, 65536})};
                        } else if (br.ok) {
                            // unaligned binding: copy into the upload ring (clamped to br.avail like the patched/aligned paths)
                            const VkDeviceSize n = std::min<VkDeviceSize>({want, br.avail, 65536});
                            UploadAlloc u = F->upload(n, std::max<VkDeviceSize>(align, 256));
                            if (u) {
                                std::memcpy(u.ptr, host_ptr(bind.addr), n);
                                bi = {u.buffer, u.offset, n};
                            }
                        }
                    }
                    break;
                }
                case NvnSource::StorageBuffer: {
                    const int slot = b.nvn.index >= 0 ? b.nvn.index : storage_slot_for(d.nvn_stage, b.nvn.c0_offset);   // NVN2 compute: index from the c[0] table
                    if (slot >= 0 && slot < 16 && S.ssbo[d.nvn_stage][slot].addr) {
                        const GState::Bind& bind = S.ssbo[d.nvn_stage][slot];
                        BufRef br = resolve_buffer(bind.addr, bind.size);
                        bi = {br.buffer, br.offset, std::min<VkDeviceSize>(bind.size ? bind.size : br.avail, br.avail)};
                        if (br.ok && mirror_capture_on()) {   // [nvn] mirror_capture: reads and writes go through the pool's copy
                            static const bool ro_cap = nxrt::config_get_bool("nvn", "gfx_ssbo_readonly", true);
                            // [nvn] mirror_gfx_storage_writeback (default on; XC2 NS2 turns it off, xc2_boot.cpp): off keeps what a graphics
                            // stage writes into storage in the copy - not written back to the pool, and the CPU's pages there are not left out
                            // of the next capture. A written binding counts its whole range, while the GPU writes a few words of it: XC2 NS2's
                            // effects update their particles from vertex shaders (points, no fragment shader) in ranges of about 300 KiB that
                            // the game also writes on the CPU. A game a frame ahead ([nvn] present_ahead) wrote the next frame's words there
                            // while the GPU still ran: the capture left those pages out and the write-back then put the old range back over
                            // them, so the effects missed their CPU updates every few frames (the waterfall mist blinking at render_scale 3,
                            // where the GPU runs long enough to overlap). The write-back went from ~65 to ~1 MB per second.
                            static const bool gfx_wb = nxrt::config_get_bool("nvn", "mirror_gfx_storage_writeback", true);
                            const bool gpu_written = !(ro_cap && d.nvn_stage != kCompute && !b.written);
                            const bool wr = gpu_written && (gfx_wb || d.nvn_stage == kCompute);
                            const bool gfx_no_wb = gpu_written && !wr;
                            const BufRef cr = mirror_capture_bind(bind.addr, bi.range, wr, gfx_no_wb);
                            if (cr.ok && cr.buffer != br.buffer) {
                                bi.buffer = cr.buffer;
                                bi.offset = cr.offset;
                                break;
                            }
                        }
                        if (!g_shadowing && br.ok && out_lazy() && out_overlaps(bind.addr, bi.range)) {   // a draw over a held compute result: the copy
                            const BufRef hb = held_or_pool(bind.addr, bi.range, kOutStorageXfb, false);
                            if (hb.buffer != br.buffer) {
                                bi.buffer = hb.buffer;
                                bi.offset = hb.offset;
                                gpu_wrote(bind.addr, bi.range, kWriteStorage);
                                out_add(bind.addr, bi.range, hb.buffer, hb.offset, br);   // what the draw writes there is held like a dispatch's
                                break;
                            }
                        }
                        if (g_shadowing && br.ok) {                                  // before gpu_wrote: the shadow reads the pool copy while valid
                            const BufRef sh = shadow_of(bind.addr, bi.range, br, true);
                            bi.buffer = sh.buffer;
                            bi.offset = sh.offset;
                        }
                        // [nvn] gfx_ssbo_readonly (upstream 0.2.0 default true): a graphics shader that never stores to or
                        // applies an atomic to the buffer (reflection: DescriptorBinding::written) leaves the pages valid -
                        // no GPU write is recorded, the device copies of the pool stay usable and the next draw need not bind again.
                        static const bool ssbo_ro = nxrt::config_get_bool("nvn", "gfx_ssbo_readonly", true);
                        const bool read_only_gfx = ssbo_ro && d.nvn_stage != kCompute && !b.written;
                        static bool ro_logged = false;   // one line, so a run says whether the reflection flag reaches here
                        if (read_only_gfx && !ro_logged) {
                            ro_logged = true;
                            NXLOG_I("nvn", "gfx_ssbo_readonly: first read-only storage buffer bind (binding %u, stage %d)", b.binding, int(d.nvn_stage));
                        }
                        if (br.ok && !read_only_gfx) gpu_wrote(bind.addr, bi.range, kWriteStorage);      // a storage buffer may be written by the shader
                    }
                    break;
                }
                case NvnSource::UniformAsStorage: {
                    // a compute shader that reads uniform block n through its c[0] descriptor: the block's buffer as a storage buffer
                    const int idx = b.nvn.index;
                    if (idx >= 0 && idx < 16 && S.ubo[d.nvn_stage][idx].addr) {
                        const GState::Bind& bind = S.ubo[d.nvn_stage][idx];
                        const BufRef br0 = resolve_buffer(bind.addr, bind.size);
                        const VkDeviceSize range = std::min<VkDeviceSize>(bind.size ? bind.size : br0.avail, br0.avail);
                        const BufRef br = held_or_pool(bind.addr, range, kOutDispatch, true);
                        bi = {br.buffer, br.offset, range};
                    }
                    break;
                }
                default:
                    // HANDOVER 90: name the source, the binding and the shader. An unbound descriptor reads as zeros in Vulkan, so a
                    // vertex/constant buffer that lands here collapses the geometry of the draws using it.
                    NXLOG_W("nvn.replay", "buffer binding not resolved: source %d (index %d, bank %d, word %d, c0 %d) shader %s set %u binding %u",
                            static_cast<int>(b.nvn.source), b.nvn.index, b.nvn.bank, b.nvn.word, b.nvn.c0_offset,
                            d.shader ? d.shader->hash.c_str() : "?", b.set, b.binding);
                    note_unsupported("shader buffer binding of an unknown NVN source");
                    break;
            }
            if (use_push0) {
                const uint32_t pi = out_push0->count++;
                out_push0->binfo[pi] = bi;
                VkWriteDescriptorSet& pw = out_push0->writes[pi];
                pw = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                pw.dstBinding = d.binding;
                pw.descriptorCount = 1;
                pw.descriptorType = d.type;
                pw.pBufferInfo = &out_push0->binfo[pi];
                continue;
            } else if (use_tpl) {
                tpl_slots[tpl_idx++].buf = bi;
            } else {
                p_binfo->push_back(bi);
                w.pBufferInfo = &p_binfo->back();
            }
            mix(reinterpret_cast<uint64_t>(bi.buffer)); mix(bi.offset); mix(bi.range);
        } else if (d.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            const int component = b.image.component == ScalarType::Sint ? 2 : b.image.component == ScalarType::Uint ? 3 : 1;
            VkDescriptorImageInfo ii = {dummy_sampler(), dummy_view(view_type_for(b), b.image.depth, component), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            TexResolved tmp_res;
            const TexResolved& r_ref = resolve_texture_ref(d, tmp_res);
            Texture* r_tex = r_ref.tex;
            bool force_no_layers = false;
            bool hw_fb_loop = false;
            if (r_tex && !g_fb_use.empty()) {
                const auto fb = g_fb_use.find(r_tex);
                if (fb != g_fb_use.end()) {
                    if (fb->second == r_tex) hw_fb_loop = true;    // VK_EXT_attachment_feedback_loop_layout: sample attachment in-place
                    else r_tex = fb->second;                       // fallback: sample the FbClone copy made before this draw
                }
            }
            if (r_tex && !g_vol3d_use.empty() && wants_vol3d(b, r_ref)) {   // a sampler3D over 2D storage samples the 3D copy (vol3d_prepare)
                const auto vc = g_vol3d_use.find(r_tex);
                if (vc != g_vol3d_use.end()) {
                    r_tex = vc->second;
                    force_no_layers = true;                        // the copy is the chosen layer, one slice deep
                }
            }
            static const std::string null_fmt = nxrt::config_get_str("nvn", "debug_null_format", "");
            if (r_tex && !null_fmt.empty() && r_tex->fmt) {
                const std::string sized = std::string(r_tex->fmt->name) + ":" + std::to_string(r_tex->b.width) + "x" + std::to_string(r_tex->b.height);
                if (null_fmt == r_tex->fmt->name || null_fmt == sized) r_tex = nullptr;
            }
            const bool feedback = !hw_fb_loop && r_tex && d.nvn_stage != kCompute && is_color_feedback(r_tex);
            if (r_tex && tex_gpu(r_tex) && !feedback) {
                TexResolved r_mod;
                const TexResolved* rp = &r_ref;
                if (r_tex != r_ref.tex || force_no_layers) {
                    r_mod = r_ref;
                    r_mod.tex = r_tex;
                    if (force_no_layers) r_mod.view.has_layers = false;
                    rp = &r_mod;
                }
                const TexResolved& r = *rp;
                ViewRequest vr = make_request(b, r);
                if (d.nvn_stage == kFragment && d.index >= 0 && d.index < 64 && ((g_draw_uint_samplers >> d.index) & 1ull)) {
                    vr.component = 3;   // shader variable was rewritten to usampler by uint_tex_patch
                }
                // [nvn] ds_depth_shaders = "hash16,hash16": fragment shaders that read a depth-stencil texture as a
                // FLOAT value (luma weights, depth effects) instead of as the raw stencil byte. The game's
                // depth_stencil_mode says which plane it means, but a shader that does arithmetic on the value wants
                // the depth plane; taking the stencil plane gives it an S8_UINT view over a float sampler, which is
                // VUID-vkCmdDrawIndexed-format-07753 and undefined. See tex_sampled_view for the other half of this.
                if (vr.component == 1 && !vr.depth_aspect && S.stage[kFragment]) {
                    static const std::string list = nxrt::config_get_str("nvn", "ds_depth_shaders", "");
                    if (!list.empty() && list.find(S.stage[kFragment]->hash) != std::string::npos) vr.depth_aspect = true;
                }
                // Which fragment shaders read a depth-stencil texture's STENCIL plane through a float sampler. That
                // descriptor is illegal in Vulkan (VUID-vkCmdDrawIndexed-format-07753: the view is S8_UINT, the shader
                // declares float) but it is what reproduces Maxwell's "the integer lands in the float register
                // unchanged". AMD tolerates it; NVIDIA is free to return anything and does - it can lose the device,
                // which reaches the player as a frozen "Loading" screen with the music still playing.
                //
                // Reported always, once per (shader, binding), because tools/shader/patch_ds_samplers.py can only
                // rewrite modules a log names. This used to sit behind [nvn] ds_debug, which meant the one run that
                // mattered - the one that died - was the one with nothing to patch it with (issue #10, 2026-11-19).
                // The set is bounded by the title's own shaders and each entry costs one line.
                if (tex_is_depth(r.tex) && vr.component == 1 && !vr.depth_aspect && S.stage[kFragment]) {
                    static std::unordered_set<uint64_t> told_ds;
                    const uint64_t tk = reinterpret_cast<uint64_t>(S.stage[kFragment]) ^ (static_cast<uint64_t>(b.set) << 56) ^ (static_cast<uint64_t>(b.binding) << 40);
                    if (told_ds.insert(tk).second)
                        NXLOG_I(kTag, "ds_debug: fs %s samples the stencil plane of texture %u through a float sampler (s%db%d)", S.stage[kFragment]->hash.c_str(),
                                r.tex->id, int(b.set), int(b.binding));
                }
                VkImageView v = tex_sampled_view(r.tex, vr);
                if (v) {
                    ii.imageView = v;
                    ii.imageLayout = hw_fb_loop ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                : (tex_layout(r.tex) == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : tex_sampled_layout(r.tex));
                    if (r.has_sampler && r.smp_ptr) {
                        const SamplerInfo& s0 = *r.smp_ptr;
                        static const bool clamp_rt = nxrt::config_get_bool("nvn", "rt_sampler_clamp", true);
                        static const bool depth_nearest = nxrt::config_get_bool("nvn", "depth_sampler_nearest", false);
                        auto wraps = [](int m) { return m == NVN_WRAP_MODE_REPEAT || m == NVN_WRAP_MODE_MIRRORED_REPEAT; };
                        const bool need_clamp = clamp_rt && (wraps(s0.wrap_s) || wraps(s0.wrap_t) || wraps(s0.wrap_r)) && tex_gpu_rendered(r.tex);
                        const bool is_depth = tex_is_depth(r.tex);
                        if (is_depth) {
                            const VkFormat df = tex_vk_format(r.tex);
                            uint32_t dbit = 0;
                            if (df == VK_FORMAT_D16_UNORM) dbit = 1u << 0;
                            else if (df == VK_FORMAT_D24_UNORM_S8_UINT) dbit = 1u << 1;
                            else if (df == VK_FORMAT_D32_SFLOAT) dbit = 1u << 2;
                            else if (df == VK_FORMAT_D32_SFLOAT_S8_UINT) dbit = 1u << 3;
                            else if (df == VK_FORMAT_X8_D24_UNORM_PACK32) dbit = 1u << 4;
                            if (dbit) {
                                static uint32_t asked_mask = 0;
                                if (!(asked_mask & dbit)) {
                                    asked_mask |= dbit;
                                    VkFormatProperties fp = {};
                                    vkGetPhysicalDeviceFormatProperties(device().physical(), df, &fp);
                                    NXLOG_I(kTag, "depth format %d sampled here: linear filtering is %s", static_cast<int>(df),
                                            (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? "supported" : "NOT supported");
                                }
                            }
                        }
                        const VkFormat ivf = tex_vk_format(r.tex);
                        const bool need_int_nearest = (vr.component == 2 || vr.component == 3 || (!is_depth && vk_format_is_integer(ivf)));
                        const bool need_nearest = depth_nearest && is_depth;
                        if (!need_clamp && !need_int_nearest && !need_nearest && r.vk_sampler) {
                            ii.sampler = r.vk_sampler;
                        } else {
                            SamplerInfo s = s0;
                            if (need_clamp) {
                                s.wrap_s = s.wrap_t = s.wrap_r = NVN_WRAP_MODE_CLAMP_TO_EDGE;
                                if (first_time_key("rt_sampler_clamp " + std::to_string(r.tex->id) + " " + (S.stage[kFragment] ? S.stage[kFragment]->hash : ""))) {
                                    NXLOG_I(kTag, "rt_sampler_clamp: fs %s samples render target t%u with wrap %d,%d,%d: clamped",
                                            S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "?", r.tex->id, s0.wrap_s, s0.wrap_t, s0.wrap_r);
                                }
                            }
                            if (need_int_nearest) {
                                s.min_filter = NVN_MIN_FILTER_NEAREST;
                                s.mag_filter = NVN_MAG_FILTER_NEAREST;
                                s.max_anisotropy = 1.0f;
                            }
                            if (need_nearest) {
                                s.min_filter = s.mag_filter = NVN_MIN_FILTER_NEAREST;
                            }
                            ii.sampler = sampler_for(s);
                        }
                    }
                    if (r.tex == g_su_tex && g_su_view) {            // the upscaled scene replaces the small one for this draw (S14)
                        ii.imageView = g_su_view;
                        ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    }
                } else if (first_time_key("dummy-view " + d.shader->hash + " " + b.name + " " + std::to_string(r.tex->id))) {
                    NXLOG_W(kTag, "sampled texture %u (%s, target %d, %dx%dx%d) has no view of the shader's type %d: shader %s binding %s (NVN slot %d, set %u binding %u) gets the dummy view",
                            r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.target, r.tex->b.width, r.tex->b.height, r.tex->b.depth,
                            static_cast<int>(view_type_for(b)), d.shader->hash.c_str(), b.name.c_str(), b.nvn.index, d.set, d.binding);
                }
            } else if (feedback) {
                log_once("color-feedback", "a texture is sampled while it is a colour attachment of the pass: the dummy texture is bound");
                static int reported = 0;
                if (reported < 40) {   // which draw: the texture, the attachment's subresource, the sampled view, the shaders (S4g)
                    ++reported;
                    char tb[96];
                    uint32_t lvl = 0, lay = 0, nlay = 0;
                    for (uint32_t i = 0; i < S.ncolor; ++i) if (S.color_tex[i].get() == r_tex) { lvl = S.color_level[i]; lay = S.color_layer[i]; nlay = S.color_layers[i]; }
                    NXLOG_I("nvn.replay", "feedback: %s binding %s vs %s fs %s | attachment level %u layer %u layers %u | sampled view levels %d+%d layers %d+%d", tex_desc(r_tex, tb, sizeof tb),
                            b.name.c_str(), S.stage[kVertex] ? S.stage[kVertex]->hash.c_str() : "-", S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-", lvl, lay, nlay,
                            r_ref.has_view && r_ref.view.has_levels ? r_ref.view.min_level : -1, r_ref.has_view && r_ref.view.has_levels ? r_ref.view.num_levels : -1,
                            r_ref.has_view && r_ref.view.has_layers ? r_ref.view.min_layer : -1, r_ref.has_view && r_ref.view.has_layers ? r_ref.view.num_layers : -1);
                }
            }
            if (use_tpl) {
                tpl_slots[tpl_idx++].img = ii;
            } else {
                p_iinfo->push_back(ii);
                w.pImageInfo = &p_iinfo->back();
            }
            mix(reinterpret_cast<uint64_t>(ii.imageView));
            mix(reinterpret_cast<uint64_t>(ii.sampler));
            mix(static_cast<uint64_t>(ii.imageLayout));
        } else if (d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            const VkFormat shader_fmt = image_format_vk(b.image.format);
            VkDescriptorImageInfo ii = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
            if (g_earlyz_scratch) {
                const VkExtent2D ext = P.extent.width && P.extent.height ? P.extent
                                       : VkExtent2D{S.color_tex[0] ? static_cast<uint32_t>(std::max(S.color_tex[0]->b.width, 1)) : 1024u,
                                                    S.color_tex[0] ? static_cast<uint32_t>(std::max(S.color_tex[0]->b.height, 1)) : 1024u};
                const VkImageView sv = earlyz_scratch_view(shader_fmt, ext.width, ext.height);
                if (sv) {
                    ii.imageView = sv;
                    if (use_tpl) {
                        tpl_slots[tpl_idx++].img = ii;
                    } else {
                        p_iinfo->push_back(ii);
                        w.pImageInfo = &p_iinfo->back();
                        p_writes->push_back(w);
                    }
                    mix(reinterpret_cast<uint64_t>(sv));
                    continue;
                }
            }
            TexResolved tmp_img;
            const TexResolved& r = resolve_image_ref(d, tmp_img);
            if (r.tex && tex_gpu(r.tex) && tex_has_storage(r.tex)) {
                ViewRequest q = make_request(b, r);
                q.swizzle = {};
                if (shader_fmt != VK_FORMAT_UNDEFINED) q.format = shader_fmt;
                q.format = tex_storage_format(r.tex, q.format);
                ii.imageView = tex_sampled_view(r.tex, q);
                if (!ii.imageView) { q.format = VK_FORMAT_UNDEFINED; q.component = 0; ii.imageView = tex_sampled_view(r.tex, q); }
            }
            if (!ii.imageView) {
                if (r.tex) NXLOG_W(kTag, "storage image: texture %u (%s %dx%dx%d) has no storage view (gpu %d, storage usage %d)", r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?",
                                   r.tex->b.width, r.tex->b.height, r.tex->b.depth, tex_gpu(r.tex) ? 1 : 0, tex_has_storage(r.tex) ? 1 : 0);
                log_once("storage-image-dummy", "a storage image binding has no usable texture: a dummy storage image is bound");
                ii.imageView = dummy_storage_view(view_type_for(b), shader_fmt);
            }
            if (use_tpl) {
                tpl_slots[tpl_idx++].img = ii;
            } else {
                p_iinfo->push_back(ii);
                w.pImageInfo = &p_iinfo->back();
            }
            mix(reinterpret_cast<uint64_t>(ii.imageView));
        } else if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
            // NVN BUFFER-target textures (samplerBuffer / imageBuffer): a texel buffer view over the pool memory that holds the texels
            const bool storage = d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            const VkFormat shader_fmt = storage ? image_format_vk(b.image.format) : VK_FORMAT_UNDEFINED;
            const int component = storage ? 0 : b.image.component == ScalarType::Float ? 1 : b.image.component == ScalarType::Sint ? 2
                                  : b.image.component == ScalarType::Uint ? 3 : 0;
            TexResolved r = storage ? resolve_image(d) : resolve_texture(d);
            uint64_t addr = 0, bytes = 0;
            VkFormat vfmt = VK_FORMAT_UNDEFINED;
            VkBufferView v = r.tex ? tex_buffer_view(r.tex, r.has_view ? &r.view : nullptr, component, shader_fmt, storage, &addr, &bytes, &vfmt) : VK_NULL_HANDLE;
            if (v && r.tex->b.target != NVN_TEXTURE_TARGET_BUFFER && first_time_key("texel-nonbuf " + d.shader->hash + " " + b.name + " " + std::to_string(r.tex->id)))
                NXLOG_I(kTag, "texture %u (%s %dx%dx%d, target %d) is read as a texel buffer by %s binding %s: it gets a buffer view over its storage (%llu bytes)",
                        r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width, r.tex->b.height, r.tex->b.depth, r.tex->b.target,
                        d.shader->hash.c_str(), b.name.c_str(), static_cast<unsigned long long>(bytes));
            if (v && g_shadowing) {                                        // before gpu_wrote: the shadow reads the pool copy while valid
                const BufRef host = resolve_buffer(addr, bytes);
                const BufRef sh = shadow_of(addr, bytes, host, storage);
                if (sh.buffer != host.buffer)
                    if (VkBufferView sv = make_buffer_view(sh.buffer, sh.offset, bytes, vfmt)) v = sv;   // offset: non-zero in the pool copy
            }
            if (v && !g_shadowing && out_lazy() && out_overlaps(addr, bytes)) {   // a draw reads a held compute result as texels: from the copy
                const MirrorSpan ms = out_servable(addr, bytes) ? mirror_span(addr, bytes) : MirrorSpan{};
                VkBufferView mv = ms.buf ? make_buffer_view(ms.buf, ms.offset, bytes, vfmt) : VK_NULL_HANDLE;
                if (mv) v = mv;
                else log_once("out-texel", "a draw reads a compute result held in the pool copy as a texel buffer the copy cannot serve: it reads the pool");
            }
            if (v && storage) gpu_wrote(addr, bytes, kWriteStorage);      // an image buffer may be written by the shader
            if (!v) {
                if (r.tex && first_time_key("texel-buffer " + d.shader->hash + " " + b.name + " " + std::to_string(r.tex->id)))
                    NXLOG_W(kTag, "texel buffer: texture %u (%s, target %d, %dx%dx%d, storage %llu bytes at %p) has no buffer view: shader %s binding %s (NVN slot %d, set %u binding %u)",
                            r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.target, r.tex->b.width, r.tex->b.height, r.tex->b.depth,
                            static_cast<unsigned long long>(r.tex->storage_size), static_cast<void*>(r.tex->mem), d.shader->hash.c_str(), b.name.c_str(),
                            b.nvn.index, d.set, d.binding);
                log_once("texel-buffer-dummy", "a texel buffer binding has no usable buffer texture: a dummy buffer view is bound");
                v = dummy_buffer_view(shader_fmt, storage);
            }
            if (!v) {
                note_unsupported("texel buffer descriptor without any view (the binding stays unwritten)");
                continue;
            }
            p_tinfo->push_back(v);
            w.pTexelBufferView = &p_tinfo->back();
            mix(reinterpret_cast<uint64_t>(v));
        } else {
            note_unsupported("descriptor of an unhandled type (the binding stays unwritten)");
            continue;
        }
        if (!use_tpl && !use_push0) p_writes->push_back(w);
    }
    ++g_rs.sets;
    if (use_push0) {
        out_push0->ready = true;
        g_rs.set_writes += out_push0->count;
        return VK_NULL_HANDLE;
    }
    if (use_tpl) {
        if (const VkDescriptorSet hit = g_set_cache.find(key)) { ++g_rs.set_hits; return hit; }
        VkDescriptorSet ds = F->alloc_set(pl.sets[set]);
        if (!ds) return VK_NULL_HANDLE;
        g_rs.set_writes += tpl_idx;
        vkUpdateDescriptorSetWithTemplate(vkdev(), ds, pl.templates[set], tpl_slots);
        g_set_cache.put(key, ds);
        return ds;
    }
    std::vector<VkWriteDescriptorSet>& writes = *p_writes;
    if (const VkDescriptorSet hit = g_set_cache.find(key)) { ++g_rs.set_hits; return hit; }
    if (rs_on()) {   // replay_stats: what differs from the last set made for this layout
        struct D { uint32_t binding, type; uint64_t a, b, c; };
        static std::unordered_map<VkDescriptorSetLayout, std::vector<D>> last;
        static std::vector<D> cur;                       // reused: the stats line must not allocate per miss
        cur.clear();
        for (const VkWriteDescriptorSet& w : writes) {
            D d{w.dstBinding, static_cast<uint32_t>(w.descriptorType), 0, 0, 0};
            if (w.pBufferInfo) { d.a = reinterpret_cast<uint64_t>(w.pBufferInfo->buffer); d.b = w.pBufferInfo->offset; d.c = w.pBufferInfo->range; }
            else if (w.pImageInfo) { d.a = reinterpret_cast<uint64_t>(w.pImageInfo->imageView); d.b = reinterpret_cast<uint64_t>(w.pImageInfo->sampler); d.c = w.pImageInfo->imageLayout; }
            else if (w.pTexelBufferView) d.a = reinterpret_cast<uint64_t>(*w.pTexelBufferView);
            cur.push_back(d);
        }
        std::vector<D>& l = last[pl.sets[set]];
        if (l.size() != cur.size()) ++g_rs.miss_first;
        else {
            bool ofs = false, buf = false, tex = false, other = false;
            for (size_t i = 0; i < cur.size(); ++i) {
                const D& x = cur[i]; const D& y = l[i];
                if (x.binding != y.binding || x.type != y.type) { other = true; continue; }
                if (x.a == y.a && x.b == y.b && x.c == y.c) continue;
                if (x.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) (x.a == y.a ? ofs : buf) = true;
                else if (x.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) tex = true;
                else other = true;
            }
            if (other) ++g_rs.miss_other;
            else if (tex) ++g_rs.miss_tex;
            else if (buf) ++g_rs.miss_ubo;
            else if (ofs) ++g_rs.miss_uboofs;
            else ++g_rs.miss_first;            // the same set as the last one (a new submission)
        }
        l.assign(cur.begin(), cur.end());
    }
    RsScope rs_alloc(kRsAlloc);   // replay_stats: the miss path (draw sets and dispatch sets)
    VkDescriptorSet ds = F->alloc_set(pl.sets[set]);
    if (!ds) return VK_NULL_HANDLE;
    g_rs.set_writes += writes.size();
    for (auto& w : writes) w.dstSet = ds;
    vkUpdateDescriptorSets(vkdev(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    g_set_cache.put(key, ds);
    return ds;
}

// ---- resolution scale: the support buffer's render_scale for a draw or dispatch (Ryujinx TextureBindingsManager.UpdateRenderScale) ----
// The translator scales texelFetch / imageLoad coordinates by render_scale[1 + i] and divides textureSize and gl_FragCoord.xy by the scale
// of texture i and of the render target ([0]). Texture i is the shader's i-th texture descriptor: the harness numbers single textures
// with one counter per shader (binding = stage slot * 128 + counter, + 64 for texel buffers), images follow the textures. Vertex shaders
// read their scales after the fragment shader's (frag_scale_count). Storage images are never scaled (tex_rescale), so images read 1.
void support_for_draw(const ProgLayout& pl, bool compute) {
    g_draw_support = UploadAlloc();
    if (render_scale() == 1.0f) return;
    float sc[73];
    for (float& v : sc) v = 1.0f;
    int frag_count = 0;
    bool any = false;
    if (!compute && P.scale != 1.0f) { sc[0] = P.scale; any = true; }
    // the fragment shader's textures first, then the vertex shader's after them
    const ShaderInfo* fs = compute ? nullptr : S.stage[kFragment];
    const ShaderInfo* vs = compute ? S.stage[kCompute] : S.stage[kVertex];
    auto tex_count = [](const ShaderInfo* sh) {
        int n = 0;
        for (const DescriptorBinding& b : sh->refl.bindings)
            if (b.count == 1 && (b.type == DescriptorType::CombinedImageSampler || b.type == DescriptorType::SampledImage || b.type == DescriptorType::UniformTexelBuffer ||
                                 b.type == DescriptorType::StorageImage || b.type == DescriptorType::StorageTexelBuffer))
                ++n;
        return n;
    };
    if (fs) frag_count = tex_count(fs);
    // the native size of the render target, for the fragment rule below
    Texture* rt = S.ncolor && S.color_tex[0] ? S.color_tex[0].get() : S.depth_tex.get();
    // Ryujinx: a texture whose size is the render target's size times one factor on both axes interpolates the scaled pixels with
    // gl_FragCoord (a negative scale tells TexelFetchScale to)
    auto frag_sign = [rt](Texture* tex, float s) {
        if (!rt || tex->b.width == 0 || tex->b.height == 0) return s;
        const float rw = static_cast<float>(rt->b.width), rh = static_cast<float>(rt->b.height);
        return rw / static_cast<float>(tex->b.width) == rh / static_cast<float>(tex->b.height) ? -s : s;
    };
    // A fragment shader's storage images an imageLoad reads at a scale (prepare_textures keeps the scale of an image only read): their
    // scales follow the textures' (the harness numbers images after the textures, in binding order).
    if (fs) {
        int tex_n = 0;
        std::vector<uint32_t> img_bindings;
        for (const DescriptorBinding& b : fs->refl.bindings) {
            if (b.count != 1) continue;
            if (b.type == DescriptorType::CombinedImageSampler || b.type == DescriptorType::SampledImage || b.type == DescriptorType::UniformTexelBuffer) ++tex_n;
            else if (b.type == DescriptorType::StorageImage || b.type == DescriptorType::StorageTexelBuffer) img_bindings.push_back(b.set << 16 | b.binding);
        }
        std::sort(img_bindings.begin(), img_bindings.end());
        for (const BindingDesc& d : pl.bindings) {
            if (d.type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || d.shader != fs) continue;
            const DescriptorBinding& b = d.shader->refl.bindings[d.index];
            if (b.count != 1) continue;
            TexResolved tmp;
            Texture* tex = resolve_image_ref(d, tmp).tex;
            if (!tex) continue;
            const float s = tex_scale(tex);
            if (s == 1.0f) continue;
            const auto at = std::find(img_bindings.begin(), img_bindings.end(), b.set << 16 | b.binding);
            const int slot = 1 + tex_n + static_cast<int>(at - img_bindings.begin());
            if (at == img_bindings.end() || slot >= 73) continue;
            sc[slot] = frag_sign(tex, s);
            any = true;
        }
    }
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.count != 1 || b.binding >= 640) continue;
        const bool is_fs = d.shader == fs;
        if (!is_fs && d.shader != vs) continue;
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        TexResolved tmp;
        Texture* tex = resolve_texture_ref(d, tmp).tex;   // only the texture is used below
        if (!tex) continue;
        if (!g_fb_use.empty()) {
            const auto fb = g_fb_use.find(tex);
            if (fb != g_fb_use.end()) tex = fb->second;
        }
        float s = tex_scale(tex);
        if (s == 1.0f) continue;
        const int idx = static_cast<int>(b.binding % 64u);
        const int slot = is_fs ? 1 + idx : (compute ? 1 + idx : 1 + frag_count + idx);
        if (slot >= 73) continue;
        if (is_fs && rt) {
            // Ryujinx: a texture whose size is the render target's size times one factor on both axes interpolates the scaled pixels
            // with gl_FragCoord (a negative scale tells TexelFetchScale to)
            const float rw = static_cast<float>(rt->b.width), rh = static_cast<float>(rt->b.height);
            if (tex->b.width > 0 && tex->b.height > 0 && rw / static_cast<float>(tex->b.width) == rh / static_cast<float>(tex->b.height)) s = -s;
        }
        sc[slot] = s;
        any = true;
    }
    if (!any) return;
    uint64_t key = 1469598103934665603ull + static_cast<uint64_t>(frag_count);
    for (float v : sc) { uint32_t u; std::memcpy(&u, &v, 4); key = (key ^ u) * 1099511628211ull; }
    auto it = g_support_scaled.find(key);
    if (it != g_support_scaled.end()) { g_draw_support = it->second; return; }
    UploadAlloc u = F->upload(sizeof(SupportBuffer), 256);
    if (!u) return;
    SupportBuffer sb = support_buffer_defaults();
    sb.frag_scale_count[0] = frag_count;
    for (int i = 0; i < 73; ++i) sb.render_scale[i][0] = sc[i];
    std::memcpy(u.ptr, &sb, sizeof sb);
    g_support_scaled[key] = u;
    g_draw_support = u;
}

// Pushes set 0's writes. [nvn] lean_draws, graphics only: while the command buffer and the pipeline layout are those of the last push,
// only the writes that differ from it are pushed (a pushed descriptor stays until its binding is pushed again or another layout is
// bound), and nothing when none differs. Adapted from Hookshot's bind_descriptors (gpu/vk/vk_draw.cpp, GPL-3).
void push_set0(VkPipelineBindPoint bp, const ProgLayout& pl, const VkWriteDescriptorSet* writes, uint32_t count) {
    if (!count) return;   // a push of no writes is not allowed
    PushedSet0& ps = g_pushed0;
    if (bp != VK_PIPELINE_BIND_POINT_GRAPHICS || !lean_on() || count > 32) {
        if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) ps.cb = VK_NULL_HANDLE;
        vkCmdPushDescriptorSetKHR(F->cb, bp, pl.layout, 0, count, writes);
        return;
    }
    const bool same_state = ps.cb == F->cb && ps.layout == pl.layout && ps.count == count;
    VkWriteDescriptorSet changed[32];
    uint32_t nchg = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const VkWriteDescriptorSet& w = writes[i];
        const PushedSet0::W now = {w.dstBinding, static_cast<uint32_t>(w.descriptorType), w.pBufferInfo->buffer, w.pBufferInfo->offset, w.pBufferInfo->range};
        PushedSet0::W& was = ps.w[i];
        if (!same_state || was.binding != now.binding || was.type != now.type || was.buffer != now.buffer || was.offset != now.offset || was.range != now.range)
            changed[nchg++] = w;
        was = now;
    }
    ps.cb = F->cb;
    ps.layout = pl.layout;
    ps.count = count;
    if (!nchg) { ++g_lean_push_none; return; }
    if (nchg == count) {
        ++g_lean_push_full;
        g_lean_push_writes += nchg;
        if (pl.templates[0] && count == pl.set_binding_idx[0].size() && vkCmdPushDescriptorSetWithTemplateKHR) {
            VkDescriptorBufferInfo bufs[32];
            for (uint32_t i = 0; i < count; ++i) bufs[i] = *writes[i].pBufferInfo;
            vkCmdPushDescriptorSetWithTemplateKHR(F->cb, pl.templates[0], pl.layout, 0, bufs);
            return;
        }
    } else {
        ++g_lean_push_delta;
        g_lean_push_writes += nchg;
    }
    vkCmdPushDescriptorSetKHR(F->cb, bp, pl.layout, 0, nchg, changed);
}

void bind_sets(const ProgLayout& pl, VkPipelineBindPoint bp) {
    VkDescriptorSet ds[kPoolSet + 1] = {};
    PushSet0 push0;
    // sets bound with another layout may disturb what set 0 holds for the layout of the last push (push_set0)
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS && g_pushed0.layout != pl.layout) g_pushed0.cb = VK_NULL_HANDLE;
    for (uint32_t set = 0; set < 4; ++set) {
        if (!pl.has_set[set]) continue;
        ds[set] = build_set(pl, set, set == 0 && pl.push_set0 ? &push0 : nullptr);
    }
    if (pl.has_set[kPoolSet]) ds[kPoolSet] = pool_set(S.tpool.get(), S.spool.get(), *F);   // the texture and sampler pools as arrays (nvn_pipeline.cpp)
    if (push0.ready && push0.count) {   // a push of no writes is not allowed
        push_set0(bp, pl, push0.writes, push0.count);
        push0.ready = false;
    }
    // consecutive sets in one call (bound sets are command buffer state: the copies build_set may record in between do not touch them)
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        if (g_bound_ds_cb != F->cb || g_bound_ds_layout != pl.layout) {
            g_bound_ds_cb = F->cb;
            g_bound_ds_layout = pl.layout;
            for (VkDescriptorSet& d : g_bound_ds) d = VK_NULL_HANDLE;
        }
        VkDescriptorSet to_bind[kPoolSet + 1] = {};
        for (uint32_t i = 0; i <= kPoolSet; ++i) {
            if (ds[i] && ds[i] != g_bound_ds[i]) {
                to_bind[i] = ds[i];
                g_bound_ds[i] = ds[i];
            }
        }
        for (uint32_t i = 0; i <= kPoolSet;) {
            if (!to_bind[i]) { ++i; continue; }
            uint32_t j = i + 1;
            while (j <= kPoolSet && to_bind[j]) ++j;
            vkCmdBindDescriptorSets(F->cb, bp, pl.layout, i, j - i, to_bind + i, 0, nullptr);
            i = j;
        }
    } else {
        for (uint32_t i = 0; i <= kPoolSet;) {
            if (!ds[i]) { ++i; continue; }
            uint32_t j = i + 1;
            while (j <= kPoolSet && ds[j]) ++j;
            vkCmdBindDescriptorSets(F->cb, bp, pl.layout, i, j - i, ds + i, 0, nullptr);
            i = j;
        }
    }
}

// ---- draws -------------------------------------------------------------------------------------------------------------------
const ShaderInfo* tcs_now();   // the program's control shader, or the generated pass-through one (below)
bool fs_list_matches(const char* key);   // defined with the dynamic state below
bool build_key(GraphicsKey& key, VkPrimitiveTopology topo) {
    std::memset(&key, 0, sizeof key);
    key.stage[0] = S.stage[kVertex];
    key.stage[1] = S.stage[kFragment];
    key.stage[2] = S.stage[kGeometry];
    if (topo == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {      // begin_draw checked: both tessellation stages, patches
        key.tess[0] = tcs_now();
        key.tess[1] = S.stage[kTessEval];
        key.patch_control_points = static_cast<uint8_t>(std::clamp<uint32_t>(S.patch_size, 1, 32));
    }
    const ShaderInfo* vs = key.stage[0];
    // attributes: those the vertex shader reads. An input array takes one location per element. A location the game did not set up
    // (no attribute, or a disabled one) still has to exist in the pipeline; it reads (0, 0, 0, 1) from a dummy binding.
    struct Missing { uint32_t location; ScalarType type; };
    Missing missing[32];
    uint32_t nmissing = 0;
    for (const ShaderIo& in : vs->refl.inputs) {
        const uint32_t elems = in.array ? in.array : 1;
        for (uint32_t e = 0; e < elems; ++e) {
            const uint32_t loc = in.location + e;
            if (loc >= 32) break;
            VkFormat f = VK_FORMAT_UNDEFINED;
            const VertexAttribInfo* a = nullptr;
            if (loc < S.attr_count && loc < 16) {
                a = &S.attr[loc];
                f = vertex_format(a->format);
            }
            if (f == VK_FORMAT_UNDEFINED) {
                if (nmissing < 32) missing[nmissing++] = {loc, in.type};
                continue;
            }
            // a float input fed by an integer attribute gets the raw bits: the integer format stays and the pipeline patches the shader
            // (spirv_int_attributes); stream-out draws use the shader's stream-out variant, which is not patched: scaled floats there
            if (in.type == ScalarType::Float && is_int_format(f) && g_xfb_active) f = scaled_for(f);
            else if ((in.type == ScalarType::Uint || in.type == ScalarType::Sint) && !is_int_format(f))
                log_once("attr-type-mismatch", "a vertex attribute with a float format feeds an integer shader input (needs a SPIR-V variant)");
            if (key.attr_count >= 16) break;
            KeyAttr& k = key.attr[key.attr_count++];
            k.location = static_cast<uint8_t>(loc);
            k.binding = static_cast<uint8_t>(a->stream & 15);
            k.offset = static_cast<uint16_t>(a->relative_offset);
            k.format = f;
            key.binding_instance[k.binding] = S.stream[k.binding].divisor > 0 ? 1 : 0;
        }
    }
    if (nmissing) {
        uint32_t used = 0;
        for (uint32_t i = 0; i < key.attr_count; ++i) used |= 1u << key.attr[i].binding;
        uint32_t free_binding = 0;
        while (free_binding < 16 && (used >> free_binding & 1)) ++free_binding;
        if (free_binding < 16) {
            key.dummy_binding = static_cast<uint8_t>(free_binding + 1);
            key.binding_instance[free_binding] = 0;
            for (uint32_t i = 0; i < nmissing && key.attr_count < 16; ++i) {
                KeyAttr& k = key.attr[key.attr_count++];
                k.location = static_cast<uint8_t>(missing[i].location);
                k.binding = static_cast<uint8_t>(free_binding);
                const bool is_float = missing[i].type != ScalarType::Uint && missing[i].type != ScalarType::Sint;
                k.offset = is_float ? 0 : 16;
                k.format = is_float ? VK_FORMAT_R32G32B32A32_SFLOAT : (missing[i].type == ScalarType::Uint ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SINT);
            }
        }
    }
    key.color_count = P.ncolor;
    if (!P.ncolor && S.stage[kFragment] && !S.stage[kFragment]->refl.outputs.empty()) key.no_out = 1;   // a depth-only pass: no outputs
    if (!key.no_out && device().features().dynamic_rendering_unused_attachments) {
        // the fragment shader may write outputs the pass has no attachment for: they are listed as unused (format UNDEFINED)
        if (const ShaderInfo* fs = S.stage[kFragment])
            for (const ShaderIo& o : fs->refl.outputs)
                for (uint32_t e = 0; e < (o.array ? o.array : 1); ++e)
                    if (o.location + e < 8 && o.location + e >= key.color_count) key.color_count = o.location + e + 1;
    }
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i) {
        if (i >= P.ncolor) {
            key.color_format[i] = VK_FORMAT_UNDEFINED;
            key.write_mask[i] = 0;
            continue;
        }
        key.color_format[i] = P.color_fmt[i];
        key.blend_enable[i] = (S.color.blend_enable_mask >> i) & 1;
        const BlendInfo& b = S.blend[i];
        key.blend_src_rgb[i] = b.src_rgb ? vk_blend_factor(b.src_rgb) : VK_BLEND_FACTOR_ONE;
        key.blend_dst_rgb[i] = b.dst_rgb ? vk_blend_factor(b.dst_rgb) : VK_BLEND_FACTOR_ZERO;
        key.blend_src_a[i] = b.src_a ? vk_blend_factor(b.src_a) : VK_BLEND_FACTOR_ONE;
        key.blend_dst_a[i] = b.dst_a ? vk_blend_factor(b.dst_a) : VK_BLEND_FACTOR_ZERO;
        key.blend_op_rgb[i] = b.eq_rgb ? vk_blend_op(b.eq_rgb) : VK_BLEND_OP_ADD;
        key.blend_op_a[i] = b.eq_a ? vk_blend_op(b.eq_a) : VK_BLEND_OP_ADD;
        key.write_mask[i] = S.chmask.mask[i] & 0xf;
        if (P.color_fmt[i] == VK_FORMAT_UNDEFINED) key.write_mask[i] = 0;
        auto uses_src1 = [](VkBlendFactor f) {
            return f == VK_BLEND_FACTOR_SRC1_COLOR || f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR || f == VK_BLEND_FACTOR_SRC1_ALPHA ||
                   f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        };
        if (key.blend_enable[i] && (uses_src1(static_cast<VkBlendFactor>(key.blend_src_rgb[i])) || uses_src1(static_cast<VkBlendFactor>(key.blend_dst_rgb[i])) ||
                                    uses_src1(static_cast<VkBlendFactor>(key.blend_src_a[i])) || uses_src1(static_cast<VkBlendFactor>(key.blend_dst_a[i]))))
            key.dual_src = 1;
    }
    key.sdr_tone_map = extras::hdr_shown_as_sdr() && S.stage[kFragment] && extras::hdr_tonemap_patched(S.stage[kFragment]->hash) ? 1 : 0;
    key.xfb = g_xfb_active && device().features().transform_feedback && S.stage[kVertex] && xfb_stride(S.stage[kVertex]) ? 1 : 0;
    key.depth_format = P.depth_fmt;
    key.stencil_format = P.has_stencil ? P.depth_fmt : VK_FORMAT_UNDEFINED;
    key.samples = P.samples;
    key.topology = static_cast<uint8_t>(topo);
    key.polygon_mode = static_cast<uint8_t>(vk_polygon_mode(S.poly.polygon_mode));
    // [nvn] wireframe_fs = "8f50dea8" (comma list of fragment shader hash prefixes): draw those with lines. The probe for "the
    // draw has sane state and data but covers no pixel" - lines show where its triangles actually are and whether they are
    // degenerate or edge-on (HANDOVER 23).
    if (fs_list_matches("wireframe_fs")) key.polygon_mode = VK_POLYGON_MODE_LINE;
    if (key.polygon_mode != VK_POLYGON_MODE_FILL && !device().features().fill_mode_non_solid) key.polygon_mode = VK_POLYGON_MODE_FILL;
    key.depth_clamp = S.depth_clamp && device().features().depth_clamp;
    key.alpha_to_coverage = S.ms.alpha_to_coverage && S.ms.enable ? 1 : 0;
    key.depth_neg_one = S.depth_mode == NVN_DEPTH_MODE_NEAR_IS_MINUS_W ? 1 : 0;
    // Diagnostic (HANDOVER 33): the global-probe draw writes the list AND the stencil; mask its colour writes to tell which one
    // makes the black region. Only with NXRT_NVN_TEST_MASK0 set.
    static const bool test_mask0 = std::getenv("NXRT_NVN_TEST_MASK0") != nullptr;   // read once: build_key runs for ~1,100 keys a frame (HANDOVER 52.1 #1)
    if (test_mask0 && S.stage[kFragment] && S.stage[kFragment]->hash == "eb74f4cbf5c86ea9")
        for (uint32_t i = 0; i < 8; ++i) key.write_mask[i] = 0;
    // Diagnostic (HANDOVER 35): the probe-list builders write 0 for "this probe does not register this pixel", which is only
    // harmless with an OR. With NXRT_NVN_TEST_LOGIC_OR an integer target is always written with OR, whatever the colour state
    // says: that tells "the game wanted OR" from "the game wanted a plain overwrite".
    static const bool test_logic_or = std::getenv("NXRT_NVN_TEST_LOGIC_OR") != nullptr;
    if (test_logic_or && device().features().logic_op && S.ncolor && S.color_tex[0] && S.color_tex[0]->fmt) {
        const std::string fn = S.color_tex[0]->fmt->name ? S.color_tex[0]->fmt->name : "";
        if (fn.find("UI") != std::string::npos) {
            key.logic_op_enable = 1;
            key.logic_op = VK_LOGIC_OP_OR;
            for (uint32_t i = 0; i < 8; ++i) key.blend_enable[i] = 0;
        }
    }
    if (S.color.logic_op != 0 && S.color.logic_op != NVN_LOGIC_OP_COPY && device().features().logic_op) {
        key.logic_op_enable = 1;
        key.logic_op = static_cast<uint8_t>(vk_logic_op(S.color.logic_op));
        static const bool test_logic_set = std::getenv("NXRT_NVN_TEST_LOGIC_SET") != nullptr;
        static const bool test_logic_off = std::getenv("NXRT_NVN_TEST_LOGIC_OFF") != nullptr;
        if (test_logic_set) key.logic_op = VK_LOGIC_OP_SET;   // diagnostic (HANDOVER 33): does the driver apply logic ops on R32G32_UINT?
        if (test_logic_off) { key.logic_op_enable = 0; key.logic_op = 0; }   // diagnostic (HANDOVER 35): is the OR applied at all?
        for (uint32_t i = 0; i < 8; ++i) key.blend_enable[i] = 0;
    }
    // [nvn] trace: what the colour state says vs what the pipeline key carries - the probe-list frame needs OR here and the
    // frame shows COPY on the last registration draws (HANDOVER 35).
    if (g_trace && S.stage[kFragment])
        TRACE("    logic: fs %s state %d -> key enable %d op %d (mask0 %x, blend0 %d)", S.stage[kFragment]->hash.c_str(), S.color.logic_op, key.logic_op_enable,
              key.logic_op, key.write_mask[0], key.blend_enable[0]);
    if (S.color.alpha_test != 0 && S.color.alpha_test != NVN_ALPHA_FUNC_ALWAYS) log_once("alpha-test", "alpha test is enabled by the game: not implemented (needs a shader variant)");
    key.uint_samplers = g_draw_uint_samplers;
    return true;
}

// [nvn] cull_off_fs = "8f50dea8" (comma list of fragment shader hash prefixes, default empty): those draws are rasterised with
// both faces. The experiment for "one decal layer is culled while its coplanar neighbour draws" - a layer that appears with
// culling off has its winding flipped (by the skinning matrices, say), not its geometry broken (HANDOVER 21).
// [nvn] depth_write_fs = "8f50dea8" (same shape): those draws get depth writes. Monolith's engine draws the eye layers with
// depth EQUAL and no depth write, relying on the depth prepass to have written the eye card's own depth; a layer that only
// shows up once it writes depth itself says the prepass and the fill disagree about that depth (HANDOVER 23).
bool fs_list_matches(const char* key) {
    // Fast path: empty config lists (depth_write_fs, depth_func_flip_fs, etc.) return false in O(1) with zero map/string work.
    static const char* empty_keys[8] = {};
    const size_t eslot = (reinterpret_cast<uintptr_t>(key) >> 4) & 7u;
    if (empty_keys[eslot] == key) return false;
    const std::string* h = S.stage[kFragment] ? &S.stage[kFragment]->hash : nullptr;
    if (!h) return false;
    static const char* last_key[8] = {};
    static const std::string* last_hash[8] = {};
    static unsigned char last_res[8] = {};
    const size_t slot = eslot;
    if (last_key[slot] == key && last_hash[slot] == h) return last_res[slot] != 0;
    static std::map<std::string, std::vector<std::string>> cache;
    auto it = cache.find(key);
    if (it == cache.end()) {
        std::vector<std::string> v;
        const std::string s = nxrt::config_get_str("nvn", key, std::strcmp(key, "earlyz_fs") == 0 ? kEarlyzDefaultFs : "");
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) v.push_back(s.substr(i, j - i));
            i = j + 1;
        }
        it = cache.emplace(key, std::move(v)).first;
    }
    if (it->second.empty()) {
        empty_keys[eslot] = key;
        return false;
    }
    const std::vector<std::string>& list = it->second;
    // "auto" (or "*"): every shader matches. Used by earlyz_fs, whose real gate is the format test in earlyz_mask_rebuild (the
    // light list is an R32G32_UINT colour target with an R32_UINT mask), so a build whose writer hashes differ from the ones
    // this list was written for still gets its mask rebuilt (HANDOVER 90: 3.0.1 binds the list in 281 shaders).
    if (list.size() == 1 && (list[0] == "auto" || list[0] == "*")) {
        last_key[slot] = key; last_hash[slot] = h; last_res[slot] = 1;
        return true;
    }
    bool res = false;
    for (const std::string& p : list)
        if (!p.empty() && h->compare(0, p.size(), p) == 0) { res = true; break; }
    last_key[slot] = key;
    last_hash[slot] = h;
    last_res[slot] = res ? 1 : 0;
    return res;
}

bool cull_off_for_current_draw() { return fs_list_matches("cull_off_fs"); }

// ---- [nvn] trace: the state a draw really runs with ---------------------------------------------------------------------
// The trace used to print the guest's request (the depth-stencil word's bits) and a "depth mode" that is a property of the
// device, not of the draw, so a frame's depth columns read all-zero and could not answer "which blended draw ignores depth".
// These helpers name the NVN enums and compute the state the draw is submitted with: the same expressions set_dynamic_state()
// uses, including the depth_off_fs / depth_write_fs / read-only-depth overrides.
const char* nvn_blend_func_name(int f) {
    switch (f) {
        case NVN_BLEND_FUNC_ZERO: return "ZERO";
        case NVN_BLEND_FUNC_ONE: return "ONE";
        case NVN_BLEND_FUNC_SRC_COLOR: return "SRC_COLOR";
        case NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR: return "1-SRC_COLOR";
        case NVN_BLEND_FUNC_SRC_ALPHA: return "SRC_ALPHA";
        case NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA: return "1-SRC_ALPHA";
        case NVN_BLEND_FUNC_DST_ALPHA: return "DST_ALPHA";
        case NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA: return "1-DST_ALPHA";
        case NVN_BLEND_FUNC_DST_COLOR: return "DST_COLOR";
        case NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR: return "1-DST_COLOR";
        case NVN_BLEND_FUNC_SRC_ALPHA_SATURATE: return "SRC_ALPHA_SAT";
        case NVN_BLEND_FUNC_SRC1_COLOR: return "SRC1_COLOR";
        case NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR: return "1-SRC1_COLOR";
        case NVN_BLEND_FUNC_SRC1_ALPHA: return "SRC1_ALPHA";
        case NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA: return "1-SRC1_ALPHA";
        default: return "?";
    }
}

const char* nvn_blend_eq_name(int e) {
    switch (e) {
        case NVN_BLEND_EQUATION_ADD: return "ADD";
        case NVN_BLEND_EQUATION_SUB: return "SUB";
        case NVN_BLEND_EQUATION_REVERSE_SUB: return "REV_SUB";
        case NVN_BLEND_EQUATION_MIN: return "MIN";
        case NVN_BLEND_EQUATION_MAX: return "MAX";
        default: return "?";
    }
}

// NVNdepthFunc / NVNcompareFunc share one numbering (1..8 = NEVER..ALWAYS).
const char* nvn_compare_name(int f) {
    switch (f) {
        case 1: return "NEVER";
        case 2: return "LESS";
        case 3: return "EQUAL";
        case 4: return "LEQUAL";
        case 5: return "GREATER";
        case 6: return "NOTEQUAL";
        case 7: return "GEQUAL";
        case 8: return "ALWAYS";
        default: return "?";
    }
}

// The depth func set_dynamic_state() actually submits (depth_func_equal may rewrite EQUAL).
int effective_depth_func() {
    struct Cfg { int func; bool mrt_only; };
    static const Cfg eq = [] {
        Cfg c{0, false};
        std::string v = nxrt::config_get_str("nvn", "depth_func_equal", "keep");
        if (v.size() > 4 && v.compare(v.size() - 4, 4, "_mrt") == 0) { c.mrt_only = true; v.resize(v.size() - 4); }
        if (v == "lequal") c.func = NVN_COMPARE_FUNC_LEQUAL;
        else if (v == "gequal") c.func = NVN_COMPARE_FUNC_GEQUAL;
        else if (v == "always") c.func = NVN_COMPARE_FUNC_ALWAYS;
        return c;
    }();
    const int f = S.ds.depth_func ? S.ds.depth_func : NVN_DEPTH_FUNC_LESS;
    return (f == NVN_COMPARE_FUNC_EQUAL && eq.func && (!eq.mrt_only || S.ncolor >= 4)) ? eq.func : f;
}

// The mirrored comparison: what a pass would test if its depth direction were the console's rather than ours.
int mirrored_depth_func(int f) {
    switch (f) {
        case NVN_COMPARE_FUNC_LESS: return NVN_COMPARE_FUNC_GREATER;
        case NVN_COMPARE_FUNC_GREATER: return NVN_COMPARE_FUNC_LESS;
        case NVN_COMPARE_FUNC_LEQUAL: return NVN_COMPARE_FUNC_GEQUAL;
        case NVN_COMPARE_FUNC_GEQUAL: return NVN_COMPARE_FUNC_LEQUAL;
        default: return f;
    }
}

// The two columns that answer "this draw ignores the scene's depth": what vkCmdSetDepthTestEnable / vkCmdSetDepthWriteEnable
// get for the draw being traced (have_depth = the pass has a depth attachment at all).
struct EffectiveDepth { bool test; bool write; int func; };
EffectiveDepth effective_depth(const Pass& P) {
    const bool have_depth = P.depth_fmt != VK_FORMAT_UNDEFINED;
    EffectiveDepth e;
    e.test = have_depth && S.ds.depth_test && !fs_list_matches("depth_off_fs");
    e.write = have_depth && (fs_list_matches("depth_write_fs") || (!P.depth_ro && S.ds.depth_test && S.ds.depth_write));
    e.func = effective_depth_func();
    return e;
}

void set_dynamic_state(uint32_t dirty, VkPrimitiveTopology topo, bool indexed, int index_type) {
    VkCommandBuffer cb = F->cb;
    if (dirty & kDynVp) set_viewports_and_scissors();
    const bool cull_off = cull_off_for_current_draw();
    // [nvn] cull_off_all (NXRT_NVN_CULL_OFF_ALL, default 0): rasterise every draw with both faces. The probe for "this prop is not
    // culled by the wrong side": XC2 leaves pieces of geometry (a prop on the brazier, structures in the dock) with no pixels at all,
    // so the frame shows the background where they should be.
    static const bool cull_off_all = nxrt::config_get_bool("nvn", "cull_off_all", false);
    static bool cull_off_prev = false;
    // A live front_face switch (hot_state_file) changes nothing about the guest state, so the dirty bits alone would never re-send
    // the front face: remember what was last applied and re-send when the mode changes.
    static int ff_applied = -1;
    const int ff_now = std::clamp(hot_int("front_face", -1), -1, 3);
    if ((dirty & kDynRaster) || cull_off != cull_off_prev || (ff_now >= 0 && ff_now != ff_applied)) {
        cull_off_prev = cull_off;
        if (ff_now >= 0 && ff_now != ff_applied) {
            NXLOG_W("nvn.replay", "front face mode %d -> %d (live)", ff_applied, ff_now);
            ff_applied = ff_now;
        }
        vkCmdSetCullMode(cb, (cull_off || cull_off_all) ? VK_CULL_MODE_NONE : vk_cull_mode(S.poly.cull_face));
        // The front face is numbered in Vulkan's framebuffer coordinates (y down), NVN's in GL's (y up), and the upper-left window
        // origin makes the viewport flip y (set_viewports_and_scissors), which reverses the winding the rasterizer computes. So the
        // Vulkan enum only matches the console's once the viewport flip is accounted for: flip the front face exactly when the
        // viewport is flipped (the lower-left origin uses a positive height and needs no flip).
        // Measured on XC2: the deferred lighting (#233) draws one full-screen triangle with cull BACK, and without the flip it is
        // culled, which left t19/t20 at their clear value and the whole frame at G-buffer albedo (HANDOVER 26).
        //   [nvn] front_face_mode: 0 = flip with the lower-left origin (the old, inverted choice), 1 = flip with the flipped
        //                          viewport (default), 2 = never, 3 = always.
        static const int ff_cfg = static_cast<int>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "front_face_mode", 1), 0, 3));
const int ff_mode = std::clamp(hot_int("front_face", ff_cfg), 0, 3);   // live: [nvn] hot_state_file, line "front_face=N"
        VkFrontFace ff = vk_front_face(S.poly.front_face);
        const bool vp_flipped = S.window_origin != NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
        const bool flip_ff = ff_mode == 3 || (ff_mode == 1 && vp_flipped) || (ff_mode == 0 && !vp_flipped);
        // Which way front face is being decided, once per change: a culling disagreement shows up as a part of a model
        // vanishing for a frame, and this line is what tells that apart from a texture or shader problem.
        {
            static int last_mode = -1, last_org = -1;
            const int org = static_cast<int>(S.window_origin);
            if (ff_mode != last_mode || org != last_org) {
                last_mode = ff_mode;
                last_org = org;
                NXLOG_I(kTag, "front face: window origin %s, [nvn] front_face_mode %d -> flip %d",
                        S.window_origin == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT ? "lower-left" : "upper-left", ff_mode, flip_ff ? 1 : 0);
            }
        }
        if (flip_ff) ff = ff == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        vkCmdSetFrontFace(cb, ff);
        vkCmdSetRasterizerDiscardEnable(cb, S.rasterizer_discard);
        vkCmdSetLineWidth(cb, device().features().wide_lines ? std::max(1.0f, S.line_width) : 1.0f);
    }
    if (topo != g_topo) {
        vkCmdSetPrimitiveTopology(cb, topo);
        g_topo = topo;
    }
    const bool restart = S.prim_restart && indexed &&
                         ((index_type == NVN_INDEX_TYPE_UNSIGNED_INT && S.restart_index == 0xffffffffu) || (index_type == NVN_INDEX_TYPE_UNSIGNED_SHORT && S.restart_index == 0xffffu) ||
                          (index_type == NVN_INDEX_TYPE_UNSIGNED_BYTE && S.restart_index == 0xffu));
    if (!g_restart_valid || restart != g_restart || (dirty & kDynRaster)) {
        vkCmdSetPrimitiveRestartEnable(cb, restart);
        g_restart = restart;
        g_restart_valid = true;
    }
    const bool have_depth = P.depth_fmt != VK_FORMAT_UNDEFINED;
    // [nvn] depth_func_equal = keep|lequal|gequal|always (default keep), optionally with a "_mrt" suffix: what an EQUAL depth test becomes.
    // Monolith's engine renders the opaque world twice - a mask=0 depth prepass with its own vertex shader, then the G-buffer fill with
    // depth EQUAL and depth write off - so the fill only lands where the prepass wrote exactly the same depth. A port whose two vertex
    // shaders round the clip position differently loses whole submeshes that way (HANDOVER 19). lequal keeps the test but accepts a fill
    // nearer than the prepass. "_mrt" applies it only to draws with 4 or more colour targets, i.e. this engine's G-buffer fill, and
    // leaves the deferred-lighting pass alone (it uses EQUAL against the depth-16 material class mask, where a looser test is wrong).
    struct EqualOverride {
        int func = 0;
        bool mrt_only = false;
    };
    static const EqualOverride equal_override = [] {
        EqualOverride e;
        std::string v = nxrt::config_get_str("nvn", "depth_func_equal", "keep");
        if (v.size() > 4 && v.compare(v.size() - 4, 4, "_mrt") == 0) { e.mrt_only = true; v.resize(v.size() - 4); }
        if (v == "lequal") e.func = NVN_COMPARE_FUNC_LEQUAL;
        else if (v == "gequal") e.func = NVN_COMPARE_FUNC_GEQUAL;
        else if (v == "always") e.func = NVN_COMPARE_FUNC_ALWAYS;
        return e;
    }();
    // [nvn] stencil_off = 1 (default 0): the debug counterpart for the other way a submesh disappears without a warning.
    static const bool stencil_off = nxrt::config_get_bool("nvn", "stencil_off", false);
    auto depth_func = [](int f) {
        return f == NVN_COMPARE_FUNC_EQUAL && equal_override.func && (!equal_override.mrt_only || S.ncolor >= 4) ? equal_override.func : f;
    };
    // [nvn] depth_off_fs = "8f50dea8" (comma list of fragment shader hash prefixes, default empty): those draws get no depth
    // test at all. The experiment for "this decal layer draws nothing while its coplanar neighbour does" - a layer that appears
    // with the test off is rejected by depth, not by geometry or the shader (XC2's eye iris over its eye card, HANDOVER 21).
    static const std::vector<std::string> depth_off_fs = [] {
        std::vector<std::string> v;
        const std::string s = nxrt::config_get_str("nvn", "depth_off_fs", "");
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) v.push_back(s.substr(i, j - i));
            i = j + 1;
        }
        return v;
    }();
    bool depth_off = false;
    if (!depth_off_fs.empty() && S.stage[kFragment])
        for (const std::string& p : depth_off_fs)
            if (!p.empty() && S.stage[kFragment]->hash.compare(0, p.size(), p) == 0) { depth_off = true; break; }
    static bool depth_off_prev = false;
    const bool depth_write_on = fs_list_matches("depth_write_fs");
    static bool depth_write_prev = false;
    // [nvn] depth_func_flip_fs = "48b8ee56" (comma list of fragment shader hash prefixes, default empty): those draws get the
    // depth comparison mirrored (LESS<->GREATER, LEQUAL<->GEQUAL; EQUAL, NOTEQUAL, ALWAYS and NEVER stay). The experiment for
    // "this pass draws nothing while the console lights the same place": a light-volume pass that marks a 640x360 list with
    // "GEQUAL" against the half-res depth (HANDOVER 42.12) is rejected by depth exactly where our depth direction differs from
    // the console's, and the region it should have marked stays dark.
    const bool depth_flip = fs_list_matches("depth_func_flip_fs");
    static bool depth_flip_prev = false;
    if ((dirty & kDynDepth) || depth_off != depth_off_prev || depth_write_on != depth_write_prev || depth_flip != depth_flip_prev) {
        depth_off_prev = depth_off;
        depth_write_prev = depth_write_on;
        depth_flip_prev = depth_flip;
        vkCmdSetDepthTestEnable(cb, have_depth && S.ds.depth_test && !depth_off);
        vkCmdSetDepthWriteEnable(cb, have_depth && (depth_write_on || (!P.depth_ro && S.ds.depth_test && S.ds.depth_write)));
        vkCmdSetDepthCompareOp(cb, vk_compare_op(depth_flip ? mirrored_depth_func(depth_func(S.ds.depth_func ? S.ds.depth_func : NVN_DEPTH_FUNC_LESS))
                                                          : depth_func(S.ds.depth_func ? S.ds.depth_func : NVN_DEPTH_FUNC_LESS)));
        vkCmdSetDepthBoundsTestEnable(cb, have_depth && S.depth_bounds && device().features().depth_bounds);
        if (S.depth_bounds && device().features().depth_bounds) vkCmdSetDepthBounds(cb, S.db_near, S.db_far);
    }
    if (dirty & kDynStencil) {
        vkCmdSetStencilTestEnable(cb, P.has_stencil && S.ds.stencil_test && !stencil_off);
        for (int f = 0; f < 2; ++f) {
            const VkStencilFaceFlags face = f == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
            vkCmdSetStencilOp(cb, face, vk_stencil_op(S.ds.stencil_fail[f]), vk_stencil_op(S.ds.depth_pass[f]), vk_stencil_op(S.ds.depth_fail[f]),
                              vk_compare_op(S.ds.stencil_func[f] ? S.ds.stencil_func[f] : NVN_STENCIL_FUNC_ALWAYS));
            vkCmdSetStencilReference(cb, face, S.stencil_ref[f]);
            vkCmdSetStencilCompareMask(cb, face, S.stencil_vmask[f]);
            // A read-only depth target zeroes the stencil write mask, but that is Vulkan's rule for one image, not the
            // console's: XC2 samples the depth and writes the stencil in the same pass. The layout choice above keeps the
            // attachment read-write when this draw writes stencil, so the mask can stay (HANDOVER 20).
            static const bool stencil_under_ro = nxrt::config_get_bool("nvn", "stencil_under_ro", true);
            const bool can_write = !P.depth_ro || (stencil_under_ro && S.stencil_wmask[f]);
            vkCmdSetStencilWriteMask(cb, face, can_write ? S.stencil_wmask[f] : 0u);
            // [nvn] trace: what is actually submitted for the stencil, not what the game asked for. The light list is built by a
            // REPLACE-64 mark followed by EQUAL-64 draws (HANDOVER 42.12); when the mark's write never lands, every gated draw
            // afterwards is dropped and a rectangle of the 640x360 list stays empty.
            if (g_trace)
                TRACE("    stencil submitted face %d: enable %d (has_stencil %d, guest test %d, stencil_off %d) ref %u vmask %x wmask %x (can_write %d, depth_ro %d) "
                      "ops fail %d pass %d depthfail %d func %d", f, P.has_stencil && S.ds.stencil_test && !stencil_off ? 1 : 0, P.has_stencil ? 1 : 0,
                      S.ds.stencil_test ? 1 : 0, stencil_off ? 1 : 0, S.stencil_ref[f], S.stencil_vmask[f], can_write ? S.stencil_wmask[f] : 0u, can_write ? 1 : 0,
                      P.depth_ro ? 1 : 0, static_cast<int>(vk_stencil_op(S.ds.stencil_fail[f])), static_cast<int>(vk_stencil_op(S.ds.depth_pass[f])),
                      static_cast<int>(vk_stencil_op(S.ds.depth_fail[f])),
                      static_cast<int>(vk_compare_op(S.ds.stencil_func[f] ? S.ds.stencil_func[f] : NVN_STENCIL_FUNC_ALWAYS)));
        }
    }
    if (dirty & kDynBias) {
        const int off = S.poly.offset_enables;
        const int mode = S.poly.polygon_mode;
        const bool bias = (mode == NVN_POLYGON_MODE_FILL && (off & NVN_POLYGON_OFFSET_ENABLE_FILL)) ||
                          (mode == NVN_POLYGON_MODE_LINE && (off & NVN_POLYGON_OFFSET_ENABLE_LINE)) ||
                          (mode == NVN_POLYGON_MODE_POINT && (off & NVN_POLYGON_OFFSET_ENABLE_POINT));
        vkCmdSetDepthBiasEnable(cb, bias);
        const float clamp = device().features().depth_bias_clamp ? S.bias_clamp : 0.0f;
        // When opted in via [nvn] depth_bias_force_unorm = 1 and supported, force UNORM depth-bias stepping (investigated for #44);
        // left off by default because forcing FORCE_UNORM globally causes severe shadow-acne bands on RTX 40/50 GPUs (#51, #54, #39).
        if (device().features().depth_bias_control_force_unorm && vkCmdSetDepthBias2EXT) {
            VkDepthBiasRepresentationInfoEXT rep = {VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT};
            rep.depthBiasRepresentation = VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
            rep.depthBiasExact = VK_FALSE;
            VkDepthBiasInfoEXT info = {VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT};
            info.pNext = &rep;
            info.depthBiasConstantFactor = S.bias_units;
            info.depthBiasClamp = clamp;
            info.depthBiasSlopeFactor = S.bias_factor;
            vkCmdSetDepthBias2EXT(cb, &info);
        } else {
            vkCmdSetDepthBias(cb, S.bias_units, clamp, S.bias_factor);
        }
    }
    if (dirty & kDynBlend) vkCmdSetBlendConstants(cb, S.blend_color);
}

// ---- [nvn] earlyz: the two draws and their state (HANDOVER 42.22.2; the mechanism is described at the globals) --------------------
// One-time setup: the bit ([nvn] earlyz_bit, default 4) and the shader list ([nvn] earlyz_fs).
void earlyz_init() {
    static bool done = false;
    if (done) return;
    done = true;
    g_earlyz_on = nxrt::config_get_bool("nvn", "earlyz", false) && !nxrt::config_get_bool("nvn", "stencil_off", false);
    const int bit = static_cast<int>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "earlyz_bit", 4), 0, 7));
    g_earlyz_bit = static_cast<uint8_t>(1u << bit);
    if (g_earlyz_on) NXLOG_I(kTag, "earlyz: emulating the console's early depth test with stencil bit %d (0x%02x) for the fragment shaders of [nvn] earlyz_fs", bit, g_earlyz_bit);
}

// Can this draw be emulated? Every condition is checked against the state that would be submitted; false keeps the draw as it is.
bool earlyz_wants(const ProgLayout& pl) {
    earlyz_init();
    if (!g_earlyz_on || !S.stage[kFragment] || !fs_list_matches("earlyz_fs")) return false;
    if (!P.open || !P.has_stencil || P.depth_fmt == VK_FORMAT_UNDEFINED || P.ncolor == 0) return false;
    const EffectiveDepth ed = effective_depth(P);
    if (P.depth_ro || !ed.test || !S.ds.stencil_test) return false;   // nothing is rejected, or the game's ops cannot be reproduced
    if (g_xfb_active) return false;   // the pipeline cannot change inside a transform feedback section
    const std::string fs = S.stage[kFragment]->hash;
    // 1. the bit must be one the draw's own stencil state does not use
    for (int f = 0; f < 2; ++f) {
        if (!((S.stencil_vmask[f] | S.stencil_wmask[f]) & g_earlyz_bit)) continue;
        if (first_time_key("earlyz-bit " + fs))
            NXLOG_W(kTag, "earlyz: fs %s tests or writes stencil bit 0x%02x itself (vmask %x wmask %x): its draws are left alone", fs.c_str(), g_earlyz_bit,
                    S.stencil_vmask[f], S.stencil_wmask[f]);
        return false;
    }
    // 2. the game's own stencil writes must come out the same from the pre-pass: one bit, tested EQUAL against itself and cleared by ZERO.
    //    Then "the test failed" means that bit is 0, so KEEP is the same as the game's ZERO on the fail path, and at the pass point every
    //    masked bit is 1, so INVERT clears exactly the bit the game's ZERO would have cleared (and sets the bit we add).
    for (int f = 0; f < 2; ++f) {
        const int fn = S.ds.stencil_func[f] ? S.ds.stencil_func[f] : NVN_STENCIL_FUNC_ALWAYS;
        const uint32_t single = S.stencil_vmask[f];
        const bool ok = S.ds.stencil_fail[f] == NVN_STENCIL_OP_ZERO && S.ds.depth_fail[f] == NVN_STENCIL_OP_ZERO && S.ds.depth_pass[f] == NVN_STENCIL_OP_ZERO &&
                        fn == NVN_COMPARE_FUNC_EQUAL && single != 0 && (single & (single - 1)) == 0 && S.stencil_wmask[f] == single && S.stencil_ref[f] == single;
        if (!ok) {
            if (first_time_key("earlyz-ops " + fs))
                NXLOG_I(kTag, "earlyz: fs %s keeps the driver's order: its stencil state (func %d ref %x vmask %x wmask %x ops %d/%d/%d) is not the one-bit EQUAL/ZERO shape the pre-pass reproduces",
                        fs.c_str(), fn, S.stencil_ref[f], S.stencil_vmask[f], S.stencil_wmask[f], S.ds.stencil_fail[f], S.ds.depth_fail[f], S.ds.depth_pass[f]);
            return false;
        }
    }
    // 3. every side effect of the fragment shader must be catchable: a storage image can go to the scratch, a storage buffer cannot
    for (const BindingDesc& d : pl.bindings)
        if (d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
            if (first_time_key("earlyz-ssbo " + fs))
                NXLOG_I(kTag, "earlyz: fs %s writes a storage buffer (set %u binding %u): its draws keep the driver's order", fs.c_str(), d.set, d.binding);
            return false;
        }
    if (first_time_key("earlyz-on " + fs))
        NXLOG_I(kTag, "earlyz: fs %s gets the depth pre-pass (stencil bit 0x%02x, depth %s%s): fragments the depth test rejects no longer write its storage images",
                fs.c_str(), g_earlyz_bit, nvn_compare_name(effective_depth_func()), ed.write ? " +write" : "");
    return true;
}

// The depth and stencil state of the pre-pass (prepass) and of the real draw.
void earlyz_depth_state(bool prepass) {
    VkCommandBuffer cb = F->cb;
    if (prepass) {   // the draw's own test, unchanged: it runs late, but the stencil operations below run after it
        const EffectiveDepth ed = effective_depth(P);
        vkCmdSetDepthTestEnable(cb, ed.test ? VK_TRUE : VK_FALSE);
        vkCmdSetDepthWriteEnable(cb, ed.write ? VK_TRUE : VK_FALSE);
        const int fn = effective_depth_func();
        vkCmdSetDepthCompareOp(cb, vk_compare_op(fs_list_matches("depth_func_flip_fs") ? mirrored_depth_func(fn) : fn));
    } else {         // the decision was taken by the pre-pass: the depth test must not reject anything a second time
        vkCmdSetDepthTestEnable(cb, VK_FALSE);
        vkCmdSetDepthWriteEnable(cb, VK_FALSE);
        if (S.depth_bounds && device().features().depth_bounds) vkCmdSetDepthBoundsTestEnable(cb, VK_FALSE);
    }
}

void earlyz_stencil_state(bool prepass) {
    VkCommandBuffer cb = F->cb;
    vkCmdSetStencilTestEnable(cb, P.has_stencil ? VK_TRUE : VK_FALSE);
    for (int f = 0; f < 2; ++f) {
        const VkStencilFaceFlags face = f == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
        if (prepass) {
            // fail: KEEP (a later face of the same light volume hits a bit an earlier face cleared; the game's ZERO would clear nothing
            // there, since a failed one-bit EQUAL test means that bit is already 0). pass: INVERT sets our bit and clears the game's.
            vkCmdSetStencilOp(cb, face, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_INVERT, VK_STENCIL_OP_ZERO,
                              vk_compare_op(S.ds.stencil_func[f] ? S.ds.stencil_func[f] : NVN_STENCIL_FUNC_ALWAYS));
            vkCmdSetStencilReference(cb, face, S.stencil_ref[f]);
            vkCmdSetStencilCompareMask(cb, face, S.stencil_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, S.stencil_wmask[f] | g_earlyz_bit);
        } else {
            vkCmdSetStencilOp(cb, face, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_COMPARE_OP_EQUAL);
            vkCmdSetStencilReference(cb, face, g_earlyz_bit);
            vkCmdSetStencilCompareMask(cb, face, g_earlyz_bit);
            vkCmdSetStencilWriteMask(cb, face, g_earlyz_bit);
        }
    }
}

// begin_draw's tail: decide, and get the pre-pass pipeline (the draw's own key with every colour write masked off).
void earlyz_plan(const ProgLayout& pl) {
    g_earlyz_pending = false;
    if (!g_pipe_key_valid || !g_pipe || !earlyz_wants(pl)) return;
    GraphicsKey k = g_pipe_key;
    for (uint32_t i = 0; i < 8; ++i) k.write_mask[i] = 0;
    const VkPipeline p = graphics_pipeline(k, pl);
    if (!p) {   // still being linked (gpl_async): this frame's draw keeps the driver's order
        return;
    }
    g_earlyz_pipe = p;
    g_earlyz_pending = true;
}

// Before the pre-pass geometry: the masked pipeline, the draw's depth test, our stencil operations, the scratch storage images.
void earlyz_prepass_begin(const ProgLayout& pl) {
    vkCmdBindPipeline(F->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_earlyz_pipe);
    g_bound_pipeline = g_earlyz_pipe;
    earlyz_depth_state(true);
    earlyz_stencil_state(true);
    g_earlyz_scratch = true;
    bind_sets(pl, VK_PIPELINE_BIND_POINT_GRAPHICS);
    g_earlyz_scratch = false;
}

// After the pre-pass geometry: the real pipeline, the game's storage images, no depth test and the stencil gate.
void earlyz_prepass_end(const ProgLayout& pl) {
    vkCmdBindPipeline(F->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe);
    g_bound_pipeline = g_pipe;
    earlyz_depth_state(false);
    earlyz_stencil_state(false);
    bind_sets(pl, VK_PIPELINE_BIND_POINT_GRAPHICS);
    g_dirty |= kDynDepth | kDynStencil;   // the next draw sets its own depth/stencil again (set_dynamic_state's "previous" copies are stale)
    g_earlyz_pending = false;
    ++g_earlyz_draws;
    static uint64_t logged = 0;           // evidence in the log without a trace: about one line every 4 s at 30 Hz
    if (g_earlyz_draws - logged >= 2000) {
        logged = g_earlyz_draws;
        NXLOG_I(kTag, "earlyz: %llu light-list draws took the depth pre-pass so far (stencil bit 0x%02x, %.1f per presented frame)", static_cast<unsigned long long>(g_earlyz_draws),
                g_earlyz_bit, static_cast<double>(g_earlyz_draws) / std::max<double>(1.0, static_cast<double>(counters().presents.load())));
    }
}

// ---- [nvn] earlyz_mask: the mask rebuild itself (HANDOVER 45.2; the why is at the globals) -------------------------------------
#include "earlyz_mask_shaders.inc"   // kEarlyzMaskComp, generated by scripts/gen_earlyz_mask_shaders.py

struct EarlyzMaskPipe {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    bool tried = false, ok = false;
};
EarlyzMaskPipe g_ezm;

bool earlyz_mask_init() {
    if (g_ezm.tried) return g_ezm.ok;
    g_ezm.tried = true;
    const VkDevice d = vkdev();
    auto fail = [](const char* what) {
        NXLOG_W(kTag, "earlyz_mask: %s failed: the registration mask stays as the shaders leave it", what);
        return false;
    };
    VkDescriptorSetLayoutBinding b[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.bindingCount = 2;
    sl.pBindings = b;
    if (vkCreateDescriptorSetLayout(d, &sl, nullptr, &g_ezm.set_layout) != VK_SUCCESS) return fail("descriptor set layout");
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_ezm.set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(d, &pli, nullptr, &g_ezm.layout) != VK_SUCCESS) return fail("pipeline layout");
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof kEarlyzMaskComp;
    mi.pCode = kEarlyzMaskComp;
    if (vkCreateShaderModule(d, &mi, nullptr, &g_ezm.module) != VK_SUCCESS) return fail("shader module");
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = g_ezm.module;
    ci.stage.pName = "main";
    ci.layout = g_ezm.layout;
    if (create_compute_pipeline_tracked(d, pipeline_cache_handle(), &ci, &g_ezm.pipe) != VK_SUCCESS) return fail("compute pipeline");
    g_ezm.ok = true;
    NXLOG_I(kTag, "earlyz_mask: the registration mask is rebuilt from the list after every list draw (the black rectangle's cause)");
    return true;
}

// Does this draw register into a storage image the mask can be derived from? Decided per draw, used after its geometry.
void earlyz_mask_plan(const ProgLayout& pl) {
    g_earlyz_mask_ready = false;
    g_earlyz_mask_tex = Ref<Texture>();
    static const bool on = [] {
        const std::string raw = nxrt::config_get_str("nvn", "earlyz_mask", "auto");
        const bool v = raw == "auto" ? !early_fragment_tests_active() : nxrt::config_get_bool("nvn", "earlyz_mask", true);
        NXLOG_I(kTag, "earlyz_mask: %s (config '%s', early_fragment_tests %s)", v ? "on (compute mask rebuild)" : "off (hardware EarlyFragmentTests)",
                raw.c_str(), early_fragment_tests_active() ? "active" : "off");
        return v;
    }();
    if (!on || !pl.has_storage_writes || !S.stage[kFragment]) return;
    // The storage image first: only a shader that writes one can be a list writer, so the "not in earlyz_fs" note stays useful
    // instead of firing for every material in the frame (HANDOVER 90: the list defaults to the eleven NS1-era hashes).
    Texture* img = nullptr;
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) continue;
        const TexResolved r = resolve_image(d);
        if (r.tex) { img = r.tex; break; }
    }
    if (!img) return;
    if (!fs_list_matches("earlyz_fs")) {
        if (first_time_key("earlyz_mask miss " + S.stage[kFragment]->hash))
            NXLOG_W(kTag, "earlyz_mask: fs %s writes a storage image but is not in earlyz_fs, so the mask is not rebuilt for it",
                    S.stage[kFragment]->hash.c_str());
        return;
    }
    if (early_fragment_tests_patched(S.stage[kFragment]->hash)) {
        if (first_time_key("earlyz_mask hw " + S.stage[kFragment]->hash))
            NXLOG_I(kTag, "earlyz_mask: fs %s uses hardware early fragment tests — skipping per-draw mask rebuild",
                    S.stage[kFragment]->hash.c_str());
        return;
    }
    g_earlyz_mask_tex = Ref<Texture>::share(img);
    g_earlyz_mask_ready = true;
    if (first_time_key("earlyz_mask " + S.stage[kFragment]->hash))
        NXLOG_I(kTag, "earlyz_mask: fs %s registers into t%u (%s): the mask is derived from the list after its draws", S.stage[kFragment]->hash.c_str(), img->id,
                img->fmt ? img->fmt->name : "?");
}

// After the draw: mask = OR over the list's four slots of (flag == 0xff) ? 1 << (index & 31) : 0.
void earlyz_mask_rebuild() {
    if (!g_earlyz_mask_ready) return;
    g_earlyz_mask_ready = false;
    Texture* list = S.color_tex[0] ? S.color_tex[0].get() : nullptr;
    Texture* mask = g_earlyz_mask_tex.get();
    if (!list || !mask || list == mask || !tex_gpu(list) || !tex_gpu(mask) || !earlyz_mask_init()) return;
    if (P.color_fmt[0] != VK_FORMAT_R32G32_UINT || tex_vk_format(mask) != VK_FORMAT_R32_UINT) return;   // not the light list
    ViewRequest q;
    q.format = VK_FORMAT_R32G32_UINT;
    const VkImageView lv = tex_sampled_view(list, q);
    q.format = VK_FORMAT_R32_UINT;
    const VkImageView mv = tex_sampled_view(mask, q);
    if (!lv || !mv) return;
    end_pass();                                     // the list is a colour attachment of the open pass
    tex_transition(list, *F, VK_IMAGE_LAYOUT_GENERAL);
    tex_transition(mask, *F, VK_IMAGE_LAYOUT_GENERAL);
    const VkDescriptorSet ds = F->alloc_set(g_ezm.set_layout);
    if (!ds) return;
    VkDescriptorImageInfo ii[2] = {};
    ii[0].imageView = lv;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[1].imageView = mv;
    ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(vkdev(), 2, w, 0, nullptr);
    const VkExtent3D e = tex_image_extent(list, 0);
    if (first_time_key("earlyz_mask extent"))
        NXLOG_I(kTag, "earlyz_mask: rebuild dispatch extent %ux%u", e.width, e.height);
    vkCmdBindPipeline(F->cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ezm.pipe);
    vkCmdBindDescriptorSets(F->cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ezm.layout, 0, 1, &ds, 0, nullptr);
    const uint32_t ext[2] = {e.width, e.height};
    vkCmdPushConstants(F->cb, g_ezm.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof ext, ext);
    vkCmdDispatch(F->cb, (e.width + 7) / 8, (e.height + 7) / 8, 1);
    tex_mark_gpu_written(mask);
    // Narrow barrier: only make the compute shader's writes to `mask` visible to the next fragment draw's storage image reads/writes
    // (instead of an ALL_COMMANDS -> ALL_COMMANDS global GPU stall).
    {
        VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(F->cb, &di);
    }
    ++g_earlyz_mask_draws;
    static uint64_t logged = 0;
    if (g_earlyz_mask_draws - logged >= 2000) {
        logged = g_earlyz_mask_draws;
        NXLOG_I(kTag, "earlyz_mask: %llu list draws had their registration mask rebuilt (%.1f per presented frame)", static_cast<unsigned long long>(g_earlyz_mask_draws),
                static_cast<double>(g_earlyz_mask_draws) / std::max<double>(1.0, static_cast<double>(counters().presents.load())));
    }
}

// A draw that runs the two draws is issued by do_draw_arrays / do_draw_elements only; the others drop the emulation for this draw.
void earlyz_refuse_indirect(const char* what) {
    if (!g_earlyz_pending) return;
    g_earlyz_pending = false;
    if (S.stage[kFragment] && first_time_key("earlyz-indirect " + S.stage[kFragment]->hash))
        NXLOG_I(kTag, "earlyz: fs %s draws through %s: those keep the driver's order", S.stage[kFragment]->hash.c_str(), what);
}

// Prepares everything a draw needs and binds it. False: skip the draw.
// [nvn] trace_min_sec = 250: trace_after_fs only starts that many seconds after the first draw (the scene to look at comes late).
bool trigger_time_ok() {
    static const auto start = std::chrono::steady_clock::now();
    static const double min_sec = static_cast<double>(nxrt::config_get_int("nvn", "trace_min_sec", 0));
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= min_sec;
}

// Scene upscaler (S14, src/gfx/extras/hdr_nvn.h): before a tone-map draw whose HDR scene input is smaller than its target, FSR 1 enlarges the scene
// to the target's size and the draw samples that instead. The override lives until the next draw.
// Temporal upscaling (FSR 3 / DLSS / DLAA, src/gfx/extras/temporal.h): a G-buffer draw (4+ colour targets and a depth target) offers its vertex
// uniform slot 0 as the camera block; the first valid one of the frame is kept with its depth target.
void temporal_gbuffer_note() {
    if (S.ncolor < 4 || !S.depth_tex || !S.stage[kVertex] || extras::temporal_camera_ready()) return;
    const GState::Bind& u = S.ubo[kVertex][0];
    if (!u.addr || u.size < 272) return;
    uint32_t w[68];
    std::memcpy(w, host_ptr(u.addr), sizeof w);
    tsr_jitter(w, 68);                                                    // the camera as the draw sees it (the bound copy is jittered the same way)
    extras::temporal_note_gbuffer(w, 68);
    if (extras::temporal_camera_ready()) g_tsr_depth = S.depth_tex;
}

// The tone-map draw with a temporal mode: the largest 2D colour input is the HDR scene; the SDK's result (target size) replaces it for this draw.
void temporal_for_draw(const ProgLayout& pl) {
    ++g_tsr_frame;                                                        // one tone-map draw per frame: the next frame's jitter phase
    tsr_cache_clear();
    Texture* target = S.color_tex[0].get();
    Texture* depth = g_tsr_depth.get();
    static bool told_intercept = false, told_input = false, told_depth = false;
    if (!told_intercept) {
        told_intercept = true;
        NXLOG_I(kTag, "temporal: tone-map draw %s intercepted (target %ux%u)", S.stage[kFragment]->hash.c_str(), target ? target->b.width : 0,
                target ? target->b.height : 0);
    }
    if (!extras::temporal_camera_ready()) {                              // counts the frames without a camera (warns after 120)
        extras::temporal_evaluate(nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, {}, {}, false);
        return;
    }
    if (!tex_gpu(target) || !depth || !tex_gpu(depth)) {
        if (!told_depth) {
            told_depth = true;
            NXLOG_W(kTag, "temporal: camera found but no usable target / G-buffer depth texture at the tone-map draw");
        }
        return;
    }
    const BindingDesc* best = nullptr;
    TexResolved best_r{};
    uint64_t best_area = 0;
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        if (view_type_for(b) != VK_IMAGE_VIEW_TYPE_2D) continue;
        TexResolved r = resolve_texture(d);
        if (!r.tex || !tex_gpu(r.tex) || tex_is_depth(r.tex) || shares_image(r.tex, depth)) continue;
        const VkExtent3D e = tex_image_extent(r.tex, 0);
        const uint64_t area = static_cast<uint64_t>(e.width) * e.height;
        if (area > best_area) { best_area = area; best = &d; best_r = r; }
    }
    if (!best) {
        if (!told_input) {
            told_input = true;
            NXLOG_W(kTag, "temporal: the tone-map draw samples no 2D colour texture to use as the scene");
        }
        return;
    }
    const DescriptorBinding& b = best->shader->refl.bindings[best->index];
    const VkExtent3D ce = tex_image_extent(best_r.tex, 0), de = tex_image_extent(depth, 0), te = tex_image_extent(target, 0);
    const VkExtent2D render{std::min(ce.width, de.width), std::min(ce.height, de.height)};
    end_pass();                                                           // compute work cannot run inside a render pass
    tex_ensure_current(best_r.tex, *F);
    tex_transition(best_r.tex, *F, tex_sampled_layout(best_r.tex));
    tex_transition(depth, *F, tex_sampled_layout(depth));
    VkImageView src = tex_sampled_view(best_r.tex, make_request(b, best_r));
    ViewRequest dr;
    dr.depth_aspect = true;
    VkImageView dv = tex_sampled_view(depth, dr);
    if (!src || !dv) return;
    VkImageView out = extras::temporal_evaluate(&device(), F->cb, src, dv, tex_sampled_layout(depth), render, VkExtent2D{te.width, te.height},
                                                S.depth_mode == NVN_DEPTH_MODE_NEAR_IS_MINUS_W);
    g_dirty = kAllDirty;                                                  // the SDKs bind their own compute state
    if (out) {
        g_su_tex = best_r.tex;
        g_su_view = out;
    }
}

void scene_upscale_for_draw(const ProgLayout& pl) {
    g_su_tex = nullptr;
    g_su_view = VK_NULL_HANDLE;
    const ShaderInfo* fs = S.stage[kFragment];
    if (fs && extras::temporal_wanted() && S.ncolor > 0 && S.color_tex[0] && extras::hdr_is_tonemap(fs->spirv, fs->hash)) {
        temporal_for_draw(pl);
        return;
    }
    if (!fs || !extras::scene_upscale_wanted() || S.ncolor == 0 || !S.color_tex[0] || !extras::hdr_is_tonemap(fs->spirv, fs->hash)) return;
    Texture* target = S.color_tex[0].get();
    if (!tex_gpu(target)) return;
    const VkExtent3D te = tex_extent(target, 0);
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        if (view_type_for(b) != VK_IMAGE_VIEW_TYPE_2D) continue;
        TexResolved r = resolve_texture(d);
        if (!r.tex || !tex_gpu(r.tex) || shares_image(r.tex, S.depth_tex.get())) continue;
        const VkExtent3D se = tex_extent(r.tex, 0);
        if (se.width >= te.width && se.height >= te.height) return;          // nothing to enlarge
        extras::SpatialUpscaler* su = extras::scene_upscaler(&device(), VkExtent2D{se.width, se.height}, VkExtent2D{te.width, te.height});
        if (!su) return;
        end_pass();                                                           // compute work cannot run inside a render pass
        tex_ensure_current(r.tex, *F);
        tex_transition(r.tex, *F, tex_sampled_layout(r.tex));
        VkImageView src = tex_sampled_view(r.tex, make_request(b, r));
        if (!src) return;
        VkImageView up = su->upscale(F->cb, src, VkExtent2D{se.width, se.height}, VkExtent2D{se.width, se.height}, VkExtent2D{te.width, te.height}, true,
                                     extras::scene_upscale_prescale(), extras::upscale_sharpness());
        if (up) {
            g_su_tex = r.tex;
            g_su_view = up;
        }
        return;
    }
}

// Evaluation-only programs (NVN tessellates with the fixed SetInner/OuterTessellationLevels): a generated pass-through control shader
// (nvn_tess_gen.h), one per (vertex shader, evaluation shader, patch size). It copies each per-vertex input of the evaluation shader from the
// vertex shader output at the same location (zero where the vertex shader writes none) and gl_Position, zeroes the per-patch inputs, and reads
// the levels from push constants (pushed before each such draw, so level changes need no new pipeline). Null when an interface variable has a
// type the generator does not handle.
const ShaderInfo* g_gen_tcs = nullptr;   // the generated control shader of the current draw (null: the program has its own, or none)
const ShaderInfo* passthrough_tcs_for(const ShaderInfo* vs, const ShaderInfo* tes) {
    struct Key {
        const ShaderInfo* vs; const ShaderInfo* tes; uint32_t n;
        bool operator<(const Key& o) const { return std::tie(vs, tes, n) < std::tie(o.vs, o.tes, o.n); }
    };
    static std::map<Key, std::unique_ptr<ShaderInfo>> cache;
    const uint32_t n = std::clamp<uint32_t>(S.patch_size, 1, 32);
    const Key key{vs, tes, n};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.get();
    auto to_io = [](const ShaderIo& v, tessgen::Io& o) {
        if (v.components < 1 || v.components > 4) return false;
        if (v.type == ScalarType::Float) o.kind = tessgen::Kind::Float;
        else if (v.type == ScalarType::Sint) o.kind = tessgen::Kind::Sint;
        else if (v.type == ScalarType::Uint) o.kind = tessgen::Kind::Uint;
        else return false;
        o.location = v.location;
        o.component = v.component;
        o.components = v.components;
        return true;
    };
    std::vector<tessgen::Io> vs_out, per_vertex, per_patch;
    bool ok = true;
    for (const ShaderIo& o : vs->refl.outputs) {   // an output array (several locations) is not matched: its readers get zero
        tessgen::Io io;
        if (o.array <= 1 && to_io(o, io)) vs_out.push_back(io);
    }
    for (const ShaderIo& in : tes->refl.inputs) {
        tessgen::Io io;
        if (!to_io(in, io)) { ok = false; break; }
        (in.array > 1 ? per_vertex : per_patch).push_back(io);   // in an evaluation shader a non-array input is per patch
    }
    std::unique_ptr<ShaderInfo> si;
    if (ok) {
        const std::vector<tessgen::Copy> copies = tessgen::match(vs_out, per_vertex);
        size_t copied = 0;
        for (const tessgen::Copy& c : copies) copied += c.has_src;
        si = std::make_unique<ShaderInfo>();
        si->hash = "tcs-pass-" + vs->hash + "-" + tes->hash + "-" + std::to_string(n);
        si->stage = ShaderStage::TessControl;
        // gl_ClipDistance / gl_CullDistance: copied when the vertex shader writes them and the evaluation shader reads gl_in[]'s
        auto uses = [](const std::vector<uint32_t>& v, uint32_t b) { return std::find(v.begin(), v.end(), b) != v.end(); };
        constexpr uint32_t kBuiltInClip = 3, kBuiltInCull = 4;
        const bool clip = uses(vs->refl.builtins_used_out, kBuiltInClip) && uses(tes->refl.builtins_used_in, kBuiltInClip);
        const bool cull = uses(vs->refl.builtins_used_out, kBuiltInCull) && uses(tes->refl.builtins_used_in, kBuiltInCull);
        si->spirv = tessgen::passthrough_tcs(copies, per_patch, n, 7, 1, clip, cull);
        si->push_bytes = tessgen::kPushBytes;
        std::string err;
        if (!reflect_spirv(si->spirv.data(), si->spirv.size(), si->refl, &err)) {
            NXLOG_W(kTag, "generated control shader for %s does not reflect: %s", tes->hash.c_str(), err.c_str());
            si.reset();
        } else {
            NXLOG_I(kTag, "evaluation shader %s (vs %s): pass-through control shader, %u vertices, %zu per-vertex locations (%zu from the vertex shader, %zu zero), %zu per-patch%s%s",
                    tes->hash.c_str(), vs->hash.c_str(), n, copies.size(), copied, copies.size() - copied, per_patch.size(), clip ? ", clip distances" : "",
                    cull ? ", cull distances" : "");
        }
    } else {
        NXLOG_W(kTag, "evaluation shader %s: an input is not a 1..4 component float/int, no pass-through control shader", tes->hash.c_str());
    }
    const ShaderInfo* out = si.get();
    cache.emplace(key, std::move(si));
    return out;
}
const ShaderInfo* tcs_now() { return S.stage[kTessCtl] ? S.stage[kTessCtl] : g_gen_tcs; }

bool begin_draw(int mode, bool indexed, int index_type, VkPrimitiveTopology* topo_out) {
    rs_start();
    const ShaderInfo* vs = S.stage[kVertex];
    if (!vs) { log_once("draw-no-vs", "draw without a vertex shader skipped"); return false; }
    if (!S.render_enable) return false;
    ++g_draw_idx;
    if (S.stage[kFragment] && skip_fs_draw(S.stage[kFragment]->hash)) return false;
    // [nvn] skip_light0 (diagnostic, default 0): drop the light-list draw whose per-light block (fs bank c4) carries index 0.
    // The game emits exactly one of those per list pass - the last draw, with no mark before it and its depth test off - and the
    // question is whether that registration is a real light. Skipping it costs nothing else: it is the last draw of the pass, so the
    // stencil bit its ZERO ops would have cleared has no consumer left.
    static const bool skip_light0 = nxrt::config_get_bool("nvn", "skip_light0", false);   // once: this runs for every draw
    if (skip_light0 && S.stage[kFragment] && fs_list_matches("earlyz_fs") && S.ubo[kFragment][1].addr &&
        S.ubo[kFragment][1].size >= 4 && *reinterpret_cast<const uint32_t*>(S.ubo[kFragment][1].addr) == 0) {
        if (first_time_key("skip-light0 " + S.stage[kFragment]->hash)) NXLOG_I(kTag, "skip_light0: dropping light-list draws whose block carries index 0");
        ++counters().draws_skipped;
        return false;
    }
    // Tessellation: Vulkan runs it only with both shaders and a patch list. NVN also allows an evaluation shader alone (fixed levels from
    // SetInner/OuterTessellationLevels): those draws get a generated pass-through control shader (passthrough_tcs_for).
    if (S.stage[kTessCtl] || S.stage[kTessEval] || mode == NVN_DRAW_PRIMITIVE_PATCHES) {
        if (mode != NVN_DRAW_PRIMITIVE_PATCHES) { note_unsupported("tessellation draw: tessellation program drawn without patches"); return false; }
        if (!S.stage[kTessEval]) { note_unsupported("tessellation draw: patches without a tessellation evaluation shader"); return false; }
        if (!S.stage[kTessCtl]) {
            const ShaderInfo* gen = passthrough_tcs_for(vs, S.stage[kTessEval]);
            if (!gen) { note_unsupported("tessellation draw: evaluation shader without a control shader, no pass-through could be made"); return false; }
            if (gen != g_gen_tcs) g_dirty |= kPipe;
            g_gen_tcs = gen;
        }
    } else {
        g_gen_tcs = nullptr;
    }
    if (g_dirty & (kPipe | kDesc)) {
        const ShaderInfo* stages[6] = {vs, S.stage[kFragment], S.stage[kGeometry], tcs_now(), S.stage[kTessEval], nullptr};
        // 64-entry direct-mapped render-thread cache (layouts live until pipeline_shutdown): avoids mutex + std::map on shader switches
        struct PlSlot { const ShaderInfo* s[6] = {}; const ProgLayout* pl = nullptr; uint64_t gen = 0; };
        constexpr size_t kPlSlots = 64;
        static PlSlot pl_cache[kPlSlots];
        const uint64_t gen = prog_layout_gen();
        uint64_t h = reinterpret_cast<uintptr_t>(stages[0]) ^ (reinterpret_cast<uintptr_t>(stages[1]) * 0x9e3779b97f4a7c15ull) ^
                     (reinterpret_cast<uintptr_t>(stages[2]) * 0xbf58476d1ce4e5b9ull) ^ (reinterpret_cast<uintptr_t>(stages[3]) << 7) ^
                     (reinterpret_cast<uintptr_t>(stages[4]) << 13);
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ull;
        PlSlot& sl = pl_cache[(h >> 27) & (kPlSlots - 1)];
        if (sl.gen == gen && sl.pl && std::memcmp(stages, sl.s, sizeof stages) == 0) {
            g_pl = sl.pl;
        } else {
            g_pl = prog_layout(stages);
            std::memcpy(sl.s, stages, sizeof stages);
            sl.pl = g_pl;
            sl.gen = gen;
            ++g_rs.layouts;
        }
    }
    if (!g_trigger_fs.empty() && !g_trigger_done && trigger_time_ok() && S.stage[kFragment] && S.stage[kFragment]->hash.compare(0, g_trigger_fs.size(), g_trigger_fs) == 0) {
        g_trigger_seen = true;
        g_trigger_present = static_cast<int64_t>(counters().presents.load());
    }
    const ProgLayout* pl = g_pl;
    if (!pl) return false;
    MemoScope memo(*pl);
    if (pl->has_set[kPoolSet] && pool_state_moved(S.tpool.get(), S.spool.get())) { g_dirty |= kDesc; ++g_bind_gen; }   // registrations or layout moves since
    // Only uniform buffers changed: set 0 is pushed again below and the sets this program bound in this command buffer stay (the textures
    // and storage buffers they hold are what the draw would resolve again). Anything else since then (a bind, a new pass or command
    // buffer, another program, a feedback loop) takes the full path.
    bool ubo_fast = false;
    if (g_dirty & kDescUbo) {
        ubo_fast = !(g_dirty & kDesc) && pl->ubo_only && pl == g_sets_pl && F->cb == g_sets_cb && !g_fb_seen && ubo_only_rebind_on();
        if (!ubo_fast) g_dirty |= kDesc;
    }
    if (extras::temporal_wanted()) temporal_gbuffer_note();
    if (S.ncolor >= 4 && S.depth_tex && S.depth_tex->b.width >= 320 && S.ds.depth_test && S.ds.depth_write) {
        g_scene_depth = S.depth_tex;
        const bool neg1 = (S.depth_mode == NVN_DEPTH_MODE_NEAR_IS_MINUS_W);
        const int dfn = effective_depth_func();
        const bool rev = (dfn == NVN_DEPTH_FUNC_GREATER || dfn == NVN_DEPTH_FUNC_GEQUAL);
        g_scene_depth_mode = (neg1 ? 2 : 0) + (rev ? 1 : 0);
    }
    rs_mark(kRsHead);
    if (g_dirty & (kDesc | kPipe)) {
        ++g_rs.draws_desc;
        g_prog_writes = pl->has_storage_writes;
        attach_targets(false);
        // Keep the depth-stencil attachment read-write when this draw writes stencil, even though it samples the depth:
        // XC2 samples the depth plane and writes the stencil plane in one pass and the console's GPU keeps them apart
        // (HANDOVER 20). With the attachment read-only the stencil write mask below is zeroed and every stencil mark the
        // deferred lighting's class mask reads back is lost, which is what rendered those objects black.
        static const bool stencil_under_ro = nxrt::config_get_bool("nvn", "stencil_under_ro", true);
        const bool stencil_writes = stencil_under_ro && S.ds.stencil_test && (S.stencil_wmask[0] || S.stencil_wmask[1]);
        const bool samples_depth = samples_depth_target(*pl);
        g_want_depth_ro = samples_depth && !stencil_writes;
        // sampled and stencil-written in the same pass: the attachment has to be GENERAL, or the sampler's layout and the
        // attachment's disagree (VUID-vkCmdDrawIndexed-imageLayout-00344). The stencil write itself is unaffected.
        g_depth_general = samples_depth && stencil_writes;
        rs_mark(kRsPrepAttach);
        prepare_textures(*pl);
        // A colour attachment that this draw also writes through a storage image (image load/store to a render
        // target) cannot stay in the colour-attachment layout: the descriptor is GENERAL, so the attachment has to be
        // too. storage_bound is what prepare_textures just moved to GENERAL.
        g_color_general = 0;
        for (uint32_t i = 0; i < S.ncolor; ++i)
            for (Texture* st : g_storage_bound)
                if (S.color_tex[i].get() == st) g_color_general |= 1u << i;
        rs_mark(kRsPrepTex);
        // [nvn] fs_debug: a fragment output whose numeric type disagrees with the colour attachment it lands in, or
        // lands in nothing. Both are Undefined-Value warnings in Vulkan; the console stores the register bits
        // unchanged, so the fix is a shader variant - but which shader is not visible from the VUID, only from here.
        static const bool fs_debug = nxrt::config_get_bool("nvn", "fs_debug", false);
        if (fs_debug && S.stage[kFragment]) {
            static std::set<std::string> told;
            for (const ShaderIo& o : S.stage[kFragment]->refl.outputs) {
                const Texture* t = o.location < S.ncolor ? S.color_tex[o.location].get() : nullptr;
                const VkFormat af = t ? tex_vk_format(const_cast<Texture*>(t)) : VK_FORMAT_UNDEFINED;
                const bool int_target = af != VK_FORMAT_UNDEFINED && is_int_format(af);
                const bool int_output = o.type != ScalarType::Float;
                const char* bad = !t ? "no attachment"
                                  : (int_output != int_target) ? (int_target ? "float output into an integer attachment"
                                                                             : "integer output into a float attachment")
                                  : (int_output && (o.type == ScalarType::Sint) != vk_format_is_signed_int(af))
                                        ? "signed output into an unsigned attachment (or the reverse)"
                                        : nullptr;
                if (!bad) continue;
                char key[128];
                snprintf(key, sizeof key, "%s:%u:%d", S.stage[kFragment]->hash.c_str(), o.location, int(af));
                if (told.insert(key).second)
                    NXLOG_I(kTag, "fs_debug: fs %s output location %u is %s, attachment %u is format %d: %s", S.stage[kFragment]->hash.c_str(),
                            o.location, o.type == ScalarType::Float ? "float" : "integer", o.location, int(af), bad);
            }
            // the light list renders into R32G32_UINT; name every shader that does, with what its outputs say
            if (S.ncolor > 0 && S.color_tex[0] && tex_vk_format(S.color_tex[0].get()) == VK_FORMAT_R32G32_UINT) {
                static std::set<std::string> told2;
                if (told2.insert(S.stage[kFragment]->hash).second) {
                    std::string outs;
                    for (const ShaderIo& o : S.stage[kFragment]->refl.outputs)
                        outs += " loc" + std::to_string(o.location) + ":" +
                                (o.type == ScalarType::Float ? "f" : o.type == ScalarType::Sint ? "i" : o.type == ScalarType::Uint ? "u" : "?") +
                                std::to_string(o.components);
                    NXLOG_I(kTag, "fs_debug: fs %s renders into R32G32_UINT (attachments %u); outputs:%s", S.stage[kFragment]->hash.c_str(), S.ncolor,
                            outs.c_str());
                }
            }
        }
        if (g_trace) {   // [nvn] trace_dump_tex: also the block-compressed level 0 as the GPU holds it, to compare with the prepared data
            static const std::string dump_dir = nxrt::config_get_str("nvn", "trace_dump_tex", "");
            static std::unordered_map<uint32_t, bool> raw_done;
            if (!dump_dir.empty()) {
                for (const BindingDesc& d : pl->bindings) {
                    if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
                    const TexResolved r = resolve_texture(d);
                    if (!r.tex || !raw_done.emplace(r.tex->id, true).second) continue;
                    end_pass();
                    const int ri = snapshot_record_raw(r.tex, *F);
                    if (ri >= 0) g_trace_snaps.push_back({ri, g_trace_n, dump_dir + "/gpu_t" + std::to_string(r.tex->id) + ".bin", "raw"});
                }
            }
        }
        scene_upscale_for_draw(*pl);
        rs_mark(kRsPrepUpscale);
    }
    rs_mark(kRsPrep);
    if (S.ncolor && hudless_candidate(S.color_tex[0].get(), (S.color.blend_enable_mask & 1u) != 0)) {   // the scene before the UI (nvn_hudless.h)
        end_pass();
        hudless_capture(*F, S.color_tex[0].get());
    }
    if (g_fb_seen || (g_dirty & (kDesc | kPipe))) feedback_prepare(*pl);
    if (!g_fb_valid.empty()) {
        for (uint32_t i = 0; i < S.ncolor; ++i) {
            Texture* t = S.color_tex[i].get();
            if (t && !g_fb_use.count(t)) g_fb_valid.erase(t);
        }
    }
    rs_mark(kRsFb);
    if (!ensure_pass(g_want_depth_ro, g_prog_writes, g_depth_general, g_color_general, g_color_feedback)) {
        // which draw, and what its targets are (the first of each vertex/fragment shader pair)
        const std::string vs = S.stage[kVertex] ? S.stage[kVertex]->hash : "-", fs = S.stage[kFragment] ? S.stage[kFragment]->hash : "-";
        if (first_time_key("draw-no-target " + vs + " " + fs)) {
            std::string tg;
            for (uint32_t i = 0; i < S.ncolor; ++i) {
                char b[96];
                const Target t = make_target(S.color_tex[i].get(), S.color_view[i], S.color_has_view[i]);
                tg += " rt" + std::to_string(i) + " " + tex_desc(t.tex, b, sizeof b) + " layer " + std::to_string(t.layer) + "+" + std::to_string(t.layers) +
                      (t.tex && !tex_gpu(t.tex) ? " (no GPU image)" : "");
            }
            char b[96];
            NXLOG_W(kTag, "draw without a usable render target skipped: vs %s fs %s, %u colour targets%s, depth %s", vs.c_str(), fs.c_str(), S.ncolor, tg.c_str(),
                    tex_desc(S.depth_tex.get(), b, sizeof b));
        }
        return false;
    }
    if (g_prog_writes || g_xfb_active) g_pass_unsafe = true;
    rs_mark(kRsPass);
    const VkPrimitiveTopology topo = vk_topology(mode);
    if (g_draw_uint_samplers != (g_pipe_key_valid ? g_pipe_key.uint_samplers : 0ull)) g_dirty |= kPipe;
    if (g_dirty & kPipe) ++g_rs.draws_pipe;
    if ((g_dirty & kPipe) || topo != g_pipe_topo || !g_pipe) {
        ++g_rs.keys;
        GraphicsKey key;
        { RsScope rs(kRsKey); build_key(key, topo); }
        { RsScope rs(kRsLookup); g_pipe = graphics_pipeline(key, *pl); }
        g_pipe_key = key;             // [nvn] earlyz: the depth pre-pass runs this key with the colour writes masked off
        g_pipe_key_valid = true;
        g_pipe_xfb = key.xfb != 0;
        g_pipe_topo = topo;
        g_key_attr_count = key.attr_count;
        g_key_dummy_binding = key.dummy_binding;
        g_key_inst_mask = 0;
        for (uint32_t i = 0; i < key.attr_count; ++i) {
            const uint8_t b = key.attr[i].binding;
            g_key_binding[i] = b;
            if (b < 16 && key.binding_instance[b] && !(g_key_dummy_binding && b == static_cast<uint8_t>(g_key_dummy_binding - 1)))
                g_key_inst_mask |= 1u << b;
        }
    }
    if (!g_pipe) return false;
    // a draw inside a stream-out section that has no stream-out variant and nothing to render to has no effect
    if (g_xfb_active && !g_pipe_xfb && !P.ncolor && !S.depth_tex) return false;
    if (g_trace) {
        char a[96], b[96], c[96];
        std::string line;
        auto remember = [](Texture* t) {
            if (!t) return;
            for (const Ref<Texture>& r : g_trace_targets) if (r.get() == t) return;
            g_trace_targets.push_back(Ref<Texture>::share(t));
        };
        for (uint32_t i = 0; i < P.ncolor; ++i) remember(S.color_tex[i].get());
        remember(S.depth_tex.get());
        for (uint32_t i = 0; i < P.ncolor; ++i) { line += tex_desc(S.color_tex[i].get(), a, sizeof a); line += " | "; }
        TRACE("draw #%d vs %s fs %s gs %s mode %d idx %d | rt %s ds %s%s | blend %u mask %x/%x/%x/%x/%x/%x/%x/%x depth t%d w%d f%d cull %d logic %d | ubo0 %llx/%llu",
              ++g_trace_n, S.stage[kVertex] ? S.stage[kVertex]->hash.c_str() : "-", S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-",
              S.stage[kGeometry] ? S.stage[kGeometry]->hash.c_str() : "-", mode, indexed ? 1 : 0, line.c_str(), tex_desc(S.depth_tex.get(), b, sizeof b),
              P.depth_ro ? " (ro)" : "", S.color.blend_enable_mask,
              S.chmask.mask[0], S.chmask.mask[1], S.chmask.mask[2], S.chmask.mask[3], S.chmask.mask[4], S.chmask.mask[5], S.chmask.mask[6], S.chmask.mask[7],
              S.ds.depth_test ? 1 : 0, S.ds.depth_write ? 1 : 0, S.ds.depth_func,
              S.poly.cull_face, S.color.logic_op, static_cast<unsigned long long>(S.ubo[kVertex][0].addr), static_cast<unsigned long long>(S.ubo[kVertex][0].size));
        // [nvn] trace: the depth state the draw is submitted with, next to the guest's request. "eff" is the column to read for
        // "this draw ignores the scene's depth": it already has have_depth, depth_off_fs, depth_write_fs and the read-only depth
        // target folded in, i.e. the values vkCmdSetDepthTestEnable / vkCmdSetDepthWriteEnable got for this draw.
        {
            const EffectiveDepth ed = effective_depth(P);
            TRACE("    depth eff test %d write %d func %d(%s) | guest test %d write %d func %d | pass has depth %d%s%s", ed.test ? 1 : 0, ed.write ? 1 : 0, ed.func,
                  nvn_compare_name(ed.func), S.ds.depth_test ? 1 : 0, S.ds.depth_write ? 1 : 0, S.ds.depth_func, P.depth_fmt != VK_FORMAT_UNDEFINED ? 1 : 0,
                  P.depth_ro ? " (read-only)" : "", fs_list_matches("depth_off_fs") ? " (depth_off_fs)" : "");
            TraceDrawRec rec;
            rec.n = g_trace_n;
            rec.rt = S.color_tex[0] ? static_cast<int>(S.color_tex[0]->id) : -1;
            rec.rt_count = static_cast<int>(P.ncolor);
            rec.fs = S.stage[kFragment] ? S.stage[kFragment]->hash : std::string("-");
            rec.blend = S.color.blend_enable_mask;
            rec.src_rgb = S.blend[0].src_rgb; rec.dst_rgb = S.blend[0].dst_rgb; rec.eq_rgb = S.blend[0].eq_rgb;
            rec.src_a = S.blend[0].src_a; rec.dst_a = S.blend[0].dst_a; rec.eq_a = S.blend[0].eq_a;
            rec.test = ed.test; rec.write = ed.write; rec.func = ed.func;
            g_trace_recs.push_back(std::move(rec));
        }
        TRACE("    poly mode %d offset %x factor %g units %g clamp %g | window depth mode %d origin %d clamp %d", S.poly.polygon_mode, S.poly.offset_enables, S.bias_factor, S.bias_units,
              S.bias_clamp, S.depth_mode, S.window_origin, S.depth_clamp ? 1 : 0);
        // The viewport and scissor in force (NVN coordinates, before the origin flip): a pass that renders into a corner of its
        // target is a viewport that was never set, or one left over from the previous pass (the shadow cascades, HANDOVER 28.9).
        {
            char vps[96] = "none (the whole target)", scs[96] = "none (the whole target)";
            if (S.nvp) std::snprintf(vps, sizeof vps, "%.1f,%.1f %.1fx%.1f (of %u)", static_cast<double>(S.vp[0][0]), static_cast<double>(S.vp[0][1]),
                                     static_cast<double>(S.vp[0][2]), static_cast<double>(S.vp[0][3]), S.nvp);
            if (S.nsc) std::snprintf(scs, sizeof scs, "%d,%d %dx%d (of %u)", S.sc[0][0], S.sc[0][1], S.sc[0][2], S.sc[0][3], S.nsc);
            TRACE("    viewport %s | scissor %s", vps, scs);
        }
        // The state a draw disappears in without a warning: a stencil test that fails everywhere, an alpha test, a sample mask, primitive
        // restart, a front-face flip. Not covered by the draw line above, so a hole in the frame can be read straight off the trace.
        TRACE("    state stencil %d func %d/%d ref %d/%d vmask %x/%x wmask %x/%x ops f %d/%d/%d b %d/%d/%d | alpha ref %g | sample mask %x | prim restart %d idx %x | front face %d attrs %u | raster discard %d", S.ds.stencil_test ? 1 : 0,
              S.ds.stencil_func[0], S.ds.stencil_func[1], S.stencil_ref[0], S.stencil_ref[1], S.stencil_vmask[0], S.stencil_vmask[1], S.stencil_wmask[0], S.stencil_wmask[1],
              S.ds.stencil_fail[0], S.ds.depth_fail[0], S.ds.depth_pass[0], S.ds.stencil_fail[1], S.ds.depth_fail[1], S.ds.depth_pass[1], S.alpha_ref, S.sample_mask,
              S.prim_restart ? 1 : 0, S.restart_index, S.poly.front_face, S.attr_count, S.rasterizer_discard ? 1 : 0);
        // [nvn] trace: the blend the draw is submitted with, by name (the numbers alone read as "5,6,1" = SRC_ALPHA,
        // ONE_MINUS_SRC_ALPHA, ADD, i.e. an ordinary translucent draw). Every target that blends gets its own line: a card that
        // only blends into rt1 does not show up in rt0's state.
        if (S.color.blend_enable_mask) {
            for (uint32_t i = 0; i < 8 && i < P.ncolor; ++i) {
                if (!(S.color.blend_enable_mask & (1u << i))) continue;
                TRACE("    blend rt%u rgb %d,%d,%d = %s, %s, %s | alpha %d,%d,%d = %s, %s, %s", i, S.blend[i].src_rgb, S.blend[i].dst_rgb, S.blend[i].eq_rgb,
                      nvn_blend_func_name(S.blend[i].src_rgb), nvn_blend_func_name(S.blend[i].dst_rgb), nvn_blend_eq_name(S.blend[i].eq_rgb), S.blend[i].src_a,
                      S.blend[i].dst_a, S.blend[i].eq_a, nvn_blend_func_name(S.blend[i].src_a), nvn_blend_func_name(S.blend[i].dst_a), nvn_blend_eq_name(S.blend[i].eq_a));
            }
        }
        g_trace_sampled.clear();
        if (trace_verbose()) {   // [nvn] trace_compact: the detail a 3,300-draw frame cannot afford - uniform blocks, storage buffers,
                                 // vertex streams, render-target memory and every sampled texture, per draw
        // [nvn] trace_ubo_bytes = N (NXRT_NVN_TRACE_UBO_BYTES, default 768, at most 65536): how many leading bytes of each uniform block the trace prints
        static const uint64_t ubo_bytes = static_cast<uint64_t>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "trace_ubo_bytes", 768), 16, 65536));
        for (int stg = kVertex; stg <= kGeometry; ++stg)
        for (uint32_t slot = 0; slot < 16; ++slot) {    // the first words of every uniform block bound to the vertex, fragment and geometry stages
            const GState::Bind& ub = S.ubo[stg][slot];
            if (!ub.addr) continue;
            BufRef ur = resolve_buffer(ub.addr, std::min<uint64_t>(ub.size, ubo_bytes));
            std::string words;
            if (ur.ok) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(ub.addr);
                for (uint32_t i = 0; i * 4 < ubo_bytes && i * 4 < ub.size; ++i) { char t[16]; std::snprintf(t, sizeof t, " %08x", w[i]); words += t; }
            }
            const char* sn = stg == kVertex ? "vs" : stg == kFragment ? "fs" : "gs";
            TRACE("    %s ubo%u %llx/%llu%s:%s", sn, slot, static_cast<unsigned long long>(ub.addr), static_cast<unsigned long long>(ub.size), ur.ok ? "" : " (outside every pool)", words.c_str());
            // [nvn] trace_ubo_dir = <dir>: the whole uniform block as a file (the log line above stops at ~2 KB, and the probe
            // constants fp_c6 sit at byte 1536 of an 8704-byte block). One file per (stage, slot, draw shader); the last draw wins.
            static const std::string ubo_dir = nxrt::config_get_str("nvn", "trace_ubo_dir", "");
            if (!ubo_dir.empty() && ur.ok && ub.size) {
                std::error_code ec;
                std::filesystem::create_directories(ubo_dir, ec);
                const std::string nm = ubo_dir + "/ubo_" + (stg == kVertex ? "vs" : "fs") + std::to_string(slot) + "_" + (S.stage[stg] ? S.stage[stg]->hash : std::string("x")) + "_" +
                                       std::to_string(static_cast<unsigned long long>(ub.size)) + ".bin";
                if (FILE* f = std::fopen(nm.c_str(), "wb")) {
                    std::fwrite(reinterpret_cast<const void*>(ub.addr), 1, static_cast<size_t>(std::min<uint64_t>(ub.size, 1u << 16)), f);
                    std::fclose(f);
                }
            }
        }
        // The storage buffers of the vertex and fragment stages, same shape as the uniform blocks above. XC2's skinned vertex
        // programs read their bone matrices from storage buffers (vp_s0/vp_s1 in the translation), so a draw whose skinning
        // collapses while its neighbours are fine shows up here as a missing or empty binding (HANDOVER 21.9).
        for (int stg = kVertex; stg <= kFragment; ++stg)
        for (uint32_t slot = 0; slot < 16; ++slot) {
            const GState::Bind& sb = S.ssbo[stg][slot];
            if (!sb.addr) continue;
            // [nvn] trace_ssbo_bytes = N (NXRT_NVN_TRACE_SSBO_BYTES, default 32, at most 4096): how many leading bytes of each
            // storage buffer the trace prints. The skinning matrices sit at bone*48, so 640 covers the first 13 bones and lets the
            // skinning of a draw be recomputed on the CPU (HANDOVER 22).
            static const uint32_t ssbo_bytes = [] {
                const int v = nxrt::config_get_int("nvn", "trace_ssbo_bytes", 32);
                return static_cast<uint32_t>(std::clamp(v, 0, 4096));
            }();
            BufRef sr = resolve_buffer(sb.addr, std::min<uint64_t>(sb.size, ssbo_bytes));
            std::string words;
            if (sr.ok) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(sb.addr);
                for (uint32_t i = 0; i * 4 < ssbo_bytes && i * 4 < sb.size; ++i) { char t[16]; std::snprintf(t, sizeof t, " %08x", w[i]); words += t; }
            }
            TRACE("    %s ssbo%u %llx/%llu%s:%s", stg == kVertex ? "vs" : "fs", slot, static_cast<unsigned long long>(sb.addr),
                  static_cast<unsigned long long>(sb.size), sr.ok ? "" : " (outside every pool)", words.c_str());
        }
        // The vertex streams' guest addresses and strides: a model's vertex buffer sits in the block the game loaded it into, so
        // this maps a draw to a model (XC2_MODEL_PROBE prints each ml::Scn::createMdl block address next to its asset path, HANDOVER 19.9).
        {
            std::string vb;
            for (uint32_t i = 0; i < S.attr_count && i < 16; ++i) {
                const VertexAttribInfo& at = S.attr[i];
                const VertexStreamInfo& st = S.stream[at.stream];
                const GState::Bind& b = S.vtx[at.stream];
                char t[72];
                std::snprintf(t, sizeof t, " s%u %llx+%td/%td bound %llu", at.stream, static_cast<unsigned long long>(b.addr), at.relative_offset, st.stride,
                              static_cast<unsigned long long>(b.size));
                vb += t;
            }
            TRACE("    streams%s", vb.c_str());
        }
        TRACE("    viewport %u: %g %g %g %g depth %g..%g scissor %u: %d %d %d %d", S.nvp, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3], S.dr[0][0], S.dr[0][1], S.nsc,
              S.sc[0][0], S.sc[0][1], S.sc[0][2], S.sc[0][3]);
        for (uint32_t i = 0; i < P.ncolor; ++i)
            if (Texture* ct = S.color_tex[i].get()) {
                const FormatInfo* vf = (S.color_has_view[i] && S.color_view[i].has_format) ? format_info(S.color_view[i].format) : nullptr;
                TRACE("    rt%u t%u mem %p storage %llu cpu-dirty %d view level %u layer %u layers %u format %s", i, ct->id, static_cast<void*>(ct->mem),
                      static_cast<unsigned long long>(ct->storage_size), ct->cpu_dirty.load() ? 1 : 0, S.color_level[i], S.color_layer[i], S.color_layers[i],
                      vf && vf->name ? vf->name : "-");
            }
        for (const BindingDesc& d : pl->bindings) {   // texel buffers (light lists and other BUFFER-target textures): the view the draw gets
            if (d.type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER && d.type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) continue;
            const bool storage = d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            const TexResolved r = storage ? resolve_image(d) : resolve_texture(d);
            if (!r.tex) { TRACE("    texel set%u b%u -> (no texture)", d.set, d.binding); continue; }
            uint64_t addr = 0, bytes = 0;
            const VkBufferView v = tex_buffer_view(r.tex, r.has_view ? &r.view : nullptr, 0, VK_FORMAT_UNDEFINED, storage, &addr, &bytes);
            std::string words;
            if (r.tex->mem) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(r.tex->mem);
                const size_t nw = std::min<size_t>(32, static_cast<size_t>(r.tex->b.width) * (r.tex->fmt ? r.tex->fmt->bytes : 4) / 4);
                char wb[12];
                for (size_t i = 0; i < nw; ++i) { std::snprintf(wb, sizeof wb, " %08x", w[i]); words += wb; }
                const size_t all = static_cast<size_t>(r.tex->b.width) * (r.tex->fmt ? r.tex->fmt->bytes : 4) / 4;
                size_t nz = 0;
                for (size_t i = 0; i < all; ++i) nz += w[i] != 0;
                std::snprintf(wb, sizeof wb, " |nz %zu", nz);
                words = std::string(wb) + "/" + std::to_string(all) + words;
            }
            TRACE("    texel set%u b%u -> t%u %s width %d @%p view %s addr %llx bytes %llu view-format %s:%s", d.set, d.binding, r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width,
                  static_cast<void*>(r.tex->mem), v ? "ok" : "NONE (dummy)", static_cast<unsigned long long>(addr), static_cast<unsigned long long>(bytes),
                  (r.has_view && r.view.has_format) ? "set" : "-", words.c_str());
        }
        for (const BindingDesc& d : pl->bindings) {
            if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && d.type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) continue;
            const TexResolved r = d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? resolve_image(d) : resolve_texture(d);
            if (r.tex) g_trace_sampled.push_back(r.tex);
            if (r.tex) {
                static const std::string dump_dir = nxrt::config_get_str("nvn", "trace_dump_tex", "");
                if (!dump_dir.empty()) trace_dump_prepared(r.tex, dump_dir);
            }
            char smp[200] = "", vw[160] = "";
            if (r.has_sampler && r.smp_ptr)
                std::snprintf(smp, sizeof smp, " | sampler min %d mag %d wrap %d,%d,%d lod %g..%g bias %g aniso %g cmp %d func %d", r.smp_ptr->min_filter, r.smp_ptr->mag_filter, r.smp_ptr->wrap_s, r.smp_ptr->wrap_t,
                              r.smp_ptr->wrap_r, r.smp_ptr->lod_min, r.smp_ptr->lod_max, r.smp_ptr->lod_bias, r.smp_ptr->max_anisotropy, r.smp_ptr->compare_mode, r.smp_ptr->compare_func);
            if (r.has_view) {
                // the view's own format by name (a view naming the sRGB partner of its texture's format, or the reverse, changes the decode)
                const FormatInfo* vf = r.view.has_format ? format_info(r.view.format) : nullptr;
                char vfn[48] = "";
                if (r.view.has_format) std::snprintf(vfn, sizeof vfn, " format %s", vf && vf->name ? vf->name : "?");
                std::snprintf(vw, sizeof vw, " | view levels %d+%d layers %d+%d%s", r.view.has_levels ? r.view.min_level : -1, r.view.has_levels ? r.view.num_levels : -1,
                              r.view.has_layers ? r.view.min_layer : -1, r.view.has_layers ? r.view.num_layers : -1, vfn);
            }
            if (r.tex) {
                const int* sw = (r.has_view && r.view.has_swizzle) ? r.view.swizzle : r.tex->b.swizzle;
                const size_t n = std::strlen(vw);
                std::snprintf(vw + n, sizeof vw - n, " swz %d%d%d%d", sw[0], sw[1], sw[2], sw[3]);
            }
            TRACE("    %s set%u b%u %s -> %s%s mem %d%%%s%s%s", d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? "image" : "tex", d.set, d.binding,
                  d.shader->refl.bindings[d.index].name.c_str(), tex_desc(r.tex, c, sizeof c), r.has_sampler ? "" : " (no sampler)", storage_nonzero_pct(r.tex),
                  r.tex && r.tex->cpu_dirty.load() ? " cpu-dirty" : "", smp, vw);
        }
        }   // trace_verbose()
    }
    if (g_xfb_active) {
        static std::unordered_map<std::string, bool> seen;
        if (seen.emplace(S.stage[kVertex]->hash, true).second)
            NXLOG_I("nvn.xfb", "draw under transform feedback: vs %s mode %d attrs %u xfb0 %llx/%llu", S.stage[kVertex]->hash.c_str(), mode, S.attr_count,
                    static_cast<unsigned long long>(S.xfb[0].addr), static_cast<unsigned long long>(S.xfb[0].size));
    }
    if (g_pipe != g_bound_pipeline) {
        RsScope rs(kRsBind);
        xfb_end_in_pass();     // the pipeline cannot change while feedback is active; the byte counter carries the position over
        if (!g_bound_pipeline) { g_pushed0.cb = VK_NULL_HANDLE; invalidate_bound_state(); }   // something else bound graphics state since: what set 0 holds is not known
        vkCmdBindPipeline(F->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe);
        ++g_sub_pipe_binds;
        g_bound_pipeline = g_pipe;
    }
    if (g_xfb_active && g_pipe_xfb && !g_xfb_began && device().features().transform_feedback && S.xfb[0].addr) {
        BufRef xb = resolve_buffer(S.xfb[0].addr, S.xfb[0].size);
        if (xb.ok) {
            gpu_wrote(S.xfb[0].addr, S.xfb[0].size ? S.xfb[0].size : xb.avail, kWriteXfb);
            if (!g_xfb_counter) {
                if (device().create_buffer(kXfbSlots * kXfbSlotBytes, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Upload,
                                           &g_xfb_counter, "nvn xfb counter") != VK_SUCCESS)
                    g_xfb_counter = gfx::Buffer();
            }
            const VkDeviceSize size = std::min<VkDeviceSize>(S.xfb[0].size ? S.xfb[0].size : xb.avail, xb.avail) & ~static_cast<VkDeviceSize>(3);
            vkCmdBindTransformFeedbackBuffersEXT(F->cb, 0, 1, &xb.buffer, &xb.offset, &size);
            VkBuffer cbuf = g_xfb_counter.buffer;
            const VkDeviceSize coff = static_cast<VkDeviceSize>(g_xfb_cur) * kXfbSlotBytes;
            if (g_xfb_have_counter && cbuf) vkCmdBeginTransformFeedbackEXT(F->cb, 0, 1, &cbuf, &coff);
            else vkCmdBeginTransformFeedbackEXT(F->cb, 0, 0, nullptr, nullptr);
            g_xfb_began = true;
        } else {
            note_unsupported("transform feedback into an address outside every pool");
        }
    }
    rs_mark(kRsPipe);
    set_dynamic_state(g_dirty, topo, indexed, index_type);
    rs_mark(kRsDyn);
    if (ubo_fast) {
        const uint16_t used_dirty = (g_ubo_dirty_mask[0] & pl->ubo_mask[0]) |
                                    (g_ubo_dirty_mask[1] & pl->ubo_mask[1]) |
                                    (g_ubo_dirty_mask[2] & pl->ubo_mask[2]) |
                                    (g_ubo_dirty_mask[3] & pl->ubo_mask[3]) |
                                    (g_ubo_dirty_mask[4] & pl->ubo_mask[4]) |
                                    (g_ubo_dirty_mask[5] & pl->ubo_mask[5]);
        if (used_dirty != 0 || g_pushed0.cb != F->cb || g_pushed0.layout != pl->layout) {
            ++g_rs.draws_ubo;
            const VkCommandBuffer cb0 = F->cb;
            PushSet0 push0;
            build_set(*pl, 0, &push0);            // leaves set 0's writes in push0 (push_set0)
            if (F->cb == cb0) {
                if (push0.ready && push0.count) push_set0(VK_PIPELINE_BIND_POINT_GRAPHICS, *pl, push0.writes, push0.count);
                for (int st = 0; st < 6; ++st) g_ubo_dirty_mask[st] &= ~pl->ubo_mask[st];
            } else {                              // building it started another command buffer: it has none of the sets, bind them all
                g_dirty |= kDesc;
            }
        } else {
            ++g_lean_push_none;
        }
    }
    if (g_dirty & kDesc) {
        support_for_draw(*pl, false);
        bind_sets(*pl, VK_PIPELINE_BIND_POINT_GRAPHICS);
        g_sets_pl = pl;
        g_sets_cb = F->cb;
        for (uint16_t& m : g_ubo_dirty_mask) m = 0;
    }
    rs_mark(kRsSets);
    if (rs_on()) {   // census of draws the scissor leaves no pixels for: scissor_rect already clamps to the pass extent
        const VkRect2D sc = S.nsc ? scissor_rect(S.sc[0]) : VkRect2D{{0, 0}, P.extent};
        if (sc.extent.width == 0 || sc.extent.height == 0) ++g_rs.culled;
    }
    if (pl->push_bytes && g_gen_tcs) {
        // the pass-through control shader's tessellation levels (NVN's fixed SetInner/OuterTessellationLevels), every draw: cheap, and
        // the layout may have changed since the last push
        tessgen::TessPush push = {};
        std::memcpy(push.outer, S.tess_outer, sizeof push.outer);
        std::memcpy(push.inner, S.tess_inner, sizeof push.inner);
        vkCmdPushConstants(F->cb, pl->layout, pl->push_stages, 0, std::min<uint32_t>(pl->push_bytes, sizeof push), &push);
        static std::set<const ShaderInfo*> logged;
        if (logged.insert(g_gen_tcs).second)
            NXLOG_I(kTag, "tessellation draw: evaluation shader %s drawn with pass-through control shader %s, patch size %u, levels %g %g %g %g / %g %g",
                    S.stage[kTessEval]->hash.c_str(), g_gen_tcs->hash.c_str(), static_cast<unsigned>(std::clamp<uint32_t>(S.patch_size, 1, 32)),
                    S.tess_outer[0], S.tess_outer[1], S.tess_outer[2], S.tess_outer[3], S.tess_inner[0], S.tess_inner[1]);
    }
    if (g_dirty & (kVb | kPipe)) {
        // each binding's buffer, then consecutive changed bindings in one call
        if (g_bound_vb_cb != F->cb) {
            g_bound_vb_cb = F->cb;
            for (BoundVbSlot& v : g_bound_vb) v.valid = false;
            g_vb_safe_mask = 0;
            g_vb_dirty_mask = 0xFFFF;
        }
        VkBuffer vbuf[16];
        VkDeviceSize voff[16], vsize[16], vstride[16];
        bool vset[16] = {};
        bool vchg[16] = {};
        const uint16_t vb_dirty_mask = (g_dirty & kVb) ? ((g_dirty == kAllDirty) ? 0xFFFFu : g_vb_dirty_mask) : 0u;
        for (uint32_t i = 0; i < g_key_attr_count; ++i) {
            const uint32_t binding = g_key_binding[i];
            if (binding >= 16 || vset[binding]) continue;
            vset[binding] = true;
            BoundVbSlot& cur = g_bound_vb[binding];
            if (g_key_dummy_binding && binding == static_cast<uint32_t>(g_key_dummy_binding - 1)) {
                vbuf[binding] = dummy_buffer();
                voff[binding] = kDummyAttrFloatOffset;
                vsize[binding] = 32;
                vstride[binding] = 0;
                g_vb_safe_mask |= 1u << binding;
                if (!cur.valid || cur.buf != vbuf[binding] || cur.off != voff[binding] || cur.size != vsize[binding] || cur.stride != vstride[binding]) {
                    cur = {vbuf[binding], voff[binding], vsize[binding], vstride[binding], 0, 0, true};
                    vchg[binding] = true;
                }
            } else {
                const GState::Bind& vb = S.vtx[binding];
                const VkDeviceSize want_stride = static_cast<VkDeviceSize>(S.stream[binding].stride);
                if (!((vb_dirty_mask >> binding) & 1u) && cur.valid && cur.nvn_addr == vb.addr && cur.nvn_size == vb.size && cur.stride == want_stride)
                    continue;
                BufRef br = vb.addr ? resolve_read(vb.addr, vb.size, S.stream[binding].divisor > 0 ? kReadVertexInst : kReadVertex) : BufRef{dummy_buffer(), 0, 65536, false, true};
                vbuf[binding] = br.buffer;
                voff[binding] = br.offset;
                vsize[binding] = vb.size ? std::min<VkDeviceSize>(vb.size, br.avail) : br.avail;
                vstride[binding] = want_stride;
                if (br.snapped || !vb.addr) g_vb_safe_mask |= 1u << binding;
                else g_vb_safe_mask &= ~(1u << binding);
                if (!cur.valid || cur.buf != vbuf[binding] || cur.off != voff[binding] || cur.size != vsize[binding] || cur.stride != vstride[binding]) {
                    cur = {vbuf[binding], voff[binding], vsize[binding], vstride[binding], vb.addr, vb.size, true};
                    vchg[binding] = true;
                } else {
                    cur.nvn_addr = vb.addr;
                    cur.nvn_size = vb.size;
                }
            }
        }
        for (uint32_t b = 0; b < 16;) {
            if (!vchg[b]) { ++b; continue; }
            uint32_t e = b + 1;
            while (e < 16 && vchg[e]) ++e;
            vkCmdBindVertexBuffers2(F->cb, b, e - b, vbuf + b, voff + b, vsize + b, vstride + b);
            g_vb_bound_in_cb = true;
            b = e;
        }
        if (!g_vb_bound_in_cb) {
            // the pipeline's vertex binding strides are dynamic: a command buffer that never bound a vertex buffer must still bind one
            VkBuffer dummy = dummy_buffer();
            VkDeviceSize off = 0, size = 65536, stride = 0;
            vkCmdBindVertexBuffers2(F->cb, 0, 1, &dummy, &off, &size, &stride);
            g_bound_vb[0] = {dummy, off, size, stride, 0, 0, true};
            g_vb_bound_in_cb = true;
        }
    }
    g_dirty = g_mirror_stale ? (kVb | kDesc) : 0;     // a storage buffer of this draw made copied pages stale: the next draw binds again
    g_vb_dirty_mask = g_mirror_stale ? 0xFFFF : 0;
    g_mirror_stale = false;
    *topo_out = topo;
    // [nvn] earlyz (HANDOVER 42.22.2): would this draw benefit from the two-draw emulation? Nothing is bound here - the caller issues the
    // pre-pass geometry between earlyz_prepass_begin and earlyz_prepass_end (do_draw_arrays / do_draw_elements only).
    earlyz_plan(*pl);
    earlyz_mask_plan(*pl);   // [nvn] earlyz_mask: is this a draw that registers into the pass's mask?
    rs_mark(kRsVb);
    ++g_rs.draws;
    ++g_split_since;          // [nvn] split_draws
    return true;
}


// [nvn] trace_snap = "253,300": after those draws of the traced frame the pass is closed and the colour targets and the depth target are
// copied (png after the frame: snap_d<draw>_t<id>_<format>.png).
void trace_snap_after_draw() {
    if (!g_trace) return;
    if (!g_trace_snap_read) {
        g_trace_snap_read = true;
        g_trace_snap_sampled = nxrt::config_get_int("nvn", "trace_snap_sampled", 0) != 0;
        // [nvn] trace_snap_rt = 50: every draw of the traced frame that renders into that texture is snapshotted,
        // which gives the whole history of one render target inside one frame (the light-probe list is rebuilt per frame).
        g_trace_snap_rt = nxrt::config_get_int("nvn", "trace_snap_rt", 0);
    g_trace_snap_rts.clear();
    {
        const std::string s = nxrt::config_get_str("nvn", "trace_snap_rt", "");
        size_t i = 0;
        while (i < s.size()) {
            const size_t j = s.find(',', i);
            const size_t e = j == std::string::npos ? s.size() : j;
            const int v = std::atoi(s.substr(i, e - i).c_str());
            if (v > 0) g_trace_snap_rts.push_back(v);
            i = e + 1;
        }
    }
        g_trace_snap_fmt = nxrt::config_get_str("nvn", "trace_snap_fmt", "");
        {   // [nvn] trace_snap_fs = "f77f9d,e72811": every draw whose fragment shader hash starts with one of these
            const std::string fl = nxrt::config_get_str("nvn", "trace_snap_fs", "");
            size_t k = 0;
            while (k < fl.size()) {
                size_t j = fl.find(',', k);
                if (j == std::string::npos) j = fl.size();
                if (j > k) g_trace_snap_fs.push_back(fl.substr(k, j - k));
                k = j + 1;
            }
        }
        const std::string list = nxrt::config_get_str("nvn", "trace_snap", "");
        size_t i = 0;
        while (i < list.size()) {
            size_t j = list.find(',', i);
            if (j == std::string::npos) j = list.size();
            if (j > i) g_trace_snap_draws.push_back(std::atoi(list.substr(i, j - i).c_str()));
            i = j + 1;
        }
    }
    bool hit = false;
    for (int d : g_trace_snap_draws) hit |= d == g_trace_n;
    if (!hit && !g_trace_snap_rts.empty())
        for (uint32_t i = 0; i < S.ncolor && !hit; ++i)
            if (const Texture* ct = S.color_tex[i].get())
                hit = std::find(g_trace_snap_rts.begin(), g_trace_snap_rts.end(), static_cast<int>(ct->id)) != g_trace_snap_rts.end();
    if (!hit && g_trace_snap_rt)
        for (uint32_t i = 0; i < S.ncolor && !hit; ++i)
            if (const Texture* ct = S.color_tex[i].get()) hit = static_cast<int>(ct->id) == g_trace_snap_rt;
    // [nvn] trace_snap_fmt: the same by format name, because the texture ids differ between runs (a window size change renumbers them all)
    if (!hit && !g_trace_snap_fmt.empty())
        for (uint32_t i = 0; i < S.ncolor && !hit; ++i)
            if (const Texture* ct = S.color_tex[i].get())
                hit = ct->fmt && ct->fmt->name && g_trace_snap_fmt == ct->fmt->name;
    if (!hit && S.stage[kFragment] && !g_trace_snap_fs.empty()) {
        for (const std::string& pre : g_trace_snap_fs) hit |= S.stage[kFragment]->hash.compare(0, pre.size(), pre) == 0;
    }
    if (!hit) return;
    end_pass();
    auto take = [&](Texture* t) {
        if (!t) return;
        const int idx = snapshot_record(t, *F);
        if (idx < 0) return;
        char buf[96];
        static const bool span_env = std::getenv("NXRT_TRACE_SPAN") && std::atoi(std::getenv("NXRT_TRACE_SPAN")) > 1;
        const std::string fpre = (span_env || g_trace_span.load(std::memory_order_relaxed) > 0)
                                     ? ("f" + std::to_string(counters().presents.load()) + "_")
                                     : std::string();
        g_trace_snaps.push_back({idx, g_trace_n, std::string("work/trace/snap_") + fpre + "d" + std::to_string(g_trace_n) + "_t" + std::to_string(t->id) + "_" + (t->fmt ? t->fmt->name : "x") + ".png", tex_desc(t, buf, sizeof buf)});
    };
    for (uint32_t i = 0; i < S.ncolor; ++i) take(S.color_tex[i].get());
    take(S.depth_tex.get());
    if (g_trace_snap_sampled) {
        for (Texture* t : g_trace_sampled) take(t);
    }
    {   // [nvn] trace_snap_tex = "51,52": copy these textures by id at the same draw. A storage image another pass writes (the
        // probe mask t51) is not sampled by the draw that consumes it, so it needs naming (HANDOVER 35).
        static const std::vector<int> ids = [] {
            std::vector<int> v;
            const std::string s = nxrt::config_get_str("nvn", "trace_snap_tex", "");
            size_t i = 0;
            while (i < s.size()) {
                size_t j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
                i = j + 1;
            }
            return v;
        }();
        for (int id : ids) {
            if (id <= 0) continue;
            uint32_t want = static_cast<uint32_t>(id);
            auto pred = [](const Texture& t, void* u) { return t.id == *static_cast<const uint32_t*>(u) && !t.destroyed.load(); };
            std::vector<Ref<Texture>> v = texture_registry_collect(pred, &want);
            if (!v.empty()) take(v.front().get());
        }
    }
}

// Counts a draw for the primitive counters, before the replay decides whether to skip it: the console generates the primitives of every
// draw it runs, and a draw of a feedback section also writes them (14). A draw with rendering disabled is not run by the console: not counted.
void count_primitives(uint32_t mode, uint32_t count, uint32_t instances) {
    if (!S.render_enable) return;
    const uint64_t per = mode == NVN_DRAW_PRIMITIVE_PATCHES ? count / std::max<uint32_t>(1, S.patch_size) : primitives_for(static_cast<int>(mode), count);   // patches: input patches
    const uint64_t prims = per * std::max<uint32_t>(1, instances);
    g_prims_generated += prims;
    if (g_xfb_active) g_prims_xfb += prims;
}

// Quads, quad strips, polygons and line loops, which Vulkan has no topology for, as the triangle (or line strip) index list of the same vertices.
// `in` holds the vertex indices of the draw in order (for DrawArrays: first, first + 1, ...). Shared by the direct and the indirect draws.
// [nvn] quad_split = n (NXRT_NVN_QUAD_SPLIT, 2..64, default 0 = off): a QUADS draw of 4 to 16 quads whose vertex shader reads
// only a float3 position at location 0 is drawn as an n x n grid of triangles per quad, the grid's positions blended from the
// quad's corners. Port of xc2-recomp's XC2_QUAD_SPLIT (runtime/vk_exec_draw.c): XC2's cloud sea composite draws the whole sea as
// six quads tens of thousands of units wide, and the attributes interpolated across those triangles move with the game's sub-pixel
// camera jitter, so the sea's distance and colour shimmer (AMD/Mesa; HANDOVER 90). 0 only counts the qualifying draws.
int quad_split_n() {
    static const int n = [] {
        const int v = static_cast<int>(nxrt::config_get_int("nvn", "quad_split", 0));
        const int r = v >= 2 ? std::min(v, 64) : 0;
        NXLOG_I(kTag, "quad_split: %s (nvn quad_split = %d)", r ? "on" : "off (0)", v);
        return r;
    }();
    return n;
}

// Does this QUADS draw qualify? Counted once per program so the log names the shaders the split would touch.
bool quad_split_candidate(uint32_t mode, uint32_t count, uint32_t instances) {
    if (mode != NVN_DRAW_PRIMITIVE_QUADS) return false;
    const ShaderInfo* vs = S.stage[kVertex];
    if (!vs) return false;
    const uint32_t quads = count / 4;
    // One line per vertex shader: whether its QUADS draws qualify for the subdivision, and why not when they do not.
    const char* why = nullptr;
    const std::vector<ShaderIo>& vin = vs->refl.vertex_inputs();
    if (instances != 1) why = "more than one instance";
    else if (quads < 1 || quads > 16) why = "not 1 to 16 quads";
    else if (vin.size() != 1 || vin[0].location != 0) why = "the vertex shader reads more than a position at location 0";
    else if (!vs->refl.builtins_in.empty()) why = "the vertex shader reads a built-in";
    else if (quads < 4 && !S.ds.depth_test) why = "1-3 quads without depth test";
    if (first_time_key("quad_split " + vs->hash)) {
        if (why) NXLOG_I(kTag, "quad_split: vs %s draws %u quads (of %u): not subdivided, %s", vs->hash.c_str(), quads, count, why);
        else NXLOG_I(kTag, "quad_split: vs %s draws %u quads (of %u): qualifies%s", vs->hash.c_str(), quads, count,
                     quad_split_n() ? ", subdivided" : " ([nvn] quad_split = 2..64 turns it on)");
    }
    return !why && quad_split_n() != 0;
}

// The subdivision itself (port of xc2-recomp's quad_split, runtime/vk_exec_draw.c, azlekayn's patch in issue #39): a QUADS draw
// of 1 to 16 quads whose vertex shader reads only a float3 position at location 0 becomes an n x n grid of triangles per quad, the
// grid's positions blended from the quad's corners (exact for a flat quad). XC2's cloud sea draws both its 5-6 quad box and its
// 1-quad 40,000-unit horizontal raymarch/depth sheets (vs 818474df / vs 34fe41df) as huge QUADS spanning w=0; without subdivision,
// AMD's triangle interpolator warps screen UV by +/-4..10 px across the sheet and jumps wildly on 1-ULP camera changes, creating
// dark silhouette ghosts next to railings/objects. For 1..3 quads, we also require varying corner Z so 2D screen-space quads
// (constant z=0.5) are left untouched. corner holds the 4 * quads vertex numbers, base vertex already added. False: draw as usual.
struct QuadSplitGrid {
    std::vector<uint8_t> verts;
    std::vector<uint32_t> idx;
};
std::unordered_map<uint64_t, QuadSplitGrid> g_qs_grids;

bool quad_split_draw(uint32_t quads, const std::vector<uint32_t>& corner, uint32_t base_instance) {
    const int split = quad_split_n();
    if (!split || quads < 1 || quads > 16 || corner.size() < size_t(quads) * 4) return false;
    if (!S.attr_count || !S.stage[kVertex]) return false;
    const VertexAttribInfo& at = S.attr[0];
    if (at.format != 0x22 || at.stream < 0 || at.stream >= 16) return false;   // 0x22 = R32G32B32_SFLOAT: a bare position
    const VertexStreamInfo& si = S.stream[at.stream];
    const GState::Bind& vb = S.vtx[at.stream];
    if (si.divisor || si.stride < at.relative_offset + 12) return false;
    const uint8_t* base = vb.addr ? static_cast<const uint8_t*>(host_ptr(vb.addr)) : nullptr;
    if (!base) return false;
    const size_t stride = static_cast<size_t>(si.stride);
    // For 1..3 quads, skip flat screen-space quads (constant Z across all 4 corners, e.g. z=0.5 NDC quads).
    if (quads < 4) {
        bool has_3d_z = false;
        for (uint32_t k = 0; k < quads && !has_3d_z; ++k) {
            float c0[3], cj[3];
            std::memcpy(c0, base + uint64_t(corner[4 * k]) * stride + at.relative_offset, 12);
            for (int j = 1; j < 4; ++j) {
                std::memcpy(cj, base + uint64_t(corner[4 * k + uint32_t(j)]) * stride + at.relative_offset, 12);
                if (std::fabs(cj[2] - c0[2]) > 1e-4f) { has_3d_z = true; break; }
            }
        }
        if (!has_3d_z) return false;
    }
    const uint32_t g = uint32_t(split) + 1, nv = quads * g * g, ni = quads * uint32_t(split) * uint32_t(split) * 6;
    // The key covers everything the grid depends on: the grid size, the stream layout and the corner positions themselves.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* d, size_t n) {
        const uint8_t* q = static_cast<const uint8_t*>(d);
        for (size_t i = 0; i < n; ++i) { h ^= q[i]; h *= 1099511628211ull; }
    };
    mix(&quads, sizeof quads);
    mix(&split, sizeof split);
    mix(&stride, sizeof stride);
    const int32_t roff = static_cast<int32_t>(at.relative_offset);
    mix(&roff, sizeof roff);
    for (uint32_t k = 0; k < quads; ++k)
        for (int j = 0; j < 4; ++j) mix(base + uint64_t(corner[4 * k + uint32_t(j)]) * stride + at.relative_offset, 12);
    auto found = g_qs_grids.find(h);
    if (found == g_qs_grids.end()) {
        if (g_qs_grids.size() >= 32) g_qs_grids.clear();   // a handful of cloud-sea draws; never grow without bound
        QuadSplitGrid gr;
        gr.verts.assign(static_cast<size_t>(nv) * stride, 0);
        for (uint32_t k = 0; k < quads; ++k) {
            float c[4][3];
            for (int j = 0; j < 4; ++j)
                std::memcpy(c[j], base + uint64_t(corner[4 * k + uint32_t(j)]) * stride + at.relative_offset, 12);
            for (uint32_t y = 0; y < g; ++y) {
                for (uint32_t x = 0; x < g; ++x) {
                    const double u = static_cast<double>(x) / split, w = static_cast<double>(y) / split;
                    float q[3];
                    for (int a = 0; a < 3; ++a)
                        q[a] = static_cast<float>((1 - u) * (1 - w) * c[0][a] + u * (1 - w) * c[1][a] + u * w * c[2][a] + (1 - u) * w * c[3][a]);
                    std::memcpy(gr.verts.data() + (static_cast<size_t>(k) * g * g + static_cast<size_t>(y) * g + x) * stride + at.relative_offset, q, 12);
                }
            }
        }
        gr.idx.resize(ni);
        uint32_t m = 0;
        for (uint32_t k = 0; k < quads; ++k)
            for (uint32_t y = 0; y < uint32_t(split); ++y)
                for (uint32_t x = 0; x < uint32_t(split); ++x) {
                    const uint32_t a = k * g * g + y * g + x, b = a + 1, cc = a + g + 1, dd = a + g;
                    gr.idx[m++] = a; gr.idx[m++] = b; gr.idx[m++] = cc; gr.idx[m++] = a; gr.idx[m++] = cc; gr.idx[m++] = dd;
                }
        found = g_qs_grids.emplace(h, std::move(gr)).first;
        NXLOG_I(kTag, "quad_split: grid built for %u quads (vs %s, n = %d): %u vertices, %u triangles",
                quads, S.stage[kVertex]->hash.c_str(), split, nv, ni / 3);
    }
    const QuadSplitGrid& grid = found->second;
    UploadAlloc vbuf = F->upload(uint64_t(grid.verts.size()), 16);
    UploadAlloc ibuf = F->upload(uint64_t(grid.idx.size()) * 4, 4);
    if (!vbuf || !ibuf) return false;
    std::memcpy(vbuf.ptr, grid.verts.data(), grid.verts.size());
    std::memcpy(ibuf.ptr, grid.idx.data(), grid.idx.size() * 4);
    const VkDeviceSize off = vbuf.offset;
    vkCmdBindVertexBuffers(F->cb, static_cast<uint32_t>(at.stream), 1, &vbuf.buffer, &off);
    if (at.stream < 16) { g_bound_vb[at.stream].valid = false; g_vb_dirty_mask |= static_cast<uint16_t>(1u << at.stream); }
    bind_index_buffer_cached(ibuf.buffer, ibuf.offset, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(F->cb, ni, 1, 0, 0, base_instance);
    g_dirty |= kVb;   // the next draw binds the game's own vertex buffers again
    static int told = 0;
    if (told < 8) { ++told; NXLOG_I(kTag, "quad_split: %u quads into %u triangles (n = %d)", quads, ni / 3, split); }
    return true;
}

std::vector<uint32_t> convert_prim_indices(PrimConversion conv, const std::vector<uint32_t>& in) {
    std::vector<uint32_t> idx;
    const size_t n = in.size();
    switch (conv) {
        case PrimConversion::Quads:
            for (size_t q = 0; q + 3 < n; q += 4) idx.insert(idx.end(), {in[q], in[q + 1], in[q + 2], in[q], in[q + 2], in[q + 3]});
            break;
        case PrimConversion::QuadStrip:
            for (size_t q = 0; q + 3 < n; q += 2) idx.insert(idx.end(), {in[q], in[q + 1], in[q + 2], in[q + 2], in[q + 1], in[q + 3]});
            break;
        case PrimConversion::Polygon:
            for (size_t k = 1; k + 1 < n; ++k) idx.insert(idx.end(), {in[0], in[k], in[k + 1]});
            break;
        case PrimConversion::LineLoop:
            idx = in;
            if (!in.empty()) idx.push_back(in[0]);
            break;
        default: idx = in; break;
    }
    return idx;
}

// Binds an index list of 32-bit indices from the upload buffer. False: no room.
bool bind_u32_indices(const std::vector<uint32_t>& idx) {
    UploadAlloc u = F->upload(idx.size() * 4, 4);
    if (!u) return false;
    std::memcpy(u.ptr, idx.data(), idx.size() * 4);
    bind_index_buffer_cached(u.buffer, u.offset, VK_INDEX_TYPE_UINT32);
    return true;
}

// [nvn] draw_ring = N (default 32, 0 = off): the last N draws/dispatches/blits the replay submitted, printed by the freeze watchdog and
// on VK_ERROR_DEVICE_LOST. A GPU that hangs gives no error of its own - the only thing left is what it was given just before it stopped.
struct RingDraw {
    enum Kind { kDraw, kDispatch, kBlit };
    Kind kind = kDraw;
    long long frame = 0;
    const ShaderInfo* sa = nullptr;   // draw: vs | dispatch: cs
    const ShaderInfo* sb = nullptr;   // draw: fs
    char a[48] = {};                  // blit: src desc
    char b[48] = {};                  // blit: dst desc
    uint32_t x = 0, y = 0, z = 0;
    uint32_t pw = 0, ph = 0, cfmt = 0, dfmt = 0;
    bool flag = false;      // draw: indexed | dispatch: indirect | blit: copy(true)/linear(false)
    uint64_t no_chunk_delta = 0;
};
constexpr size_t kDrawRingCap = 256;
RingDraw g_ring[kDrawRingCap];
std::atomic<uint64_t> g_ring_total{0};
uint64_t g_ring_last_no_chunk = 0;
size_t draw_ring_cap() {
    static const size_t n = static_cast<size_t>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "draw_ring", 32), 0, static_cast<int64_t>(kDrawRingCap)));
    return n;
}
inline RingDraw& draw_ring_alloc() {
    const uint64_t idx = g_ring_total.load(std::memory_order_relaxed);
    RingDraw& d = g_ring[idx & (kDrawRingCap - 1)];
    const uint64_t nc = gpu::mirror_capture_no_chunk_escapes();
    d.no_chunk_delta = nc >= g_ring_last_no_chunk ? (nc - g_ring_last_no_chunk) : 0;
    g_ring_last_no_chunk = nc;
    d.a[0] = d.b[0] = '\0';
    d.sa = d.sb = nullptr;
    return d;
}
inline void draw_ring_commit() {
    g_ring_total.store(g_ring_total.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}
void draw_ring_note(uint32_t count, uint32_t instances, bool indexed) {
    if (!draw_ring_cap()) return;
    RingDraw& d = draw_ring_alloc();
    d.kind = RingDraw::kDraw;
    d.frame = static_cast<long long>(counters().presents.load(std::memory_order_relaxed));
    d.sa = S.stage[kVertex];
    d.sb = S.stage[kFragment];
    d.x = count;
    d.y = instances;
    d.pw = P.extent.width;
    d.ph = P.extent.height;
    d.cfmt = (S.ncolor > 0) ? static_cast<uint32_t>(P.color_fmt[0]) : 0u;
    d.dfmt = S.depth_tex ? static_cast<uint32_t>(P.depth_fmt) : 0u;
    d.flag = indexed;
    draw_ring_commit();
}
void draw_ring_note_dispatch(const ShaderInfo& cs, uint32_t gx, uint32_t gy, uint32_t gz, bool indirect) {
    if (!draw_ring_cap()) return;
    RingDraw& d = draw_ring_alloc();
    d.kind = RingDraw::kDispatch;
    d.frame = static_cast<long long>(counters().presents.load(std::memory_order_relaxed));
    d.sa = &cs;
    d.x = gx;
    d.y = gy;
    d.z = gz;
    d.flag = indirect;
    draw_ring_commit();
}
void draw_ring_note_blit(Texture* src, Texture* dst, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh, bool copy_only) {
    if (!draw_ring_cap()) return;
    RingDraw& d = draw_ring_alloc();
    d.kind = RingDraw::kBlit;
    d.frame = static_cast<long long>(counters().presents.load(std::memory_order_relaxed));
    std::snprintf(d.a, sizeof d.a, "t%u(%ux%u f%u)", src ? src->id : 0u, sw, sh, src ? static_cast<unsigned>(tex_vk_format(src)) : 0u);
    std::snprintf(d.b, sizeof d.b, "t%u(%ux%u f%u)", dst ? dst->id : 0u, dw, dh, dst ? static_cast<unsigned>(tex_vk_format(dst)) : 0u);
    d.flag = copy_only;
    draw_ring_commit();
}
std::string draw_ring_report() {
    const size_t cap = draw_ring_cap();
    if (!cap) return "";
    std::string out = "\n--- the last draws/dispatches/blits the replay submitted ([nvn] draw_ring, oldest first) ---\n";
    const uint64_t total = g_ring_total.load(std::memory_order_acquire);
    if (total == 0) return out + "(none recorded)\n";
    const uint64_t n = std::min<uint64_t>(total, cap);
    const uint64_t start = total - n;
    char line[512];
    for (uint64_t i = start; i < total; ++i) {
        const RingDraw& d = g_ring[i & (kDrawRingCap - 1)];
        if (d.kind == RingDraw::kDraw) {
            const char* fsh = (d.sb && !d.sb->hash.empty()) ? d.sb->hash.c_str() : "-";
            const char* vsh = (d.sa && !d.sa->hash.empty()) ? d.sa->hash.c_str() : "-";
            std::snprintf(line, sizeof line, "frame %lld  draw  fs %.47s  vs %.47s  %s  count %u  instances %u  pass %ux%u c%u d%u  no_chunk +%llu\n",
                          d.frame, fsh, vsh, d.flag ? "indexed" : "arrays", d.x, d.y,
                          d.pw, d.ph, d.cfmt, d.dfmt, static_cast<unsigned long long>(d.no_chunk_delta));
        } else if (d.kind == RingDraw::kDispatch) {
            const char* csh = (d.sa && !d.sa->hash.empty()) ? d.sa->hash.c_str() : "-";
            std::snprintf(line, sizeof line, "frame %lld  dispatch  cs %.47s  groups %ux%ux%u%s  no_chunk +%llu\n",
                          d.frame, csh, d.x, d.y, d.z, d.flag ? " (indirect)" : "",
                          static_cast<unsigned long long>(d.no_chunk_delta));
        } else {
            std::snprintf(line, sizeof line, "frame %lld  %s  %s -> %s  no_chunk +%llu\n",
                          d.frame, d.flag ? "copy" : "blit", d.a, d.b,
                          static_cast<unsigned long long>(d.no_chunk_delta));
        }
        out += line;
    }
    return out;
}
static const bool g_ring_registered = (nxrt::diag_add_report(draw_ring_report), true);

// [nvn] draw_log_vs = <comma list of vertex shader hash prefixes> (debug, empty = off): every draw with one of them is logged at Info under
// nvn.drawlog with the presented frame, the counts and the vertex stage's first four uniform blocks (notes/fps120.md, typewriter hunt).
void draw_log(uint32_t count, uint32_t instances, bool indexed) {
    draw_ring_note(count, instances, indexed);
    static const std::vector<std::string> prefixes = [] {
        std::vector<std::string> v;
        const std::string s = nxrt::config_get_str("nvn", "draw_log_vs", "");
        for (size_t i = 0; i < s.size();) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) v.push_back(s.substr(i, j - i));
            i = j + 1;
        }
        return v;
    }();
    if (prefixes.empty() || !S.stage[kVertex]) return;
    const std::string& h = S.stage[kVertex]->hash;
    bool hit = false;
    for (const std::string& p : prefixes) hit = hit || h.compare(0, p.size(), p) == 0;
    if (!hit) return;
    NXLOG_I("nvn.drawlog", "frame %lld vs %s fs %s %s count %u instances %u ubo %llx/%llu %llx/%llu %llx/%llu %llx/%llu",
            static_cast<long long>(counters().presents.load()), h.c_str(), S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-",
            indexed ? "indexed" : "arrays", count, instances, static_cast<unsigned long long>(S.ubo[kVertex][0].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][0].size), static_cast<unsigned long long>(S.ubo[kVertex][1].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][1].size), static_cast<unsigned long long>(S.ubo[kVertex][2].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][2].size), static_cast<unsigned long long>(S.ubo[kVertex][3].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][3].size));
}

static void trace_dump_geometry(uint32_t mode, uint32_t count, bool indexed, uint32_t index_type, uint64_t index_address,
                                int32_t base_vertex, uint32_t base_instance);

bool snapshot_instances_on() {
    // auto (XC2 NS2's default, xc2_boot.cpp): off when [nvn] mirror_capture is on (vertex streams already read the captured copy)
    static const bool on = nxrt::config_get_str("nvn", "snapshot_instances", "1") == "auto" ? !mirror_capture_on()
                                                                                          : nxrt::config_get_bool("nvn", "snapshot_instances", true);
    return on;
}
uint64_t g_snap_draws = 0, g_snap_bytes = 0, g_snap_big = 0;
void snapshot_instances(uint32_t first_instance, uint32_t instances) {
    // Fast skip: only copy instanced streams whose vertex buffer was NOT already served from a GPU-safe copy (VRAM mirror / SnapMap snapshot).
    const uint32_t need = g_key_inst_mask & ~g_vb_safe_mask;
    if (!need || !snapshot_instances_on() || !F) return;
    bool any = false;
    for (uint32_t mask = need; mask; mask &= mask - 1) {
        const uint32_t b = static_cast<uint32_t>(std::countr_zero(mask));
        const VertexStreamInfo& st = S.stream[b];
        const GState::Bind& vb = S.vtx[b];
        if (st.divisor <= 0 || !vb.addr) continue;
        const uint8_t* host = static_cast<const uint8_t*>(host_ptr(vb.addr));
        if (!host) continue;
        const uint64_t div = static_cast<uint64_t>(st.divisor), stride = static_cast<uint64_t>(std::max<ptrdiff_t>(0, st.stride));
        const uint64_t recs = (static_cast<uint64_t>(first_instance) + instances + div - 1) / div;
        uint64_t bytes = (stride ? recs * stride : 0) + 256;            // + the attribute offsets inside the last record
        if (vb.size) bytes = std::min<uint64_t>(bytes, vb.size);
        if (bytes > (4ull << 20)) { ++g_snap_big; continue; }
        UploadAlloc u = F->upload(bytes, 16);
        if (!u) continue;
        std::memcpy(u.ptr, host, static_cast<size_t>(bytes));
        const VkDeviceSize off = u.offset, size = bytes, vstride = stride;
        vkCmdBindVertexBuffers2(F->cb, b, 1, &u.buffer, &off, &size, &vstride);
        g_bound_vb[b].valid = false;
        g_vb_dirty_mask |= static_cast<uint16_t>(1u << b);
        g_snap_bytes += bytes;
        any = true;
    }
    if (any) {
        ++g_snap_draws;
        g_dirty |= kVb;   // the next draw binds the game's own buffers again
        static uint64_t next = 20000;
        if (g_snap_draws >= next) {
            next += 200000;
            NXLOG_I(kTag, "snapshot_instances: %llu draws, %llu MiB of per-instance data copied at replay, %llu ranges over 4 MiB left in place",
                    (unsigned long long)g_snap_draws, (unsigned long long)(g_snap_bytes >> 20), (unsigned long long)g_snap_big);
        }
    }
}

void do_draw_arrays(const P_DrawArrays& p) {
    count_primitives(p.mode, p.count, p.instance_count);
    VkPrimitiveTopology topo;
    if (!begin_draw(static_cast<int>(p.mode), false, 0, &topo)) { counters().draws_skipped.fetch_add(1); return; }
    snapshot_instances(p.base_instance, std::max<uint32_t>(1, p.instance_count));
    draw_log(p.count, p.instance_count, false);
    quad_split_candidate(p.mode, p.count, std::max<uint32_t>(1, p.instance_count));   // [nvn] quad_split: does this draw qualify?
    const PrimConversion conv = prim_conversion(static_cast<int>(p.mode));
    const uint32_t inst = std::max<uint32_t>(1, p.instance_count);
    if (conv == PrimConversion::None) {
        TRACE("    vertices %u instances %u first %u", p.count, p.instance_count, p.first);
        if (g_trace) trace_dump_geometry(p.mode, p.count, false, 0, 0, static_cast<int32_t>(p.first), p.base_instance);
        auto geo = [&] { vkCmdDraw(F->cb, p.count, inst, p.first, p.base_instance); };
        if (g_earlyz_pending) { earlyz_prepass_begin(*g_pl); geo(); earlyz_prepass_end(*g_pl); }   // [nvn] earlyz: the depth pre-pass first
        {
            DrawOccScope docc_scope;   // [nvn] draw_occ: samples this draw passes (reported after the frame)
            geo();
        }
        earlyz_mask_rebuild();   // [nvn] earlyz_mask: the registration mask, derived from what the list just got
    } else {
        // QUADS / QUAD_STRIP / POLYGON / LINE_LOOP: generate an index buffer
        std::vector<uint32_t> in(p.count);
        for (uint32_t k = 0; k < p.count; ++k) in[k] = p.first + k;
        if (conv == PrimConversion::Quads && quad_split_candidate(p.mode, p.count, inst) &&
            quad_split_draw(p.count / 4, in, p.base_instance)) return;   // [nvn] quad_split
        const std::vector<uint32_t> idx = convert_prim_indices(conv, in);
        if (idx.empty()) return;
        if (!bind_u32_indices(idx)) return;
        const uint32_t n = static_cast<uint32_t>(idx.size());
        auto geo = [&] { vkCmdDrawIndexed(F->cb, n, inst, 0, 0, p.base_instance); };
        if (g_earlyz_pending) { earlyz_prepass_begin(*g_pl); geo(); earlyz_prepass_end(*g_pl); }   // [nvn] earlyz: the depth pre-pass first
        {
            DrawOccScope docc_scope;   // [nvn] draw_occ: samples this draw passes (reported after the frame)
            geo();
        }
        earlyz_mask_rebuild();   // [nvn] earlyz_mask: the registration mask, derived from what the list just got
    }
    rs_mark(kRsCall);
    ++g_sub_draws;
    g_pass_draws++;
    g_pass_verts += static_cast<uint64_t>(p.count) * std::max<uint32_t>(1, p.instance_count);
    trace_snap_after_draw();
}

// [nvn] trace_vb_dir + trace_vb_fs: dump the geometry and the vertex-stage constant data of every traced draw whose
// fragment shader matches one of the trace_vb_fs prefixes (comma separated). With the vertex buffer, the index buffer, the
// uniform blocks and the skinning storage buffers on disk, a draw's clip positions can be recomputed on the CPU without
// guessing which NVN uniform block the shader's c[k] is (HANDOVER 25.5: #147/#148 vs #151).
static void trace_dump_geometry(uint32_t mode, uint32_t count, bool indexed, uint32_t index_type, uint64_t index_address,
                                int32_t base_vertex, uint32_t base_instance) {
    static const std::string dir = nxrt::config_get_str("nvn", "trace_vb_dir", "");
    static const std::string filter = nxrt::config_get_str("nvn", "trace_vb_fs", "");
    if (dir.empty() || filter.empty()) return;
    const ShaderInfo* fs = S.stage[kFragment];
    const ShaderInfo* vs = S.stage[kVertex];
    // A filter entry is a fragment-shader hash prefix, "nofs" (a vertex-only draw such as the skinning passes, whose
    // writes are what the character's draws read) or "vs:<prefix>" (a vertex-shader prefix).
    bool match = false;
    for (size_t pos = 0; pos <= filter.size() && !match;) {
        const size_t comma = filter.find(',', pos);
        std::string_view f(filter.data() + pos, (comma == std::string::npos ? filter.size() : comma) - pos);
        while (!f.empty() && (f.front() == ' ' || f.front() == '\t')) f.remove_prefix(1);
        while (!f.empty() && (f.back() == ' ' || f.back() == '\t')) f.remove_suffix(1);
        if (f == "nofs") match = fs == nullptr;
        else if (f.size() > 3 && f.substr(0, 3) == "vs:") match = vs && vs->hash.compare(0, f.size() - 3, f.substr(3)) == 0;
        else match = fs && !f.empty() && fs->hash.compare(0, f.size(), f) == 0;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (!match) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const int n = g_trace_n;
    char name[64];
    {
        std::snprintf(name, sizeof name, "/d%d.txt", n);
        std::ofstream m(dir + name);
        if (!m) return;
        m << "draw " << n << " vs " << (S.stage[kVertex] ? S.stage[kVertex]->hash : "-") << " fs " << (fs ? fs->hash : "-")
          << " mode " << mode << " count " << count << " type " << (indexed ? index_type : 0xffffffffu) << " base_vertex " << base_vertex
          << " base_instance " << base_instance << " index_address 0x" << std::hex << index_address << std::dec << "\n";
        m << "attr_count " << S.attr_count << "\n";
        for (uint32_t i = 0; i < S.attr_count && i < 16; ++i) {
            const VertexAttribInfo& at = S.attr[i];
            const VertexStreamInfo& st = S.stream[at.stream];
            const GState::Bind& b = S.vtx[at.stream];
            m << "attr " << i << " stream " << at.stream << " format " << at.format << " offset " << at.relative_offset
              << " stride " << st.stride << " addr 0x" << std::hex << b.addr << " size " << std::dec << b.size << "\n";
        }
        for (uint32_t slot = 0; slot < 16; ++slot) {
            const GState::Bind& ub = S.ubo[kVertex][slot];
            if (ub.addr) m << "ubo " << slot << " addr 0x" << std::hex << ub.addr << " size " << std::dec << ub.size << "\n";
            const GState::Bind& sb = S.ssbo[kVertex][slot];
            if (sb.addr) m << "ssbo " << slot << " addr 0x" << std::hex << sb.addr << " size " << std::dec << sb.size << "\n";
        }
    }
    bool stream_done[16] = {};
    for (uint32_t i = 0; i < S.attr_count && i < 16; ++i) {
        const int stream = S.attr[i].stream;
        if (stream < 0 || stream >= 16 || stream_done[stream]) continue;
        stream_done[stream] = true;
        const GState::Bind& b = S.vtx[stream];
        const uint8_t* src = b.addr ? host_ptr(b.addr) : nullptr;
        if (!src) continue;
        std::snprintf(name, sizeof name, "/d%d_s%d.bin", n, stream);
        std::ofstream f(dir + name, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(std::min<uint64_t>(b.size, 4u << 20)));
    }
    if (indexed) if (const uint8_t* ix = host_ptr(index_address)) {
        const uint32_t isz = index_size_bytes(static_cast<int>(index_type));
        std::snprintf(name, sizeof name, "/d%d_ib.bin", n);
        std::ofstream f(dir + name, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(ix), static_cast<std::streamsize>(uint64_t(count) * isz));
    }
    for (uint32_t slot = 0; slot < 16; ++slot) {
        const GState::Bind& ub = S.ubo[kVertex][slot];
        if (const uint8_t* s = ub.addr ? host_ptr(ub.addr) : nullptr) {
            std::snprintf(name, sizeof name, "/d%d_ubo%u.bin", n, slot);
            std::ofstream f(dir + name, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(s), static_cast<std::streamsize>(std::min<uint64_t>(ub.size, 1u << 20)));
        }
        const GState::Bind& sb = S.ssbo[kVertex][slot];
        if (const uint8_t* s = sb.addr ? host_ptr(sb.addr) : nullptr) {
            std::snprintf(name, sizeof name, "/d%d_ssbo%u.bin", n, slot);
            std::ofstream f(dir + name, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(s), static_cast<std::streamsize>(std::min<uint64_t>(sb.size, 1u << 20)));
        }
    }
    NXLOG_I("nvn.trace", "geometry of draw #%d (%s) dumped to %s", n, fs ? fs->hash.c_str() : "no fs", dir.c_str());
}

// Decode one vertex attribute for the trace, by NVN format: the character draws feed RG16UI bone indices, RGBA8 decal
// UVs, RGBA8SN normals and 32-bit floats, and reading every one as three floats printed NaNs.
static int trace_attr_components(const uint8_t* p, int fmt, double out[4], const char** kind) {
    auto half = [](uint16_t h) {
        const int e = (h >> 10) & 31, m = h & 1023;
        const double sg = (h & 0x8000) ? -1.0 : 1.0;
        if (e == 0) return sg * m * (1.0 / 16777216.0);
        if (e == 31) return sg * 65504.0;
        return sg * (1.0 + m / 1024.0) * double(int64_t(1) << (e - 15));
    };
    switch (fmt) {
        case 0x22: for (int i = 0; i < 3; ++i) { float f; std::memcpy(&f, p + 4 * i, 4); out[i] = f; } out[3] = 1; *kind = "f32"; return 3;
        case 0x16: for (int i = 0; i < 2; ++i) { float f; std::memcpy(&f, p + 4 * i, 4); out[i] = f; } out[2] = 0; out[3] = 1; *kind = "f32"; return 2;
        case 0x0a: { float f; std::memcpy(&f, p, 4); out[0] = f; out[1] = out[2] = 0; out[3] = 1; *kind = "f32"; return 1; }
        case 0x25: for (int i = 0; i < 4; ++i) out[i] = p[i] / 255.0; *kind = "un8"; return 4;
        case 0x26: for (int i = 0; i < 4; ++i) out[i] = std::max(-1.0, int8_t(p[i]) / 127.0); *kind = "sn8"; return 4;
        case 0x27: for (int i = 0; i < 4; ++i) out[i] = p[i]; *kind = "u8"; return 4;
        case 0x28: for (int i = 0; i < 4; ++i) out[i] = int8_t(p[i]); *kind = "s8"; return 4;
        case 0x14: { uint16_t a, b; std::memcpy(&a, p, 2); std::memcpy(&b, p + 2, 2); out[0] = a; out[1] = b; out[2] = out[3] = 0; *kind = "u16"; return 2; }
        case 0x2a: { uint16_t v[4]; std::memcpy(v, p, 8); for (int i = 0; i < 4; ++i) out[i] = v[i] / 65535.0; *kind = "un16"; return 4; }
        case 0x11: { uint16_t v[2]; std::memcpy(v, p, 4); out[0] = half(v[0]); out[1] = half(v[1]); out[2] = 0; out[3] = 1; *kind = "f16"; return 2; }
        case 0x29: { uint16_t v[4]; std::memcpy(v, p, 8); for (int i = 0; i < 4; ++i) out[i] = half(v[i]); *kind = "f16"; return 4; }
        default: for (int i = 0; i < 3; ++i) { float f; std::memcpy(&f, p + 4 * i, 4); out[i] = f; } out[3] = 1; *kind = "raw"; return 3;
    }
}

void do_draw_elements(const P_DrawElements& p) {
    count_primitives(p.mode, p.count, p.instance_count);
    VkPrimitiveTopology topo;
    if (!begin_draw(static_cast<int>(p.mode), true, static_cast<int>(p.type), &topo)) { counters().draws_skipped.fetch_add(1); return; }
    snapshot_instances(p.base_instance, std::max<uint32_t>(1, p.instance_count));
    draw_log(p.count, p.instance_count, true);
    if (g_trace) {   // the indices and the first attributes of the first vertices
        const uint32_t isz0 = index_size_bytes(static_cast<int>(p.type));
        const uint8_t* ix = host_ptr(p.index_address);
        TRACE("    indexed: count %u type %d instances %u base_vertex %d base_instance %u index address %llx", p.count, static_cast<int>(p.type), p.instance_count, static_cast<int>(p.base_vertex),
              p.base_instance, static_cast<unsigned long long>(p.index_address));
        if (trace_verbose()) {   // [nvn] trace_compact: the geometry dump and every vertex of the first 12 are the expensive part
        trace_dump_geometry(p.mode, p.count, true, p.type, p.index_address, p.base_vertex, p.base_instance);
        for (uint32_t i = 0; ix && i < std::min<uint32_t>(p.count, 12); ++i) {
            uint32_t v = 0;
            if (isz0 == 1) v = ix[i]; else if (isz0 == 2) { uint16_t t; std::memcpy(&t, ix + i * 2, 2); v = t; } else std::memcpy(&v, ix + i * 4, 4);
            std::string a;
            for (uint32_t k = 0; k < S.attr_count && k < 6; ++k) {
                const VertexAttribInfo& at = S.attr[k];
                const auto& st = S.stream[at.stream];
                const GState::Bind& vb = S.vtx[at.stream];
                if (!vb.addr) continue;
                const uint8_t* ve = reinterpret_cast<const uint8_t*>(vb.addr + static_cast<uint64_t>(static_cast<int64_t>(v) + p.base_vertex) * st.stride + at.relative_offset);
                double c[4] = {0, 0, 0, 1};
                const char* kind = "?";
                trace_attr_components(ve, at.format, c, &kind);
                char t[192]; std::snprintf(t, sizeof t, " a%u(f%d %s: %g %g %g %g)", k, at.format, kind, c[0], c[1], c[2], c[3]);
                a += t;
            }
            TRACE("      index %u:%s", v, a.c_str());
        }
        }   // trace_verbose()
    }
    if (g_trace && trace_verbose()) {   // bounds of the first attributes over all indices of the draw (half and float formats decoded), to see what part of a texture a mesh maps
        const uint32_t isz1 = index_size_bytes(static_cast<int>(p.type));
        const uint8_t* ix = host_ptr(p.index_address);
        if (ix && p.count) {
            // The largest index of the draw: with the stream's bound size this shows whether the GPU can fetch every vertex the
            // draw references. A vertex buffer bound shorter than (idx_max+1)*stride feeds zeros, which collapses the triangle -
            // XC2's eye cards fail exactly that way while the face mesh beside them is fine (HANDOVER 21.10).
            uint32_t vmax = 0;
            for (uint32_t i = 0; i < p.count; ++i) {
                uint32_t v = 0;
                if (isz1 == 1) v = ix[i]; else if (isz1 == 2) { uint16_t t; std::memcpy(&t, ix + i * 2, 2); v = t; } else std::memcpy(&v, ix + i * 4, 4);
                vmax = std::max(vmax, v);
            }
            std::string a;
            for (uint32_t k = 0; k < S.attr_count && k < 3; ++k) {
                const VertexAttribInfo& at = S.attr[k];
                const auto& st = S.stream[at.stream];
                const GState::Bind& vb = S.vtx[at.stream];
                if (!vb.addr) continue;
                float lo[4] = {1e30f, 1e30f, 1e30f, 1e30f}, hi[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                int nc = at.format == 41 ? 4 : at.format == 34 ? 3 : at.format == 17 ? 2 : at.format == 22 ? 2 : 0;
                for (uint32_t i = 0; nc && i < p.count && i < 20000; ++i) {
                    uint32_t v = 0;
                    if (isz1 == 1) v = ix[i]; else if (isz1 == 2) { uint16_t t; std::memcpy(&t, ix + i * 2, 2); v = t; } else std::memcpy(&v, ix + i * 4, 4);
                    const uint8_t* ve = reinterpret_cast<const uint8_t*>(vb.addr + static_cast<uint64_t>(static_cast<int64_t>(v) + p.base_vertex) * st.stride + at.relative_offset);
                    for (int c = 0; c < nc; ++c) {
                        float f;
                        if (at.format == 41 || at.format == 17) {
                            uint16_t h; std::memcpy(&h, ve + c * 2, 2);
                            const int e = (h >> 10) & 31, m = h & 1023; const float sg = (h & 0x8000) ? -1.f : 1.f;
                            f = e == 0 ? sg * m * (1.f / 16777216.f) : e == 31 ? sg * 65504.f : sg * (1.f + m / 1024.f) * std::ldexp(1.f, e - 15);
                        } else std::memcpy(&f, ve + c * 4, 4);
                        lo[c] = std::min(lo[c], f); hi[c] = std::max(hi[c], f);
                    }
                }
                char t[200];
                if (nc) std::snprintf(t, sizeof t, " a%u[%g..%g, %g..%g, %g..%g, %g..%g]", k, lo[0], hi[0], lo[1], hi[1], nc > 2 ? lo[2] : 0.f, nc > 2 ? hi[2] : 0.f, nc > 3 ? lo[3] : 0.f, nc > 3 ? hi[3] : 0.f);
                else std::snprintf(t, sizeof t, " a%u(fmt %d: not decoded)", k, at.format);
                a += t;
            }
            TRACE("    bounds:%s idx_max %u", a.c_str(), vmax);
        }
    }
    const uint32_t isz = index_size_bytes(static_cast<int>(p.type));
    const PrimConversion conv = prim_conversion(static_cast<int>(p.mode));
    if (conv != PrimConversion::None || isz == 1) {
        const uint8_t* src = host_ptr(p.index_address);
        // widen to 32 bit and/or convert the primitive
        std::vector<uint32_t> in(p.count);
        for (uint32_t i = 0; i < p.count; ++i) {
            uint32_t v = 0;
            if (isz == 1) v = src[i];
            else if (isz == 2) { uint16_t t; std::memcpy(&t, src + i * 2, 2); v = t; }
            else std::memcpy(&v, src + i * 4, 4);
            in[i] = v;
        }
        if (conv == PrimConversion::Quads && quad_split_candidate(p.mode, p.count, std::max<uint32_t>(1, p.instance_count))) {   // [nvn] quad_split: qualifies?
            std::vector<uint32_t> corner(in.begin(), in.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(in.size(), 64)));
            for (uint32_t& v : corner) v += static_cast<uint32_t>(p.base_vertex);
            if (quad_split_draw(static_cast<uint32_t>(in.size() / 4), corner, p.base_instance)) return;
        }
        const std::vector<uint32_t> idx = convert_prim_indices(conv, in);
        if (idx.empty()) return;
        if (!bind_u32_indices(idx)) return;
        const uint32_t n = static_cast<uint32_t>(idx.size());
        const uint32_t inst = std::max<uint32_t>(1, p.instance_count);
        auto geo = [&] { vkCmdDrawIndexed(F->cb, n, inst, 0, p.base_vertex, p.base_instance); };
        if (g_earlyz_pending) { earlyz_prepass_begin(*g_pl); geo(); earlyz_prepass_end(*g_pl); }   // [nvn] earlyz: the depth pre-pass first
        {
            DrawOccScope docc_scope;   // [nvn] draw_occ: samples this draw passes (reported after the frame)
            geo();
        }
        earlyz_mask_rebuild();   // [nvn] earlyz_mask: the registration mask, derived from what the list just got
    } else {
        BufRef br = resolve_read(p.index_address, static_cast<uint64_t>(p.count) * isz, kReadIndex);
        bind_index_buffer_cached(br.buffer, br.offset, vk_index_type(static_cast<int>(p.type)));
        const uint32_t inst = std::max<uint32_t>(1, p.instance_count);
        auto geo = [&] { vkCmdDrawIndexed(F->cb, p.count, inst, 0, p.base_vertex, p.base_instance); };
        if (g_earlyz_pending) { earlyz_prepass_begin(*g_pl); geo(); earlyz_prepass_end(*g_pl); }   // [nvn] earlyz: the depth pre-pass first
        {
            DrawOccScope docc_scope;   // [nvn] draw_occ: samples this draw passes (reported after the frame)
            geo();
        }
        earlyz_mask_rebuild();   // [nvn] earlyz_mask: the registration mask, derived from what the list just got
    }
    rs_mark(kRsCall);
    ++g_sub_draws;
    g_pass_draws++;
    g_pass_verts += static_cast<uint64_t>(p.count) * std::max<uint32_t>(1, p.instance_count);
    trace_snap_after_draw();
}

// The bytes at a GPU address as the CPU sees them, when they are current: inside a pool, and not a range a submission that has not completed
// may write (gw_settled). Null otherwise. For decisions that need data the GPU would otherwise hold (indirect arguments, draw counts).
// Compute results held in the device copy ([nvn] compute_out_lazy) are newer than the pool: they are fetched back first (out_fetch: a
// completed submission's result, or one whose copy-back is recorded but has not run); one the submission being recorded writes cannot be
// read (null). Without the fetch an unguarded held result would be read stale, and a guarded one through a fault.
const uint8_t* cpu_view(uint64_t addr, uint64_t size) {
    if (!addr || !size) return nullptr;
    const PoolHit hit = pool_lookup(addr);
    if (!hit.pool || hit.pool->is_virtual || !hit.pool->base || hit.offset + size > hit.pool->size) return nullptr;
    if (!gw_settled(addr, size)) return nullptr;
    if (out_any()) {
        out_fetch(addr, size, false);
        if (out_overlaps(addr, size)) return nullptr;
    }
    return host_ptr(addr);
}

// An indirect draw Vulkan cannot take as it is (quads, quad strips, polygons and line loops have no topology; 8-bit indices need
// VK_EXT_index_type_uint8): each draw's arguments (and its indices) are read by the CPU and the draw is made as a direct indexed draw of
// the converted 32-bit index list, like DrawArrays / DrawElements. That needs the arguments in memory the CPU can read: arguments a compute
// shader or a copy wrote and no completed submission has handed back are not (skipped, logged). A multi-draw reads its count the same way.
void draw_indirect_cpu(const P_DrawIndirect& p, PrimConversion conv, uint32_t isz, bool indexed) {
    static bool noted = false;
    if (!noted) {
        noted = true;
        NXLOG_I("nvn.replay", "replay: indirect draw by CPU index conversion (first): mode %u, %s, index size %u, %s", p.mode, indexed ? "indexed" : "arrays", isz,
                p.multi ? "multi-draw" : "single");
    }
    const uint64_t cmd_size = indexed ? 20 : 16;
    const uint64_t stride = p.stride > 0 ? static_cast<uint64_t>(p.stride) : cmd_size;
    uint32_t n = 1;
    if (p.multi && p.count_address) {
        const uint8_t* c = cpu_view(p.count_address, 4);
        if (!c) { note_unsupported("indirect draw conversion: the draw count is not readable by the CPU (written by the GPU)"); counters().draws_skipped.fetch_add(1); return; }
        std::memcpy(&n, c, 4);
        n = std::min(n, p.max_count);
    }
    bool drew = false;
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* a = cpu_view(p.indirect_address + static_cast<uint64_t>(i) * stride, cmd_size);
        if (!a) { note_unsupported("indirect draw conversion: the arguments are not readable by the CPU (written by the GPU)"); break; }
        uint32_t v[5] = {};
        std::memcpy(v, a, cmd_size);
        const uint32_t count = v[0], instances = v[1];
        if (!count || !instances) continue;
        std::vector<uint32_t> in(count);
        int32_t base_vertex = 0;
        uint32_t base_instance = v[3];
        if (indexed) {
            base_vertex = static_cast<int32_t>(v[3]);
            base_instance = v[4];
            const uint8_t* src = cpu_view(p.index_address + static_cast<uint64_t>(v[2]) * isz, static_cast<uint64_t>(count) * isz);
            if (!src) { note_unsupported("indirect draw conversion: the indices are not readable by the CPU (written by the GPU)"); break; }
            for (uint32_t k = 0; k < count; ++k) {
                uint32_t x = 0;
                if (isz == 1) x = src[k];
                else if (isz == 2) { uint16_t t; std::memcpy(&t, src + k * 2, 2); x = t; }
                else std::memcpy(&x, src + k * 4, 4);
                in[k] = x;
            }
        } else {
            for (uint32_t k = 0; k < count; ++k) in[k] = v[2] + k;
        }
        count_primitives(p.mode, count, instances);
        const std::vector<uint32_t> idx = convert_prim_indices(conv, in);
        if (idx.empty() || !bind_u32_indices(idx)) continue;
        DrawOccScope docc_scope;   // [nvn] draw_occ: samples this draw passes (reported after the frame)
        vkCmdDrawIndexed(F->cb, static_cast<uint32_t>(idx.size()), instances, 0, base_vertex, base_instance);
        drew = true;
    }
    if (drew) {
        rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
        g_pass_draws++;
    } else {
        counters().draws_skipped.fetch_add(1);
    }
    trace_snap_after_draw();
}

void do_draw_indirect(const P_DrawIndirect& p) {
    const bool indexed = p.type != 0xffffffffu;
    VkPrimitiveTopology topo;
    if (!begin_draw(static_cast<int>(p.mode), indexed, indexed ? static_cast<int>(p.type) : 0, &topo)) { counters().draws_skipped.fetch_add(1); return; }
    earlyz_refuse_indirect("an indirect draw");   // [nvn] earlyz: only the direct paths issue the pre-pass geometry
    const PrimConversion conv = prim_conversion(static_cast<int>(p.mode));
    const uint32_t isz = indexed ? index_size_bytes(static_cast<int>(p.type)) : 0;
    if (conv != PrimConversion::None || (indexed && isz == 1 && !device().features().index_type_uint8)) {
        draw_indirect_cpu(p, conv, isz, indexed);
        return;
    }
    // GPU path. The primitives of these draws are not known to the CPU: the primitive counters see them through a query window (ResetCounter).
    static bool noted_indirect = false, noted_u8 = false;
    if (!noted_indirect) { noted_indirect = true; NXLOG_I("nvn.replay", "replay: indirect draw (first): mode %u, %s%s", p.mode, indexed ? "indexed" : "arrays", p.multi ? ", multi-draw" : ""); }
    if (indexed && isz == 1 && !noted_u8) { noted_u8 = true; NXLOG_I("nvn.replay", "replay: indirect draw with 8-bit indices (first): VK_EXT_index_type_uint8"); }
    const uint64_t cmd_size = indexed ? 20 : 16;
    const uint64_t n_cmds = p.multi && p.count_address ? std::max<uint64_t>(1, p.max_count) : 1;
    BufRef ib = held_or_pool(p.indirect_address, (p.stride ? static_cast<uint64_t>(p.stride) : cmd_size) * (n_cmds - 1) + cmd_size, kOutIndirect, false);
    if (indexed) {
        BufRef xb = resolve_buffer(p.index_address, 4);
        bind_index_buffer_cached(xb.buffer, xb.offset, vk_index_type(static_cast<int>(p.type)));
    }
    if (p.multi && p.count_address) {
        BufRef cb = held_or_pool(p.count_address, 4, kOutIndirect, false);
        const uint32_t stride = static_cast<uint32_t>(p.stride ? p.stride : cmd_size);
        DrawOccScope docc_scope;   // one query covers the whole multi-draw
        if (indexed) vkCmdDrawIndexedIndirectCount(F->cb, ib.buffer, ib.offset, cb.buffer, cb.offset, p.max_count, stride);
        else vkCmdDrawIndirectCount(F->cb, ib.buffer, ib.offset, cb.buffer, cb.offset, p.max_count, stride);
    } else if (indexed) {
        DrawOccScope docc_scope;   // [nvn] draw_occ: samples this draw passes (reported after the frame)
        vkCmdDrawIndexedIndirect(F->cb, ib.buffer, ib.offset, 1, 20);
    } else {
        DrawOccScope docc_scope;   // [nvn] draw_occ: samples this draw passes (reported after the frame)
        vkCmdDrawIndirect(F->cb, ib.buffer, ib.offset, 1, 16);
    }
    rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
    g_pass_draws++;
    trace_snap_after_draw();
}

// DrawTransformFeedback(mode, control): draws what the feedback section begun with this control buffer wrote, as many vertices as it wrote
// (vkCmdDrawIndirectByteCountEXT over the section's byte counter, divided by the stride of the vertex stream that reads the feedback buffer). The game
// binds the feedback buffer as an ordinary vertex buffer. The feedback writes are ordered before the vertex fetch by ending the pass first.
void do_draw_xfb(const P_DrawXfb& p) {
    static bool noted = false;
    if (!noted) { noted = true; NXLOG_I("nvn.xfb", "replay: transform feedback draw (first): mode %u control %llx", p.mode, static_cast<unsigned long long>(p.control)); }
    if (!S.render_enable) return;
    if (!device().features().transform_feedback) { note_unsupported("DrawTransformFeedback (the device has no VK_EXT_transform_feedback)"); counters().draws_skipped.fetch_add(1); return; }
    const auto it = g_xfb_slot_of.find(p.control);
    if (!g_xfb_counter || it == g_xfb_slot_of.end() || !g_xfb_slot_written[it->second]) {
        TRACE("DrawTransformFeedback control %llx: the section wrote nothing", static_cast<unsigned long long>(p.control));   // zero bytes: zero vertices
        counters().draws_skipped.fetch_add(1);
        return;
    }
    if (prim_conversion(static_cast<int>(p.mode)) != PrimConversion::None) { note_unsupported("DrawTransformFeedback of a quad/polygon/line-loop primitive"); return; }
    const uint32_t slot = it->second;
    end_pass();
    memory_barrier(F->cb);   // the section's writes (feedback buffer, counter) before the vertex fetch and the indirect read
    VkPrimitiveTopology topo;
    if (!begin_draw(static_cast<int>(p.mode), false, 0, &topo)) { counters().draws_skipped.fetch_add(1); return; }
    earlyz_refuse_indirect("DrawTransformFeedback");   // [nvn] earlyz: only the direct paths issue the pre-pass geometry
    uint32_t stride = g_key_attr_count ? S.stream[g_key_binding[0]].stride : 0;
    for (uint32_t b = 0; !stride && b < 16; ++b) stride = S.stream[b].stride;
    if (!stride) { note_unsupported("DrawTransformFeedback with no vertex stream stride"); counters().draws_skipped.fetch_add(1); return; }
    vkCmdDrawIndirectByteCountEXT(F->cb, 1, 0, g_xfb_counter.buffer, static_cast<VkDeviceSize>(slot) * kXfbSlotBytes, 0, stride);
    rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
    g_pass_draws++;
    trace_snap_after_draw();
}

std::string g_cs_missing;   // trace: the untranslated block(s) of the bound compute program ("cs -" dispatches)

// [nvn] trace_snap_cs = "98acab99,e8d64471": after every dispatch of the traced frame whose compute shader hash starts with one of these,
// the textures and images it binds are copied (png after the frame: snap_c<dispatch>_t<id>_<format>.png; float formats also as .bin).
void trace_snap_after_dispatch(const ShaderInfo& cs, const ProgLayout& pl) {
    static const std::vector<std::string> list = [] {
        std::vector<std::string> v;
        const std::string s = nxrt::config_get_str("nvn", "trace_snap_cs", "");
        for (size_t k = 0; k < s.size();) {
            size_t j = s.find(',', k);
            if (j == std::string::npos) j = s.size();
            if (j > k) v.push_back(s.substr(k, j - k));
            k = j + 1;
        }
        return v;
    }();
    static int n = 0;
    ++n;
    bool hit = false;
    for (const std::string& pre : list) hit |= cs.hash.compare(0, pre.size(), pre) == 0;
    if (!hit) return;
    std::vector<Texture*> done;
    for (const BindingDesc& d : pl.bindings) {
        const bool img = d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (!img && d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        Texture* t = (img ? resolve_image(d) : resolve_texture(d)).tex;
        if (!t || std::find(done.begin(), done.end(), t) != done.end()) continue;
        done.push_back(t);
        const int idx = snapshot_record(t, *F);
        if (idx < 0) continue;
        char buf[96];
        g_trace_snaps.push_back({idx, -n, "work/trace/snap_c" + std::to_string(n) + "_" + cs.hash.substr(0, 8) + "_t" + std::to_string(t->id) + "_" +
                                                 (t->fmt ? t->fmt->name : "x") + ".png", tex_desc(t, buf, sizeof buf)});
    }
}

// XC2 NS2 dispatches its spatial upscale compute variants (0794aaf86ae7fd58, 6139946183e76e3d, 9d2efe26d94e3ac0;
// cp_t_cb0_20 -> cp_i_cb0_24_) to scale 1080p/1440p colour targets up to the 3840x2160 output buffer (t1/t2).
// When attachment scaling is active, `src` is already at the target resolution (or output size). Running the translated
// compute shader would force `dst` back to 1x 3840x2160 via `tex_rescale` and point-subsample `src` at `render_scale`
// multiples, throwing away 4K detail and causing 4K HUD overhead at 1080p/1440p (#55, #56).
bool try_bypass_upscaled_output_compute(const ShaderInfo& cs, const ProgLayout& pl, uint64_t indirect) {
    if (indirect) return false;
    Texture* src = nullptr;
    Texture* dst = nullptr;
    const DescriptorBinding* src_b = nullptr;
    const DescriptorBinding* dst_b = nullptr;
    int sampler_count = 0, storage_count = 0;
    for (const BindingDesc& d : pl.bindings) {
        if (d.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            ++sampler_count;
            src_b = &d.shader->refl.bindings[d.index];
            src = resolve_texture(d).tex;
        } else if (d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            ++storage_count;
            dst_b = &d.shader->refl.bindings[d.index];
            dst = resolve_image(d).tex;
        }
    }
    if (sampler_count != 1 || storage_count != 1 || !src || !dst || !tex_gpu(src)) return false;
    const bool known_hash = (cs.hash == "0794aaf86ae7fd58" || cs.hash == "6139946183e76e3d" || cs.hash == "9d2efe26d94e3ac0");
    const bool known_names = (src_b && dst_b && src_b->name == "cp_t_cb0_20" && dst_b->name == "cp_i_cb0_24_");
    if (!known_hash && !known_names) return false;
    if (src->b.width < 1920 || src->b.height < 1080 || src->b.width >= dst->b.width || dst->b.width != 3840 || dst->b.height != 2160)
        return false;
    const float src_sc = tex_scale(src);
    const float out_sc = output_scale_live();
    const uint32_t q_out_h = quality_output_height_live();
    if (src_sc == 1.0f && out_sc == 1.0f && q_out_h == 0) return false;
    const VkExtent3D se = tex_image_extent(src, 0);
    const float dst_want_est = (q_out_h > 0) ? (static_cast<float>(q_out_h) / static_cast<float>(dst->b.height))
                                             : (out_sc > 1.0f ? std::max(1.0f, std::min(out_sc, std::round(out_sc * 1080.0f) / static_cast<float>(dst->b.height))) : 1.0f);
    if (src_sc == 1.0f && std::fabs(dst_want_est - 1.0f) < 0.01f) return false;
    const uint32_t est_w = static_cast<uint32_t>(std::lround(static_cast<double>(dst->b.width) * dst_want_est));
    if (se.width == 0 || se.height == 0 || (se.width != est_w && src_sc < 1.99f)) return false;
    out_texture_read(src);
    end_pass(true);
    tex_ensure_current(src, *F);
    if (!tex_gpu_attachment(dst)) return false;
    const float dst_want = tex_scale_wanted(dst);
    if (tex_scale(dst) != dst_want) tex_rescale(dst, *F, dst_want, "upscaler already at output size");
    tex_transition(src, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    tex_transition(dst, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    tex_mark_gpu_written(dst);
    const VkExtent3D de = tex_image_extent(dst, 0);
    if (se.width == de.width && se.height == de.height && tex_vk_format(src) == tex_vk_format(dst)) {
        VkImageCopy c = {};
        c.srcSubresource = {tex_aspect(src), 0, 0, 1};
        c.dstSubresource = {tex_aspect(dst), 0, 0, 1};
        c.extent = {se.width, se.height, 1};
        vkCmdCopyImage(F->cb, tex_image(src), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(dst), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    } else {
        VkImageBlit b = {};
        b.srcSubresource = {tex_aspect(src), 0, 0, 1};
        b.dstSubresource = {tex_aspect(dst), 0, 0, 1};
        b.srcOffsets[1] = {static_cast<int32_t>(se.width), static_cast<int32_t>(se.height), 1};
        b.dstOffsets[1] = {static_cast<int32_t>(de.width), static_cast<int32_t>(de.height), 1};
        const VkFilter filter = (!tex_is_depth(src) && !vk_format_is_integer(tex_vk_format(src))) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        vkCmdBlitImage(F->cb, tex_image(src), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(dst), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                       filter);
    }
    memory_barrier(F->cb);
    if (gpass_on()) gstamp("cs 0794->copy");
    counters().copies.fetch_add(1, std::memory_order_relaxed);
    TRACE("    cs %s bypassed -> direct copy %ux%u to t%u (%ux%u)", cs.hash.c_str(), se.width, se.height, dst->id, de.width, de.height);
    if (first_time_key("cs-upscale-bypass-" + cs.hash)) {
        NXLOG_I(kTag, "cs %s: bypassed 1080p->4K upscale (t%u %ux%u -> t%u %ux%u)",
                cs.hash.c_str(), src->id, se.width, se.height, dst->id, de.width, de.height);
    }
    return true;
}

void do_dispatch(uint32_t x, uint32_t y, uint32_t z, uint64_t indirect) {
    const ShaderInfo* cs = S.stage[kCompute];
    TRACE("dispatch %u %u %u indirect %llx cs %s%s%s", x, y, z, static_cast<unsigned long long>(indirect), cs ? cs->hash.c_str() : "-",
          !cs && !g_cs_missing.empty() ? " missing " : "", !cs ? g_cs_missing.c_str() : "");
    if (!cs) return;
    const ShaderInfo* stages[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, cs};
    const ProgLayout* pl = prog_layout(stages);
    if (!pl) return;
    MemoScope memo(*pl);
    if (try_bypass_upscaled_output_compute(*cs, *pl, indirect)) return;
    prepare_textures(*pl);
    end_pass();
    tsr_cache_clear();
    if (g_trace) {   // the compute bindings: buffers (source, slot, address), texel buffers and textures, and the first words of each uniform block
        for (const BindingDesc& d : pl->bindings) {
            const DescriptorBinding& b = d.shader->refl.bindings[d.index];
            if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                const int slot = b.nvn.source == NvnSource::StorageBuffer ? (b.nvn.index >= 0 ? b.nvn.index : storage_slot_for(d.nvn_stage, b.nvn.c0_offset)) : b.nvn.index;
                const bool ssbo = b.nvn.source == NvnSource::StorageBuffer;
                const GState::Bind* bd = (slot >= 0 && slot < 16) ? (ssbo ? &S.ssbo[d.nvn_stage][slot] : &S.ubo[d.nvn_stage][slot]) : nullptr;
                std::string words;
                if (bd && bd->addr && resolve_buffer(bd->addr, 4).ok) {
                    const uint32_t* w = reinterpret_cast<const uint32_t*>(host_ptr(bd->addr));
                    for (uint32_t i = 0; i < 128 && i * 4 < bd->size; ++i) { char t[16]; std::snprintf(t, sizeof t, " %08x", w[i]); words += t; }
                }
                TRACE("    cs b%u %s src %d slot %d c0 %d addr %llx/%llu:%s", d.binding, b.name.c_str(), static_cast<int>(b.nvn.source), slot, b.nvn.c0_offset,
                      bd ? static_cast<unsigned long long>(bd->addr) : 0ull, bd ? static_cast<unsigned long long>(bd->size) : 0ull, words.c_str());
            } else {
                const bool img = d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                const TexResolved r = img ? resolve_image(d) : resolve_texture(d);
                char c[96];
                std::string words;
                if (r.tex && r.tex->mem) {
                    const uint32_t* w = reinterpret_cast<const uint32_t*>(r.tex->mem);
                    for (uint32_t i = 0; i < 64; ++i) { char t[16]; std::snprintf(t, sizeof t, " %08x", w[i]); words += t; }
                }
                TRACE("    cs b%u %s type %d -> %s @%p:%s", d.binding, b.name.c_str(), static_cast<int>(d.type), tex_desc(r.tex, c, sizeof c), r.tex ? static_cast<void*>(r.tex->mem) : nullptr, words.c_str());
            }
        }
    }
    VkPipeline pipe = compute_pipeline(cs, *pl);
    if (!pipe) return;
    vkCmdBindPipeline(F->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    g_bound_pipeline = VK_NULL_HANDLE;
    support_for_draw(*pl, true);
    const BufRef ind = indirect ? held_or_pool(indirect, 12, kOutIndirect, true) : BufRef{};   // arguments a dispatch wrote: from the copy
    g_shadowing = compute_shadows_on();
    g_shadow_uses.clear();
    g_sh_barrier = g_sh_copied = false;
    // NVN2 compute c[0]: the work-group counts at 0x4 (x, y, z; the linearised group index of ee541a5d reads x and y there), the push
    // constant words from 0x10 (control +0xa9), then the resource table, whose descriptor words stay 0: storage buffers are bound at their
    // exact address, so the translated code's base - (c[0] word & ~alignment) offset must see 0 there. Indirect counts are not known here.
    g_cs_c0 = UploadAlloc();
    if (cs->control.nvn2 && std::any_of(cs->refl.bindings.begin(), cs->refl.bindings.end(),
                                        [](const DescriptorBinding& b) { return b.used && b.nvn.source == NvnSource::DriverConstants; })) {
        const VkDeviceSize n = std::max<VkDeviceSize>(0x400, (cs->control.c0_table_size + 0xffu) & ~0xffu);
        g_cs_c0 = F->upload(n, std::max<VkDeviceSize>(device().properties().limits.minUniformBufferOffsetAlignment, 256));
        if (g_cs_c0) {
            std::memset(g_cs_c0.ptr, 0, static_cast<size_t>(n));
            const uint32_t groups[3] = {x, y, z};
            std::memcpy(g_cs_c0.ptr + 4, groups, sizeof groups);
            std::memcpy(g_cs_c0.ptr + 0x10, S.push_cmp, std::min<size_t>(4u * cs->control.c0_push_words, sizeof S.push_cmp));
            if (indirect) log_once("cs-c0-indirect", "indirect dispatch of an NVN2 compute shader that reads c[0]: its work-group counts there read 0");
        }
    }
    bind_sets(*pl, VK_PIPELINE_BIND_POINT_COMPUTE);
    if (g_sh_copied) {
        memory_barrier(F->cb);     // the copies into the shadows before the dispatch reads them
        gstamp("shadow in");
    }
    {
        char cp[96];
        std::snprintf(cp, sizeof cp, "cs %s %ux%ux%u%s", cs->hash.substr(0, 8).c_str(), x, y, z, indirect ? " ind" : "");
        device().cmd_checkpoint(F->cb, cp);
    }
    if (indirect) {
        vkCmdDispatchIndirect(F->cb, ind.buffer, ind.offset);
    } else {
        vkCmdDispatch(F->cb, x, y, z);
        if (gpass_on()) gstamp(("cs " + cs->hash.substr(0, 8)).c_str());
    }
    draw_ring_note_dispatch(*cs, x, y, z, indirect != 0);
    shadows_after_dispatch();
    memory_barrier(F->cb);
    if (g_trace) trace_snap_after_dispatch(*cs, *pl);
    g_cs_c0 = UploadAlloc();
    invalidate_bound_state();
    g_dirty = kAllDirty;
    g_vb_dirty_mask = 0xFFFF;
    counters().dispatches.fetch_add(1, std::memory_order_relaxed);
}

// ---- conditional render ---------------------------------------------------------------------------------------------------------
void note_report(uint64_t address, uint64_t counter) {
    if (!address) return;
    if (g_reports.size() > 65536) g_reports.clear();   // bounded: a forgotten address reads from memory again
    g_reports[address] = counter;
}

// The counter word at a report address; false when it is not known before the GPU runs.
bool report_counter(uint64_t address, uint64_t* out) {
    auto it = g_reports.find(address);
    if (it != g_reports.end()) {
        if (it->second == kReportUnknown) return false;
        *out = it->second;
        return true;
    }
    const PoolHit hit = pool_lookup(address);
    if (!hit.pool || hit.pool->is_virtual || !hit.pool->base || hit.offset + 8 > hit.pool->size) return false;
    std::memcpy(out, host_ptr(address), 8);
    return true;
}

void do_write_memory(uint64_t gpu_addr, const void* data, uint32_t size);

// A counter report whose counter word is a query's result: the placeholder (counter 0, the timestamp), the result over the counter word
// (query < 0: none), and with stamp_after the timestamp again (the feedback query's second result lands on it).
// With [nvn] mirror_capture the GPU reads the pool through its device-local copy, so the result has to land there (the cloud sea and fog
// volumes were not drawn when it went to the pool only). The report is assembled in the copy and then copied to the pool in one piece:
// written into the pool as a placeholder and completed only by the write-back at the end of the submission, the game's memory read
// "0 samples passed, new timestamp" for most of the frame, and a game a frame ahead ([nvn] present_ahead) culled the effects behind those
// queries (flames, smoke and water reflections flickering in and out).
void report_from_query(uint64_t address, uint64_t timestamp, VkQueryPool qp, int query, bool stamp_after) {
    struct { uint64_t counter, timestamp; } rep = {0, timestamp};
    constexpr VkQueryResultFlags kFlags = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;
    g_reports[address] = kReportUnknown;   // the query result lands on the GPU
    if (mirror_capture_on()) {
        const BufRef pool = resolve_buffer(address, sizeof rep);
        const BufRef copy = pool.ok ? mirror_capture_bind(address, sizeof rep, true) : BufRef{};
        if (copy.ok && copy.buffer != pool.buffer) {
            end_pass();
            memory_barrier(F->cb);
            gpu_wrote(address, sizeof rep, kWriteUpdate);
            vkCmdUpdateBuffer(F->cb, copy.buffer, copy.offset, sizeof rep, &rep);
            if (query >= 0) {
                memory_barrier(F->cb);
                vkCmdCopyQueryPoolResults(F->cb, qp, static_cast<uint32_t>(query), 1, copy.buffer, copy.offset, 16, kFlags);
                if (stamp_after) {
                    memory_barrier(F->cb);
                    vkCmdUpdateBuffer(F->cb, copy.buffer, copy.offset + 8, 8, &rep.timestamp);
                }
            }
            memory_barrier(F->cb);
            const VkBufferCopy c{copy.offset, pool.offset, sizeof rep};
            vkCmdCopyBuffer(F->cb, copy.buffer, pool.buffer, 1, &c);
            memory_barrier(F->cb);
            gstamp("report");
            return;
        }
    }
    do_write_memory(address, &rep, sizeof rep);
    if (query < 0) return;
    flush_write_mem_barrier();
    const BufRef b = resolve_buffer(address, sizeof rep);
    if (!b.ok) return;
    vkCmdCopyQueryPoolResults(F->cb, qp, static_cast<uint32_t>(query), 1, b.buffer, b.offset, 16, kFlags);
    memory_barrier(F->cb);
    if (stamp_after) do_write_memory(address + 8, &rep.timestamp, 8);
}

bool conditional_render_enabled(uint32_t mode, uint64_t address) {
    uint64_t a = 0, b = 0;
    if (!report_counter(address, &a) || !report_counter(address + 16, &b)) return true;
    if (mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL) return a == b;
    if (mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL) return a != b;
    return true;
}

// ---- clears -------------------------------------------------------------------------------------------------------------------
VkClearRect full_rect() {
    VkClearRect r = {};
    r.rect = S.nsc ? scissor_rect(S.sc[0]) : VkRect2D{{0, 0}, P.extent};
    r.baseArrayLayer = 0;
    r.layerCount = 1;
    return r;
}

void do_clear_color(const P_ClearColor& p) {
    if (g_trace) {   // the value (as floats for kind 0, else as words) and the rectangle the clear covers (the scissor when one is set)
        char a[96], v[96];
        if (p.kind == 0) {
            float f[4];
            std::memcpy(f, p.value, sizeof f);
            std::snprintf(v, sizeof v, "%g %g %g %g", f[0], f[1], f[2], f[3]);
        } else {
            std::snprintf(v, sizeof v, "%08x %08x %08x %08x", p.value[0], p.value[1], p.value[2], p.value[3]);
        }
        const VkClearRect cr = full_rect();
        TRACE("clear color rt%u mask %x kind %u %s value %s rect %d %d %u %u", p.target, p.mask, p.kind, tex_desc(p.target < 8 ? S.color_tex[p.target].get() : nullptr, a, sizeof a), v,
              cr.rect.offset.x, cr.rect.offset.y, cr.rect.extent.width, cr.rect.extent.height);
    }
    if (!S.render_enable) return;   // render enable gates clears as it gates draws (yuzu DrawManager::Clear)
    if (p.target >= 8 || !S.color_tex[p.target]) return;
    if (p.mask == 0) return;
    if (!ensure_pass()) return;
    if (!P.color_fmt[p.target]) return;
    if (p.mask != 0xf) {
        // vkCmdClearAttachments ignores the colour write mask: a masked clear is a draw with that mask
        if (!g_mask_ready) { g_mask_ok = g_mask.init(device()); g_mask_ready = true; }
        const VkClearRect cr = full_rect();
        if (g_mask_ok && g_mask.record(F->cb, P.color_fmt[p.target], P.ncolor, p.target, p.mask, p.kind, p.value, cr.rect, P.samples)) {
            static bool noted = false;
            if (!noted) { noted = true; NXLOG_I("nvn.replay", "replay: ClearColor with a partial channel mask (first): rt%u mask %x, drawn with that colour write mask", p.target, p.mask); }
            counters().clears.fetch_add(1, std::memory_order_relaxed);
            g_bound_pipeline = VK_NULL_HANDLE;
            invalidate_bound_state();
            g_dirty = kAllDirty;
            g_vb_dirty_mask = 0xFFFF;
            g_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;   // binding a pipeline without these dynamic states invalidates them
            g_restart_valid = false;
            return;
        }
        note_unsupported("ClearColor with a partial channel mask (all channels are cleared)");
    }
    VkClearAttachment ca = {};
    ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ca.colorAttachment = p.target;
    std::memcpy(ca.clearValue.color.uint32, p.value, 16);
    VkClearRect r = full_rect();
    vkCmdClearAttachments(F->cb, 1, &ca, 1, &r);
    counters().clears.fetch_add(1, std::memory_order_relaxed);
}

void do_clear_depth_stencil(const P_ClearDepthStencil& p) {
    TRACE("clear depth/stencil d %g (mask %d) s %d (mask %x)", p.depth, p.depth_mask, p.stencil, p.stencil_mask);
    if (!S.render_enable) return;
    if (!S.depth_tex) return;
    if (!ensure_pass()) return;
    VkClearAttachment ca = {};
    if (p.depth_mask) ca.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (p.stencil_mask && P.has_stencil) ca.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!ca.aspectMask) return;
    if (p.stencil_mask && p.stencil_mask != 0xff && P.has_stencil && !P.depth_ro) {
        // vkCmdClearAttachments has no stencil write mask: a masked stencil clear is a draw with that write mask (depth goes along)
        if (!g_mask_ready) { g_mask_ok = g_mask.init(device()); g_mask_ready = true; }
        const VkClearRect cr = full_rect();
        if (g_mask_ok && g_mask.record_depth_stencil(F->cb, P.color_fmt, P.ncolor, P.depth_fmt, true, P.samples, p.depth_mask != 0, p.depth, p.stencil_mask,
                                                    static_cast<uint32_t>(p.stencil), cr.rect)) {
            static bool noted = false;
            if (!noted) { noted = true; NXLOG_I("nvn.replay", "replay: ClearDepthStencil with a partial stencil mask (first): mask %x, drawn with that stencil write mask", p.stencil_mask); }
            counters().clears.fetch_add(1, std::memory_order_relaxed);
            g_bound_pipeline = VK_NULL_HANDLE;
            invalidate_bound_state();
            g_dirty = kAllDirty;
            g_vb_dirty_mask = 0xFFFF;
            g_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;   // binding a pipeline without these dynamic states invalidates them
            g_restart_valid = false;
            return;
        }
        note_unsupported("ClearDepthStencil with a partial stencil mask (the whole stencil is cleared)");
    }
    ca.clearValue.depthStencil = {p.depth, static_cast<uint32_t>(p.stencil)};
    VkClearRect r = full_rect();
    vkCmdClearAttachments(F->cb, 1, &ca, 1, &r);
    counters().clears.fetch_add(1, std::memory_order_relaxed);
}

// ---- copies ------------------------------------------------------------------------------------------------------------------
struct SubRes { uint32_t level = 0, layer = 0, layers = 1; };
SubRes sub_of(const TextureViewInfo& v, bool has_view) {
    SubRes s;
    if (has_view) {
        if (v.has_levels) s.level = static_cast<uint32_t>(v.min_level);
        if (v.has_layers) { s.layer = static_cast<uint32_t>(v.min_layer); s.layers = 1; }
    }
    return s;
}

void do_clear_texture(const P_ClearTexture& p) {
    Texture* t = p.tex;
    if (!t || !tex_gpu(t)) return;
    if (g_trace) {   // the value as floats (kind 0) or words, like do_clear_color: a colour target cleared outside a draw
        char a[96], v[96];
        if (p.kind == 0) {
            float f[4];
            std::memcpy(f, p.value, sizeof f);
            std::snprintf(v, sizeof v, "%g %g %g %g", f[0], f[1], f[2], f[3]);
        } else {
            std::snprintf(v, sizeof v, "%08x %08x %08x %08x", p.value[0], p.value[1], p.value[2], p.value[3]);
        }
        TRACE("clear texture %s kind %u value %s", tex_desc(p.tex, a, sizeof a), p.kind, v);
    }
    end_pass();
    SubRes s = sub_of(p.view, p.has_view);
    tex_ensure_current(t, *F);
    tex_transition(t, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    tex_mark_gpu_written(t);
    if (p.has_region) {
        const VkExtent3D e = tex_extent(t, s.level);
        if (p.region.x != 0 || p.region.y != 0 || (p.region.w && static_cast<uint32_t>(p.region.w) < e.width) || (p.region.h && static_cast<uint32_t>(p.region.h) < e.height))
            note_unsupported("ClearTexture of a sub-region (the whole level is cleared)");
    }
    VkImageSubresourceRange rng = {tex_aspect(t), s.level, 1, s.layer, p.has_view && p.view.has_layers ? static_cast<uint32_t>(std::max(1, p.view.num_layers)) : tex_layers(t)};
    if (tex_is_depth(t)) {
        VkClearDepthStencilValue v = {};
        std::memcpy(&v.depth, &p.value[0], 4);
        vkCmdClearDepthStencilImage(F->cb, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &rng);
    } else {
        VkClearColorValue v = {};
        std::memcpy(v.uint32, p.value, 16);
        vkCmdClearColorImage(F->cb, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &rng);
    }
    counters().clears.fetch_add(1, std::memory_order_relaxed);
    gstamp("clear texture");
}

void do_copy_buffer_texture(const P_CopyBufTex& p, bool to_texture) {
    Texture* t = p.tex;
    if (t && t->b.target == NVN_TEXTURE_TARGET_3D) {
        static int logged = 0;
        if (logged < 30) {
            ++logged;
            char a[96];
            NXLOG_I("nvn.copy", "%s 3D %s region %s (%d,%d,%d %dx%dx%d) buffer %llx row %lld image %lld", to_texture ? "buffer -> texture" : "texture -> buffer", tex_desc(t, a, sizeof a),
                    p.has_region ? "given" : "whole", p.has_region ? p.region.x : 0, p.has_region ? p.region.y : 0, p.has_region ? p.region.z : 0,
                    p.has_region ? p.region.w : 0, p.has_region ? p.region.h : 0, p.has_region ? p.region.d : 0, static_cast<unsigned long long>(p.address),
                    static_cast<long long>(p.row_stride), static_cast<long long>(p.image_stride));
        }
    }
    if (!t || !tex_gpu(t)) return;
    end_pass();
    if (tex_scale(t) != 1.0f) {
        static int logged = 0;
        if (logged++ < 10) {
            char a[96];
            NXLOG_W(kTag, "%s a scaled render target %s: the copy uses NVN texel coordinates (wrong part of the image)", to_texture ? "buffer copied into" : "copy from",
                    tex_desc(t, a, sizeof a));
        }
    }
    SubRes s = sub_of(p.view, p.has_view);
    const VkExtent3D ext = tex_extent(t, s.level);
    P_CopyRegion r = p.has_region ? p.region : P_CopyRegion{0, 0, 0, static_cast<int32_t>(ext.width), static_cast<int32_t>(ext.height), static_cast<int32_t>(ext.depth)};
    const FormatInfo* fi = t->fmt;
    const uint32_t bw = fi ? fi->block_w : 1, bh = fi ? fi->block_h : 1, bpb = fi ? fi->bytes : 4;
    uint32_t rw = static_cast<uint32_t>(r.w), rh = static_cast<uint32_t>(r.h), rd = std::max<int32_t>(1, r.d);
    VkBufferImageCopy c = {};
    const uint64_t row_blocks = (rw + bw - 1) / bw;
    const uint64_t row_bytes = row_blocks * bpb;
    const uint64_t stride = p.row_stride > 0 ? static_cast<uint64_t>(p.row_stride) : row_bytes;
    c.bufferRowLength = static_cast<uint32_t>(stride / bpb * bw);
    c.bufferImageHeight = p.image_stride > 0 ? static_cast<uint32_t>(static_cast<uint64_t>(p.image_stride) / stride * bh) : 0;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, s.level, s.layer, 1};
    if (tex_is_depth(t)) c.imageSubresource.aspectMask = tex_aspect(t) & VK_IMAGE_ASPECT_DEPTH_BIT;
    c.imageOffset = {r.x, r.y, r.z};
    c.imageExtent = {rw, rh, rd};
    if (t->b.target != NVN_TEXTURE_TARGET_3D && t->layers > 1 && !p.has_view) { c.imageSubresource.layerCount = tex_layers(t); c.imageExtent.depth = 1; }
    if (t->b.target != NVN_TEXTURE_TARGET_3D && rd > 1) { c.imageSubresource.layerCount = rd; c.imageSubresource.baseArrayLayer = s.layer + static_cast<uint32_t>(std::max(0, r.z)); c.imageOffset.z = 0; c.imageExtent.depth = 1; }
    const uint64_t buf_bytes = stride * ((rh + bh - 1) / bh) * rd;
    if (!to_texture) compute_out_guard(p.address, buf_bytes, kOutBufferWrite);     // written: held results there go back first
    BufRef br = to_texture ? held_or_pool(p.address, buf_bytes, kOutBufferRead, true) : resolve_buffer(p.address, buf_bytes);
    c.bufferOffset = br.offset;
    if (to_texture) {
        tex_ensure_current(t, *F);
        tex_transition(t, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        tev(t, "copy_buffer_to_texture", p.address, p.has_view ? 1 : 0);
        for (uint32_t i = 0; i < c.imageSubresource.layerCount; ++i) tex_sparse_before_copy(t, c.imageSubresource.baseArrayLayer + i, s.level);
        tex_mark_gpu_written(t);
        vkCmdCopyBufferToImage(F->cb, br.buffer, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    } else {
        tex_ensure_current(t, *F);
        tex_transition(t, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vkCmdCopyImageToBuffer(F->cb, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, br.buffer, 1, &c);
        gpu_wrote(p.address, stride * ((rh + bh - 1) / bh) * rd, kWriteImageCopy);
    }
    memory_barrier(F->cb);
    counters().copies.fetch_add(1, std::memory_order_relaxed);
    gstamp(to_texture ? "copy buffer->texture" : "copy texture->buffer");
}

void do_copy_texture_texture(const P_CopyTexTex& p) {
    Texture *s = p.src, *d = p.dst;
    if (!s || !d || !tex_gpu(s) || !tex_gpu(d)) return;
    if (g_trace) {   // a colour target written outside a draw is invisible in the draw list otherwise
        char a[96], b[96];
        TRACE("copy texture %s -> %s", tex_desc(s, a, sizeof a), tex_desc(d, b, sizeof b));
    }
    end_pass();
    const SubRes ss = sub_of(p.src_view, p.has_src_view), ds = sub_of(p.dst_view, p.has_dst_view);
    const VkExtent3D se = tex_extent(s, ss.level), de = tex_extent(d, ds.level);
    P_CopyRegion sr = p.has_src_region ? p.src_region : P_CopyRegion{0, 0, 0, static_cast<int32_t>(se.width), static_cast<int32_t>(se.height), static_cast<int32_t>(se.depth)};
    P_CopyRegion dr = p.has_dst_region ? p.dst_region : P_CopyRegion{0, 0, 0, static_cast<int32_t>(de.width), static_cast<int32_t>(de.height), static_cast<int32_t>(de.depth)};
    // scaled images: the NVN regions in image texels (x, y and the size times each image's scale, clipped to the image)
    auto scale_region = [](P_CopyRegion& r, Texture* t, uint32_t level) {
        const float sc = tex_scale(t);
        if (sc == 1.0f) return;
        const VkExtent3D ie = tex_image_extent(t, level);
        const int32_t x1 = std::min<int32_t>(static_cast<int32_t>(std::lround((r.x + r.w) * sc)), static_cast<int32_t>(ie.width));
        const int32_t y1 = std::min<int32_t>(static_cast<int32_t>(std::lround((r.y + r.h) * sc)), static_cast<int32_t>(ie.height));
        r.x = static_cast<int32_t>(std::lround(r.x * sc));
        r.y = static_cast<int32_t>(std::lround(r.y * sc));
        r.w = std::max(0, x1 - r.x);
        r.h = std::max(0, y1 - r.y);
    };
    scale_region(sr, s, ss.level);
    scale_region(dr, d, ds.level);
    if (sr.w <= 0 || sr.h <= 0 || dr.w <= 0 || dr.h <= 0) return;
    tex_ensure_current(s, *F);
    if (s != d) tex_ensure_current(d, *F);
    const bool same = s == d;
    tex_transition(s, *F, same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if (!same) tex_transition(d, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    tev(d, "copy_texture_to_texture", reinterpret_cast<uint64_t>(s), 0);
    for (int32_t i = 0; i < std::max(1, d->b.target != NVN_TEXTURE_TARGET_3D ? dr.d : 1); ++i) tex_sparse_before_copy(d, ds.layer + static_cast<uint32_t>(i), ds.level);
    tex_mark_gpu_written(d);
    const VkImageLayout sl = same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageLayout dl = same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageAspectFlags sa = tex_aspect(s), da = tex_aspect(d);
    const bool same_size = sr.w == dr.w && sr.h == dr.h && sr.d == dr.d;
    const bool same_fmt = tex_vk_format(s) == tex_vk_format(d);
    // MIRROR_X/Y/Z: the 2D engine writes the region reversed along that axis (a blit with the destination corners swapped)
    const uint32_t mirror = p.flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT | NVN_COPY_FLAGS_MIRROR_Z_BIT);
    if (mirror) log_once("copy-mirror", "CopyTextureToTexture with mirror flags: copied as a mirrored blit");
    const bool copy_only = same_size && same_fmt && !mirror;
    {
        char cp[96];
        std::snprintf(cp, sizeof cp, "%s t%u(%ux%u)->t%u(%ux%u)", copy_only ? "copy" : "blit",
                      s ? s->id : 0u, static_cast<unsigned>(sr.w), static_cast<unsigned>(sr.h),
                      d ? d->id : 0u, static_cast<unsigned>(dr.w), static_cast<unsigned>(dr.h));
        device().cmd_checkpoint(F->cb, cp);
    }
    if (copy_only) {
        VkImageCopy c = {};
        c.srcSubresource = {sa, ss.level, ss.layer, 1};
        c.dstSubresource = {da, ds.level, ds.layer, 1};
        c.srcOffset = {sr.x, sr.y, sr.z};
        c.dstOffset = {dr.x, dr.y, dr.z};
        c.extent = {static_cast<uint32_t>(sr.w), static_cast<uint32_t>(sr.h), static_cast<uint32_t>(std::max(1, sr.d))};
        if (s->b.target != NVN_TEXTURE_TARGET_3D && sr.d > 1) { c.srcSubresource.layerCount = c.dstSubresource.layerCount = sr.d; c.extent.depth = 1; c.srcOffset.z = c.dstOffset.z = 0; }
        vkCmdCopyImage(F->cb, tex_image(s), sl, tex_image(d), dl, 1, &c);
    } else {
        VkImageBlit b = {};
        b.srcSubresource = {sa, ss.level, ss.layer, 1};
        b.dstSubresource = {da, ds.level, ds.layer, 1};
        b.srcOffsets[0] = {sr.x, sr.y, sr.z};
        b.srcOffsets[1] = {sr.x + sr.w, sr.y + sr.h, sr.z + std::max(1, sr.d)};
        b.dstOffsets[0] = {dr.x, dr.y, dr.z};
        b.dstOffsets[1] = {dr.x + dr.w, dr.y + dr.h, dr.z + std::max(1, dr.d)};
        if (mirror & NVN_COPY_FLAGS_MIRROR_X_BIT) std::swap(b.dstOffsets[0].x, b.dstOffsets[1].x);
        if (mirror & NVN_COPY_FLAGS_MIRROR_Y_BIT) std::swap(b.dstOffsets[0].y, b.dstOffsets[1].y);
        if (mirror & NVN_COPY_FLAGS_MIRROR_Z_BIT) std::swap(b.dstOffsets[0].z, b.dstOffsets[1].z);
        const bool linear = ((p.flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) || tex_scale(s) != tex_scale(d)) &&
                            !tex_is_depth(s) && !tex_is_depth(d) && !vk_format_is_integer(tex_vk_format(s)) && !vk_format_is_integer(tex_vk_format(d));
        vkCmdBlitImage(F->cb, tex_image(s), sl, tex_image(d), dl, 1, &b, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    draw_ring_note_blit(s, d, static_cast<uint32_t>(sr.w), static_cast<uint32_t>(sr.h),
                        static_cast<uint32_t>(dr.w), static_cast<uint32_t>(dr.h), copy_only);
    memory_barrier(F->cb);
    counters().copies.fetch_add(1, std::memory_order_relaxed);
    gstamp("copy texture");
}

void do_downsample(const P_Downsample& p) {
    Texture *s = p.src, *d = p.dst;
    if (!s || !d || !tex_gpu(s) || !tex_gpu(d)) return;
    end_pass();
    if (tex_is_depth(s)) { note_unsupported("Downsample of a depth texture"); return; }
    tex_transition(s, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    tex_transition(d, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    tex_mark_gpu_written(d);
    VkImageResolve rs = {};
    rs.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rs.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    if (tex_scale(s) != tex_scale(d)) log_once("downsample-scale", "Downsample between a scaled and an unscaled texture: only the common part is resolved");
    const VkExtent3D e = tex_image_extent(d, 0), es = tex_image_extent(s, 0);
    rs.extent = {std::min(e.width, es.width), std::min(e.height, es.height), 1};
    vkCmdResolveImage(F->cb, tex_image(s), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(d), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rs);
    memory_barrier(F->cb);
    counters().copies.fetch_add(1, std::memory_order_relaxed);
    gstamp("downsample");
}

void do_write_memory(uint64_t gpu_addr, const void* data, uint32_t size) {
    // a GPU-ordered write of up to 64 KiB into pool memory (UpdateUniformBuffer, counters, events).
    // Narrow coalesced transfer barriers: avoids TWO full ALL_COMMANDS->ALL_COMMANDS GPU stalls per write.
    const bool had_barrier = end_pass(true);
    if (g_pending_write_mem) {
        const bool overlaps = g_wm_last_size && gpu_addr < g_wm_last_addr + g_wm_last_size && g_wm_last_addr < gpu_addr + size;
        if (overlaps) transfer_post_barrier(F->cb);
    } else if (!had_barrier) {
        transfer_pre_barrier(F->cb);
    }
    tsr_cache_clear();
    BufRef b = resolve_buffer(gpu_addr, size);
    if (!b.ok) return;
    gpu_wrote(gpu_addr, size, kWriteUpdate);
    vkCmdUpdateBuffer(F->cb, b.buffer, b.offset, size, data);
    mirror_capture_update(F->cb, gpu_addr, data, size);   // [nvn] mirror_capture: the same bytes into the copy
    for (uint16_t& m : g_ubo_dirty_mask) m = 0xFFFF;
    if (!mirror_capture_on()) g_dirty |= kDescUbo;
    g_pending_write_mem = true;
    g_wm_last_addr = gpu_addr;
    g_wm_last_size = size;
    gstamp("write memory");
}

// ---- the interpreter ----------------------------------------------------------------------------------------------------------
void apply_bind_program(const P_BindProgram& p) {
    for (int st = 0; st < 6; ++st)
        if (p.stages & (1u << st)) S.stage[st] = p.program ? p.program->stage[st] : nullptr;
    if (g_trace && (p.stages & (1u << kCompute))) {
        g_cs_missing = (p.program && !p.program->stage[kCompute]) ? std::string(p.program->missing) : std::string();
        if (!g_cs_missing.empty()) TRACE("bind program: compute block(s) %s not in the shader cache", g_cs_missing.c_str());
    }
}

// [nvn] same_binds (default on): a bind that sets what is bound already marks nothing dirty. The game binds its program, buffers,
// textures and state objects again before most draws; each such bind made the next draw look up its layout and pipeline and build its
// descriptor sets and vertex buffers again from the same state (notes/rt.md). What the draw's bindings read at replay time (pool entries,
// uniform data) is read at the same time either way; whatever else must rebind (a new pass or submission, GPU writes to copied pages,
// dispatches, DLSS) sets the bits itself.
bool same_binds_on() {
    static const bool on = nxrt::config_get_bool("nvn", "same_binds", true);
    return on;
}
struct RawLast { bool valid = false; uint8_t b[sizeof(P_RawState::bytes)] = {}; };
RawLast g_raw_state[6], g_raw_blend[8];   // the raw state blocks S was last set from (BindColor .. BindMultisample; BindBlend per target)
uint32_t g_raw_attr_n = UINT32_MAX, g_raw_stream_n = UINT32_MAX;
uint32_t g_raw_attr[16] = {};
uint64_t g_raw_stream[16] = {};

bool raw_same(RawLast& l, const uint8_t* bytes) {
    const bool same = l.valid && std::memcmp(l.b, bytes, sizeof l.b) == 0;
    std::memcpy(l.b, bytes, sizeof l.b);
    l.valid = true;
    return same;
}
template <typename T> bool raw_list_same(uint32_t& last_n, T* last, uint32_t count, const uint8_t* blob) {
    const uint32_t n = std::min<uint32_t>(count, 16);
    const bool same = last_n == count && std::memcmp(last, blob, n * sizeof(T)) == 0;
    last_n = count;
    std::memcpy(last, blob, n * sizeof(T));
    return same;
}
void raw_binds_reset() {
    for (RawLast& l : g_raw_state) l.valid = false;
    for (RawLast& l : g_raw_blend) l.valid = false;
    g_raw_attr_n = g_raw_stream_n = UINT32_MAX;
}

// Is the bind op `r` (before it is applied) a no-op on S? Also keeps the raw state blocks up to date: call it for every bind op or never.
bool bind_is_same(const RecordReader& r) {
    switch (r.op()) {
        case Op::BindProgram: {
            const auto& p = r.payload<P_BindProgram>();
            for (int st = 0; st < 6; ++st)
                if ((p.stages & (1u << st)) && S.stage[st] != (p.program ? p.program->stage[st] : nullptr)) return false;
            return true;
        }
        case Op::BindBuffers: {
            const auto& p = r.payload<P_BindBuffers>();
            if (p.kind == kBufXfb) return false;
            const NVNbufferRange* rg = reinterpret_cast<const NVNbufferRange*>(r.blob<P_BindBuffers>());
            for (uint32_t i = 0; i < p.count; ++i) {
                const uint32_t k = p.first + i;
                const GState::Bind* b = nullptr;
                if (p.kind == kBufVertex && k < 16) b = &S.vtx[k];
                else if (p.kind == kBufUniform && p.stage < 6 && k < 16) b = &S.ubo[p.stage][k];
                else if (p.kind == kBufStorage && p.stage < 6 && k < 16) b = &S.ssbo[p.stage][k];
                if (b && (b->addr != rg[i].address || b->size != rg[i].size)) return false;
            }
            return true;
        }
        case Op::BindTextures: case Op::BindImages: {
            const auto& p = r.payload<P_BindTextures>();
            const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_BindTextures>());
            const bool tex = r.op() == Op::BindTextures;
            for (uint32_t i = 0; i < p.count; ++i) {
                if (p.stage >= 6 || p.first + i >= (tex ? 64u : 16u)) continue;
                if ((tex ? S.tex[p.stage][p.first + i] : S.img[p.stage][p.first + i]) != h[i]) return false;
            }
            return true;
        }
        case Op::SetTexturePool: return S.tpool.get() == static_cast<TexturePool*>(r.payload<P_Pool>().pool);
        case Op::SetSamplerPool: return S.spool.get() == static_cast<SamplerPool*>(r.payload<P_Pool>().pool);
        case Op::BindColor: return raw_same(g_raw_state[0], r.payload<P_RawState>().bytes);
        case Op::BindChannelMask: return raw_same(g_raw_state[1], r.payload<P_RawState>().bytes);
        case Op::BindDepthStencil: return raw_same(g_raw_state[2], r.payload<P_RawState>().bytes);
        case Op::BindPolygon: return raw_same(g_raw_state[3], r.payload<P_RawState>().bytes);
        case Op::BindMultisample: return raw_same(g_raw_state[4], r.payload<P_RawState>().bytes);
        case Op::BindBlend: {
            NVNblendState st;
            std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st);
            const int t = decode(&st).target;
            if (t < 0 || t >= 8) return false;
            return raw_same(g_raw_blend[t], r.payload<P_RawState>().bytes);
        }
        case Op::BindVertexAttribs:
            return raw_list_same(g_raw_attr_n, g_raw_attr, r.payload<P_BindVertexAttribs>().count, r.blob<P_BindVertexAttribs>());
        case Op::BindVertexStreams:
            return raw_list_same(g_raw_stream_n, g_raw_stream, r.payload<P_BindVertexStreams>().count, r.blob<P_BindVertexStreams>());
        default: return false;
    }
}

bool is_bind_op(Op op) {
    switch (op) {
        case Op::BindProgram: case Op::BindBuffers: case Op::BindTextures: case Op::BindImages: case Op::SetTexturePool: case Op::SetSamplerPool:
        case Op::BindColor: case Op::BindChannelMask: case Op::BindDepthStencil: case Op::BindPolygon: case Op::BindMultisample: case Op::BindBlend:
        case Op::BindVertexAttribs: case Op::BindVertexStreams:
            return true;
        default: return false;
    }
}

void mark_dirty(const RecordReader& r) {
    if ((same_binds_on() || rs_on()) && is_bind_op(r.op())) {
        ++g_rs.binds;
        if (bind_is_same(r)) {
            ++g_rs.binds_same;
            if (same_binds_on()) return;
        }
    }
    switch (r.op()) {
        case Op::BindProgram: g_dirty |= kPipe | kDesc | kVb; break;
        case Op::BindBuffers: {
            const uint32_t kind = r.payload<P_BindBuffers>().kind;
            g_dirty |= kind == kBufVertex ? kVb : kind == kBufUniform ? kDescUbo : kDesc;
            if (kind != kBufVertex) ++g_bind_gen;   // [nvn] fb_memo: a uniform block can carry a texture handle (HandleInUniform)
            break;
        }
        case Op::BindTextures: case Op::BindImages: case Op::SetTexturePool: case Op::SetSamplerPool:
            g_dirty |= kDesc;
            ++g_bind_gen;                            // [nvn] fb_memo: what a sampler resolves to may differ from now on
            break;
        case Op::BindVertexAttribs: case Op::BindVertexStreams: g_dirty |= kPipe | kVb; break;
        case Op::BindBlend: case Op::BindColor: case Op::BindChannelMask: case Op::BindMultisample: case Op::SetDepthClamp: g_dirty |= kPipe; break;
        case Op::BindPolygon: g_dirty |= kPipe | kDynRaster | kDynBias; break;
        case Op::BindDepthStencil: case Op::SetDepthBounds: g_dirty |= kDynDepth | kDynStencil; break;
        case Op::SetViewports: case Op::SetScissors: case Op::SetDepthRanges: g_dirty |= kDynVp; break;
        case Op::SetStencil: g_dirty |= kDynStencil; break;
        case Op::SetBlendColor: g_dirty |= kDynBlend; break;
        case Op::SetLineWidth: case Op::SetRasterizerDiscard: case Op::SetPrimitiveRestart: g_dirty |= kDynRaster; break;
        case Op::SetPatchSize: g_dirty |= kPipe; break;   // patch control points are part of the pipeline
        case Op::SetPolygonOffsetClamp: g_dirty |= kDynBias; break;
        case Op::SetRenderTargets: g_dirty = kAllDirty; ++g_bind_gen; break;   // [nvn] fb_memo: the attachment set is part of the key
        default: break;
    }
}

// ---- windowed GPU counters (see g_cw) ----------------------------------------------------------------------------------------
bool cw_ready(CounterWindow& w, bool xfb) {
    if (w.pool) return true;
    if (w.tried) return false;
    w.tried = true;
    const gfx::DeviceFeatures& f = device().features();
    if (xfb ? !(f.transform_feedback && f.transform_feedback_queries) : !f.pipeline_statistics_query) return false;
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = xfb ? VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT : VK_QUERY_TYPE_PIPELINE_STATISTICS;
    qi.queryCount = kCounterSlots;
    if (!xfb) qi.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT;
    if (vkCreateQueryPool(vkdev(), &qi, nullptr, &w.pool) != VK_SUCCESS) w.pool = VK_NULL_HANDLE;
    return w.pool != VK_NULL_HANDLE;
}

void cw_end_query(CounterWindow& w, bool xfb) {
    if (w.active < 0) return;
    if (xfb) vkCmdEndQueryIndexedEXT(F->cb, w.pool, static_cast<uint32_t>(w.active), 0);
    else vkCmdEndQuery(F->cb, w.pool, static_cast<uint32_t>(w.active));
}

// Does the window that a ResetCounter of `type` opens (up to the next report or reset of that type in the same list) hold a draw whose
// primitives the CPU cannot count: an indirect draw or a DrawTransformFeedback? The record reader is copied; the list itself is not touched.
bool counter_window_has_gpu_draw(RecordReader look, uint32_t type) {
    while (look.next()) {
        const Op op = look.op();
        if ((op == Op::ReportCounter || op == Op::ResetCounter) && look.payload<P_Counter>().type == type) return false;
        if (op == Op::DrawIndirect || op == Op::DrawXfb) return true;
    }
    return false;
}

void run_list(const CmdList& list, int depth) {
    if (depth > kMaxDepth) { log_once("depth", "CallCommands nesting too deep"); return; }
    ++g_sub_lists;
    const bool sb_on = same_binds_on();
    const bool check_same = sb_on || rs_on();
    for (RecordReader r(list); r.next();) {
        const Op op = r.op();
        if (op >= Op::DrawArrays && out_any()) {
            RsScope rs(kRsOut);
            if (out_lazy()) out_before_op(r);                                               // compute shadows: results back to the pool first
            else if (!compute_out_may_wait(op)) flush_compute_out(kOutOther);
        }
        switch (op) {
            case Op::Nop: break;
            case Op::BindProgram: {
                const auto& p = r.payload<P_BindProgram>();
                bool same = check_same;
                if (check_same) ++g_rs.binds;
                for (int st = 0; st < 6; ++st) {
                    if (p.stages & (1u << st)) {
                        const ShaderInfo* ns = p.program ? p.program->stage[st] : nullptr;
                        if (S.stage[st] != ns) { same = false; S.stage[st] = ns; }
                    }
                }
                if (g_trace && (p.stages & (1u << kCompute))) {
                    g_cs_missing = (p.program && !p.program->stage[kCompute]) ? std::string(p.program->missing) : std::string();
                    if (!g_cs_missing.empty()) TRACE("bind program: compute block(s) %s not in the shader cache", g_cs_missing.c_str());
                }
                if (same) { ++g_rs.binds_same; if (sb_on) break; }
                g_dirty |= kPipe | kDesc | kVb;
                break;
            }
            case Op::BindBuffers: {
                const auto& p = r.payload<P_BindBuffers>();
                const NVNbufferRange* rg = reinterpret_cast<const NVNbufferRange*>(r.blob<P_BindBuffers>());
                bool same = check_same && p.kind != kBufXfb;
                if (check_same) ++g_rs.binds;
                if (p.kind == kBufUniform) {
                    if (p.stage < 6) {
                        GState::Bind* dst = S.ubo[p.stage];
                        uint32_t bm = g_bm_ubo[p.stage];
                        uint16_t dm = g_ubo_dirty_mask[p.stage];
                        for (uint32_t i = 0; i < p.count; ++i) {
                            const uint32_t k = p.first + i;
                            if (k < 16) {
                                const uint64_t a = rg[i].address, sz = rg[i].size;
                                if (dst[k].addr != a || dst[k].size != sz) {
                                    same = false;
                                    dst[k] = {a, sz};
                                    bm = (bm & ~(1u << k)) | (a ? (1u << k) : 0u);
                                    dm |= static_cast<uint16_t>(1u << k);
                                }
                            }
                        }
                        g_bm_ubo[p.stage] = bm;
                        g_ubo_dirty_mask[p.stage] = dm;
                    }
                    if (same) { ++g_rs.binds_same; if (sb_on) break; }
                    g_dirty |= kDescUbo;
                    ++g_ubo_bind_gen;
                } else if (p.kind == kBufVertex) {
                    uint32_t bm = g_bm_vtx;
                    for (uint32_t i = 0; i < p.count; ++i) {
                        const uint32_t k = p.first + i;
                        if (k < 16) {
                            const uint64_t a = rg[i].address, sz = rg[i].size;
                            if (S.vtx[k].addr != a || S.vtx[k].size != sz) {
                                same = false;
                                S.vtx[k] = {a, sz};
                                bm = (bm & ~(1u << k)) | (a ? (1u << k) : 0u);
                                g_vb_dirty_mask |= static_cast<uint16_t>(1u << k);
                            }
                        }
                    }
                    g_bm_vtx = bm;
                    if (same) { ++g_rs.binds_same; if (sb_on) break; }
                    g_dirty |= kVb;
                } else if (p.kind == kBufStorage) {
                    if (p.stage < 6) {
                        GState::Bind* dst = S.ssbo[p.stage];
                        uint32_t bm = g_bm_ssbo[p.stage];
                        for (uint32_t i = 0; i < p.count; ++i) {
                            const uint32_t k = p.first + i;
                            if (k < 16) {
                                const uint64_t a = rg[i].address, sz = rg[i].size;
                                if (dst[k].addr != a || dst[k].size != sz) {
                                    same = false;
                                    dst[k] = {a, sz};
                                    bm = (bm & ~(1u << k)) | (a ? (1u << k) : 0u);
                                }
                            }
                        }
                        g_bm_ssbo[p.stage] = bm;
                    }
                    if (same) { ++g_rs.binds_same; if (sb_on) break; }
                    g_dirty |= kDesc;
                    ++g_bind_gen;
                } else if (p.kind == kBufXfb) {
                    for (uint32_t i = 0; i < p.count; ++i) {
                        const uint32_t k = p.first + i;
                        if (k < 4) S.xfb[k] = {rg[i].address, rg[i].size};
                    }
                    g_dirty |= kDesc;
                    ++g_bind_gen;
                }
                break;
            }
            case Op::BindTextures: {
                const auto& p = r.payload<P_BindTextures>();
                const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_BindTextures>());
                bool same = check_same;
                if (check_same) ++g_rs.binds;
                if (p.stage < 6) {
                    uint64_t* dst = S.tex[p.stage];
                    for (uint32_t i = 0; i < p.count; ++i) {
                        const uint32_t k = p.first + i;
                        if (k < 64 && dst[k] != h[i]) { same = false; dst[k] = h[i]; }
                    }
                }
                if (same) { ++g_rs.binds_same; if (sb_on) break; }
                g_dirty |= kDesc;
                ++g_bind_gen;
                break;
            }
            case Op::BindImages: {
                const auto& p = r.payload<P_BindTextures>();
                const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_BindTextures>());
                bool same = check_same;
                if (check_same) ++g_rs.binds;
                if (p.stage < 6) {
                    uint64_t* dst = S.img[p.stage];
                    for (uint32_t i = 0; i < p.count; ++i) {
                        const uint32_t k = p.first + i;
                        if (k < 16 && dst[k] != h[i]) { same = false; dst[k] = h[i]; }
                    }
                }
                if (same) { ++g_rs.binds_same; if (sb_on) break; }
                g_dirty |= kDesc;
                ++g_bind_gen;
                break;
            }
            case Op::BindVertexAttribs: {
                const auto& p = r.payload<P_BindVertexAttribs>();
                const uint8_t* bl = r.blob<P_BindVertexAttribs>();
                if (check_same) {
                    ++g_rs.binds;
                    if (raw_list_same(g_raw_attr_n, g_raw_attr, p.count, bl)) { ++g_rs.binds_same; if (sb_on) break; }
                }
                g_dirty |= kPipe | kVb;
                const NVNvertexAttribState* a = reinterpret_cast<const NVNvertexAttribState*>(bl);
                S.attr_count = std::min<uint32_t>(p.count, 16);
                for (uint32_t i = 0; i < S.attr_count; ++i) S.attr[i] = decode(&a[i]);
                break;
            }
            case Op::BindVertexStreams: {
                const auto& p = r.payload<P_BindVertexStreams>();
                const uint8_t* bl = r.blob<P_BindVertexStreams>();
                if (check_same) {
                    ++g_rs.binds;
                    if (raw_list_same(g_raw_stream_n, g_raw_stream, p.count, bl)) { ++g_rs.binds_same; if (sb_on) break; }
                }
                g_dirty |= kPipe | kVb;
                const NVNvertexStreamState* a = reinterpret_cast<const NVNvertexStreamState*>(bl);
                S.stream_count = std::min<uint32_t>(p.count, 16);
                for (uint32_t i = 0; i < S.stream_count; ++i) S.stream[i] = decode(&a[i]);
                break;
            }
            case Op::BindBlend: {
                NVNblendState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st);
                BlendInfo b = decode(&st);
                if (check_same) {
                    ++g_rs.binds;
                    if (b.target >= 0 && b.target < 8 && raw_same(g_raw_blend[b.target], r.payload<P_RawState>().bytes)) {
                        ++g_rs.binds_same;
                        if (sb_on) break;
                    }
                }
                g_dirty |= kPipe;
                if (b.target >= 0 && b.target < 8) S.blend[b.target] = b;
                break;
            }
            case Op::BindColor: {
                if (check_same) { ++g_rs.binds; if (raw_same(g_raw_state[0], r.payload<P_RawState>().bytes)) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kPipe;
                NVNcolorState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.color = decode(&st);
                break;
            }
            case Op::BindChannelMask: {
                if (check_same) { ++g_rs.binds; if (raw_same(g_raw_state[1], r.payload<P_RawState>().bytes)) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kPipe;
                NVNchannelMaskState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.chmask = decode(&st);
                break;
            }
            case Op::BindDepthStencil: {
                if (check_same) { ++g_rs.binds; if (raw_same(g_raw_state[2], r.payload<P_RawState>().bytes)) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDynDepth | kDynStencil;
                NVNdepthStencilState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.ds = decode(&st);
                break;
            }
            case Op::BindPolygon: {
                if (check_same) { ++g_rs.binds; if (raw_same(g_raw_state[3], r.payload<P_RawState>().bytes)) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kPipe | kDynRaster | kDynBias;
                NVNpolygonState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.poly = decode(&st);
                break;
            }
            case Op::BindMultisample: {
                if (check_same) { ++g_rs.binds; if (raw_same(g_raw_state[4], r.payload<P_RawState>().bytes)) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kPipe;
                NVNmultisampleState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.ms = decode(&st);
                break;
            }
            case Op::SetTexturePool: {
                TexturePool* tp = static_cast<TexturePool*>(r.payload<P_Pool>().pool);
                if (check_same) { ++g_rs.binds; if (S.tpool.get() == tp) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDesc; ++g_bind_gen;
                S.tpool = Ref<TexturePool>::share(tp);
                break;
            }
            case Op::SetSamplerPool: {
                SamplerPool* sp = static_cast<SamplerPool*>(r.payload<P_Pool>().pool);
                if (check_same) { ++g_rs.binds; if (S.spool.get() == sp) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDesc; ++g_bind_gen;
                S.spool = Ref<SamplerPool>::share(sp);
                break;
            }
            case Op::SetViewports: {
                const auto& p = r.payload<P_Rects>();
                const float* v = reinterpret_cast<const float*>(r.blob<P_Rects>());
                bool same = check_same;
                const uint32_t new_n = std::max(S.nvp, std::min<uint32_t>(p.first + p.count, 16));
                if (new_n != S.nvp) { same = false; S.nvp = new_n; }
                for (uint32_t i = 0; i < p.count && p.first + i < 16; ++i) {
                    if (std::memcmp(S.vp[p.first + i], v + 4 * i, 16) != 0) {
                        same = false;
                        std::memcpy(S.vp[p.first + i], v + 4 * i, 16);
                    }
                }
                if (check_same) { ++g_rs.binds; if (same) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDynVp;
                break;
            }
            case Op::SetScissors: {
                const auto& p = r.payload<P_Rects>();
                const int32_t* v = reinterpret_cast<const int32_t*>(r.blob<P_Rects>());
                bool same = check_same;
                const uint32_t new_n = std::max(S.nsc, std::min<uint32_t>(p.first + p.count, 16));
                if (new_n != S.nsc) { same = false; S.nsc = new_n; }
                for (uint32_t i = 0; i < p.count && p.first + i < 16; ++i) {
                    if (std::memcmp(S.sc[p.first + i], v + 4 * i, 16) != 0) {
                        same = false;
                        std::memcpy(S.sc[p.first + i], v + 4 * i, 16);
                    }
                }
                if (check_same) { ++g_rs.binds; if (same) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDynVp;
                break;
            }
            case Op::SetDepthRanges: {
                const auto& p = r.payload<P_DepthRanges>();
                const float* v = reinterpret_cast<const float*>(r.blob<P_DepthRanges>());
                bool same = check_same;
                for (uint32_t i = 0; i < p.count && p.first + i < 16; ++i) {
                    if (std::memcmp(S.dr[p.first + i], v + 2 * i, 8) != 0) {
                        same = false;
                        std::memcpy(S.dr[p.first + i], v + 2 * i, 8);
                    }
                }
                if (check_same) { ++g_rs.binds; if (same) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDynVp;
                break;
            }
            case Op::SetDepthBounds: {
                const auto& p = r.payload<P_DepthBounds>();
                const bool en = p.enable != 0;
                const bool same = check_same && S.depth_bounds == en && S.db_near == p.near_ && S.db_far == p.far_;
                if (check_same) { ++g_rs.binds; if (same) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDynDepth | kDynStencil; S.depth_bounds = en; S.db_near = p.near_; S.db_far = p.far_;
                break;
            }
            case Op::SetStencil: {
                const auto& p = r.payload<P_Stencil>();
                bool same = check_same;
                for (int f = 0; f < 2; ++f) {
                    const bool hit = p.face == NVN_FACE_FRONT_AND_BACK || (f == 0 && p.face == NVN_FACE_FRONT) || (f == 1 && p.face == NVN_FACE_BACK);
                    if (!hit) continue;
                    uint32_t& dst = (p.what == 0) ? S.stencil_ref[f] : (p.what == 1) ? S.stencil_vmask[f] : S.stencil_wmask[f];
                    if (dst != p.value) { same = false; dst = p.value; }
                }
                if (check_same) { ++g_rs.binds; if (same) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDynStencil;
                break;
            }
            case Op::SetBlendColor: {
                const float* c = r.payload<P_BlendColor>().c;
                const bool same = check_same && std::memcmp(S.blend_color, c, 16) == 0;
                if (check_same) { ++g_rs.binds; if (same) { ++g_rs.binds_same; if (sb_on) break; } }
                g_dirty |= kDynBlend; std::memcpy(S.blend_color, c, 16);
                break;
            }
            case Op::SetPointSize: S.point_size = r.payload<P_Float>().v; break;
            case Op::SetLineWidth: {
                const float v = r.payload<P_Float>().v;
                if (sb_on && S.line_width == v) break;
                g_dirty |= kDynRaster; S.line_width = v;
                break;
            }
            case Op::SetAlphaRef: S.alpha_ref = r.payload<P_Float>().v; break;
            case Op::SetPolygonOffsetClamp: {
                const auto& p = r.payload<P_PolygonOffsetClamp>();
                if (sb_on && S.bias_factor == p.factor && S.bias_units == p.units && S.bias_clamp == p.clamp) break;
                g_dirty |= kDynBias; S.bias_factor = p.factor; S.bias_units = p.units; S.bias_clamp = p.clamp;
                break;
            }
            case Op::SetSampleMask: S.sample_mask = r.payload<P_U32>().v; break;
            case Op::SetRasterizerDiscard: {
                const bool v = r.payload<P_U32>().v != 0;
                if (sb_on && S.rasterizer_discard == v) break;
                g_dirty |= kDynRaster; S.rasterizer_discard = v;
                break;
            }
            case Op::SetDepthClamp: {
                const bool v = r.payload<P_U32>().v != 0;
                if (sb_on && S.depth_clamp == v) break;
                g_dirty |= kPipe; S.depth_clamp = v;
                break;
            }
            case Op::SetPrimitiveRestart: {
                const auto& p = r.payload<P_PrimitiveRestart>();
                const bool en = p.enable != 0;
                if (sb_on && S.prim_restart == en && S.restart_index == p.index) break;
                g_dirty |= kDynRaster; S.prim_restart = en; S.restart_index = p.index;
                break;
            }
            case Op::SetPatchSize: S.patch_size = r.payload<P_U32>().v; g_dirty |= kPipe; break;   // the patch size is in the pipeline key
            case Op::SetTessLevels: {
                const auto& p = r.payload<P_TessLevels>();
                if (p.outer) std::memcpy(S.tess_outer, p.v, sizeof S.tess_outer);
                else std::memcpy(S.tess_inner, p.v, sizeof S.tess_inner);
                break;
            }
            case Op::SetRenderEnable:
                if (r.size() >= sizeof(RecHeader) + sizeof(P_RenderEnableCond)) {
                    const auto& p = r.payload<P_RenderEnableCond>();
                    S.render_enable = conditional_render_enabled(p.mode, p.address);
                } else {
                    S.render_enable = r.payload<P_U32>().v != 0;
                }
                break;
            case Op::SetCoverageModulationTable: case Op::SetConservativeRaster: break;
            case Op::SetRenderTargets: {
                g_dirty = kAllDirty;
                g_vb_dirty_mask = 0xFFFF;
                ++g_bind_gen;
                P.targets_clean = false;
                const auto& p = r.payload<P_RenderTargets>();
                bool color_changed = (S.ncolor != p.count);
                for (uint32_t i = 0; !color_changed && i < p.count && i < 8; ++i) {
                    const TextureViewInfo& v = p.color[i].view;
                    const bool hv = p.color[i].has_view != 0;
                    const uint32_t lv = hv && v.has_levels ? static_cast<uint32_t>(v.min_level) : 0;
                    const uint32_t la = hv && v.has_layers ? static_cast<uint32_t>(v.min_layer) : 0;
                    if (S.color_tex[i].get() != p.color[i].tex || S.color_level[i] != lv || S.color_layer[i] != la)
                        color_changed = true;
                }
                if (color_changed) g_fb_valid.clear();
                S.ncolor = p.count;
                for (uint32_t i = 0; i < 8; ++i) {
                    S.color_tex[i] = i < p.count ? Ref<Texture>::share(p.color[i].tex) : Ref<Texture>();
                    if (i < p.count) { S.color_view[i] = p.color[i].view; S.color_has_view[i] = p.color[i].has_view != 0; }
                    else S.color_has_view[i] = false;
                    const TextureViewInfo& v = S.color_view[i];
                    const bool hv = S.color_has_view[i];
                    S.color_level[i] = hv && v.has_levels ? static_cast<uint32_t>(v.min_level) : 0;
                    S.color_layer[i] = hv && v.has_layers ? static_cast<uint32_t>(v.min_layer) : 0;
                    S.color_layers[i] = hv && v.has_layers ? static_cast<uint32_t>(v.num_layers) : (S.color_tex[i] ? static_cast<uint32_t>(S.color_tex[i]->layers) : 0u);
                }
                S.depth_tex = Ref<Texture>::share(p.depth.tex);
                S.depth_view = p.depth.view;
                S.depth_has_view = p.depth.has_view != 0;
                break;
            }
            case Op::DrawArrays: do_draw_arrays(r.payload<P_DrawArrays>()); break;
            case Op::DrawElements: do_draw_elements(r.payload<P_DrawElements>()); break;
            case Op::DrawIndirect: do_draw_indirect(r.payload<P_DrawIndirect>()); break;
            case Op::DrawXfb: do_draw_xfb(r.payload<P_DrawXfb>()); break;
            case Op::PushDebugGroup: {
                static const bool emit_dbg_labels = nxrt::config_get_bool("nvn", "debug_labels", device().validation_enabled());
                if (emit_dbg_labels && device().features().debug_labels && vkCmdBeginDebugUtilsLabelEXT) {
                    VkDebugUtilsLabelEXT l = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
                    l.pLabelName = reinterpret_cast<const char*>(r.blob<P_DebugGroup>());
                    vkCmdBeginDebugUtilsLabelEXT(F->cb, &l);
                    ++g_dbg_depth;
                }
                break;
            }
            case Op::PopDebugGroup:
                if (g_dbg_depth > 0 && vkCmdEndDebugUtilsLabelEXT) { vkCmdEndDebugUtilsLabelEXT(F->cb); --g_dbg_depth; }   // a pop whose push was in an earlier submission has nothing to close
                break;
            case Op::DrawTexture: note_unsupported("DrawTexture"); break;
            case Op::ClearColor: g_fb_valid.clear(); do_clear_color(r.payload<P_ClearColor>()); break;
            case Op::ClearDepthStencil: do_clear_depth_stencil(r.payload<P_ClearDepthStencil>()); break;
            case Op::ClearTexture: g_fb_valid.clear(); do_clear_texture(r.payload<P_ClearTexture>()); break;
            case Op::ClearBuffer: {
                const auto& p = r.payload<P_ClearBuffer>();
                end_pass();
                tsr_cache_clear();
                BufRef b = resolve_buffer(p.address, p.size);
                if (b.ok) vkCmdFillBuffer(F->cb, b.buffer, b.offset, p.size & ~3ull, p.value);
                if (b.ok) gpu_wrote(p.address, p.size, kWriteFill);
                // [nvn] trace: where a fill / copy lands. The light list's vertex shader multiplies its cube by the block bound
                // at slot 4 (HANDOVER 42.15); if the game updates that block on the GPU, this is the line that says so.
                if (g_trace) TRACE("fill buffer %llx size %llu value %x%s", static_cast<unsigned long long>(p.address),
                                   static_cast<unsigned long long>(p.size), p.value, b.ok ? "" : " (not resolvable)");
                memory_barrier(F->cb);
                counters().clears.fetch_add(1);
                gstamp("fill buffer");
                break;
            }
            case Op::CopyBufferToBuffer: {
                const auto& p = r.payload<P_CopyBufferToBuffer>();
                end_pass();
                tsr_cache_clear();
                BufRef s = held_or_pool(p.src, p.size, kOutBufferRead, true), d = resolve_buffer(p.dst, p.size);
                if (s.ok && d.ok) {
                    VkBufferCopy c = {s.offset, d.offset, p.size};
                    vkCmdCopyBuffer(F->cb, s.buffer, d.buffer, 1, &c);
                    gpu_wrote(p.dst, p.size, kWriteCopy);
                }
                if (g_trace) TRACE("copy buffer %llx -> %llx size %llu%s", static_cast<unsigned long long>(p.src),
                                   static_cast<unsigned long long>(p.dst), static_cast<unsigned long long>(p.size),
                                   s.ok && d.ok ? "" : " (not resolvable)");
                memory_barrier(F->cb);
                counters().copies.fetch_add(1);
                gstamp("copy buffer");
                break;
            }
            case Op::CopyBufferToTexture: g_fb_valid.clear(); do_copy_buffer_texture(r.payload<P_CopyBufTex>(), true); break;
            case Op::CopyTextureToBuffer: do_copy_buffer_texture(r.payload<P_CopyBufTex>(), false); break;
            case Op::CopyTextureToTexture: g_fb_valid.clear(); do_copy_texture_texture(r.payload<P_CopyTexTex>()); break;
            case Op::Downsample: g_fb_valid.clear(); do_downsample(r.payload<P_Downsample>()); break;
            case Op::UpdateUniformBuffer: {
                const auto& p = r.payload<P_UpdateUniformBuffer>();
                const uint8_t* data = r.blob<P_UpdateUniformBuffer>();
                const uint32_t n = static_cast<uint32_t>(p.size & ~3ull);
                if (n) do_write_memory(p.address + static_cast<uint64_t>(p.offset), data, n);
                break;
            }
            case Op::PushConstants: {
                const auto& p = r.payload<P_PushConstants>();
                const uint8_t* src = p.address ? cpu_view(p.address, p.size) : r.blob<P_PushConstants>();
                if (!src) {
                    log_once("push-deferred", "a Deferred push constant reads memory the GPU has not finished writing: skipped");
                    break;
                }
                const uint32_t off = std::min<uint32_t>(p.offset, 256), n = std::min<uint32_t>(p.size, 256 - off);
                if (g_trace) {
                    std::string words;
                    for (uint32_t i = 0; i + 4 <= n; i += 4) { uint32_t v; std::memcpy(&v, src + i, 4); char t[12]; std::snprintf(t, sizeof t, " %08x", v); words += t; }
                    TRACE("push constants %s slot %u offset %u size %u:%s", p.compute ? "compute" : "graphics", p.slot, p.offset, p.size, words.c_str());
                }
                if (p.compute) {
                    std::memcpy(S.push_cmp + 256 * (p.slot & 7) + off, src, n);
                } else {
                    std::memcpy(S.push_gfx[p.slot & 7] + off, src, n);
                    g_push_buf[p.slot & 7] = UploadAlloc();
                }
                g_dirty |= kDesc;
                break;
            }
            case Op::Dispatch: { RsScope rs(kRsDispatch); ++g_rs.dispatches; g_fb_valid.clear(); const auto& p = r.payload<P_Dispatch>(); do_dispatch(p.x, p.y, p.z, 0); break; }
            case Op::DispatchIndirect: { RsScope rs(kRsDispatch); ++g_rs.dispatches; g_fb_valid.clear(); do_dispatch(0, 0, 0, r.payload<P_DispatchIndirect>().address); break; }
            case Op::ReportCounter: {
                const auto& p = r.payload<P_Counter>();
                struct { uint64_t counter, timestamp; } rep = {1, timestamp_ns()};   // counters other than the ones handled below are reported as non-zero
                if (p.type == NVN_COUNTER_TYPE_SAMPLES_PASSED && occ_ready()) {
                    end_pass();
                    if (g_occ_active >= 0) {
                        vkCmdEndQuery(F->cb, g_occ_pool, static_cast<uint32_t>(g_occ_active));
                        g_occ_last = g_occ_active;
                        g_occ_active = -1;
                    }
                    if (p.address) report_from_query(p.address, rep.timestamp, g_occ_pool, g_occ_last, false);
                    break;
                }
                if (p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED || p.type == NVN_COUNTER_TYPE_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) {
                    CounterWindow& cw = g_cw[p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED ? 0 : 1];
                    if (cw.active >= 0) {   // the GPU counted this window (it holds indirect or feedback draws): the report is the query's result
                        end_pass();
                        cw_end_query(cw, &cw == &g_cw[1]);
                        cw.last = cw.active;
                        cw.active = -1;
                        // the feedback query has two results (written, needed): the second lands on the timestamp word, written again after it
                        if (p.address) report_from_query(p.address, rep.timestamp, cw.pool, cw.last, &cw == &g_cw[1]);
                        break;
                    }
                    rep.counter = p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED ? g_prims_generated : g_prims_xfb;
                    note_report(p.address, rep.counter);
                    if (p.address) do_write_memory(p.address, &rep, sizeof rep);
                    break;
                }
                // TIMESTAMP_TOP is a timestamp too (taken when the command reaches the top of the pipe): its counter word is 0 like TIMESTAMP's
                const bool stamp = p.type == NVN_COUNTER_TYPE_TIMESTAMP || p.type == NVN_COUNTER_TYPE_TIMESTAMP_TOP;
                if (!stamp) note_unsupported("statistics counters other than samples passed, primitives generated and feedback primitives written are reported as non-zero");
                if (stamp) rep.counter = 0;
                note_report(p.address, rep.counter);
                if (p.address) do_write_memory(p.address, &rep, sizeof rep);
                break;
            }
            case Op::ReportValue: {
                const auto& p = r.payload<P_ReportValue>();
                struct { uint64_t counter, timestamp; } rep = {p.value, timestamp_ns()};
                note_report(p.address, rep.counter);
                if (p.address) do_write_memory(p.address, &rep, sizeof rep);
                break;
            }
            case Op::ResetCounter: {
                const auto& p = r.payload<P_Counter>();
                if (p.type == NVN_COUNTER_TYPE_SAMPLES_PASSED && occ_ready()) {
                    end_pass();
                    if (g_occ_active >= 0) vkCmdEndQuery(F->cb, g_occ_pool, static_cast<uint32_t>(g_occ_active));
                    const uint32_t slot = g_occ_next++ % kOccSlots;
                    vkCmdResetQueryPool(F->cb, g_occ_pool, slot, 1);
                    vkCmdBeginQuery(F->cb, g_occ_pool, slot, device().features().occlusion_query_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
                    g_occ_active = static_cast<int32_t>(slot);
                    g_occ_last = -1;
                } else if (p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED || p.type == NVN_COUNTER_TYPE_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) {
                    const bool gen = p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED;
                    (gen ? g_prims_generated : g_prims_xfb) = 0;
                    CounterWindow& cw = g_cw[gen ? 0 : 1];
                    if (cw.active >= 0) { end_pass(); cw_end_query(cw, !gen); cw.active = -1; }   // an unreported window is over
                    if (counter_window_has_gpu_draw(r, p.type) && cw_ready(cw, !gen)) {
                        static bool noted = false;
                        if (!noted) { noted = true; NXLOG_I("nvn.replay", "replay: primitive counter %u counted by a GPU query (first): its window holds an indirect or feedback draw", p.type); }
                        end_pass();
                        const uint32_t slot = cw.next++ % kCounterSlots;
                        vkCmdResetQueryPool(F->cb, cw.pool, slot, 1);
                        if (gen) vkCmdBeginQuery(F->cb, cw.pool, slot, 0);
                        else vkCmdBeginQueryIndexedEXT(F->cb, cw.pool, slot, 0, 0);
                        cw.active = static_cast<int32_t>(slot);
                    }
                }
                break;
            }
            case Op::Barrier: {
                const auto& pb = r.payload<P_Barrier>();
                TRACE("barrier bits %x", pb.bits);
                g_fb_valid.clear();
                // With VK_EXT_attachment_feedback_loop_layout active, an NVN tiled-fragment / texture-invalidate barrier
                // (e.g. 0x14 = ORDER_FRAGMENTS_TILED | INVALIDATE_TEXTURE) only needs a narrow color-to-sampled barrier
                // between dynamic rendering passes (no FbClone copy, no ALL_COMMANDS stall, and spec-valid without
                // VK_KHR_dynamic_rendering_local_read).
                if (P.open && P.color_feedback != 0 && !P.depth_ro && P.color_general == 0 &&
                    (pb.bits & ~(0x01u | 0x02u | 0x04u | 0x10u)) == 0) {
                    Texture* fb_color[8] = {};
                    const uint32_t fb_n = std::min<uint32_t>(P.ncolor, 8u);
                    for (uint32_t i = 0; i < fb_n; ++i) fb_color[i] = P.color[i].tex;
                    end_pass(true);
                    // Only clear the current color attachments covered by the narrow color feedback barrier below;
                    // preserve P.depth and any earlier passes' unflushed entries in g_pending.
                    g_pending.erase(std::remove_if(g_pending.begin(), g_pending.end(), [&](const PendingWrite& w) {
                        for (uint32_t i = 0; i < fb_n; ++i) if (fb_color[i] && same_image(w.tex.get(), fb_color[i])) return true;
                        return false;
                    }), g_pending.end());
                    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                    mb.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                    mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    mb.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.memoryBarrierCount = 1;
                    di.pMemoryBarriers = &mb;
                    vkCmdPipelineBarrier2(F->cb, &di);
                    break;
                }
                if (P.open) {
                    if (barrier_keeps_pass()) {
                        memory_barrier_in_pass(F->cb);
                        g_pass_unsafe = true;
                    } else {
                        end_pass();
                    }
                } else {
                    memory_barrier(F->cb);
                    g_pending.clear();
                    g_pending_write_mem = false;
                    g_wm_last_addr = g_wm_last_size = 0;
                }
                break;
            }
            case Op::FenceSync: {
                const auto& p = r.payload<P_Sync>();
                if (p.sync) { p.sync->serial.store(F->serial); p.sync->armed.store(true); }
                break;
            }
            case Op::WaitSync: break;
            case Op::SignalEvent: {
                const auto& p = r.payload<P_Event>();
                if (p.event && p.event->pool && p.event->pool->base) {
                    Event* ev = p.event;
                    uint32_t v = p.value;
                    if (p.mode != NVN_EVENT_SIGNAL_MODE_WRITE) {
                        // ADD: the GPU cannot add into memory; the sum is computed here from the value after the signals recorded so far
                        uint32_t base = 0;
                        if (ev->shadow_valid.load()) base = ev->shadow.load();
                        else std::memcpy(&base, ev->pool->base + ev->offset, 4);
                        v = base + p.value;
                    }
                    ev->shadow.store(v);
                    ev->shadow_valid.store(true);
                    do_write_memory(ev->pool->gpu_base + static_cast<uint64_t>(ev->offset), &v, 4);
                }
                break;
            }
            case Op::WaitEvent: note_unsupported("GPU-side WaitEvent"); break;
            case Op::CallCommands: {
                const auto& p = r.payload<P_CallCommands>();
                const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_CallCommands>());
                for (uint32_t i = 0; i < p.count; ++i) {
                    Ref<CmdList> l = handle_lookup(h[i]);
                    if (l) run_list(*l, depth + 1);
                    else log_once("missing-handle", "CallCommands of an unknown handle skipped");
                }
                break;
            }
            case Op::DiscardColor: case Op::DiscardDepthStencil: break;
            case Op::ResolveDepth: note_unsupported("ResolveDepthBuffer"); break;
            case Op::EvaluateDlss:                                                    // the host upscaler (api_dlss_replay.cpp) binds its own compute state
                end_pass();
                dlss_replay(*F, r.payload<P_EvaluateDlss>().eval);
                g_dirty = kAllDirty;
                gstamp("dlss");
                break;
            case Op::BeginXfb: case Op::EndXfb: case Op::PauseXfb: case Op::ResumeXfb: {
                const auto& p = r.payload<P_Counter>();
                static std::unordered_map<std::string, bool> seen;
                const std::string key = std::string(op_name(r.op())) + (S.stage[kVertex] ? S.stage[kVertex]->hash : "-");
                if (seen.emplace(key, true).second)
                    NXLOG_I("nvn.xfb", "%s control %llx vs %s xfb buffer0 %llx/%llu", op_name(r.op()), static_cast<unsigned long long>(p.address),
                            S.stage[kVertex] ? S.stage[kVertex]->hash.c_str() : "-", static_cast<unsigned long long>(S.xfb[0].addr),
                            static_cast<unsigned long long>(S.xfb[0].size));
                TRACE("%s control %llx", op_name(r.op()), static_cast<unsigned long long>(p.address));
                if (!device().features().transform_feedback) {
                    note_unsupported("transform feedback (the device has no VK_EXT_transform_feedback)");
                    break;
                }
                if (r.op() == Op::BeginXfb) {
                    xfb_end_in_pass();    // a section still open in the pass ends where the new one starts
                    auto sit = g_xfb_slot_of.find(p.address);
                    if (sit == g_xfb_slot_of.end()) {
                        if (g_xfb_slot_next >= kXfbSlots) {   // all counters in use: start over (a DrawTransformFeedback of an older control buffer then draws nothing)
                            g_xfb_slot_of.clear();
                            g_xfb_slot_next = 0;
                        }
                        sit = g_xfb_slot_of.emplace(p.address, g_xfb_slot_next++).first;
                    }
                    g_xfb_cur = sit->second;
                    g_xfb_slot_written[g_xfb_cur] = false;
                    g_xfb_active = true;
                    g_xfb_have_counter = false;
                } else if (r.op() == Op::EndXfb) {
                    xfb_end_in_pass();
                    g_xfb_active = false;
                    g_xfb_have_counter = false;
                }
                break;
            }
            default: note_unsupported(op_name(r.op())); break;
        }
    }
}

}  // namespace

// [display] unpaced_loading (default on): the limiter stops waiting while the game loads, since its loading code advances once per
// presented frame. Loading = presented frames with fewer than 16 draws (boot's black screen draws 3: exposure, bloom, tone map; the
// loading screen a few more) AND the game still stalling: a present gap over 100 ms in the last 1.5 s. A light screen that presents
// steadily (the title screen, a menu) is paced again within 1.5 s, so it does not run at 1000 fps; so is any frame with 16+ draws.
void note_frame_for_pacing() {
    static const bool on = nxrt::config_get_bool("display", "unpaced_loading", true);
    if (!on) return;
    using clk = std::chrono::steady_clock;
    static uint64_t last_p = 0, last_d = 0;
    static int low = 0;
    static clk::time_point last_present{}, last_stall{};
    const uint64_t p = counters().presents.load(std::memory_order_relaxed);
    if (p == last_p) return;
    const auto now = clk::now();
    const uint64_t d = counters().draws.load(std::memory_order_relaxed) + counters().draws_skipped.load(std::memory_order_relaxed);
    const uint64_t per = (d - last_d) / (p - last_p);
    last_p = p;
    last_d = d;
    if (last_present == clk::time_point{} || now - last_present > std::chrono::milliseconds(100)) last_stall = now;
    last_present = now;
    low = per < 16 ? low + 1 : 0;
    const bool want = low >= 3 && now - last_stall < std::chrono::milliseconds(1500);
    if (want != nxrt::frame_limiter_unpaced()) {
        nxrt::frame_limiter_set_unpaced(want);
        ft::mark(want ? "pace.loading" : "pace.playing", static_cast<uint32_t>(per));
        NXLOG_I(kTag, "pacing %s (present %llu, %llu draws)", want ? "off: loading" : "on", static_cast<unsigned long long>(p),
                static_cast<unsigned long long>(per));
    }
    static bool full = false;
    if (!full && per >= 16) {
        full = true;
        NXLOG_I(kTag, "first frame with 16+ draws (present %llu, %llu draws)", static_cast<unsigned long long>(p), static_cast<unsigned long long>(per));
    }
}

void replay_begin_submit(Frame& f) {
    gpl_note_presents(counters().presents.load(std::memory_order_relaxed));   // [nvn] gpl_async defers draws only while frames are flowing
    note_frame_for_pacing();
    rs_report();
    F = &f;
    tsr_cache_clear();
    tex_poll_cpu_writes();
    if (extras::LowLatency* ll = extras::lowlat()) ll->submit_begin();     // RENDERSUBMIT_START of the frame (S14)
    if (g_trace_frame == -2) {
        // [nvn] trace_frame = "6050" or "6050,9300": every listed presented frame is traced (the first one is the main one)
        const std::string list = nxrt::config_get_str("nvn", "trace_frame", "");
        g_trace_frame = -1;
        size_t i = 0;
        while (i < list.size()) {
            size_t j = list.find(',', i);
            if (j == std::string::npos) j = list.size();
            const long long v = std::atoll(list.substr(i, j - i).c_str());
            if (g_trace_frame < 0) g_trace_frame = v; else g_trace_extra.push_back(v);
            i = j + 1;
        }
    }
    if (!g_trigger_read) {
        g_trigger_read = true;
        g_trigger_fs = nxrt::config_get_str("nvn", "trace_after_fs", "");
    }
    if (!g_trigger_fs.empty() && g_trigger_seen && !g_trigger_done && static_cast<int64_t>(counters().presents.load()) > g_trigger_present) {
        g_trigger_done = true;
        g_trace_frame = static_cast<int64_t>(counters().presents.load());
    }
    {   // [nvn] trace_sec = N (NXRT_NVN_TRACE_SEC): trace the first frame presented N seconds after the first draw (frame numbers differ between runs)
        static const double trace_sec = static_cast<double>(nxrt::config_get_int("nvn", "trace_sec", 0));
        static const auto t0 = std::chrono::steady_clock::now();
        static bool trace_sec_done = false;
        if (trace_sec > 0 && !trace_sec_done && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= trace_sec) {
            trace_sec_done = true;
            g_trace_frame = static_cast<int64_t>(counters().presents.load());
        }
    }
    if (g_trace_req.exchange(false)) g_trace_frame = static_cast<int64_t>(counters().presents.load()) + 1;   // armed from a script (@shot nvntrace_*)
    g_trace = g_trace_frame >= 0 && static_cast<int64_t>(counters().presents.load()) == g_trace_frame;
    // A span from one key press: the frames a flicker alternates over cannot be hit by hand, and with trace_targets set to a
    // target that is never drawn the cost is the draw list only.
    if (const int64_t span = g_trace_span.load(std::memory_order_relaxed); span > 0) {
        g_trace_span.store(span - 1, std::memory_order_relaxed);
        g_trace = true;
    }
    for (int64_t e : g_trace_extra) g_trace |= static_cast<int64_t>(counters().presents.load()) == e;
    if (g_trace) {
        g_trace_n = 0;
        g_trace_recs.clear();
        static const int compact_mode = static_cast<int>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "trace_compact", 0), 0, 2));
        NXLOG_I("nvn.trace", "---- frame %lld ---- (trace_compact %d: %s detail after 600 draws)", static_cast<long long>(counters().presents.load()), compact_mode,
                compact_mode == 1 ? "short" : compact_mode == 2 ? "long" : "short");
    }
    {   // [nvn] draw_occ: the traced frame's occlusion-query window, reset once per traced frame outside any render pass
        static int64_t occ_frame = -1;
        const int64_t now_frame = static_cast<int64_t>(counters().presents.load());
        if (g_trace && occ_frame != now_frame) {
            occ_frame = now_frame;
            draw_occ_frame_begin(f.cb);
        }
    }
    {   // S38: device-local pool copies and tracked barriers. [nvn] gpuperf_ab = N: both are off in even windows of N presents and on in odd
        // ones (the paired-window measurement of S29); without it they follow their own settings.
        static const int64_t ab = config_get_int("nvn", "gpuperf_ab", 0);
        static const bool tracked = config_get_str("nvn", "pass_barriers", "tracked") != "full";
        static int last_on = -1;
        bool on = true;
        if (ab > 0) on = ((counters().presents.load() / static_cast<uint64_t>(ab)) & 1) != 0;
        if (static_cast<int>(on) != last_on) {
            last_on = on ? 1 : 0;
            if (ab > 0) ft::mark(on ? "gpuperf.on" : "gpuperf.off", static_cast<uint32_t>(counters().presents.load()));
        }
        g_tracked = on && tracked;
        set_narrow_barriers(g_tracked);
        g_pending.clear();
        g_pass_unsafe = false;
        mirror_begin_submit(f, on);
    }
    S.window_origin = window_origin_mode();
    S.depth_mode = depth_mode();
    g_draw_idx = 0;
    ++g_dbg_frame;
    P = Pass();
    g_set_cache.clear();
    {   // [nvn] tex_preupload (default on): textures whose background preparation has finished are uploaded here, outside every pass. An area
        // load (XC2 NS2: 402 textures, 256 MiB in one frame) otherwise ended and restarted the render pass at the first draw of each one
        // (prepare_textures): hundreds of pass breaks inside the G-buffer pass, ~90 ms of GPU time for one frame.
        static const bool pre_on = config_get_bool("nvn", "tex_preupload", false);   // off: it uploaded textures in bursts during play, before they were needed (more 20-45 ms frames)
        if (pre_on && texprep::enabled()) {
            static std::vector<Ref<Texture>> ready;
            ready.clear();
            static const size_t max_pre_n = static_cast<size_t>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "tex_preupload_max", 12), 1, 1024));
            static const uint64_t max_pre_bytes = static_cast<uint64_t>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "tex_preupload_kb", 4096), 256, 524288)) << 10;
            texprep::ready_textures(ready, max_pre_n);
            if (!ready.empty()) {
                NXFT("tex.preupload");
                uint32_t n = 0;
                uint64_t bytes = 0;
                for (const Ref<Texture>& rt : ready) {
                    Texture* t = rt.get();
                    if (!t || t->destroyed.load() || !tex_gpu(t)) continue;
                    if (n > 0 && bytes + t->storage_size > max_pre_bytes) break;
                    out_texture_read(t);
                    tex_ensure_current(t, f);
                    tex_transition(t, f, tex_sampled_layout(t));
                    bytes += t->storage_size;
                    ++n;
                }
                static uint64_t total = 0, calls = 0;
                total += n;
                if ((++calls & 255) == 1) NXLOG_I(kTag, "tex_preupload: %llu textures uploaded at submission start so far", (unsigned long long)total);
            }
        }
    }
    {   // [nvn] lean_draws: the setting, the bound-slot masks from the state itself, and the pushed set 0 (a new command buffer has none)
        g_lean = lean_setting();
        g_bm_vtx = 0;
        for (uint32_t k = 0; k < 16; ++k) g_bm_vtx |= S.vtx[k].addr ? 1u << k : 0u;
        for (int st = 0; st < 6; ++st) {
            g_bm_ubo[st] = g_bm_ssbo[st] = 0;
            for (uint32_t k = 0; k < 16; ++k) {
                g_bm_ubo[st] |= S.ubo[st][k].addr ? 1u << k : 0u;
                g_bm_ssbo[st] |= S.ssbo[st][k].addr ? 1u << k : 0u;
            }
        }
        g_pushed0.cb = VK_NULL_HANDLE;
    }
    g_sets_pl = nullptr;
    g_sets_cb = VK_NULL_HANDLE;
    g_bound_pipeline = VK_NULL_HANDLE;
    invalidate_bound_state();
    g_dirty = kAllDirty;
    g_vb_dirty_mask = 0xFFFF;
    g_pl = nullptr;
    g_pipe = VK_NULL_HANDLE;
    g_topo = g_pipe_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    g_restart_valid = false;
    g_vb_bound_in_cb = false;
    g_want_depth_ro = false;
    g_depth_general = false;
    g_color_general = 0;
    g_color_feedback = 0;
    g_fb_use.clear();
    g_fb_valid.clear();
    g_fb_seen = false;
    g_pending_write_mem = false;
    g_wm_last_addr = g_wm_last_size = 0;
    g_key_dummy_binding = 0;
    g_split_since = 0;
    g_split_t0 = std::chrono::steady_clock::now();
    for (UploadAlloc& u : g_push_buf) u = UploadAlloc();   // the upload memory of the last submission
    g_cs_c0 = UploadAlloc();
    // one support buffer per submission (Ryujinx SupportBuffer: render scales 1.0, no BGRA remap)
    g_support = f.upload(sizeof(SupportBuffer), 256);
    g_support_scaled.clear();
    g_draw_support = UploadAlloc();
    if (g_support) {
        SupportBuffer sb = support_buffer_defaults();
        std::memcpy(g_support.ptr, &sb, sizeof sb);
    }
}

void replay_lists(Frame& f, const std::vector<Ref<CmdList>>& lists) {
    F = &f;
    {   // [nvn] spv_hot: a .spv edited while the game runs is read again here, between frames, and the modules and pipelines made from the old
        // words are dropped; the next draw that needs the shader builds them from the new file (ShaderCache::hot_poll).
        static std::vector<const ShaderInfo*> reloaded;
        reloaded.clear();
        shader_cache().hot_poll(&reloaded);
        for (const ShaderInfo* s : reloaded) pipeline_hot_shader_changed(s);
    }
    RsScope rs(kRsAll);
    for (const Ref<CmdList>& l : lists)
        if (l) run_list(*l, 0);
}

void replay_request_trace() { g_trace_req.store(true); }
void replay_request_trace_span(int count) { g_trace_span.store(count > 0 ? count : 1); }

// [nvn] trace: the end-of-frame answer to "which blended draws ignore the scene's depth" (HANDOVER 42.3). One line per group of
// (target, fragment shader, blend, effective depth state), with the draw numbers, sorted by how many draws the group has: a card
// that draws over the stairs without a depth test is then one line instead of a grep over every draw block of a 3,300-draw frame.
void trace_summary() {
    if (g_trace_recs.empty()) return;
    size_t blended = 0, no_test = 0, no_test_blended = 0, always_blended = 0;
    for (const TraceDrawRec& r : g_trace_recs) {
        const bool b = r.blend != 0;
        if (b) ++blended;
        if (!r.test) { ++no_test; if (b) ++no_test_blended; }
        else if (b && r.func == NVN_COMPARE_FUNC_ALWAYS) ++always_blended;
    }
    NXLOG_I("nvn.trace", "summary: %zu draws, %zu blended; %zu without an effective depth test (%zu of them blended); %zu blended with depth func ALWAYS",
            g_trace_recs.size(), blended, no_test, no_test_blended, always_blended);
    NXLOG_I("nvn.trace", "summary: earlyz (the console's early depth test by stencil) %s, bit 0x%02x: %llu draws emulated so far this session", g_earlyz_on ? "on" : "off",
            g_earlyz_bit, static_cast<unsigned long long>(g_earlyz_draws));
    struct Group { int rt, rt_count, src, dst, eq; std::string fs; bool test, write; int func; };
    std::map<std::string, std::pair<Group, std::vector<int>>> groups;
    for (const TraceDrawRec& r : g_trace_recs) {
        if (!r.blend) continue;
        const Group g{r.rt, r.rt_count, r.src_rgb, r.dst_rgb, r.eq_rgb, r.fs, r.test, r.write, r.func};
        char key[256];
        std::snprintf(key, sizeof key, "t%d/%d %s %d,%d,%d t%dw%df%d", g.rt, g.rt_count, g.fs.c_str(), g.src, g.dst, g.eq, g.test ? 1 : 0, g.write ? 1 : 0, g.func);
        auto it = groups.find(key);
        if (it == groups.end()) it = groups.emplace(key, std::make_pair(g, std::vector<int>())).first;
        it->second.second.push_back(r.n);
    }
    std::vector<std::pair<Group, std::vector<int>>> sorted;
    sorted.reserve(groups.size());
    for (auto& kv : groups) sorted.push_back(std::move(kv.second));
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
    NXLOG_I("nvn.trace", "summary: blended draws by (target, shader, blend, effective depth): %zu groups", sorted.size());
    size_t shown = 0;
    for (const auto& g : sorted) {
        if (shown++ >= 24) break;
        const std::vector<int>& d = g.second;
        std::string list;
        for (size_t i = 0; i < d.size() && i < 8; ++i) list += " #" + std::to_string(d[i]);
        if (d.size() > 8) list += " ...";
        NXLOG_I("nvn.trace", "summary: %4zu x t%d(%d rt) fs %s | blend %s,%s,%s | depth eff test %d write %d %s | draws%s", d.size(), g.first.rt, g.first.rt_count,
                g.first.fs.c_str(), nvn_blend_func_name(g.first.src), nvn_blend_func_name(g.first.dst), nvn_blend_eq_name(g.first.eq), g.first.test ? 1 : 0,
                g.first.write ? 1 : 0, nvn_compare_name(g.first.func), list.c_str());
    }
}

void trace_dump_targets() {
    trace_summary();
    g_trace_recs.clear();
    draw_occ_report();   // [nvn] draw_occ: the traced frame's per-draw sample counts, before the (slow) target dumps
    // The snapshot copies were recorded into the submission that just ended: read their buffers back only after the GPU has run it.
    // Without this the mapped memory still holds zeroes or an older frame - what looked like "t50 changes after a draw that does not
    // write it" (HANDOVER 33.6).
    if (!g_trace_snaps.empty() || !g_trace_targets.empty()) {
        std::error_code ec;
        std::filesystem::create_directories("work/trace", ec);
    }
    if (!g_trace_snaps.empty()) finish();
    for (const TraceSnap& sn : g_trace_snaps) {
        std::string stats;
        const bool ok = snapshot_write(sn.index, sn.name, &stats);
        NXLOG_I("nvn.trace", "snapshot after draw %d: %s -> %s: %s", sn.draw, sn.desc.c_str(), ok ? sn.name.c_str() : "(not written)", ok ? stats.c_str() : "");
    }
    g_trace_snaps.clear();
    if (g_trace_targets.empty()) return;
    std::vector<Ref<Texture>> list;
    list.swap(g_trace_targets);
    const int64_t frame = static_cast<int64_t>(counters().presents.load());
    // [nvn] trace_targets = "1,8,9,12,13" (default empty = every target the frame draws to): a multi-frame trace
    // ([nvn] trace_frame = "8300,8450,8600") otherwise writes hundreds of images per frame.
    static const std::vector<int> want_ids = [] {
        std::vector<int> v;
        const std::string s = nxrt::config_get_str("nvn", "trace_targets", "");
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
            i = j + 1;
        }
        return v;
    }();
    // The first traced frame of a run keeps the plain trace_t<id>_<format>.png names every tool expects; a second one gets its
    // frame number in the name, so a multi-frame trace keeps every frame's targets side by side.
    static int traced = 0;
    const bool numbered = traced++ > 0;
    for (const Ref<Texture>& t : list) {
        if (!want_ids.empty() && std::find(want_ids.begin(), want_ids.end(), static_cast<int>(t->id)) == want_ids.end()) continue;
        char buf[96];
        std::string stats;
        const std::string png = std::string("work/trace/") + (numbered ? "trace_f" + std::to_string(frame) : std::string("trace")) + "_t" + std::to_string(t->id) + "_" +
                                (t->fmt ? t->fmt->name : "x") + ".png";
        const bool ok = dump_texture(t.get(), png, &stats);
        NXLOG_I("nvn.trace", "target %s -> %s (frame %lld): %s", tex_desc(t.get(), buf, sizeof buf), ok ? png.c_str() : "(not dumped)",
                static_cast<long long>(frame), ok ? stats.c_str() : "");
    }
}

// The copies of feedback attachments are keyed by the attachment's address. A texture the game finalizes (every area load makes new render
// targets) leaves its copy behind, a full-size image each (S34 soak: 500 an hour, about 3 GB an hour of GPU memory). A copy that no submission
// has used for kFbIdleSubmits is released; one that is still in use stays (and one released too early is made again on its next use).
constexpr uint64_t kFbIdleSubmits = 600;
void fb_sweep(uint64_t serial) {
    g_fb_use.clear();       // raw pointers into the copies: none is in use between two submissions
    g_fb_valid.clear();
    for (auto it = g_fb_clone.begin(); it != g_fb_clone.end();) {
        if (serial > it->second.last + kFbIdleSubmits) it = g_fb_clone.erase(it);
        else ++it;
    }
    g_fb_clone_n.store(static_cast<int64_t>(g_fb_clone.size()), std::memory_order_relaxed);
}

void replay_end_submit(Frame& f) {
    F = &f;
    if (out_any()) {         // every compute result is in game memory, or held and guarded, before the submission's serial signals
        end_pass();
        out_end_submit(F->cb, F->serial);
    }
    end_pass();
    if (mirror_capture_on()) mirror_capture_end(F->cb);   // GPU writes in the copies back to the pools before the serial signals
    mirror_end_submit();
    if ((f.serial & 127) == 0 && !g_fb_clone.empty()) fb_sweep(f.serial);
    if ((f.serial & 127) == 0 && !g_vol3d.empty()) {   // 3D copies of 2D textures no submission used for kFbIdleSubmits are released
        g_vol3d_use.clear();
        for (auto it = g_vol3d.begin(); it != g_vol3d.end();) {
            if (f.serial > it->second.last + kFbIdleSubmits) it = g_vol3d.erase(it);
            else ++it;
        }
    }
    if (g_occ_active >= 0) {   // a query cannot outlive its command buffer: the count of an unreported counter is dropped
        vkCmdEndQuery(f.cb, g_occ_pool, static_cast<uint32_t>(g_occ_active));
        g_occ_active = -1;
    }
    for (int k = 0; k < 2; ++k) {
        cw_end_query(g_cw[k], k == 1);
        g_cw[k].active = -1;
    }
    for (; g_dbg_depth > 0; --g_dbg_depth) vkCmdEndDebugUtilsLabelEXT(f.cb);   // labels cannot outlive their command buffer either
    flush_sub_counters();
    F = nullptr;
}

// A recording that threw: its command buffer is thrown away, so nothing is ended in it. The state that would otherwise make the next
// submission end a pass, query or label it never began is cleared.
void replay_abort_submit() {
    flush_sub_counters();
    P.open = false;
    P.targets_clean = false;
    invalidate_bound_state();
    tsr_cache_clear();
    g_xfb_began = false;
    g_pending.clear();
    g_occ_active = -1;
    for (int k = 0; k < 2; ++k) g_cw[k].active = -1;
    g_dbg_depth = 0;
    g_fb_use.clear();
    g_fb_valid.clear();
    g_pending_write_mem = false;
    g_wm_last_addr = g_wm_last_size = 0;
    g_vb_dirty_mask = 0xFFFF;
    F = nullptr;
}

void replay_shutdown() {
    g_tpool_snap.clear();
    g_spool_snap.clear();
    if (g_xfb_counter) device().destroy_buffer(g_xfb_counter);
    g_xfb_counter = gfx::Buffer();
    g_xfb_began = g_xfb_active = g_xfb_have_counter = false;
    g_xfb_slot_of.clear();
    g_xfb_slot_next = g_xfb_cur = 0;
    std::memset(g_xfb_slot_written, 0, sizeof g_xfb_slot_written);
    for (CounterWindow& w : g_cw) {
        if (w.pool) vkDestroyQueryPool(vkdev(), w.pool, nullptr);
        w = CounterWindow();
    }
    g_dbg_depth = 0;
    g_gw_next = g_gw_count = 0;
    g_gw_lost_serial = 0;
    if (g_occ_pool) vkDestroyQueryPool(vkdev(), g_occ_pool, nullptr);
    g_occ_pool = VK_NULL_HANDLE;
    g_occ_tried = false;
    g_occ_active = g_occ_last = -1;
    g_prims_generated = g_prims_xfb = 0;
    g_mask.shutdown();
    g_mask_ready = false;
    flush_sub_counters();
    g_ring_total.store(0, std::memory_order_release);
    dlss_replay_shutdown();
    pipeline_shutdown();
    g_pl = nullptr;                // the layouts are gone with the pipelines
    S = GState();
    raw_binds_reset();
    P = Pass();
    g_pending.clear();
    out_reset();
    for (auto& kv : g_shadows) {   // left at exit otherwise (the validation layer reports them at vkDestroyDevice)
        if (!kv.second.buf) continue;
        forget_buffer_views(kv.second.buf.buffer);
        device().destroy_buffer(kv.second.buf);
    }
    g_shadows.clear();
    g_set_cache.clear();
    tsr_cache_clear();
    g_fb_use.clear();
    g_fb_valid.clear();
    g_fb_clone.clear();
    g_vol3d_use.clear();
    g_vol3d.clear();
    g_scene_depth = Ref<Texture>{};
    g_fb_clone_n.store(0, std::memory_order_relaxed);
}

// Gauge (nxrt/gauge.h): reads the atomic mirror above, never the map. Removed before the globals are destroyed.
namespace {
struct ReplayGauges {
    ReplayGauges() {
        nxrt::gauge_register("nvn.replay.fb_clones", [] { return g_fb_clone_n.load(std::memory_order_relaxed); });
    }
    ~ReplayGauges() { nxrt::gauge_unregister("nvn.replay.fb_clones"); }
};
ReplayGauges g_replay_gauges;
}  // namespace

Ref<Texture> take_scene_depth(int* out_depth_mode) {
    if (out_depth_mode) *out_depth_mode = g_scene_depth_mode;
    return g_scene_depth;
}

}  // namespace nxrt::nvn::be::gpu
