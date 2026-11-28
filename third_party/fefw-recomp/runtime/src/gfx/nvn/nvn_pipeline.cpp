// SPDX-License-Identifier: GPL-3.0-or-later
// Pipeline layouts, shader modules and pipelines (see nvn_pipeline.h). Pipeline creation is synchronous on the render thread; a
// VkPipelineCache persists between runs ([nvn] pipeline_cache, default "pipeline_cache.bin").
#include "nvn_pipeline.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "nxrt/config.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nvn_enums.h"
#include "nvn_format.h"
#include "nvn_gpu_internal.h"
#include "nvn_spirv_patch.h"
#include "nvn_subgroup.h"
#include "vk_gauges.h"
#include "extras/hdr_nvn.h"
#include "nxrt/extras.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.pipe";

struct LayoutKey {
    const ShaderInfo* s[6];
    bool operator<(const LayoutKey& o) const { return std::memcmp(s, o.s, sizeof s) < 0; }
};

// Shader modules and layouts are also made by the prewarm workers (nvn_prewarm.cpp), so they have locks; the pipeline buckets below stay
// render thread only.
std::mutex g_layout_mu;
std::recursive_mutex g_module_mu;   // every module cache below; the variant makers call shader_module
std::map<LayoutKey, ProgLayout> g_layouts;
// Descriptor set layouts and the pipeline layout built from them, shared by every program whose descriptors are the same (S24): two programs with
// the same signature get the same VkPipelineLayout handle, so the shader parts compiled for one are found again for the other.
struct SharedLayout { VkDescriptorSetLayout sets[5] = {}; VkDescriptorUpdateTemplate templates[4] = {}; VkPipelineLayout layout = VK_NULL_HANDLE; };
std::map<std::string, SharedLayout> g_shared_layouts;
std::unordered_map<const ShaderInfo*, VkShaderModule> g_modules;
std::mutex g_module_id_mu;
std::unordered_map<VkShaderModule, SubgroupShader> g_module_subgroups;
std::unordered_map<VkShaderModule, uint64_t> g_module_ids;   // every module create_module made: the hash of its words (module_content_id)
std::atomic<uint64_t> g_built_at_draw{0};                    // whole pipelines built by the render thread at a draw
std::atomic<bool> g_cache_dirty{false};
std::atomic<uint32_t> g_new_since_save{0};      // pipelines and parts created since the cache file was last written
std::atomic<bool> g_cache_big{false};         // the file loaded at start was large (a shader preparation made it): rewritten rarely, see pipeline_cache_dirty
// Producer modules with extra outputs (nvn_spirv_patch.h): (shader, slots as location * 4 + scalar type).
std::map<std::pair<const ShaderInfo*, std::vector<uint32_t>>, VkShaderModule> g_variants;
std::unordered_map<const ShaderInfo*, VkShaderModule> g_dual_src;
std::unordered_map<const ShaderInfo*, std::pair<VkShaderModule, bool>> g_no_out;   // (module, owned by this map)      // fragment shaders with private outputs (passes without colour attachments)
std::map<std::pair<const ShaderInfo*, size_t>, std::pair<VkShaderModule, uint32_t>> g_xfb_mod;   // vertex shaders with the captured outputs (+ how many integer attributes were patched in), their stride       // fragment shaders with the second blend source at Location 0, Index 1
// One pipeline per key. `up`: a linked pipeline (nvn_gpl.cpp) that a background thread is replacing with an optimised one; the lookup swaps it in.
struct GfxEntry {
    GraphicsKey first;
    VkPipeline second = VK_NULL_HANDLE;
    VkPipeline alt = VK_NULL_HANDLE;                 // measurement only ([nvn] gpl_ab): the optimised pipeline, kept next to the linked one
    std::shared_ptr<PipeUpgrade> up;
};
std::unordered_map<uint64_t, std::vector<GfxEntry>> g_gfx;
// Direct-mapped render-thread L1 cache for settled graphics pipelines (avoids unordered_map bucket lookup + atomic RMW per draw).
struct GfxFastSlot {
    uint64_t hash = 0;
    uint64_t gen = 0;
    VkPipeline pipe = VK_NULL_HANDLE;
    GraphicsKey key;
};
constexpr size_t kGfxFastSlots = 256;
GfxFastSlot g_gfx_fast[kGfxFastSlots];
uint64_t g_gfx_fast_gen = 1;
uint32_t g_gfx_fast_hits = 0;
std::atomic<uint64_t> g_layout_gen{1};
std::map<std::pair<const ShaderInfo*, VkPipelineLayout>, VkPipeline> g_compute;
VkPipelineCache g_cache = VK_NULL_HANDLE;
bool g_cache_tried = false;
std::string g_cache_path;
VkPipelineCache g_gpl_cache = VK_NULL_HANDLE;
bool g_gpl_cache_tried = false;
nxrt::GaugeCounter g_pipeline_count{"nvn.pipelines"};   // live VkPipelines (graphics and compute, linked ones included); render thread
std::unordered_map<const ShaderInfo*, std::pair<gfx::Buffer, VkDeviceSize>> g_const_bufs;
gfx::Buffer g_driver_const;
bool g_driver_const_made = false;
// Counters mirrored at the sites that change the maps above (they are render thread only, so the gauge reads these instead).
nxrt::GaugeCounter g_pipes_created{"nvn.pipelines_created"};     // ever; same events as counters().pipelines_created (shader parts included)
nxrt::GaugeCounter g_gfx_keys{"nvn.pipeline_keys"};              // entries in g_gfx: one per graphics key tried (a failed build keeps its entry)
nxrt::GaugeCounter g_compute_keys{"nvn.compute_pipeline_keys"};  // entries in g_compute
nxrt::GaugeCounter g_const_n{"nvn.const_buffers"};               // entries in g_const_bufs
nxrt::GaugeCounter g_const_bytes{"nvn.const_buffers.bytes"};     // bytes of the constant buffers in g_const_bufs

VkShaderStageFlagBits stage_bit(ShaderStage s) {
    switch (s) {
        case ShaderStage::Vertex: return VK_SHADER_STAGE_VERTEX_BIT;
        case ShaderStage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
        case ShaderStage::Geometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
        case ShaderStage::TessControl: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case ShaderStage::TessEval: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        case ShaderStage::Compute: return VK_SHADER_STAGE_COMPUTE_BIT;
        default: return VK_SHADER_STAGE_VERTEX_BIT;
    }
}

std::mutex g_cache_mu;
std::atomic<bool> g_cache_ready{false};
std::atomic<bool> g_gpl_cache_ready{false};
std::atomic<bool> g_cache_save_on{true};
std::shared_mutex g_cache_rw_mu;   // shared by pipeline compiles, exclusive for vkGetPipelineCacheData & shutdown
std::mutex g_save_mu;              // serialises writing pipeline_cache.bin.tmp -> pipeline_cache.bin

std::string trim_lower(std::string s) {
    auto not_ws = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_ws));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_ws).base(), s.end());
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool is_explicit_save_on(const std::string& m) { return m == "1" || m == "true" || m == "on" || m == "yes"; }
bool is_explicit_save_off(const std::string& m) { return m == "0" || m == "false" || m == "off" || m == "no"; }

void ensure_cache() {
    if (g_cache_ready.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_cache_mu);       // the prewarm workers and the render thread may get here together
    if (g_cache_tried) return;
    g_cache_tried = true;
    g_cache_path = config_get_str("nvn", "pipeline_cache", "pipeline_cache.bin");
    const std::string raw_save = config_get_str("nvn", "pipeline_cache_save", "auto");
    const std::string norm_save = trim_lower(raw_save);
    if (!norm_save.empty() && norm_save != "auto" && !is_explicit_save_on(norm_save) && !is_explicit_save_off(norm_save))
        NXLOG_W(kTag, "[nvn] pipeline_cache_save = %s is not known: using auto", raw_save.c_str());
    const uint32_t vendor_id = device().gpu().vendor_id;
    const bool save_on = should_save_pipeline_cache(norm_save, vendor_id);
    g_cache_save_on.store(save_on, std::memory_order_relaxed);
    if (!g_cache_path.empty() && !save_on) {
        if (!is_explicit_save_off(norm_save) && vendor_id == 0x8086u)
            NXLOG_I(kTag, "pipeline cache disk save off by default on Intel GPU (vendor 0x8086, #49 workaround; set [nvn] pipeline_cache_save = true to override)");
        else
            NXLOG_I(kTag, "pipeline cache disk save off ([nvn] pipeline_cache_save = %s)", raw_save.c_str());
    }
    std::vector<char> data;
    if (!g_cache_path.empty()) {
        if (FILE* f = std::fopen(g_cache_path.c_str(), "rb")) {          // one read: the file can be hundreds of MB after a shader preparation
            std::fseek(f, 0, SEEK_END);
            const long len = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (len > 0) {
                data.resize(static_cast<size_t>(len));
                if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
            }
            std::fclose(f);
        }
    }
    g_cache_big.store(data.size() >= (64u << 20), std::memory_order_relaxed);
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(vkdev(), &ci, nullptr, &g_cache) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        vkCreatePipelineCache(vkdev(), &ci, nullptr, &g_cache);
    } else if (!data.empty()) {
        NXLOG_I(kTag, "pipeline cache loaded: %zu bytes from %s", data.size(), g_cache_path.c_str());
    }
    g_cache_ready.store(true, std::memory_order_release);
}

// Separate in-memory cache for runtime GPL shader parts and linked pipelines: keeps library blobs out of the persisted
// pipeline cache (and out of vkGetPipelineCacheData on drivers where serialising GPL parts crashes, issue #49).
void ensure_gpl_cache() {
    if (g_gpl_cache_ready.load(std::memory_order_acquire)) return;
    ensure_cache();
    std::lock_guard<std::mutex> lk(g_cache_mu);
    if (g_gpl_cache_tried) return;
    g_gpl_cache_tried = true;
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    vkCreatePipelineCache(vkdev(), &ci, nullptr, &g_gpl_cache);
    if (g_gpl_cache != VK_NULL_HANDLE && g_cache != VK_NULL_HANDLE) {
        std::shared_lock<std::shared_mutex> cache_lk(g_cache_rw_mu);
        vkMergePipelineCaches(vkdev(), g_gpl_cache, 1, &g_cache);
    }
    g_gpl_cache_ready.store(true, std::memory_order_release);
}

void save_cache() {
    if (!g_cache_save_on.load(std::memory_order_relaxed) || g_cache_path.empty()) return;
    NXFT("pipe.cache.save");
    std::vector<char> data;
    size_t n = 0;
    std::string path;
    {
        std::unique_lock<std::shared_mutex> lk(g_cache_rw_mu);
        if (!g_cache || g_cache_path.empty() || !g_cache_save_on.load(std::memory_order_relaxed)) return;
        g_cache_dirty.store(false);
        g_new_since_save.store(0);
        if (vkGetPipelineCacheData(vkdev(), g_cache, &n, nullptr) != VK_SUCCESS || !n) return;
        data.resize(n);
        if (vkGetPipelineCacheData(vkdev(), g_cache, &n, data.data()) != VK_SUCCESS || !n) return;
        path = g_cache_path;
    }
    std::lock_guard<std::mutex> save_lk(g_save_mu);
    // written to a temporary file and renamed, so a process that ends in the middle of it leaves the old cache intact
        const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write(data.data(), static_cast<std::streamsize>(n));
        if (!f) return;
    }
    if (n >= (64u << 20)) g_cache_big.store(true, std::memory_order_relaxed);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
    }
}

}  // namespace

bool should_save_pipeline_cache(const std::string& mode, uint32_t vendor_id) {
    const std::string m = trim_lower(mode);
    if (is_explicit_save_on(m)) return true;
    if (is_explicit_save_off(m)) return false;
    return vendor_id != 0x8086u;
}

// A cache of hundreds of MB (the shader preparation's) is rewritten only after thousands of new pipelines: the new ones are few next to it and
// the write takes seconds. The ordinary small cache is marked dirty when anything new appeared (background saver throttles writes).
bool pipeline_cache_dirty() {
    if (!g_cache_save_on.load(std::memory_order_relaxed)) return false;
    if (!g_cache_dirty.load(std::memory_order_relaxed)) return false;
    return !g_cache_big.load(std::memory_order_relaxed) || g_new_since_save.load(std::memory_order_relaxed) >= 3000;
}
void pipeline_cache_save() { save_cache(); }
// The write at the end of a session: skipped when the cache is the big one of a preparation and little was added.
void pipeline_cache_flush() {
    if (!g_cache_big || g_new_since_save.load(std::memory_order_relaxed) >= 300) save_cache();
}

bool GraphicsKey::operator==(const GraphicsKey& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
// Word by word: byte by byte it was ~300 dependent multiplies a lookup, and the render thread looks up a key for most draws (~1,700 a
// frame in Dagsion; rt agent, notes/rt.md). Only the in-memory map g_gfx uses it. Keys are zeroed before they are filled (padding too).
uint64_t GraphicsKey::hash() const {
    static_assert(sizeof(GraphicsKey) % 8 == 0, "GraphicsKey is hashed in 8-byte words");
    const uint8_t* p = reinterpret_cast<const uint8_t*>(this);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof *this; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
    }
    return h;
}

namespace {
std::mutex g_early_ft_mu;
std::unordered_set<std::string> g_early_ft_hashes;

void note_early_fragment_tests_patched(const char* what) {
    if (!what || !*what) return;
    std::lock_guard<std::mutex> lk(g_early_ft_mu);
    g_early_ft_hashes.emplace(what);
}
}  // namespace

bool early_fragment_tests_patched(const std::string& fs_hash) {
    std::lock_guard<std::mutex> lk(g_early_ft_mu);
    return g_early_ft_hashes.find(fs_hash) != g_early_ft_hashes.end();
}

// A module for SPIR-V words. The translator declares OpCapability Float64 in every module although no module uses a 64-bit float;
// devices without shaderFloat64 (Intel iGPUs) reject that, so the capability is dropped when no OpTypeFloat 64 exists.
static VkShaderModule create_module(const std::vector<uint32_t>& words, const char* what) {
    std::vector<uint32_t> fixed;
    const std::vector<uint32_t>* code = &words;
    {
        const auto& w = words;
        size_t cap_at = 0;
        bool uses64 = false;
        for (size_t i = 5; i < w.size();) {
            const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            if (len == 0) break;
            if (op == 17 && w[i + 1] == 10) cap_at = i;                       // OpCapability Float64
            if (op == 22 && i + 2 < w.size() && w[i + 2] == 64) uses64 = true;  // OpTypeFloat 64
            i += len;
        }
        if (cap_at && !uses64) {
            fixed = w;
            fixed.erase(fixed.begin() + static_cast<std::ptrdiff_t>(cap_at), fixed.begin() + static_cast<std::ptrdiff_t>(cap_at) + 2);
            code = &fixed;
        }
    }
    // Every Location output of a vertex / tessellation evaluation / geometry shader starts at (0,0,0,1) as on the console: a component the
    // shader never writes is undefined in Vulkan, and RADV hands NaN where AMD's Windows driver hands 0 (spirv_init_outputs). Every
    // module goes through here, the ones spirv_add_outputs / spirv_gs_passthrough rewrote included. [nvn] init_outputs = false: off (A/B).
    // [nvn] early_fragment_tests_fs = "eb74f4cb,...": those fragment shaders (hash prefixes) get EarlyFragmentTests, so the depth /
    // stencil test runs before them as on the console. Without it a driver may run the shader for fragments the test rejects: their image
    // writes land while their colour does not, which is how the light-list OR shaders spent the four per-tile slots and left the black
    // rectangle (HANDOVER 42.21). Only the listed shaders are touched - the general rule crashed amdvlk while it built pipelines.
    std::vector<uint32_t> with_early;
    {
        static const std::string list = config_get_str("nvn", "early_fragment_tests_fs",
            "eb74f4cb,077ceb59,d3d6d5bf,a6db222b,1eb6d5fe,238b7791,fe2136f8,3c39992a,b8843fe5,766314ee,a629f4ac");
        // [nvn] early_fragment_tests = off | early | amd | auto (default off).
        // "amd" (and "auto" on AMD GPUs exposing VK_AMD_shader_early_and_late_fragment_tests) uses EarlyAndLateFragmentTestsAMD.
        static const std::string cfg_mode = config_get_str("nvn", "early_fragment_tests", "off");
        const std::string mode = (cfg_mode == "auto")
            ? (device().features().early_and_late_fragment_tests_amd ? std::string("amd") : std::string("off"))
            : cfg_mode;
        const bool amd = (mode == "amd") && device().features().early_and_late_fragment_tests_amd;
        const bool active = (mode == "early") || amd;
        bool want = false;
        if (active && what) {
            if (list == "auto" || list == "*") {
                want = true;
            } else {
                const std::string h(what);
                for (size_t k = 0; k < list.size() && !want;) {
                    size_t j = list.find(',', k);
                    if (j == std::string::npos) j = list.size();
                    if (j > k) want = h.compare(0, j - k, list, k, j - k) == 0;
                    k = j + 1;
                }
            }
        }
        if (want && spirv_add_early_fragment_tests(*code, with_early, amd)) {
            code = &with_early;
            note_early_fragment_tests_patched(what);
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 20)
                NXLOG_I(kTag, "early_fragment_tests (%s): patched fs %s", amd ? "EarlyAndLateFragmentTestsAMD" : "EarlyFragmentTests", what);
        }
    }
    // [gfx] render_scale (issue #94): fragment shaders performing 2D storage-image atomics at gl_FragCoord.xy / scale
    // (XC2's 640x360 light-list registration mask) are rewritten to raw ivec2(gl_FragCoord.xy) so the mask, list, and
    // shared half-res depth/effect targets stay at the attachment scale instead of falling back to 1x.
    std::vector<uint32_t> with_scaled_atomic;
    if (spirv_scale_frag_image_atomic(*code, with_scaled_atomic)) {
        code = &with_scaled_atomic;
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 20)
            NXLOG_I(kTag, "frag_image_atomic_scale: patched fs %s to raw gl_FragCoord.xy", what ? what : "?");
    }
    static const bool init_outputs = config_get_bool("nvn", "init_outputs", true);
    std::vector<uint32_t> initialised;
    if (init_outputs && spirv_init_outputs(*code, initialised)) code = &initialised;
    // [nvn] spirv_salt (measurement only): an OpSourceExtension with the salt in every module gives the driver a new shader hash, so a run
    // compiles every shader from scratch without touching the driver's own cache on disk.
    static const std::string salt = config_get_str("nvn", "spirv_salt", "");
    std::vector<uint32_t> salted;
    if (!salt.empty() && code->size() > 5) {
        salted = *code;
        size_t at = 5;
        while (at < salted.size()) {
            const uint32_t op = salted[at] & 0xffff, len = salted[at] >> 16;
            if (len == 0) break;
            if (op != 17 && op != 10 && op != 11 && op != 14 && op != 15 && op != 16 && op != 331) break;   // capability .. execution mode
            at += len;
        }
        std::vector<uint32_t> ins((salt.size() + 1 + 3) / 4 + 1, 0);
        ins[0] = (static_cast<uint32_t>(ins.size()) << 16) | 4u;                 // OpSourceExtension
        std::memcpy(&ins[1], salt.c_str(), salt.size());
        salted.insert(salted.begin() + static_cast<std::ptrdiff_t>(at), ins.begin(), ins.end());
        code = &salted;
    }
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code->size() * 4;
    ci.pCode = code->data();
    VkShaderModule m = VK_NULL_HANDLE;
    const VkResult r = vkCreateShaderModule(vkdev(), &ci, nullptr, &m);
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "vkCreateShaderModule failed (%d) for %s", static_cast<int>(r), what);
        return m;
    }
    uint64_t id = 1469598103934665603ull ^ (static_cast<uint64_t>(code->size()) * 0x9e3779b97f4a7c15ull);
    for (uint32_t w : *code) id = (id ^ w) * 1099511628211ull;
    {
        std::lock_guard<std::mutex> lk(g_module_id_mu);
        const auto subgroup = subgroup_shader(*code);
        // GPL keys include the guest subgroup policy. Capabilities are immutable
        // for the lifetime of these device-local caches.
        g_module_ids[m] = subgroup.used ? id ^ 0x6775657374333201ull : id;
        g_module_subgroups[m] = subgroup;
    }
    return m;
}

VkShaderModule shader_module(const ShaderInfo* s) {
    if (!s) return VK_NULL_HANDLE;
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_modules.find(s);
    if (it != g_modules.end()) return it->second;
    VkShaderModule m = create_module(s->spirv, s->hash.c_str());
    g_modules[s] = m;
    return m;
}

// The transform feedback variant, and on top of it the integer vertex attributes when this draw feeds any: a
// stream-out draw still has to declare a uint input where the attribute is an integer format, or the pipeline is
// VUID-VkGraphicsPipelineCreateInfo-Input-08733 (the transform and the attribute patch touch different things, so
// both apply).
static const std::pair<VkShaderModule, uint32_t>& xfb_variant(const ShaderInfo* vs, const std::vector<SpvIntSlot>& ints) {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    const auto key = std::make_pair(vs, ints.size());
    auto it = g_xfb_mod.find(key);
    if (it != g_xfb_mod.end()) return it->second;
    std::vector<uint32_t> patched;
    std::vector<SpvXfbOutput> lay;
    uint32_t stride = 0;
    VkShaderModule m = VK_NULL_HANDLE;
    if (spirv_add_xfb(vs->spirv, patched, &lay, &stride)) {
        if (!ints.empty()) {
            std::vector<uint32_t> with_int;
            if (spirv_int_attributes(patched, ints, with_int)) {
                patched.swap(with_int);
            } else {
                NXLOG_W(kTag, "%s: the integer vertex attributes could not be patched into the transform feedback variant",
                        vs->hash.c_str());
            }
        }
        m = create_module(patched, vs->hash.c_str());
        std::string desc;
        for (const SpvXfbOutput& o : lay) desc += " loc" + std::to_string(o.location) + "x" + std::to_string(o.components) + "@" + std::to_string(o.offset);
        NXLOG_I(kTag, "%s: transform feedback variant, %u bytes per vertex:%s", vs->hash.c_str(), stride, desc.c_str());
    } else {
        NXLOG_W(kTag, "%s: no transform feedback variant could be made (no Location outputs?)", vs->hash.c_str());
    }
    return g_xfb_mod.emplace(key, std::make_pair(m, stride)).first->second;
}

uint32_t xfb_stride(const ShaderInfo* vs) { return vs ? xfb_variant(vs, {}).second : 0; }

static VkShaderModule no_output_module(const ShaderInfo* fs) {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_no_out.find(fs);
    if (it != g_no_out.end()) return it->second.first;
    std::vector<uint32_t> patched;
    std::pair<VkShaderModule, bool> e = {VK_NULL_HANDLE, false};
    if (spirv_outputs_to_private(fs->spirv, patched)) e = {create_module(patched, fs->hash.c_str()), true};
    else e = {shader_module(fs), false};
    g_no_out[fs] = e;
    return e.first;
}

// The tone-map fragment shader runs as its HDR variant while the HDR scene is active (extras/hdr_nvn.h); every other fragment shader is the translated one.
std::unordered_map<const ShaderInfo*, VkShaderModule> g_hdr_mod;   // file scope: pipeline_hot_shader_changed drops it ([nvn] spv_hot)

static VkShaderModule hdr_fragment_module(const ShaderInfo* fs) {
    if (!extras::hdr_scene_active() || fs->spirv.size() > 65536) return shader_module(fs);   // Pokopia's composite tone map is about 24.5k words
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_hdr_mod.find(fs);
    if (it != g_hdr_mod.end()) return it->second;
    std::vector<uint32_t> patched;
    VkShaderModule m = extras::hdr_tonemap_variant(fs->spirv, fs->hash, patched) ? create_module(patched, fs->hash.c_str()) : shader_module(fs);
    g_hdr_mod[fs] = m;
    return m;
}

// The integer-output variant (nvn_spirv_patch.h): a colour target with an integer format, written by a draw whose colour state asks
// for a logic op. Vulkan only defines logic operations on integer attachments, so the game's float outputs are bitcast to the target's
// integer type rather than reinterpreted by the ROP (HANDOVER 35: the light-probe list t50 is OR-ed together this way).
std::unordered_map<const ShaderInfo*, VkShaderModule> g_int_out_mod;

VkShaderModule fs_int_output_module(const ShaderInfo* fs) {
    if (!fs) return VK_NULL_HANDLE;
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_int_out_mod.find(fs);
    if (it != g_int_out_mod.end()) return it->second;
    std::vector<uint32_t> patched;
    VkShaderModule m = spirv_outputs_to_uint(fs->spirv, patched) ? create_module(patched, fs->hash.c_str()) : VK_NULL_HANDLE;
    if (!m) NXLOG_W(kTag, "%s: no integer-output variant (%s; the integer target is written through its float alias instead)",
                        fs->hash.c_str(), spirv_outputs_to_uint_reason());
    else {
        NXLOG_I(kTag, "%s: integer-output variant (%zu words)", fs->hash.c_str(), patched.size());
        if (const char* dir = std::getenv("NXRT_NVN_DUMP_INT_OUT")) {   // diagnostic: the rewritten module, for spirv-val
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (FILE* f = std::fopen((std::string(dir) + "/int_out_" + fs->hash + ".spv").c_str(), "wb")) {
                std::fwrite(patched.data(), 4, patched.size(), f);
                std::fclose(f);
            }
        }
    }
    g_int_out_mod[fs] = m;
    return m;
}

bool fs_int_outputs_available(const ShaderInfo* fs) { return fs_int_output_module(fs) != VK_NULL_HANDLE; }

// Any colour attachment of the key with an integer format?
static bool key_int_colour(const GraphicsKey& key) {
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i) {
        switch (static_cast<VkFormat>(key.color_format[i])) {
            case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT:
            case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
            case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
            case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT:
            case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
            case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32A32_SINT: return true;
            default: break;
        }
    }
    return false;
}

static VkShaderModule dual_source_module(const ShaderInfo* fs) {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_dual_src.find(fs);
    if (it != g_dual_src.end()) return it->second;
    VkShaderModule m = VK_NULL_HANDLE;
    std::vector<uint32_t> patched;
    if (spirv_dual_source_output(fs->spirv, patched)) {
        m = create_module(patched, fs->hash.c_str());
    } else {
        NXLOG_W(kTag, "%s: dual-source blending is on but the fragment shader has no second output; the unpatched module is used", fs->hash.c_str());
        m = shader_module(fs);
    }
    g_dual_src[fs] = m;
    return m;
}

// The slots `consumer` reads that `producer` does not write: the console gives them undefined values, Vulkan needs an output.
static std::vector<SpvOutputSlot> missing_outputs(const ShaderInfo* producer, const ShaderInfo* consumer) {
    std::vector<SpvOutputSlot> missing;
    if (!producer || !consumer) return missing;
    bool covered[64] = {};
    for (const ShaderIo& o : producer->refl.outputs)
        for (uint32_t e = 0; e < (o.array ? o.array : 1); ++e)
            if (o.location + e < 64) covered[o.location + e] = true;
    for (const ShaderIo& in : consumer->refl.inputs) {
        for (uint32_t e = 0; e < (in.array ? in.array : 1); ++e) {
            const uint32_t loc = in.location + e;
            if (loc >= 64 || covered[loc]) continue;
            covered[loc] = true;
            SpvOutputSlot slot;
            slot.location = loc;
            slot.type = in.type == ScalarType::Sint ? SpvScalar::Sint : in.type == ScalarType::Uint ? SpvScalar::Uint : SpvScalar::Float;
            missing.push_back(slot);
        }
    }
    return missing;
}

// The module of `producer` to use in a pipeline whose next stage is `consumer`: the translated one, or a copy with the outputs the
// consumer reads and the producer lacks (and for geometry shaders, with its input primitive execution mode rewritten to match `gs_topo`).
// Null on a Vulkan error.
static VkShaderModule producer_module(const ShaderInfo* producer, const ShaderInfo* consumer, const std::vector<SpvIntSlot>& ints = {},
                                      uint32_t gs_topo = 0xffu) {
    const std::vector<SpvOutputSlot> missing = missing_outputs(producer, consumer);
    const bool check_gs = producer->stage == ShaderStage::Geometry && gs_topo != 0xffu;
    if (missing.empty() && ints.empty() && !check_gs) return shader_module(producer);
    std::vector<uint32_t> id;
    for (const SpvOutputSlot& m : missing) id.push_back(m.location * 4 + static_cast<uint32_t>(m.type));
    for (const SpvIntSlot& n : ints) id.push_back(0x80000000u | (n.location * 2 + (n.is_signed ? 1u : 0u)));
    if (check_gs) id.push_back(0x40000000u | gs_topo);
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_variants.find({producer, id});
    if (it != g_variants.end()) return it->second;
    VkShaderModule m = VK_NULL_HANDLE;
    std::vector<uint32_t> base_words;
    const std::vector<uint32_t>* words = &producer->spirv;
    bool gs_patched = false;
    if (check_gs) {
        std::vector<uint32_t> topo_words;
        if (spirv_gs_input_topology(*words, gs_topo, topo_words)) {
            base_words = std::move(topo_words);
            words = &base_words;
            gs_patched = true;
            NXLOG_I(kTag, "%s: geometry input topology rewritten for VkPrimitiveTopology %u (%zu words)", producer->hash.c_str(), gs_topo, base_words.size());
        }
    }
    if (!ints.empty()) {
        // integer vertex attributes: the shader gets the raw bits the console delivers (nvn_spirv_patch.h)
        std::vector<uint32_t> int_words;
        if (!spirv_int_attributes(*words, ints, int_words)) {
            NXLOG_E(kTag, "%s: the integer vertex attributes could not be patched in; draws with this shader are skipped", producer->hash.c_str());
            g_variants[{producer, id}] = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
        base_words = std::move(int_words);
        words = &base_words;
    }
    if (missing.empty()) {
        m = (gs_patched || !ints.empty()) ? create_module(*words, producer->hash.c_str()) : shader_module(producer);
        g_variants[{producer, id}] = m;
        return m;
    }
    std::vector<uint32_t> patched;
    // a passthrough geometry shader: the hardware forwards the attributes of the input vertices, the translation has no code for it
    const bool pass = producer->stage == ShaderStage::Geometry && spirv_gs_passthrough(*words, missing, patched);
    if (pass || spirv_add_outputs(*words, missing, patched)) {
        m = create_module(patched, producer->hash.c_str());
        NXLOG_D(kTag, "%s: %zu outputs added for the next stage%s", producer->hash.c_str(), missing.size(), pass ? " (copied from the input vertices)" : "");
    } else {
        NXLOG_W(kTag, "%s: the %zu varyings the next stage reads could not be added; the unpatched module is used", producer->hash.c_str(), missing.size());
        m = (gs_patched || !ints.empty()) ? create_module(*words, producer->hash.c_str()) : shader_module(producer);
    }
    g_variants[{producer, id}] = m;
    return m;
}

// The device's push descriptor limit (0 without VK_KHR_push_descriptor).
uint32_t push_descriptor_max() {
    static const uint32_t n = [] {
        if (!device().features().push_descriptor || !vkCmdPushDescriptorSetKHR) return 0u;
        VkPhysicalDevicePushDescriptorPropertiesKHR pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pp};
        vkGetPhysicalDeviceProperties2(device().physical(), &p2);
        return pp.maxPushDescriptors;
    }();
    return n;
}

// [nvn] push_uniforms (default FALSE since the measurement in HANDOVER 71: neutral cost - 11.52 ms against 11.83 at the same
// draw count - so the behaviour change is not paid for; the code stays for the day a scene binds many more descriptors).
// Its purpose: the uniform buffers of a draw are pushed (vkCmdPushDescriptorSetKHR) rather than written into a descriptor
// set. Read once (the pipeline layouts depend on it; changing it rebuilds the pipelines once).
bool push_uniforms_on() {
    static const bool on = [] {
        const bool want = nxrt::config_get_bool("nvn", "push_uniforms", false);
        const uint32_t max = push_descriptor_max();
        const bool ok = want && max >= 8;
        NXLOG_I(kTag, "uniform buffers: %s ([nvn] push_uniforms = %s, VK_KHR_push_descriptor %s, %u push descriptors a set)",
                ok ? "pushed with each draw" : "descriptor sets", want ? "true" : "false", device().features().push_descriptor ? "yes" : "no", max);
        return ok;
    }();
    return on;
}

bool early_fragment_tests_active() {
    static const bool on = [] {
        const std::string cfg_mode = config_get_str("nvn", "early_fragment_tests", "off");
        const std::string mode = (cfg_mode == "auto")
            ? (device().features().early_and_late_fragment_tests_amd ? std::string("amd") : std::string("off"))
            : cfg_mode;
        if (mode != "early" && !((mode == "amd") && device().features().early_and_late_fragment_tests_amd)) return false;
        const std::string list = config_get_str("nvn", "early_fragment_tests_fs",
            "eb74f4cb,077ceb59,d3d6d5bf,a6db222b,1eb6d5fe,238b7791,fe2136f8,3c39992a,b8843fe5,766314ee,a629f4ac");
        return list == "auto" || list == "*";
    }();
    return on;
}

uint64_t prog_layout_gen() { return g_layout_gen.load(std::memory_order_relaxed); }

const ProgLayout* prog_layout(const ShaderInfo* const stages[6]) {
    LayoutKey key;
    for (int i = 0; i < 6; ++i) key.s[i] = stages[i];
    std::lock_guard<std::mutex> lk(g_layout_mu);
    auto it = g_layouts.find(key);
    if (it != g_layouts.end()) return &it->second;

    ProgLayout pl;
    std::map<std::pair<uint32_t, uint32_t>, size_t> index;
    for (int st = 0; st < 6; ++st) {
        const ShaderInfo* s = stages[st];
        if (!s) continue;
        for (size_t i = 0; i < s->refl.bindings.size(); ++i) {
            const DescriptorBinding& b = s->refl.bindings[i];
            if (!b.used || (b.set > 3 && b.set != kPoolSet)) continue;
            const auto k = std::make_pair(b.set, b.binding);
            auto f = index.find(k);
            if (f == index.end()) {
                BindingDesc d;
                d.set = b.set;
                d.binding = b.binding;
                d.type = static_cast<VkDescriptorType>(b.type);
                d.stages = stage_bit(s->stage);
                d.shader = s;
                d.index = static_cast<int>(i);
                d.nvn_stage = st;
                index[k] = pl.bindings.size();
                pl.has_sampler |= d.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                pl.has_storage_writes |= d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                pl.bindings.push_back(d);
            } else {
                pl.bindings[f->second].stages |= stage_bit(s->stage);   // shared binding (the support buffer)
            }
        }
    }
    std::vector<VkDescriptorSetLayoutBinding> lbs[4];
    std::string sig;                                   // the descriptors of the program, as bytes: the key of the shared layout
    for (int set = 0; set < 4; ++set) {
        for (uint32_t bi = 0; bi < static_cast<uint32_t>(pl.bindings.size()); ++bi) {
            const BindingDesc& d = pl.bindings[bi];
            if (d.set != static_cast<uint32_t>(set)) continue;
            VkDescriptorSetLayoutBinding b = {};
            b.binding = d.binding;
            b.descriptorType = d.type;
            const DescriptorBinding& db = d.shader->refl.bindings[d.index];
            b.descriptorCount = std::max<uint32_t>(1, db.count);
            b.stageFlags = d.stages;
            lbs[set].push_back(b);
            pl.set_binding_idx[set].push_back(bi);
            if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                const NvnSource src = db.nvn.source;
                if (src == NvnSource::UniformBlock || src == NvnSource::UniformAsStorage) {
                    if (d.nvn_stage >= 0 && d.nvn_stage < 6 && db.nvn.index >= 0 && db.nvn.index < 16)
                        pl.ubo_mask[d.nvn_stage] |= static_cast<uint16_t>(1u << db.nvn.index);
                } else if (src == NvnSource::StorageBuffer) {
                    if (d.nvn_stage >= 0 && d.nvn_stage < 6) {
                        if (db.nvn.index >= 0 && db.nvn.index < 16)
                            pl.ssbo_mask[d.nvn_stage] |= static_cast<uint16_t>(1u << db.nvn.index);
                        else
                            pl.ssbo_mask[d.nvn_stage] = 0xFFFFu;
                    }
                }
            }
            if (db.nvn.source == NvnSource::HandleInUniform) {
                pl.has_handle_in_ubo = true;
                if (d.nvn_stage >= 0 && d.nvn_stage < 6 && db.nvn.index >= 0 && db.nvn.index < 16)
                    pl.ubo_mask[d.nvn_stage] |= static_cast<uint16_t>(1u << db.nvn.index);
            }
            if (d.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                const NvnSource src = db.nvn.source;
                if (src == NvnSource::SamplerBinding || src == NvnSource::HandleInUniform) {
                    pl.sampler_binding_idx.push_back(bi);
                    if (db.image.dim == 2) pl.has_sampler_3d = true;
                }
                if (src == NvnSource::SamplerBinding || src == NvnSource::HandleInUniform || src == NvnSource::ImageBinding)
                    pl.prep_tex_idx.push_back(bi);
            } else if (d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                pl.prep_tex_idx.push_back(bi);
            }
            pl.has_set[set] = true;
            const uint32_t w[4] = {b.binding, static_cast<uint32_t>(b.descriptorType), b.descriptorCount, b.stageFlags};
            sig.append(reinterpret_cast<const char*>(w), sizeof w);
        }
        sig.append("|", 1);
    }
    // set 0 pushed: every binding a single buffer, no more of them than the device pushes in one set
    if (pl.has_set[0] && push_uniforms_on()) {
        bool ok = lbs[0].size() <= push_descriptor_max();
        for (const VkDescriptorSetLayoutBinding& b : lbs[0])
            ok = ok && b.descriptorCount == 1 && (b.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        pl.push_set0 = ok;
        if (!ok) {
            static int logged = 0;
            if (logged++ < 4)
                NXLOG_I(kTag, "a program's uniform set (%zu bindings) is not pushed: more than the device's %u push descriptors, or not only buffers",
                        lbs[0].size(), push_descriptor_max());
        }
        if (ok) sig.append("push0", 5);
        // a uniform buffer rebind changes nothing outside set 0 unless another set reads a uniform block (a texture handle in one)
        bool ubo_only = ok;
        for (const BindingDesc& d : pl.bindings) {
            if (d.set == 0) continue;
            const NvnSource src = d.shader->refl.bindings[d.index].nvn.source;
            if (src == NvnSource::HandleInUniform || src == NvnSource::UniformBlock || src == NvnSource::UniformAsStorage ||
                src == NvnSource::PushConstants)
                ubo_only = false;
        }
        pl.ubo_only = ubo_only;
    }
    // the pool arrays: the shared layout of kPoolSet (pool_set_layout), whatever arrays the program's stages use
    for (const BindingDesc& d : pl.bindings) pl.has_set[kPoolSet] |= d.set == kPoolSet;
    VkDescriptorSetLayout pool_layout = VK_NULL_HANDLE;
    if (pl.has_set[kPoolSet]) {
        pool_layout = pool_set_layout();
        if (!pool_layout) {
            static bool logged = false;
            if (!logged) { logged = true; NXLOG_E(kTag, "a program samples through the texture pool but the device has no pool set layout: its draws are skipped"); }
            return nullptr;
        }
        sig.append("pools", 5);
    }
    // push constants of generated stages (the pass-through tessellation control shader): one range from offset 0 over those stages
    VkPushConstantRange pcr = {0, 0, 0};
    for (int st = 0; st < 6; ++st)
        if (stages[st] && stages[st]->push_bytes) {
            pcr.stageFlags |= stage_bit(stages[st]->stage);
            pcr.size = std::max(pcr.size, stages[st]->push_bytes);
        }
    if (pcr.size) {
        const uint32_t w[2] = {pcr.stageFlags, pcr.size};
        sig.append("P", 1);
        sig.append(reinterpret_cast<const char*>(w), sizeof w);
    }
    auto sh = g_shared_layouts.find(sig);
    if (sh == g_shared_layouts.end()) {
        SharedLayout nl;
        for (int set = 0; set < 4; ++set) {
            VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            if (set == 0 && pl.push_set0) ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            ci.bindingCount = static_cast<uint32_t>(lbs[set].size());
            ci.pBindings = lbs[set].data();
            if (vkCreateDescriptorSetLayout(vkdev(), &ci, nullptr, &nl.sets[set]) != VK_SUCCESS) {
                NXLOG_E(kTag, "descriptor set layout %d failed", set);
                return nullptr;
            }
            if (!lbs[set].empty() && !(set == 0 && pl.push_set0) && vkCreateDescriptorUpdateTemplate) {
                bool tpl_ok = true;
                std::vector<VkDescriptorUpdateTemplateEntry> entries;
                entries.reserve(lbs[set].size());
                for (size_t i = 0; i < lbs[set].size(); ++i) {
                    const VkDescriptorSetLayoutBinding& lb = lbs[set][i];
                    if (lb.descriptorCount != 1 ||
                        (lb.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                         lb.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                         lb.descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                         lb.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)) {
                        tpl_ok = false;
                        break;
                    }
                    VkDescriptorUpdateTemplateEntry te = {};
                    te.dstBinding = lb.binding;
                    te.dstArrayElement = 0;
                    te.descriptorCount = 1;
                    te.descriptorType = lb.descriptorType;
                    te.offset = i * sizeof(DescTemplateSlot);
                    te.stride = sizeof(DescTemplateSlot);
                    entries.push_back(te);
                }
                if (tpl_ok) {
                    VkDescriptorUpdateTemplateCreateInfo tci = {VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
                    tci.descriptorUpdateEntryCount = static_cast<uint32_t>(entries.size());
                    tci.pDescriptorUpdateEntries = entries.data();
                    tci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
                    tci.descriptorSetLayout = nl.sets[set];
                    if (vkCreateDescriptorUpdateTemplate(vkdev(), &tci, nullptr, &nl.templates[set]) != VK_SUCCESS)
                        nl.templates[set] = VK_NULL_HANDLE;
                }
            }
        }
        nl.sets[kPoolSet] = pool_layout;            // not owned (pool_sets_shutdown destroys it)
        VkPipelineLayoutCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pci.setLayoutCount = pool_layout ? 5 : 4;
        pci.pSetLayouts = nl.sets;
        pci.pushConstantRangeCount = pcr.size ? 1 : 0;
        pci.pPushConstantRanges = pcr.size ? &pcr : nullptr;
        if (vkCreatePipelineLayout(vkdev(), &pci, nullptr, &nl.layout) != VK_SUCCESS) {
            NXLOG_E(kTag, "pipeline layout failed");
            return nullptr;
        }
        if (pl.push_set0 && !lbs[0].empty() && vkCreateDescriptorUpdateTemplate && vkCmdPushDescriptorSetWithTemplateKHR) {
            bool pt_ok = true;
            std::vector<VkDescriptorUpdateTemplateEntry> entries;
            entries.reserve(lbs[0].size());
            for (size_t i = 0; i < lbs[0].size(); ++i) {
                const VkDescriptorSetLayoutBinding& lb = lbs[0][i];
                if (lb.descriptorCount != 1 ||
                    (lb.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                     lb.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)) {
                    pt_ok = false;
                    break;
                }
                VkDescriptorUpdateTemplateEntry te = {};
                te.dstBinding = lb.binding;
                te.dstArrayElement = 0;
                te.descriptorCount = 1;
                te.descriptorType = lb.descriptorType;
                te.offset = i * sizeof(VkDescriptorBufferInfo);
                te.stride = sizeof(VkDescriptorBufferInfo);
                entries.push_back(te);
            }
            if (pt_ok) {
                VkDescriptorUpdateTemplateCreateInfo tci = {VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
                tci.descriptorUpdateEntryCount = static_cast<uint32_t>(entries.size());
                tci.pDescriptorUpdateEntries = entries.data();
                tci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
                tci.pipelineBindPoint = (lbs[0][0].stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
                    ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;
                tci.pipelineLayout = nl.layout;
                tci.set = 0;
                if (vkCreateDescriptorUpdateTemplate(vkdev(), &tci, nullptr, &nl.templates[0]) != VK_SUCCESS)
                    nl.templates[0] = VK_NULL_HANDLE;
            }
        }
        sh = g_shared_layouts.emplace(sig, nl).first;
    }
    for (int set = 0; set < 5; ++set) pl.sets[set] = sh->second.sets[set];
    for (int set = 0; set < 4; ++set) pl.templates[set] = sh->second.templates[set];
    pl.layout = sh->second.layout;
    pl.push_stages = pcr.stageFlags;
    pl.push_bytes = pcr.size;
    return &g_layouts.emplace(key, std::move(pl)).first->second;
}

std::vector<SpvIntSlot> key_int_slots(const GraphicsKey& key) {
    // vertex inputs the shader declares as floats but the game feeds with integer attributes (key formats UINT / SINT)
    std::vector<SpvIntSlot> int_slots;
    // A vertex attribute with an integer format feeding a float shader input is legal on the console (the bits land in
    // the float register) but not in Vulkan: the module has to declare the input as uint (spirv_int_attributes).
    // Stream-out draws need it too - xfb_variant applies both transforms - so there is no xfb exception here.
    if (key.stage[0])
        for (uint32_t k = 0; k < key.attr_count && k < 16; ++k) {
            const VkFormat f = static_cast<VkFormat>(key.attr[k].format);
            if (!vk_format_is_integer(f)) continue;
            for (const ShaderIo& in : key.stage[0]->refl.inputs)
                if (in.type == ScalarType::Float && key.attr[k].location >= in.location && key.attr[k].location < in.location + (in.array ? in.array : 1)) {
                    SpvIntSlot sl;
                    sl.location = key.attr[k].location;
                    sl.is_signed = vk_format_is_signed_int(f);
                    int_slots.push_back(sl);
                    break;
                }
        }
    return int_slots;
}

static uint32_t active_color_mask(const GraphicsKey& key) {
    uint32_t m = 0;
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i)
        if (key.color_format[i] != VK_FORMAT_UNDEFINED) m |= (1u << i);
    return m;
}

static bool fs_has_unattached_outputs(const ShaderInfo* fs, uint32_t keep_mask) {
    if (!fs) return false;
    for (const ShaderIo& o : fs->refl.outputs) {
        const uint32_t count = o.array ? o.array : 1;
        for (uint32_t e = 0; e < count; ++e) {
            const uint32_t loc = o.location + e;
            if (loc >= 32 || !((keep_mask >> loc) & 1u)) return true;
        }
    }
    return false;
}

static VkShaderModule fragment_module(const GraphicsKey& key) {
    const ShaderInfo* fs = key.stage[1];
    if (!fs) return VK_NULL_HANDLE;
    const bool want_int_out = !key.no_out && key.logic_op_enable && key_int_colour(key);
    const uint32_t keep_mask = key.no_out ? 0u : active_color_mask(key);
    // Prewarm/prebuild keys leave color_count == 0 (real no-color draws set no_out = 1),
    // so skip output pruning there or precompiled FS parts will never match draw-time keys.
    const bool want_prune = !key.no_out && key.color_count > 0 && !key.dual_src &&
                            fs_has_unattached_outputs(fs, keep_mask);
    const bool want_uint_smp = key.uint_samplers != 0;

    if (!want_prune && !want_uint_smp) {
        if (key.no_out) return no_output_module(fs);
        if (key.sdr_tone_map) return shader_module(fs);
        if (want_int_out) return fs_int_output_module(fs);
        if (key.dual_src) return dual_source_module(fs);
        return hdr_fragment_module(fs);
    }

    std::vector<SpvSamplerSlot> slots;
    std::vector<uint32_t> id;
    const uint32_t flags = (key.no_out ? 1u : 0u) | (want_int_out ? 2u : 0u) | (key.dual_src ? 4u : 0u) |
                           (extras::hdr_scene_active() && !key.sdr_tone_map && fs->spirv.size() <= 65536 ? 8u : 0u);
    if (flags) id.push_back(0x30000000u | flags);
    if (want_prune) id.push_back(0x50000000u | keep_mask);
    if (want_uint_smp) {
        for (uint32_t i = 0; i < 64 && i < fs->refl.bindings.size(); ++i) {
            if (!((key.uint_samplers >> i) & 1ull)) continue;
            const DescriptorBinding& b = fs->refl.bindings[i];
            slots.push_back({b.set, b.binding});
            id.push_back(0x20000000u | ((b.set & 0xffu) << 16) | (b.binding & 0xffffu));
        }
    }
    if (id.empty()) return shader_module(fs);

    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    if (auto it = g_variants.find({fs, id}); it != g_variants.end()) return it->second;

    std::vector<uint32_t> cur = fs->spirv;
    std::vector<uint32_t> tmp;
    bool changed = false;
    if (key.no_out) {
        if (spirv_outputs_to_private(cur, tmp, 0)) { cur = std::move(tmp); changed = true; }
    } else if (want_prune) {
        if (spirv_outputs_to_private(cur, tmp, keep_mask)) { cur = std::move(tmp); changed = true; }
    }
    if (want_int_out) {
        if (spirv_outputs_to_uint(cur, tmp)) { cur = std::move(tmp); changed = true; }
    } else if (key.dual_src) {
        if (spirv_dual_source_output(cur, tmp)) { cur = std::move(tmp); changed = true; }
    } else if (extras::hdr_scene_active() && !key.sdr_tone_map && fs->spirv.size() <= 65536) {
        if (extras::hdr_tonemap_variant(cur, fs->hash, tmp)) { cur = std::move(tmp); changed = true; }
    }
    if (!slots.empty()) {
        if (spirv_uint_samplers(cur, slots, tmp)) { cur = std::move(tmp); changed = true; }
    }

    VkShaderModule m = changed ? create_module(cur, fs->hash.c_str()) : shader_module(fs);
    g_variants[{fs, id}] = m;
    return m;
}

bool resolve_stage_modules(const GraphicsKey& key, StageModules* out) {
    *out = StageModules{};
    // the stage that feeds the fragment shader gets outputs for the varyings the fragment shader reads and it does not write
    // (geometry, else tessellation evaluation, else vertex)
    const int feeder = key.stage[2] ? 2 : key.tess[1] ? 4 : 0;
    const std::vector<SpvIntSlot> int_slots = key_int_slots(key);
    for (int i = 0; i < 3; ++i) {
        if (!key.stage[i]) continue;
        if (key.xfb && i != 0) continue;     // stream-out draws run the vertex stage only
        if (key.xfb) {
            VkShaderModule m = xfb_variant(key.stage[0], int_slots).first;
            if (!m) return false;
            out->m[0] = m;
            continue;
        }
        VkShaderModule m = (i == 2) ? producer_module(key.stage[2], (i == feeder && key.stage[1]) ? key.stage[1] : nullptr, {}, key.topology)
                           : (i == feeder && key.stage[1]) ? producer_module(key.stage[i], key.stage[1], i == 0 ? int_slots : std::vector<SpvIntSlot>())
                           : (i == 0 && !int_slots.empty()) ? producer_module(key.stage[i], nullptr, int_slots)
                           : (i == 1)                    ? fragment_module(key)
                                                         : shader_module(key.stage[i]);
        if (!m) return false;
        out->m[i] = m;
    }
    // tessellation stages (never under stream-out: the replay does not build such keys)
    for (int t = 0; t < 2 && !key.xfb; ++t) {
        const ShaderInfo* s = key.tess[t];
        if (!s) continue;
        VkShaderModule m = (t == 1 && feeder == 4 && key.stage[1]) ? producer_module(s, key.stage[1], std::vector<SpvIntSlot>()) : shader_module(s);
        if (!m) return false;
        out->m[3 + t] = m;
    }
    return true;
}

const VkPipelineDynamicStateCreateInfo* pipeline_dynamic_state() {
    static const VkDynamicState dyn[] = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE, VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BOUNDS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_LINE_WIDTH};
    static const VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                        sizeof(dyn) / sizeof(dyn[0]), dyn};
    return &dy;
}

// A constant bank read past its bound size gives zero on the console; translated shaders index banks with values read from them (the
// light loops of the game's lighting compute shaders run past their 896-byte block), and without robust access those reads go anywhere.
const void* ubo_robustness() {
    auto make = [](VkPipelineRobustnessBufferBehaviorEXT ubo) {
        VkPipelineRobustnessCreateInfoEXT r = {VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT};
        r.storageBuffers = r.vertexInputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT_EXT;
        r.uniformBuffers = ubo;
        r.images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DEVICE_DEFAULT_EXT;
        return r;
    };
    static const VkPipelineRobustnessCreateInfoEXT v1 = make(VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT),
                                                   v2 = make(VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT);
    static const bool on = nxrt::config_get_bool("nvn", "robust_ubo", true);
    const int level = on ? device().features().ubo_robustness : 0;
    return level == 2 ? &v2 : level == 1 ? &v1 : nullptr;
}

VkPipelineCache pipeline_cache_handle() {
    ensure_cache();
    return g_cache;
}

VkPipelineCache pipeline_library_cache_handle() {
    ensure_gpl_cache();
    return g_gpl_cache;
}

void note_pipeline_created(double ms, const char* what, bool persisted) {
    counters().pipelines_created.fetch_add(1);
    g_pipes_created.add(1);
    if (persisted) {
        g_cache_dirty.store(true, std::memory_order_relaxed);
        if (ms >= 2.0) g_new_since_save.fetch_add(1, std::memory_order_relaxed);      // a part the cache already had comes back in well under that
    }
    NXLOG_D(kTag, "%s created in %.1f ms", what, ms);
}

VkResult create_graphics_pipeline_tracked(VkDevice dev, VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* gi, VkPipeline* out) {
    compiles_in_flight().fetch_add(1, std::memory_order_relaxed);
    const uint64_t t0 = nxrt::hitch_ticks();
    // Centralised here so whole pipelines, GPL parts and prewarm workers agree.
    auto info = *gi;
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> sizes(gi->stageCount);
    if (gi->stageCount) stages.assign(gi->pStages, gi->pStages + gi->stageCount);
    for (size_t i = 0; i < stages.size(); ++i) configure_guest_subgroup(stages[i], sizes[i]);
    info.pStages = stages.data();
    std::shared_lock<std::shared_mutex> cache_lk(g_cache_rw_mu, std::defer_lock);
    if (cache != VK_NULL_HANDLE) cache_lk.lock();
    const VkResult r = vkCreateGraphicsPipelines(dev, cache, 1, &info, nullptr, out);
    if (nxrt::hitch_on_render_thread()) nxrt::hitch_phase_add(nxrt::kHpRenderPipeline, nxrt::hitch_ticks() - t0);
    compiles_in_flight().fetch_sub(1, std::memory_order_relaxed);
    return r;
}

VkResult create_compute_pipeline_tracked(VkDevice dev, VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* out) {
    compiles_in_flight().fetch_add(1, std::memory_order_relaxed);
    const uint64_t t0 = nxrt::hitch_ticks();
    std::shared_lock<std::shared_mutex> cache_lk(g_cache_rw_mu, std::defer_lock);
    if (cache != VK_NULL_HANDLE) cache_lk.lock();
    const VkResult r = vkCreateComputePipelines(dev, cache, 1, ci, nullptr, out);
    if (nxrt::hitch_on_render_thread()) nxrt::hitch_phase_add(nxrt::kHpRenderPipeline, nxrt::hitch_ticks() - t0);
    compiles_in_flight().fetch_sub(1, std::memory_order_relaxed);
    return r;
}

void configure_guest_subgroup(VkPipelineShaderStageCreateInfo& stage,
                              VkPipelineShaderStageRequiredSubgroupSizeCreateInfo& size) {
    SubgroupShader shader;
    {
        std::lock_guard<std::mutex> lk(g_module_id_mu);
        auto it = g_module_subgroups.find(stage.module);
        if (it == g_module_subgroups.end()) return;
        shader = it->second;
    }
    if (!shader.used) return;
    if (!require_guest_subgroup(stage, size, shader, device().features(), device().limits_ext())) {
        static std::atomic<uint32_t> warned{0};
        if (!(warned.fetch_or(stage.stage) & stage.stage))
            NXLOG_W(kTag, "cannot require guest subgroup size 32 for stage 0x%x (device support or compute workgroup limit); native size %u",
                    stage.stage, device().limits_ext().subgroup_size);
    }
}

VkPipeline build_graphics_pipeline(const GraphicsKey& key, const ProgLayout& layout) {
    ensure_cache();
    const auto t0 = std::chrono::steady_clock::now();

    VkPipelineShaderStageCreateInfo stages[5] = {};
    uint32_t ns = 0;
    const VkShaderStageFlagBits bits[5] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
                                           VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
    StageModules sm;
    if (!resolve_stage_modules(key, &sm)) return VK_NULL_HANDLE;
    for (int i = 0; i < 5; ++i) {
        if (!sm.m[i]) continue;
        stages[ns] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, ubo_robustness()};
        stages[ns].stage = bits[i];
        stages[ns].module = sm.m[i];
        stages[ns].pName = "main";
        ++ns;
    }

    // vertex input
    VkVertexInputBindingDescription vb[16];
    VkVertexInputAttributeDescription va[16];
    uint32_t nvb = 0, nva = 0;
    bool binding_seen[16] = {};
    for (uint32_t i = 0; i < key.attr_count && i < 16; ++i) {
        const KeyAttr& a = key.attr[i];
        va[nva++] = {a.location, a.binding, static_cast<VkFormat>(a.format), a.offset};
        if (!binding_seen[a.binding]) {
            binding_seen[a.binding] = true;
            vb[nvb++] = {a.binding, 0, key.binding_instance[a.binding] ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        }
    }
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = nvb;
    vi.pVertexBindingDescriptions = vb;
    vi.vertexAttributeDescriptionCount = nva;
    vi.pVertexAttributeDescriptions = va;

    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = static_cast<VkPrimitiveTopology>(key.topology);
    VkPipelineTessellationStateCreateInfo ts = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    ts.patchControlPoints = key.patch_control_points;

    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
    dcc.negativeOneToOne = key.depth_neg_one ? VK_TRUE : VK_FALSE;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    if (device().features().depth_clip_control) vp.pNext = &dcc;   // viewport and scissor counts are dynamic

    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = key.depth_clamp ? VK_TRUE : VK_FALSE;
    rs.polygonMode = static_cast<VkPolygonMode>(key.polygon_mode);
    rs.lineWidth = 1.0f;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = static_cast<VkSampleCountFlagBits>(key.samples ? key.samples : 1);
    ms.alphaToCoverageEnable = key.alpha_to_coverage ? VK_TRUE : VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState cba[8];
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i) {
        cba[i] = {};
        cba[i].blendEnable = key.blend_enable[i] ? VK_TRUE : VK_FALSE;
        cba[i].srcColorBlendFactor = static_cast<VkBlendFactor>(key.blend_src_rgb[i]);
        cba[i].dstColorBlendFactor = static_cast<VkBlendFactor>(key.blend_dst_rgb[i]);
        cba[i].colorBlendOp = static_cast<VkBlendOp>(key.blend_op_rgb[i]);
        cba[i].srcAlphaBlendFactor = static_cast<VkBlendFactor>(key.blend_src_a[i]);
        cba[i].dstAlphaBlendFactor = static_cast<VkBlendFactor>(key.blend_dst_a[i]);
        cba[i].alphaBlendOp = static_cast<VkBlendOp>(key.blend_op_a[i]);
        cba[i].colorWriteMask = key.write_mask[i];
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.logicOpEnable = key.logic_op_enable ? VK_TRUE : VK_FALSE;
    cb.logicOp = static_cast<VkLogicOp>(key.logic_op);
    cb.attachmentCount = key.color_count;
    cb.pAttachments = cba;

    VkFormat cf[8];
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i) cf[i] = static_cast<VkFormat>(key.color_format[i]);
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = key.color_count;
    ri.pColorAttachmentFormats = cf;
    ri.depthAttachmentFormat = static_cast<VkFormat>(key.depth_format);
    ri.stencilAttachmentFormat = static_cast<VkFormat>(key.stencil_format);

    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri};
    if (device().features().attachment_feedback_loop_layout)
        gi.flags |= VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    gi.stageCount = ns;
    gi.pStages = stages;
    gi.pVertexInputState = &vi;
    gi.pInputAssemblyState = &ia;
    gi.pTessellationState = key.patch_control_points ? &ts : nullptr;
    gi.pViewportState = &vp;
    gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms;
    gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cb;
    gi.pDynamicState = pipeline_dynamic_state();
    gi.layout = layout.layout;
    // Diagnostic (HANDOVER 33): say exactly what the colour-blend state of a logic-op pipeline is; the probe list t50
    // relies on OR surviving a (0,0) write.
    if (key.logic_op_enable)
        NXLOG_I(kTag, "logic-op pipeline: fs %s op %d blend0 %d mask0 %x fmt0 %u", key.stage[1] ? key.stage[1]->hash.c_str() : "-", static_cast<int>(cb.logicOp),
                static_cast<int>(cba[0].blendEnable), cba[0].colorWriteMask, key.color_format[0]);
    VkPipeline pipe = VK_NULL_HANDLE;
    const VkResult r = create_graphics_pipeline_tracked(vkdev(), g_cache, &gi, &pipe);
    const double ms_taken = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "graphics pipeline failed (%d): vs %s fs %s, %u colour targets", static_cast<int>(r), key.stage[0] ? key.stage[0]->hash.c_str() : "-",
                key.stage[1] ? key.stage[1]->hash.c_str() : "-", key.color_count);
        pipe = VK_NULL_HANDLE;
    } else {
        counters().pipelines_created.fetch_add(1);
        g_cache_dirty.store(true, std::memory_order_relaxed);
    g_new_since_save.fetch_add(1, std::memory_order_relaxed);
        g_pipes_created.add(1);
        NXLOG_D(kTag, "graphics pipeline created in %.1f ms (vs %s fs %s)", ms_taken, key.stage[0] ? key.stage[0]->hash.c_str() : "-",
                key.stage[1] ? key.stage[1]->hash.c_str() : "-");
    }
    return pipe;
}

VkPipeline graphics_pipeline(const GraphicsKey& key, const ProgLayout& layout) {
    const uint64_t h = key.hash();
    GfxFastSlot& fs = g_gfx_fast[h & (kGfxFastSlots - 1)];
    if (fs.gen == g_gfx_fast_gen && fs.hash == h && fs.key == key) {
        if ((++g_gfx_fast_hits & 255u) == 0) {
            counters().pipeline_cache_hits.fetch_add(256, std::memory_order_relaxed);
            g_gfx_fast_hits = 0;
        }
        return fs.pipe;
    }
    ensure_cache();
    auto& bucket = g_gfx[h];
    for (auto& e : bucket)
        if (e.first == key) {
            counters().pipeline_cache_hits.fetch_add(1, std::memory_order_relaxed);
            if (e.up && e.up->state.load(std::memory_order_acquire) != 0) {
                gpl_take_upgrade(e.second, e.alt, e.up);
                ++g_gfx_fast_gen;
            }
            if (e.alt && gpl_ab_whole()) return e.alt;
            if (!e.up && !e.alt && e.second) {
                fs.hash = h;
                fs.gen = g_gfx_fast_gen;
                fs.pipe = e.second;
                fs.key = key;
            }
            return e.second;
        }
    prewarm_init();
    VkPipeline pipe = VK_NULL_HANDLE;
    std::shared_ptr<PipeUpgrade> later;          // the key file's whole pipeline, still being built: it replaces the linked one
    // compiled ahead on a worker; one being compiled is linked from its parts meanwhile when they are built (else waited for)
    if (prewarm_take(key, &pipe, [&] { return gpl_ready(key, layout); }, &later)) {
        ++g_pipeline_count;
        g_gfx_keys.add(1);
        bucket.push_back({key, pipe, VK_NULL_HANDLE, nullptr});
        return pipe;
    }
    std::shared_ptr<PipeUpgrade> up;
    bool pending = false;
    pipe = gpl_link(key, layout, later ? nullptr : &up, &pending); // linked from shader parts compiled ahead (null: not eligible or a part failed)
    if (later) up = later;
    prewarm_record(key);                         // every key a draw asked for: the next run builds it from boot
    if (pending && !later) return VK_NULL_HANDLE;   // its parts are compiling in the background: the draw is skipped, the next one asks again
    g_gfx_keys.add(1);                           // every path below adds one entry to the bucket
    if (pipe) {
        ++g_pipeline_count;
        bucket.push_back({key, pipe, VK_NULL_HANDLE, up});
        // Diagnostic (HANDOVER 31): the probe list relies on the logic op; say once per new pipeline whether it went in.
        if (key.logic_op_enable) {
            static int logged = 0;
            if (logged++ < 8) NXLOG_I("nvn.pipe", "logic op enabled: op %d, fs %s, color0 fmt %u", key.logic_op, key.stage[1] ? key.stage[1]->hash.c_str() : "-", key.color_format[0]);
        }
        return pipe;
    }
    if (pending) {                               // (rare) parts gone missing between the check and the link: skipped until the whole pipeline is in
        bucket.push_back({key, VK_NULL_HANDLE, VK_NULL_HANDLE, up});
        return VK_NULL_HANDLE;
    }
    NXFT("pipe.create");
    pipe = build_graphics_pipeline(key, layout);
    if (pipe) {
        ++g_pipeline_count;
        g_built_at_draw.fetch_add(1, std::memory_order_relaxed);
    }
    bucket.push_back({key, pipe, VK_NULL_HANDLE, up});   // up: the key file's build, when one is under way, replaces this one (same pipeline)
    return pipe;
}

VkPipeline compute_pipeline(const ShaderInfo* cs, const ProgLayout& layout) {
    ensure_cache();
    const auto k = std::make_pair(cs, layout.layout);
    auto it = g_compute.find(k);
    if (it != g_compute.end()) return it->second;
    VkShaderModule m = shader_module(cs);
    VkPipeline pipe = VK_NULL_HANDLE;
    if (m) {
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, ubo_robustness()};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, ubo_robustness()};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size{};
        configure_guest_subgroup(ci.stage, subgroup_size);
        ci.layout = layout.layout;
        const VkResult cr = create_compute_pipeline_tracked(vkdev(), g_cache, &ci, &pipe);
        if (cr != VK_SUCCESS) {
            NXLOG_E(kTag, "compute pipeline failed for %s", cs->hash.c_str());
            pipe = VK_NULL_HANDLE;
        } else {
            counters().pipelines_created.fetch_add(1);
            ++g_pipeline_count;
            g_pipes_created.add(1);
            g_cache_dirty.store(true, std::memory_order_relaxed);
    g_new_since_save.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_compute[k] = pipe;                         // new key: the lookup above missed
    g_compute_keys.add(1);
    return pipe;
}

VkBuffer constant_data_buffer(const ShaderInfo* s, VkDeviceSize* range) {
    auto it = g_const_bufs.find(s);
    if (it == g_const_bufs.end()) {
        const VkDeviceSize size = std::max<VkDeviceSize>(256, (s->constants.size() + 15) & ~15ull);
        gfx::Buffer b;
        if (device().create_buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gfx::MemoryUse::Upload, &b, "nvn c1") != VK_SUCCESS) {
            *range = 0;
            return VK_NULL_HANDLE;
        }
        std::memset(b.mapped, 0, size);
        if (!s->constants.empty()) std::memcpy(b.mapped, s->constants.data(), s->constants.size());
        it = g_const_bufs.emplace(s, std::make_pair(b, size)).first;
        g_const_n.add(1);
        g_const_bytes.add(static_cast<int64_t>(size));
    }
    *range = it->second.second;
    return it->second.first.buffer;
}

VkBuffer driver_constants_buffer(VkDeviceSize* range) {
    if (!g_driver_const_made) {
        g_driver_const_made = true;
        if (device().create_buffer(65536, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gfx::MemoryUse::Upload, &g_driver_const, "nvn c0") == VK_SUCCESS) {
            std::memset(g_driver_const.mapped, 0, 65536);
            // c[0]+0x610: maximum point size read by the agl vertex shaders (gl_PointSize = min(size, c0[0x610]))
            const float max_point = 64.0f;
            std::memcpy(static_cast<uint8_t*>(g_driver_const.mapped) + 0x610, &max_point, 4);
        }
    }
    *range = 65536;
    return g_driver_const.buffer;
}

// ---- texture and sampler pools as descriptor arrays (kPoolSet) -------------------------------------------------------------------
// The GPU may still run submissions that read a pool set when the game registers something new, so each pool pair keeps a few sets and
// only writes one that no submission in flight reads (or one that only the submission being recorded has bound: the bindings are update
// after bind). A set is brought up to date with the ids registered since it was last current (the pools' change lists); a new set is
// written whole once. An element whose texture is not in its sampled layout yet (an attachment of the open pass) gets the dummy and is
// written again once it is. Render thread, except pool_set_layout (pipeline layouts are also made by the prewarm workers).
namespace {

constexpr VkShaderStageFlags kPoolStages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
constexpr uint64_t kPoolIdle = 2000;            // submissions after which an unused pool pair or texture pool record is dropped

std::mutex g_pool_layout_mu;
VkDescriptorSetLayout g_pool_layout = VK_NULL_HANDLE;
bool g_pool_layout_tried = false;
uint32_t g_pool_copies = 2;                     // sets per pool pair (2..8, from the device's update-after-bind budget)

struct PoolCopy {
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool written = false;                       // every element written once
    uint64_t tex_version = 0, smp_version = 0;  // the registrations it holds
    uint64_t used = 0, used_before = 0;         // serial of the last submission that bound it, and of the one before that
    std::vector<uint32_t> pending;              // texture ids holding the dummy until their texture is in its sampled layout
};
struct PoolPair {
    Ref<TexturePool> tp;
    Ref<SamplerPool> sp;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    std::vector<PoolCopy> copies;
    int cur = -1;
    uint64_t last_use = 0;
};
// What pool_textures_to_prepare knows of a texture pool.
struct PoolRes {
    Ref<TexturePool> tp;
    uint64_t version = ~0ull;                   // registrations reflected in by_id / reg
    std::vector<Texture*> by_id;                // the texture at each id (compared only; reg holds the references)
    std::unordered_map<Texture*, std::pair<Ref<Texture>, uint32_t>> reg;   // registered textures, the number of ids each sits at
    std::vector<Ref<Texture>> watch;            // to look at next time
    uint64_t epoch = 0;                         // g_tex_scan_epoch when every texture was last looked at
    bool all = true;                            // look at every texture next time
    uint64_t last_use = 0;
};
std::vector<std::unique_ptr<PoolPair>> g_pool_pairs;
std::vector<std::unique_ptr<PoolRes>> g_pool_res;
std::vector<Ref<Texture>> g_layout_moves;       // pool_note_layout, drained by pool_textures_to_prepare
nxrt::GaugeCounter g_pool_writes{"nvn.pool_descriptors_written"};
nxrt::GaugeCounter g_pool_sets{"nvn.pool_sets"};

template <class Pool>
uint64_t version_of(Pool* p) {
    if (!p) return 0;
    std::shared_lock<std::shared_mutex> lk(p->lock);
    return p->version;
}

// Texture id `id` as an element of the 2D or cube array: a view of that shape when the texture (or its view) has it and it is in its
// sampled layout, else the dummy (*not_ready: it may become usable, write it again later).
VkDescriptorImageInfo pool_image(const TexturePool::Entry& e, VkImageViewType type, bool* not_ready) {
    VkDescriptorImageInfo ii = {VK_NULL_HANDLE, dummy_view(type, false), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    Texture* t = e.used ? e.tex.get() : nullptr;
    if (!t || !t->fmt) return ii;
    // finalized and still registered: the image of a live texture on the same storage (tex_gpu takes it), else the dummy, not a new image
    if (t->destroyed.load(std::memory_order_relaxed) && !t->gpu && !texture_registry_find_alias(t)) return ii;
    const int target = e.has_view && e.view.has_target ? e.view.target : t->b.target;
    // a square 2D array of 6n layers is also a cube (its image is made cube compatible, shape_of): the game registers its environment
    // maps that way and samples them through the cube array, as it binds them to cube samplers
    const bool cube_array = target == NVN_TEXTURE_TARGET_2D_ARRAY && t->b.width == t->b.height && t->b.samples <= 1 && t->layers >= 6 &&
                            t->layers % 6 == 0;
    const bool cube = target == NVN_TEXTURE_TARGET_CUBEMAP || target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY || cube_array;
    // a cube read through the 2D array gives its first face: Maxwell keeps a cube's faces as layers and a non-array sample reads layer 0, as
    // tex_sampled_view does for bound slots (nxnative-penis6 6b0cffbb5e); the dummy read zero
    const bool flat = target == NVN_TEXTURE_TARGET_2D || target == NVN_TEXTURE_TARGET_2D_ARRAY || target == NVN_TEXTURE_TARGET_RECTANGLE ||
                      target == NVN_TEXTURE_TARGET_CUBEMAP || target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY;
    if (type == VK_IMAGE_VIEW_TYPE_CUBE ? !cube : !flat) return ii;
    if (!tex_gpu(t)) return ii;
    if (tex_layout(t) != tex_sampled_layout(t)) {
        *not_ready = true;
        return ii;
    }
    ViewRequest q;
    q.type = type;
    int sw[4] = {t->b.swizzle[0], t->b.swizzle[1], t->b.swizzle[2], t->b.swizzle[3]};
    int dsmode = t->b.depth_stencil_mode;
    if (e.has_view) {
        const TextureViewInfo& v = e.view;
        if (v.has_levels) { q.base_level = static_cast<uint32_t>(v.min_level); q.level_count = static_cast<uint32_t>(v.num_levels); }
        if (v.has_layers) { q.base_layer = static_cast<uint32_t>(v.min_layer); q.layer_count = static_cast<uint32_t>(v.num_layers); }
        if (v.has_format)
            if (const FormatInfo* fi = format_info(v.format)) q.format = fi->vk;
        if (v.has_swizzle) for (int i = 0; i < 4; ++i) sw[i] = v.swizzle[i];
        if (v.has_depth_stencil_mode) dsmode = v.depth_stencil_mode;
    }
    q.swizzle = {vk_swizzle(sw[0]), vk_swizzle(sw[1]), vk_swizzle(sw[2]), vk_swizzle(sw[3])};
    q.depth_aspect = dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
    q.component = 1;                            // the arrays are float images
    if (VkImageView v = tex_sampled_view(t, q)) {
        ii.imageView = v;
        ii.imageLayout = tex_sampled_layout(t);
    }
    return ii;
}

// Writes the elements of `ids` (all ids when null) of one set from the pools as they are now.
void write_pool_textures(PoolCopy& c, TexturePool* tp, const std::vector<uint32_t>* ids) {
    static std::vector<VkDescriptorImageInfo> flat, cube;
    std::vector<uint32_t> pending;
    std::vector<VkWriteDescriptorSet> writes;
    auto element = [&](const TexturePool::Entry* e, uint32_t id, size_t at) {
        bool not_ready = false;
        static const TexturePool::Entry kNone;
        flat[at] = pool_image(e ? *e : kNone, VK_IMAGE_VIEW_TYPE_2D, &not_ready);
        cube[at] = pool_image(e ? *e : kNone, VK_IMAGE_VIEW_TYPE_CUBE, &not_ready);
        if (not_ready) pending.push_back(id);
    };
    std::shared_lock<std::shared_mutex> lk;
    if (tp) lk = std::shared_lock<std::shared_mutex>(tp->lock);
    const size_t count = tp ? std::min<size_t>(tp->entries.size(), kPoolTextureCount) : 0;
    if (!ids) {
        flat.resize(kPoolTextureCount);
        cube.resize(kPoolTextureCount);
        for (uint32_t id = 0; id < kPoolTextureCount; ++id) element(id < count ? &tp->entries[id] : nullptr, id, id);
        size_t used = 0, n2d = 0, ncube = 0;
        const VkImageView d2 = dummy_view(VK_IMAGE_VIEW_TYPE_2D, false), dc = dummy_view(VK_IMAGE_VIEW_TYPE_CUBE, false);
        for (uint32_t id = 0; id < count; ++id) {
            used += tp->entries[id].used ? 1 : 0;
            n2d += flat[id].imageView != d2 ? 1 : 0;
            ncube += cube[id].imageView != dc ? 1 : 0;
        }
        NXLOG_I(kTag, "pool arrays: set written whole: %zu of %zu ids registered, %zu sampled as 2D, %zu as cubes, %zu waiting for their sampled layout",
                used, count, n2d, ncube, pending.size());
        for (uint32_t b : {uint32_t(kPoolTex2D), uint32_t(kPoolTexCube)}) {
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = c.set;
            w.dstBinding = b;
            w.descriptorCount = kPoolTextureCount;
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            w.pImageInfo = b == kPoolTex2D ? flat.data() : cube.data();
            writes.push_back(w);
        }
        c.pending.clear();
    } else {
        flat.resize(ids->size());
        cube.resize(ids->size());
        for (size_t k = 0; k < ids->size(); ++k) {
            const uint32_t id = (*ids)[k];
            element(id < count ? &tp->entries[id] : nullptr, id, k);
            for (uint32_t b : {uint32_t(kPoolTex2D), uint32_t(kPoolTexCube)}) {
                VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = c.set;
                w.dstBinding = b;
                w.dstArrayElement = id;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                w.pImageInfo = b == kPoolTex2D ? &flat[k] : &cube[k];
                writes.push_back(w);
            }
        }
        // the ids written now are no longer pending (unless they still are)
        c.pending.erase(std::remove_if(c.pending.begin(), c.pending.end(), [&](uint32_t id) { return std::find(ids->begin(), ids->end(), id) != ids->end(); }),
                        c.pending.end());
    }
    lk = {};
    vkUpdateDescriptorSets(vkdev(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    g_pool_writes.add(static_cast<int64_t>(ids ? ids->size() * 2 : 2ull * kPoolTextureCount));
    c.pending.insert(c.pending.end(), pending.begin(), pending.end());
}

void write_pool_samplers(PoolCopy& c, SamplerPool* sp, const std::vector<uint32_t>* ids) {
    static std::vector<VkDescriptorImageInfo> info;
    std::vector<VkWriteDescriptorSet> writes;
    std::shared_lock<std::shared_mutex> lk;
    if (sp) lk = std::shared_lock<std::shared_mutex>(sp->lock);
    auto element = [&](uint32_t id) {
        VkDescriptorImageInfo ii = {dummy_sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        if (sp && id < sp->used.size() && sp->used[id]) ii.sampler = sampler_for(sp->entries[id]);
        return ii;
    };
    if (!ids) {
        info.resize(kPoolSamplerCount);
        for (uint32_t id = 0; id < kPoolSamplerCount; ++id) info[id] = element(id);
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = c.set;
        w.dstBinding = kPoolSamplers;
        w.descriptorCount = kPoolSamplerCount;
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w.pImageInfo = info.data();
        writes.push_back(w);
    } else {
        info.resize(ids->size());
        for (size_t k = 0; k < ids->size(); ++k) {
            info[k] = element((*ids)[k]);
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = c.set;
            w.dstBinding = kPoolSamplers;
            w.dstArrayElement = (*ids)[k];
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            w.pImageInfo = &info[k];
            writes.push_back(w);
        }
    }
    lk = {};
    vkUpdateDescriptorSets(vkdev(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    g_pool_writes.add(static_cast<int64_t>(ids ? ids->size() : kPoolSamplerCount));
}

// The ids registered after `from` up to the pool's current version (sorted, unique); false when the change list no longer reaches back
// that far (write every id).
template <class Pool>
bool changed_since(Pool* p, uint64_t from, uint64_t* now, std::vector<uint32_t>& ids) {
    ids.clear();
    if (!p) { *now = 0; return true; }
    std::shared_lock<std::shared_mutex> lk(p->lock);
    *now = p->version;
    const uint64_t n = p->version - from;
    if (from > p->version || n > p->changes.size()) return false;
    ids.assign(p->changes.end() - static_cast<std::ptrdiff_t>(n), p->changes.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    ids.erase(std::remove_if(ids.begin(), ids.end(), [](uint32_t id) { return id >= kPoolTextureCount; }), ids.end());
    return true;
}

// True when some pending element of the set has become writable.
bool pending_ready(const PoolCopy& c, TexturePool* tp) {
    if (c.pending.empty() || !tp) return false;
    std::shared_lock<std::shared_mutex> lk(tp->lock);
    for (uint32_t id : c.pending) {
        if (id >= tp->entries.size()) continue;
        Texture* t = tp->entries[id].used ? tp->entries[id].tex.get() : nullptr;
        if (!t || (t->gpu && tex_layout(t) == tex_sampled_layout(t))) return true;
    }
    return false;
}

void update_copy(PoolCopy& c, TexturePool* tp, SamplerPool* sp) {
    std::vector<uint32_t> ids;
    uint64_t tv = 0, sv = 0;
    if (!c.written) {
        changed_since(tp, 0, &tv, ids);
        changed_since(sp, 0, &sv, ids);
        write_pool_textures(c, tp, nullptr);
        write_pool_samplers(c, sp, nullptr);
        c.written = true;
    } else {
        if (changed_since(tp, c.tex_version, &tv, ids)) {
            for (uint32_t id : c.pending) ids.push_back(id);
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            if (!ids.empty()) write_pool_textures(c, tp, &ids);
        } else {
            write_pool_textures(c, tp, nullptr);
        }
        if (changed_since(sp, c.smp_version, &sv, ids)) {
            ids.erase(std::remove_if(ids.begin(), ids.end(), [](uint32_t id) { return id >= kPoolSamplerCount; }), ids.end());
            if (!ids.empty()) write_pool_samplers(c, sp, &ids);
        } else {
            write_pool_samplers(c, sp, nullptr);
        }
    }
    c.tex_version = tv;
    c.smp_version = sv;
}

PoolPair* pool_pair(TexturePool* tp, SamplerPool* sp) {
    for (auto& p : g_pool_pairs)
        if (p->tp.get() == tp && p->sp.get() == sp) return p.get();
    auto p = std::make_unique<PoolPair>();
    p->tp = Ref<TexturePool>::share(tp);
    p->sp = Ref<SamplerPool>::share(sp);
    const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_SAMPLER, g_pool_copies * kPoolSamplerCount},
                                           {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, g_pool_copies * 2 * kPoolTextureCount}};
    VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    ci.maxSets = g_pool_copies;
    ci.poolSizeCount = 2;
    ci.pPoolSizes = sizes;
    const VkResult r = vkCreateDescriptorPool(vkdev(), &ci, nullptr, &p->dpool);
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "pool arrays: descriptor pool for %u sets failed (%d)", g_pool_copies, static_cast<int>(r));
        return nullptr;
    }
    vk_gauge_descriptor_pools().add(1);
    g_pool_pairs.push_back(std::move(p));
    return g_pool_pairs.back().get();
}

PoolRes& pool_res(TexturePool* tp) {
    for (auto& r : g_pool_res)
        if (r->tp.get() == tp) return *r;
    auto r = std::make_unique<PoolRes>();
    r->tp = Ref<TexturePool>::share(tp);
    g_pool_res.push_back(std::move(r));
    return *g_pool_res.back();
}

// Drops the pairs and texture pool records nothing used for kPoolIdle submissions (their sets once the GPU is done with them).
void pool_prune(uint64_t serial) {
    static uint64_t last = 0;
    if (serial - last < 256) return;
    last = serial;
    for (auto it = g_pool_pairs.begin(); it != g_pool_pairs.end();) {
        if (serial - (*it)->last_use < kPoolIdle) { ++it; continue; }
        const VkDescriptorPool dp = (*it)->dpool;
        g_pool_sets.sub(static_cast<int64_t>((*it)->copies.size()));
        defer_destroy([dp] { vkDestroyDescriptorPool(vkdev(), dp, nullptr); vk_gauge_descriptor_pools().sub(1); });
        it = g_pool_pairs.erase(it);
    }
    g_pool_res.erase(std::remove_if(g_pool_res.begin(), g_pool_res.end(), [&](const std::unique_ptr<PoolRes>& r) { return serial - r->last_use >= kPoolIdle; }),
                     g_pool_res.end());
}

}  // namespace

VkDescriptorSetLayout pool_set_layout() {
    std::lock_guard<std::mutex> lk(g_pool_layout_mu);
    if (g_pool_layout_tried) return g_pool_layout;
    g_pool_layout_tried = true;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    vkGetPhysicalDeviceFeatures2(device().physical(), &f2);
    VkPhysicalDeviceVulkan12Properties p12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p12};
    vkGetPhysicalDeviceProperties2(device().physical(), &p2);
    // the pool arrays plus room for a program's own textures and samplers (sets 2 and 3) in the same stage
    const uint64_t images = 2ull * kPoolTextureCount, per_set = images + kPoolSamplerCount;
    const bool feats = f12.descriptorIndexing && f12.descriptorBindingPartiallyBound && f12.descriptorBindingSampledImageUpdateAfterBind &&
                       f12.shaderSampledImageArrayNonUniformIndexing;
    const bool limits = p12.maxDescriptorSetUpdateAfterBindSampledImages >= images + 512 && p12.maxPerStageDescriptorUpdateAfterBindSampledImages >= images + 256 &&
                        p12.maxDescriptorSetUpdateAfterBindSamplers >= kPoolSamplerCount + 512 && p12.maxPerStageDescriptorUpdateAfterBindSamplers >= kPoolSamplerCount + 256 &&
                        p12.maxPerStageUpdateAfterBindResources >= per_set + 512 && p12.maxUpdateAfterBindDescriptorsInAllPools >= 2 * per_set;
    g_pool_copies = static_cast<uint32_t>(std::clamp<uint64_t>(p12.maxUpdateAfterBindDescriptorsInAllPools / (2 * per_set), 2, 8));
    NXLOG_I(kTag, "pool arrays: descriptor indexing %d, partially bound %d, sampled images update after bind %d, non-uniform indexing %d; update-after-bind limits: "
            "%u sampled images per set, %u per stage, %u samplers per set, %u per stage, %u resources per stage, %u in all pools: %s",
            f12.descriptorIndexing, f12.descriptorBindingPartiallyBound, f12.descriptorBindingSampledImageUpdateAfterBind, f12.shaderSampledImageArrayNonUniformIndexing,
            p12.maxDescriptorSetUpdateAfterBindSampledImages, p12.maxPerStageDescriptorUpdateAfterBindSampledImages, p12.maxDescriptorSetUpdateAfterBindSamplers,
            p12.maxPerStageDescriptorUpdateAfterBindSamplers, p12.maxPerStageUpdateAfterBindResources, p12.maxUpdateAfterBindDescriptorsInAllPools,
            feats && limits ? "usable" : "NOT usable: programs that sample through the pools are not drawn");
    if (!feats || !limits) return VK_NULL_HANDLE;
    const VkDescriptorSetLayoutBinding b[kPoolBindingCount] = {
        {kPoolSamplers, VK_DESCRIPTOR_TYPE_SAMPLER, kPoolSamplerCount, kPoolStages, nullptr},
        {kPoolTex2D, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kPoolTextureCount, kPoolStages, nullptr},
        {kPoolTexCube, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kPoolTextureCount, kPoolStages, nullptr}};
    const VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    const VkDescriptorBindingFlags fl[kPoolBindingCount] = {flags, flags, flags};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bf.bindingCount = kPoolBindingCount;
    bf.pBindingFlags = fl;
    VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bf};
    ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    ci.bindingCount = kPoolBindingCount;
    ci.pBindings = b;
    if (vkCreateDescriptorSetLayout(vkdev(), &ci, nullptr, &g_pool_layout) != VK_SUCCESS) {
        NXLOG_E(kTag, "pool arrays: descriptor set layout failed");
        g_pool_layout = VK_NULL_HANDLE;
    }
    return g_pool_layout;
}

VkDescriptorSet pool_set(TexturePool* tp, SamplerPool* sp, Frame& f) {
    if (!pool_set_layout()) return VK_NULL_HANDLE;
    pool_prune(f.serial);
    PoolPair* pp = pool_pair(tp, sp);
    if (!pp) return VK_NULL_HANDLE;
    pp->last_use = f.serial;
    auto bind = [&](int i) {
        PoolCopy& c = pp->copies[static_cast<size_t>(i)];
        if (c.used != f.serial) { c.used_before = c.used; c.used = f.serial; }
        pp->cur = i;
        return c.set;
    };
    const uint64_t tv = version_of(tp), sv = version_of(sp);
    if (pp->cur >= 0) {
        const PoolCopy& c = pp->copies[static_cast<size_t>(pp->cur)];
        if (c.written && c.tex_version == tv && c.smp_version == sv && !pending_ready(c, tp)) return bind(pp->cur);
    }
    // a set no submission in flight (nor this open recording) reads: prefer the current idle copy if valid, else any idle copy
    const uint64_t done = poll_completed();
    auto idle = [&](const PoolCopy& c) { return c.used != f.serial && c.used <= done; };
    int pick = pp->cur >= 0 && idle(pp->copies[static_cast<size_t>(pp->cur)]) ? pp->cur : -1;
    for (size_t i = 0; pick < 0 && i < pp->copies.size(); ++i)
        if (idle(pp->copies[i]) && pp->copies[i].written) pick = static_cast<int>(i);
    if (pick < 0 && pp->copies.size() < g_pool_copies) {
        PoolCopy c;
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pp->dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g_pool_layout;
        if (vkAllocateDescriptorSets(vkdev(), &ai, &c.set) == VK_SUCCESS) {
            device().set_name(VK_OBJECT_TYPE_DESCRIPTOR_SET, reinterpret_cast<uint64_t>(c.set), "nvn pool arrays");
            pp->copies.push_back(std::move(c));
            g_pool_sets.add(1);
            pick = static_cast<int>(pp->copies.size() - 1);
        }
    }
    if (pick < 0) {
        // every copy is in flight or bound in this recording: wait for the oldest prior submission (never wait on f.serial itself, which is still recording!)
        size_t oldest = pp->copies.size();
        for (size_t i = 0; i < pp->copies.size(); ++i) {
            if (pp->copies[i].used == f.serial) continue;
            if (oldest == pp->copies.size() || pp->copies[i].used < pp->copies[oldest].used) oldest = i;
        }
        if (oldest < pp->copies.size()) {
            static int logged = 0;
            if (logged++ < 10) NXLOG_W(kTag, "pool arrays: all %zu sets of a pool pair are in flight; waiting for submission %llu", pp->copies.size(),
                                       static_cast<unsigned long long>(pp->copies[oldest].used));
            submit_recorded();
            while (!device().device_lost() && poll_completed() < pp->copies[oldest].used) std::this_thread::yield();
            if (device().device_lost())
                return pp->cur >= 0 ? pp->copies[static_cast<size_t>(pp->cur)].set : pp->copies[oldest].set;
            pick = static_cast<int>(oldest);
        } else {
            // Every copy was bound earlier inside this unsubmitted recording; waiting on f.serial would deadlock.
            static int logged_same = 0;
            if (logged_same++ < 10)
                NXLOG_W(kTag, "pool arrays: all %zu sets bound in open submission %llu; updating update-after-bind copy",
                        pp->copies.size(), static_cast<unsigned long long>(f.serial));
            pick = (pp->cur == 0 && pp->copies.size() > 1) ? 1 : 0;
        }
    }
    update_copy(pp->copies[static_cast<size_t>(pick)], tp, sp);
    return bind(pick);
}

void pool_note_layout(Texture* t, VkImageLayout layout) {
    if (!t || g_pool_res.empty() || layout == tex_sampled_layout(t)) return;   // nothing samples through a pool yet
    if (g_layout_moves.size() >= 4096) {
        for (auto& r : g_pool_res) r->all = true;
        g_layout_moves.clear();
    }
    g_layout_moves.push_back(Ref<Texture>::share(t));
}

void pool_texture_remade(Texture* t) {
    if (!t) return;
    const TexGpu* g = t->gpu;   // textures sharing the image hold the same views
    for (auto& pp : g_pool_pairs) {
        TexturePool* tp = pp->tp.get();
        if (!tp) continue;
        if (!g) {   // released (many at a scene change): skip the scan when the pool's record says t is not registered in it
            const PoolRes* r = nullptr;
            for (auto& x : g_pool_res)
                if (x->tp.get() == tp) r = x.get();
            if (r && !r->reg.count(t) && r->version == version_of(tp)) continue;
        }
        std::unique_lock<std::shared_mutex> lk(tp->lock);
        const size_t n = std::min<size_t>(tp->entries.size(), kPoolTextureCount);
        for (size_t id = 0; id < n; ++id) {
            Texture* x = tp->entries[id].used ? tp->entries[id].tex.get() : nullptr;
            if (x && (x == t || (g && x->gpu == g))) pool_note_change(*tp, static_cast<int>(id));
        }
    }
}

void pool_textures_to_prepare(TexturePool* tp, std::vector<Texture*>& out) {
    if (!tp) return;
    PoolRes& r = pool_res(tp);
    r.last_use = current_serial();
    for (const Ref<Texture>& t : g_layout_moves)        // moved since the last call: the pools they are registered in look at them
        for (auto& pr : g_pool_res)
            if (pr->reg.count(t.get())) pr->watch.push_back(t);
    g_layout_moves.clear();
    {
        std::shared_lock<std::shared_mutex> lk(tp->lock);
        if (r.version != tp->version) {
            const uint64_t n = tp->version - r.version;
            const bool partial = r.version != ~0ull && r.version <= tp->version && n <= tp->changes.size() && r.by_id.size() == tp->entries.size();
            auto set_id = [&](uint32_t id) {
                const TexturePool::Entry& e = tp->entries[id];
                Texture* now = e.used ? e.tex.get() : nullptr;
                Texture*& was = r.by_id[id];
                if (was == now) return;
                if (was) {
                    auto it = r.reg.find(was);
                    if (it != r.reg.end() && --it->second.second == 0) r.reg.erase(it);
                }
                if (now) {
                    auto& slot = r.reg[now];
                    if (!slot.second++) slot.first = Ref<Texture>::share(now);
                    if (partial) r.watch.push_back(slot.first);
                }
                was = now;
            };
            if (partial) {
                for (size_t k = tp->changes.size() - n; k < tp->changes.size(); ++k)
                    if (tp->changes[k] < tp->entries.size()) set_id(tp->changes[k]);
            } else {
                r.by_id.assign(tp->entries.size(), nullptr);
                r.reg.clear();
                for (uint32_t id = 0; id < tp->entries.size(); ++id) set_id(id);
                r.all = true;
            }
            r.version = tp->version;
        }
    }
    const uint64_t ep = g_tex_scan_epoch.load(std::memory_order_relaxed);
    if (ep != r.epoch) {
        r.all = true;
        r.epoch = ep;
    }
    const size_t first = out.size();
    auto look = [&](Texture* t) {
        if ((t->destroyed.load(std::memory_order_relaxed) && !t->gpu && !texture_registry_find_alias(t)) || !tex_gpu(t)) return;
        if (t->cpu_dirty.load(std::memory_order_relaxed) || tex_layout(t) != tex_sampled_layout(t)) out.push_back(t);
    };
    if (r.all) {
        for (auto& [t, e] : r.reg) look(t);
        r.all = false;
    } else {
        for (const Ref<Texture>& t : r.watch) look(t.get());
    }
    // not ready now (the replay leaves an attachment of the open pass as it is): looked at again next time
    r.watch.clear();
    for (size_t i = first; i < out.size(); ++i) r.watch.push_back(Ref<Texture>::share(out[i]));
}

bool pool_state_moved(TexturePool* tp, SamplerPool* sp) {
    if (!g_layout_moves.empty()) return true;
    PoolRes* r = nullptr;
    for (auto& x : g_pool_res)
        if (x->tp.get() == tp) r = x.get();
    if (tp && (!r || r->all || !r->watch.empty() || r->epoch != g_tex_scan_epoch.load(std::memory_order_relaxed) || r->version != version_of(tp))) return true;
    for (auto& p : g_pool_pairs) {
        if (p->tp.get() != tp || p->sp.get() != sp) continue;
        if (p->cur < 0) return true;
        const PoolCopy& c = p->copies[static_cast<size_t>(p->cur)];
        return c.tex_version != version_of(tp) || c.smp_version != version_of(sp) || pending_ready(c, tp);
    }
    return true;
}

void pool_sets_shutdown() {
    VkDevice d = vkdev();
    for (auto& p : g_pool_pairs) {
        if (p->dpool) { vkDestroyDescriptorPool(d, p->dpool, nullptr); vk_gauge_descriptor_pools().sub(1); }
        g_pool_sets.sub(static_cast<int64_t>(p->copies.size()));
    }
    g_pool_pairs.clear();
    g_pool_res.clear();
    g_layout_moves.clear();
    std::lock_guard<std::mutex> lk(g_pool_layout_mu);
    if (g_pool_layout) vkDestroyDescriptorSetLayout(d, g_pool_layout, nullptr);
    g_pool_layout = VK_NULL_HANDLE;
    g_pool_layout_tried = false;
}

size_t pipeline_count() { return static_cast<size_t>(g_pipeline_count.get()); }

// Collect every live VkShaderModule across all seven maps before destroying so shared fallback handles (variant maps returning
// shader_module(s)) are destroyed at most once and no map retains a dangling handle between gpl_prepare batches. Caller holds g_module_mu.
static void clear_all_shader_modules_locked(VkDevice d) {
    std::unordered_set<VkShaderModule> unique;
    auto drain = [&](auto& map) {
        for (auto& [k, h] : map)
            if (h) unique.insert(h);
        map.clear();
    };
    drain(g_modules);
    drain(g_variants);
    drain(g_dual_src);
    drain(g_hdr_mod);
    drain(g_int_out_mod);
    for (auto& [k, e] : g_no_out)
        if (e.first) unique.insert(e.first);
    g_no_out.clear();
    for (auto& [k, v] : g_xfb_mod)
        if (v.first) unique.insert(v.first);
    g_xfb_mod.clear();
    for (VkShaderModule h : unique) vkDestroyShaderModule(d, h, nullptr);
    std::lock_guard<std::mutex> lk2(g_module_id_mu);
    g_module_ids.clear();
    g_module_subgroups.clear();
}

void module_cache_clear() {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    clear_all_shader_modules_locked(vkdev());
}

uint64_t module_content_id(VkShaderModule m) {
    std::lock_guard<std::mutex> lk(g_module_id_mu);
    auto it = g_module_ids.find(m);
    return it != g_module_ids.end() ? it->second : reinterpret_cast<uint64_t>(m);   // not made by create_module: the handle stands for itself
}

uint64_t pipelines_built_at_draw() { return g_built_at_draw.load(std::memory_order_relaxed); }

// [nvn] spv_hot: ShaderCache::hot_poll replaced this shader's SPIR-V words. Every module made from the old words goes, and every pipeline that
// used it, so the next draw that needs it builds both again from the new words. The program layouts (ProgLayout, the descriptor set layouts and
// the pipeline layout) are keyed by the ShaderInfo, whose address did not change and whose interface hot_poll checked: they stay valid.
void pipeline_hot_shader_changed(const ShaderInfo* s) {
    if (!s) return;
    const VkDevice d = vkdev();
    size_t dropped = 0;
    // The handles go through the deferred destruction the rest of the back end uses: they are freed once the submission being recorded now (the
    // last one that can still name them) has completed, so this costs no stall and is safe from the render thread inside a submission.
    auto later_pipeline = [d](VkPipeline p) {
        if (p) defer_destroy([d, p] { vkDestroyPipeline(d, p, nullptr); });
    };
    {
        std::lock_guard<std::recursive_mutex> lk(g_module_mu);
        // The modules are only dropped from the caches, not destroyed: several of those maps hand out the same handle (a variant maker that cannot
        // rewrite the shader stores shader_module(s) itself), so a destroy per map would destroy one module twice, and the pipeline-library
        // workers compile parts from these handles on background threads, which a serial deferral does not cover. A module is a few hundred
        // KiB and a debugging session swaps a handful of times, so they are left to the device teardown (pipeline_shutdown).
        auto drop = [&](VkShaderModule m) {
            if (!m) return;
            std::lock_guard<std::mutex> lk2(g_module_id_mu);
            g_module_ids.erase(m);
            g_module_subgroups.erase(m);
        };
        auto plain = [&](std::unordered_map<const ShaderInfo*, VkShaderModule>& m) {
            auto it = m.find(s);
            if (it == m.end()) return;
            drop(it->second);
            m.erase(it);
        };
        plain(g_modules);
        plain(g_dual_src);
        plain(g_int_out_mod);
        plain(g_hdr_mod);
        if (auto it = g_no_out.find(s); it != g_no_out.end()) {                  // (module, owned by this map)
            if (it->second.second) drop(it->second.first);
            g_no_out.erase(it);
        }
        for (auto it = g_xfb_mod.begin(); it != g_xfb_mod.end();) {                // transform feedback variants: (shader, slots)
            if (it->first.first != s) {
                ++it;
                continue;
            }
            drop(it->second.first);
            it = g_xfb_mod.erase(it);
        }
        for (auto it = g_variants.begin(); it != g_variants.end();) {            // producer modules: (shader, slots)
            if (it->first.first != s) {
                ++it;
                continue;
            }
            drop(it->second);
            it = g_variants.erase(it);
        }
    }
    for (auto it = g_gfx.begin(); it != g_gfx.end();) {
        auto& bucket = it->second;
        for (auto e = bucket.begin(); e != bucket.end();) {
            const GraphicsKey& k = e->first;
            if (k.stage[0] != s && k.stage[1] != s && k.stage[2] != s && k.tess[0] != s && k.tess[1] != s) {
                ++e;
                continue;
            }
            if (e->second) {
                later_pipeline(e->second);
                g_pipeline_count.sub(1);
                ++dropped;
            }
            e = bucket.erase(e);
            g_gfx_keys.sub(1);
        }
        if (bucket.empty()) it = g_gfx.erase(it);
        else ++it;
    }
    for (auto it = g_compute.begin(); it != g_compute.end();) {
        if (it->first.first != s) {
            ++it;
            continue;
        }
        if (it->second) {
            later_pipeline(it->second);
            g_pipeline_count.sub(1);
            ++dropped;
        }
        it = g_compute.erase(it);
        g_compute_keys.sub(1);
    }
    ++g_gfx_fast_gen;
    prewarm_hot_disable();   // the key file's pipelines and the parts built ahead were made from the old words
    NXLOG_I(kTag, "spv_hot: %s: modules dropped, %zu pipeline(s) destroyed, prewarm off for this run", s->hash.c_str(), dropped);
}

void pipeline_shutdown() {
    if (g_gfx_fast_hits) {
        counters().pipeline_cache_hits.fetch_add(g_gfx_fast_hits, std::memory_order_relaxed);
        g_gfx_fast_hits = 0;
    }
    ++g_gfx_fast_gen;
    g_layout_gen.fetch_add(1, std::memory_order_relaxed);
    VkDevice d = vkdev();
    prewarm_shutdown();
    gpl_stop();
    if (!g_cache_big || g_new_since_save.load() >= 300) save_cache();
    for (auto& [h, bucket] : g_gfx)
        for (auto& e : bucket) {
            if (e.second) vkDestroyPipeline(d, e.second, nullptr);
            if (e.alt) vkDestroyPipeline(d, e.alt, nullptr);
            if (e.up && e.up->state.load() == 1 && e.up->pipe) vkDestroyPipeline(d, e.up->pipe, nullptr);     // built and never taken
        }
    g_gfx.clear();
    g_gfx_keys.sub(g_gfx_keys.get());
    gpl_shutdown();                              // the parts the pipelines were linked from
    for (auto& [k, p] : g_compute)
        if (p) vkDestroyPipeline(d, p, nullptr);
    g_compute.clear();
    g_compute_keys.sub(g_compute_keys.get());
    g_layouts.clear();
    for (auto& [k, l] : g_shared_layouts) {          // the handles in g_layouts were copies of these
        if (l.layout) vkDestroyPipelineLayout(d, l.layout, nullptr);
        for (int set = 0; set < 4; ++set) {          // kPoolSet's is pool_sets_shutdown's
            if (l.templates[set] && vkDestroyDescriptorUpdateTemplate) vkDestroyDescriptorUpdateTemplate(d, l.templates[set], nullptr);
            if (l.sets[set]) vkDestroyDescriptorSetLayout(d, l.sets[set], nullptr);
        }
    }
    g_shared_layouts.clear();
    pool_sets_shutdown();
    {
        std::lock_guard<std::recursive_mutex> lk(g_module_mu);
        clear_all_shader_modules_locked(d);
    }
    for (auto& [k, b] : g_const_bufs) device().destroy_buffer(b.first);
    g_const_bufs.clear();
    g_const_n.sub(g_const_n.get());
    g_const_bytes.sub(g_const_bytes.get());
    if (g_driver_const_made) device().destroy_buffer(g_driver_const);
    g_driver_const_made = false;
    {
        std::unique_lock<std::shared_mutex> lk(g_cache_rw_mu);
        if (g_cache) vkDestroyPipelineCache(d, g_cache, nullptr);
        if (g_gpl_cache) vkDestroyPipelineCache(d, g_gpl_cache, nullptr);
        g_cache = VK_NULL_HANDLE;
        g_gpl_cache = VK_NULL_HANDLE;
        g_cache_tried = false;
        g_gpl_cache_tried = false;
        g_cache_ready.store(false);
        g_gpl_cache_ready.store(false);
        g_cache_save_on.store(true);
        g_cache_dirty.store(false);
        g_new_since_save.store(0);
        g_cache_path.clear();
    }
    g_pipeline_count.sub(g_pipeline_count.get());
}

// Gauges (nxrt/gauge.h). The callbacks read the layout and module maps under the locks that guard them; they are removed before those maps
// are destroyed.
namespace {
struct PipelineGauges {
    PipelineGauges() {
        nxrt::gauge_register("nvn.pipeline_layouts", [] {
            std::lock_guard<std::mutex> lk(g_layout_mu);
            return static_cast<int64_t>(g_layouts.size());
        });
        nxrt::gauge_register("nvn.shared_layouts", [] {
            std::lock_guard<std::mutex> lk(g_layout_mu);
            return static_cast<int64_t>(g_shared_layouts.size());
        });
        nxrt::gauge_register("nvn.shader_modules", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_modules.size());
        });
        nxrt::gauge_register("nvn.shader_modules.variants", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_variants.size());
        });
        nxrt::gauge_register("nvn.shader_modules.dual_src", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_dual_src.size());
        });
        nxrt::gauge_register("nvn.shader_modules.no_out", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_no_out.size());
        });
        nxrt::gauge_register("nvn.shader_modules.xfb", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_xfb_mod.size());
        });
    }
    ~PipelineGauges() {
        for (const char* n : {"nvn.pipeline_layouts", "nvn.shared_layouts", "nvn.shader_modules", "nvn.shader_modules.variants",
                              "nvn.shader_modules.dual_src", "nvn.shader_modules.no_out", "nvn.shader_modules.xfb"})
            nxrt::gauge_unregister(n);
    }
};
PipelineGauges g_pipeline_gauges;
}  // namespace

}  // namespace nxrt::nvn::be::gpu
