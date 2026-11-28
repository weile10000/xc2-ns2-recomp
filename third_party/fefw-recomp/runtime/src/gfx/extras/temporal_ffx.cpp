// SPDX-License-Identifier: GPL-3.0-or-later
// AMD FSR 3.1 through the FidelityFX API (temporal.h). The Vulkan DLLs are loaded at run time: the FSR SDK v2.4.0 pair
// amd_fidelityfx_loader_vk.dll + amd_fidelityfx_upscaler_vk.dll (MIT, signed), or the v1.1.4 amd_fidelityfx_vk.dll; [upscale] fsr_dll names
// one explicitly, otherwise upscalers\fsr3\ (the release package) or the executable's folder is searched. The upscaler version is the newest the DLL offers for this GPU (FSR 3.1.x; FSR 4 where AMD supports it).
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "extras/temporal.h"
#include "nxrt/config.h"
#include "nxrt/extras.h"
#include "nxrt/gfx.h"
#include "nxrt/log.h"

#include "../../../third_party/amd_ffx_api/api/include/ffx_api.h"
#include "../../../third_party/amd_ffx_api/api/include/ffx_api_types.h"
#include "../../../third_party/amd_ffx_api/api/include/vk/ffx_api_vk.h"
#include "../../../third_party/amd_ffx_api/upscalers/include/ffx_upscale.h"

namespace nxrt::extras {
namespace {

constexpr const char* kTag = "fsr3";

std::mutex g_ver_mu;
std::string g_live_want_version;
std::string g_active_version;
std::atomic<uint32_t> g_ffx_gen{0};

std::string effective_want_version() {
    {
        std::lock_guard<std::mutex> lk(g_ver_mu);
        if (!g_live_want_version.empty()) return g_live_want_version;
    }
    return nxrt::config_get_str("upscale", "fsr_version", "3.1");
}

std::wstring exe_dir() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring p(buf, n);
    const size_t s = p.find_last_of(L"\\/");
    return s == std::wstring::npos ? std::wstring() : p.substr(0, s + 1);
}
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}
bool exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }

// The candidates, in order: [upscale] fsr_dll, then the release package's upscalers\fsr3\ folder next to the executable, then the executable's own
// folder; in each the v2.4.0 loader, the v2.4.0 standalone upscaler, then the v1.1.4 single DLL.
//
// The upscaler exports the same five ffx* entry points as the loader (measured 2026-11-19 with llvm-readobj), so it can serve as
// the entry point when the 39 KB loader is absent and the player only dropped the one file they had (issue #10 follow-up).
// Preferring it before amd_fidelityfx_vk.dll avoids loading an incomplete v1/FG API wrapper when both sit next to each other
// without the loader (issue #18). The loader still wins when present: it is the documented front door.
std::wstring find_dll() {
    const std::string cfg = settings().upscale.fsr_dll;
    if (!cfg.empty()) return exists(widen(cfg)) ? widen(cfg) : std::wstring();
    const std::wstring exe = exe_dir();
    for (const std::wstring& dir : {exe + L"upscalers\\fsr3\\", exe})
        for (const wchar_t* n : {L"amd_fidelityfx_loader_vk.dll", L"amd_fidelityfx_upscaler_vk.dll", L"amd_fidelityfx_vk.dll"})
            if (exists(dir + n)) return dir + n;
    return {};
}

void on_message(uint32_t type, const wchar_t* msg) {
    char buf[512];
    WideCharToMultiByte(CP_UTF8, 0, msg ? msg : L"", -1, buf, sizeof buf, nullptr, nullptr);
    if (type == FFX_API_MESSAGE_TYPE_ERROR) NXLOG_E(kTag, "%s", buf);
    else NXLOG_W(kTag, "%s", buf);
}

FfxApiResource resource(const TemporalImage& t, uint32_t state, uint32_t usage) {
    FfxApiResource r = {};
    r.resource = reinterpret_cast<void*>(t.image);
    r.state = state;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = t.format == VK_FORMAT_R16G16B16A16_SFLOAT ? FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT
                         : t.format == VK_FORMAT_R16G16_SFLOAT     ? FFX_API_SURFACE_FORMAT_R16G16_FLOAT
                         : t.format == VK_FORMAT_R32_SFLOAT        ? FFX_API_SURFACE_FORMAT_R32_FLOAT
                                                                   : FFX_API_SURFACE_FORMAT_UNKNOWN;
    r.description.width = t.width;
    r.description.height = t.height;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage = usage;
    return r;
}

// --- FSR create/dispatch flags, A/B-able while the game runs (HANDOVER 95.9) -----------------------
// [upscale] fsr_version picks the provider; the flag set decides which *code path* the provider takes.
// xc2-recomp 0.1.13 (the NS1 recomp, unpacked in work\ref\xc2-0113, runtime\vk_fsr3.c) runs FSR Upscaling
// 4.1.2 on this exact driver and GPU with FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE alone -- no HDR, no auto
// exposure -- while we enable HDR + auto exposure on a linear RGBA16F buffer. The HDR/auto-exposure path is
// therefore the first suspect for the amdvlk crash, and the cheapest way to test it is to switch the flags.
// Read from a file next to [dlss] hot_file (work\fsr_flags.txt) every time a context is made:
//   hdr = 0|1            default 1   FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE
//   auto_exposure = 0|1  default 1   FFX_UPSCALE_ENABLE_AUTO_EXPOSURE
//   non_linear = 0|1     default 0   FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE (+ the dispatch's sRGB flag)
// A new context is needed for a change to apply: tools\run\up.cmd -> 1 (taau) then 4 (fsr3).
struct FsrFlags {
    bool hdr = true;
    bool auto_exposure = true;
    bool non_linear = false;
};

std::string flag_file_path() {
    const char* hot = std::getenv("NXRT_DLSS_HOT_FILE");
    std::string p = hot && *hot ? hot : "";
    if (p.empty()) return {};
    const size_t cut = p.find_last_of("\\/");
    return (cut == std::string::npos ? std::string() : p.substr(0, cut + 1)) + "fsr_flags.txt";
}

// "key = value": 1/0, true/false, on/off, yes/no. Returns false when the line names another key.
bool flag_line(const char* line, const char* key, bool* value) {
    const size_t n = std::strlen(key);
    if (strncmp(line, key, n) != 0) return false;
    const char* v = line + n;
    while (*v == ' ' || *v == '\t' || *v == '=') ++v;
    *value = *v == '1' || (*v == 't') || (*v == 'y') || (*v == 'o' && v[1] == 'n');
    return true;
}

FsrFlags read_fsr_flags() {
    FsrFlags fl;
    const std::string path = flag_file_path();
    if (path.empty()) return fl;
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) return fl;
    char line[256];
    while (std::fgets(line, sizeof line, fp)) {
        for (char* c = line; *c; ++c)
            if (*c >= 'A' && *c <= 'Z') *c = static_cast<char>(*c - 'A' + 'a');
        bool v = false;
        if (flag_line(line, "hdr", &v)) fl.hdr = v;
        else if (flag_line(line, "auto_exposure", &v)) fl.auto_exposure = v;
        else if (flag_line(line, "non_linear", &v)) fl.non_linear = v;
    }
    std::fclose(fp);
    return fl;
}

// --- FidelityFX barrier watchdog (HANDOVER 95, amdvlk crash) -------------------------------------
// amdvlk (Adrenalin 26.10.1, amdvlk64.dll 9.2.10.353, Vulkan 1.4.315 / driver 2.0.353) crashes with
// 0xC0000005 reading 0x250 inside vkCmdPipelineBarrier, on the barrier list the FSR 4.1.2 upscaler
// rebuilds during ffxDispatch: the driver dereferences a NULL resource from that list. Measured
// 4 crashes in 21.1 minutes of FSR 4.1.2 (15 sessions) against 0 in 236 minutes of FSR 3.1.6 / TAAU
// (docs 41, HANDOVER 95). Mode 1 on a crash session saw no NULL handle at all, so the pointer comes
// from inside the driver and dropping entries (mode 2) would not fix it; 3.1.6 is the default instead.
// This sits between FidelityFX and the driver, i.e. it is the vkDeviceProcAddr we hand to FFX:
//   [upscale] barrier_check = 0  wrapper installed, pass-through (default: no behaviour change)
//                             1  log every barrier that carries a NULL image or buffer handle
//                             2  log it and drop those entries before the driver sees them
using PfnCmdBarrier = void (*)(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t,
                               const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t,
                               const VkImageMemoryBarrier*);
PfnCmdBarrier g_barrier = nullptr;
VkDevice g_barrier_device = VK_NULL_HANDLE;
uint64_t g_barrier_calls = 0, g_barrier_null_images = 0, g_barrier_null_buffers = 0;
bool g_barrier_first_logged = false;

int barrier_mode() {
    static int mode = -1;
    if (mode < 0) mode = static_cast<int>(nxrt::config_get_int("upscale", "barrier_check", 0));
    return mode;
}

// mode >= 3 also tracks the images FidelityFX creates and destroys itself (it asks us for vkCreateImage through the
// same proc address), so a barrier can be printed together with what its image actually is. HANDOVER 95.10: the
// barrier that kills amdvlk is a one-off, UNDEFINED -> TRANSFER_SRC on an image that no earlier barrier mentions -- so
// "which image is that" is the question worth answering.
struct FfxImageInfo {
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0, mips = 0, layers = 0;
    VkImageUsageFlags usage = 0;
    bool destroyed = false;
};
std::map<uint64_t, FfxImageInfo> g_ffx_images;
uint64_t g_ffx_created = 0, g_ffx_destroyed = 0;

using PfnCreateImage = VkResult (*)(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*, VkImage*);
using PfnDestroyImage = void (*)(VkDevice, VkImage, const VkAllocationCallbacks*);
PfnCreateImage g_create_image = nullptr;
PfnDestroyImage g_destroy_image = nullptr;

VkResult image_created(VkDevice dev, const VkImageCreateInfo* ci, const VkAllocationCallbacks* alloc, VkImage* out) {
    if (!g_create_image) return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult r = g_create_image(dev, ci, alloc, out);
    if (r == VK_SUCCESS && out && *out && barrier_mode() >= 3) {
        FfxImageInfo info;
        info.format = ci->format;
        info.width = ci->extent.width;
        info.height = ci->extent.height;
        info.mips = ci->mipLevels;
        info.layers = ci->arrayLayers;
        info.usage = ci->usage;
        g_ffx_images[reinterpret_cast<uint64_t>(*out)] = info;
        ++g_ffx_created;
        NXLOG_I(kTag, "image #%llu created by FidelityFX: 0x%llx %ux%u format %d mips %u layers %u usage 0x%x", static_cast<unsigned long long>(g_ffx_created),
                static_cast<unsigned long long>(reinterpret_cast<uint64_t>(*out)), ci->extent.width, ci->extent.height, static_cast<int>(ci->format),
                ci->mipLevels, ci->arrayLayers, static_cast<unsigned>(ci->usage));
    }
    return r;
}

void image_destroyed(VkDevice dev, VkImage img, const VkAllocationCallbacks* alloc) {
    if (!g_destroy_image) return;
    const uint64_t h = reinterpret_cast<uint64_t>(img);
    auto it = g_ffx_images.find(h);
    if (it != g_ffx_images.end()) {
        it->second.destroyed = true;
        ++g_ffx_destroyed;
        if (barrier_mode() >= 3)
            NXLOG_I(kTag, "image destroyed by FidelityFX: 0x%llx (%llu destroyed so far)", static_cast<unsigned long long>(h),
                    static_cast<unsigned long long>(g_ffx_destroyed));
    }
    g_destroy_image(dev, img, alloc);
}

const char* image_describe(uint64_t h, char* buf, size_t n) {
    auto it = g_ffx_images.find(h);
    if (it == g_ffx_images.end()) {
        const char* app = gfx::image_trace_describe(h);   // one of the runtime's own images?
        snprintf(buf, n, "%s", app ? app : "not created through us");
        return buf;
    }
    const FfxImageInfo& i = it->second;
    snprintf(buf, n, "%ux%u fmt %d mips %u layers %u usage 0x%x%s", i.width, i.height, static_cast<int>(i.format), i.mips, i.layers,
             static_cast<unsigned>(i.usage), i.destroyed ? " DESTROYED" : "");
    return buf;
}

// mode 3: print the barrier's shape every time it *changes* (run length). FidelityFX records the same handful of
// barriers every frame, so a session produces a handful of lines instead of millions -- and the line right before a
// crash is the shape of the barrier that killed the driver.
uint64_t g_trace_img = 0, g_trace_buf = 0;
int g_trace_img_layout = -1, g_trace_img_aspect = -1;
VkDeviceSize g_trace_buf_range = 0;
uint64_t g_trace_stages = 0;

void trace_barrier(uint32_t mem_n, uint32_t buf_n, const VkBufferMemoryBarrier* bufs, uint32_t img_n, const VkImageMemoryBarrier* imgs,
                   VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    const uint64_t img = img_n && imgs ? reinterpret_cast<uint64_t>(imgs[0].image) : 0;
    const int lay = img_n && imgs ? static_cast<int>(static_cast<int>(imgs[0].oldLayout) * 32 + static_cast<int>(imgs[0].newLayout)) : -1;
    const int asp = img_n && imgs ? static_cast<int>(imgs[0].subresourceRange.aspectMask) : -1;
    const uint64_t buf = buf_n && bufs ? reinterpret_cast<uint64_t>(bufs[0].buffer) : 0;
    const VkDeviceSize range = buf_n && bufs ? bufs[0].offset ^ bufs[0].size : 0;
    const uint64_t stages = (static_cast<uint64_t>(src_stage) << 32) ^ dst_stage;
    if (img == g_trace_img && lay == g_trace_img_layout && asp == g_trace_img_aspect && buf == g_trace_buf && range == g_trace_buf_range &&
        stages == g_trace_stages)
        return;
    g_trace_img = img, g_trace_img_layout = lay, g_trace_img_aspect = asp;
    g_trace_buf = buf, g_trace_buf_range = range, g_trace_stages = stages;
    char desc[160] = "";
    if (img) image_describe(img, desc, sizeof desc);
    NXLOG_I(kTag, "barrier trace #%llu: %u memory, %u buffer, %u image; stages 0x%x -> 0x%x; image 0x%llx layout %u->%u aspect 0x%x%s [%s]; buffer 0x%llx off %llu size %llu%s",
            static_cast<unsigned long long>(g_barrier_calls), mem_n, buf_n, img_n, static_cast<unsigned>(src_stage),
            static_cast<unsigned>(dst_stage), static_cast<unsigned long long>(img), img_n && imgs ? static_cast<unsigned>(imgs[0].oldLayout) : 0,
            img_n && imgs ? static_cast<unsigned>(imgs[0].newLayout) : 0, img_n && imgs ? static_cast<unsigned>(imgs[0].subresourceRange.aspectMask) : 0,
            img_n > 1 ? " (+more)" : "", desc, static_cast<unsigned long long>(buf),
            buf_n && bufs ? static_cast<unsigned long long>(bufs[0].offset) : 0,
            buf_n && bufs ? static_cast<unsigned long long>(bufs[0].size) : 0, buf_n > 1 ? " (+more)" : "");
}

void cmd_pipeline_barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                          VkDependencyFlags flags, uint32_t mem_n, const VkMemoryBarrier* mem, uint32_t buf_n,
                          const VkBufferMemoryBarrier* bufs, uint32_t img_n, const VkImageMemoryBarrier* imgs) {
    if (!g_barrier && g_barrier_device != VK_NULL_HANDLE)
        g_barrier = reinterpret_cast<PfnCmdBarrier>(vkGetDeviceProcAddr(g_barrier_device, "vkCmdPipelineBarrier"));
    if (!g_barrier) return;
    ++g_barrier_calls;
    const int mode = barrier_mode();
    if (mode >= 1 && !g_barrier_first_logged) {
        g_barrier_first_logged = true;
        NXLOG_I(kTag, "barrier_check: FidelityFX records barriers through us (first one: %u memory, %u buffer, %u image)", mem_n,
                buf_n, img_n);
    }
    if (mode < 1) {
        g_barrier(cb, src_stage, dst_stage, flags, mem_n, mem, buf_n, bufs, img_n, imgs);
        return;
    }
    if (mode >= 3) trace_barrier(mem_n, buf_n, bufs, img_n, imgs, src_stage, dst_stage);
    uint32_t null_bufs = 0, null_imgs = 0;
    for (uint32_t i = 0; buf_n && i < buf_n; ++i) null_bufs += bufs[i].buffer == VK_NULL_HANDLE ? 1u : 0u;
    for (uint32_t i = 0; img_n && i < img_n; ++i) null_imgs += imgs[i].image == VK_NULL_HANDLE ? 1u : 0u;
    if (null_bufs || null_imgs) {
        const uint64_t seen = null_bufs ? ++g_barrier_null_buffers : ++g_barrier_null_images;
        if (seen <= 8 || seen % 100 == 0) {
            NXLOG_E(kTag, "barrier_check: barrier #%llu holds %u/%u NULL buffer and %u/%u NULL image handles (src 0x%x dst 0x%x dep 0x%x, %u memory); mode %d",
                    static_cast<unsigned long long>(g_barrier_calls), null_bufs, buf_n, null_imgs, img_n,
                    static_cast<unsigned>(src_stage), static_cast<unsigned>(dst_stage), static_cast<unsigned>(flags), mem_n, mode);
        }
        if (mode >= 2) {
            std::vector<VkBufferMemoryBarrier> kept_bufs;
            std::vector<VkImageMemoryBarrier> kept_imgs;
            for (uint32_t i = 0; buf_n && i < buf_n; ++i)
                if (bufs[i].buffer != VK_NULL_HANDLE) kept_bufs.push_back(bufs[i]);
            for (uint32_t i = 0; img_n && i < img_n; ++i)
                if (imgs[i].image != VK_NULL_HANDLE) kept_imgs.push_back(imgs[i]);
            g_barrier(cb, src_stage, dst_stage, flags, mem_n, mem, static_cast<uint32_t>(kept_bufs.size()),
                      kept_bufs.empty() ? nullptr : kept_bufs.data(), static_cast<uint32_t>(kept_imgs.size()),
                      kept_imgs.empty() ? nullptr : kept_imgs.data());
            return;
        }
    }
    g_barrier(cb, src_stage, dst_stage, flags, mem_n, mem, buf_n, bufs, img_n, imgs);
}

PFN_vkVoidFunction fx_device_proc(VkDevice dev, const char* name) {
    const PFN_vkVoidFunction p = vkGetDeviceProcAddr(dev, name);
    if (name && std::strcmp(name, "vkCmdPipelineBarrier") == 0) {
        g_barrier = reinterpret_cast<PfnCmdBarrier>(p);
        g_barrier_device = dev;
        if (!g_barrier) return nullptr;
        return reinterpret_cast<PFN_vkVoidFunction>(&cmd_pipeline_barrier);
    }
    if (name && std::strcmp(name, "vkCreateImage") == 0 && p) {
        g_create_image = reinterpret_cast<PfnCreateImage>(p);
        return reinterpret_cast<PFN_vkVoidFunction>(&image_created);
    }
    if (name && std::strcmp(name, "vkDestroyImage") == 0 && p) {
        g_destroy_image = reinterpret_cast<PfnDestroyImage>(p);
        return reinterpret_cast<PFN_vkVoidFunction>(&image_destroyed);
    }
    return p;
}

class Ffx final : public TemporalBackend {
public:
    const char* name() const override { return version_.empty() ? "FSR 3" : version_.c_str(); }

    bool load(gfx::Device& dev) {
        dev_ = &dev;
        const std::wstring path = find_dll();
        if (path.empty()) {
            NXLOG_W(kTag, "no FidelityFX DLL: put amd_fidelityfx_loader_vk.dll (39 KB) and amd_fidelityfx_upscaler_vk.dll (36 MB) from the "
                          "FidelityFX SDK v2.4.0 signedbin in upscalers\\fsr3 next to the executable, or name the DLL in [upscale] fsr_dll; "
                          "the upscaler alone is enough if that is the only file you have");
            return false;
        }
        mod_ = LoadLibraryW(path.c_str());
        if (!mod_) {
            // Error 126 is "module not found" for the DLL *or one of its dependencies*, so name the file that was tried:
            // a player cannot tell a wrong folder from a missing dependency otherwise (issue #18).
            const std::error_code ec(static_cast<int>(GetLastError()), std::system_category());
            char narrow_path[520] = {};
            WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, narrow_path, sizeof narrow_path - 1, nullptr, nullptr);
            NXLOG_E(kTag, "LoadLibrary(%s) failed: %s (error %lu). Error 126 means the file or one of the DLLs it needs is missing",
                    narrow_path, ec.message().c_str(), GetLastError());
            return false;
        }
        create_ = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(mod_, "ffxCreateContext"));
        destroy_ = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(mod_, "ffxDestroyContext"));
        query_ = reinterpret_cast<PfnFfxQuery>(GetProcAddress(mod_, "ffxQuery"));
        dispatch_ = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(mod_, "ffxDispatch"));
        if (!create_ || !destroy_ || !query_ || !dispatch_) {
            NXLOG_E(kTag, "the DLL does not export the FidelityFX API");
            return false;
        }
        char n[MAX_PATH];
        WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, n, sizeof n, nullptr, nullptr);
        NXLOG_I(kTag, "FidelityFX API loaded from %s", n);
        return true;
    }

    bool evaluate(VkCommandBuffer cb, const TemporalFrame& f) override {
        // HANDOVER 95.14: pin the game's dynamic resolution before the upscaler runs (no-op unless [upscale] pin_render_scale)
        nxrt::extras::pin_render_scale_tick();
        nxrt::extras::pin_render_size_tick();
        const uint32_t gen = g_ffx_gen.load(std::memory_order_acquire);
        if (gen != seen_gen_) {
            seen_gen_ = gen;
            failed_ = false;
            reset_contexts();
        }
        if (failed_) return false;
        // HANDOVER 95.19: the game's dynamic resolution changes the input size every few dozen frames, and FSR 4.1.2 on
        // amdvlk dies on the barrier it records a few milliseconds after such a change (docs/handover/41 95.10). The NS1
        // recomp never sees that because it tears the FidelityFX context down whenever the size changes; [upscale]
        // fsr_follow_size does the same here. Always enable per-size isolation for FSR 4.x to protect amdvlk.
        const std::string want_v = effective_want_version();
        bool follow_size = nxrt::config_get_bool("upscale", "fsr_follow_size", false) || (!want_v.empty() && want_v[0] == '4') ||
                           version_.rfind("FSR 4", 0) == 0;
        if (follow_size != follow_size_active_) {
            reset_contexts();
            follow_size_active_ = follow_size;
        }
        if (follow_size && !slot_for(f.render, f.out)) {
            failed_ = true;
            return false;
        }
        // A context this size has never seen starts with no history. The game only sets its own reset flag when it
        // changes scene, which is not the same moment: FSR 4's model reads uninitialised history and can keep writing
        // black for as long as that screen lasts (issue #96), so force the first dispatch of a new context to reset.
        bool force_reset = false;
        if (cur_slot_ < slots_.size()) {
            Slot& cs = slots_[cur_slot_];
            if (cs.fresh) { cs.fresh = false; force_reset = true; }
        }
        if (!follow_size && (!ctx_ || f.render.width > max_render_.width || f.render.height > max_render_.height || f.out.width != max_out_.width ||
                             f.out.height != max_out_.height)) {
            if (!make(f.render, f.out)) {
                failed_ = true;
                return false;
            }
            if (version_.rfind("FSR 4", 0) == 0 && !follow_size_active_) {
                // Provider resolved to FSR 4.x even without an explicit fsr_version = 4*: isolate future resizes immediately.
                follow_size_active_ = true;
                slots_.push_back(Slot{ctx_, f.render, f.out, ++slot_tick_});
                ctx_ = nullptr;
                active_ = &slots_.back().ctx;
            }
        }
        ffxDispatchDescUpscale d = {};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = reinterpret_cast<void*>(cb);
        d.color = resource(f.color, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
        d.depth = resource(f.depth, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
        d.motionVectors = resource(f.motion, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
        d.output = resource(f.output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
        d.jitterOffset = {f.jitter_px[0], f.jitter_px[1]};
        d.motionVectorScale = {f.mv_scale[0], f.mv_scale[1]};
        d.renderSize = {f.render.width, f.render.height};
        d.upscaleSize = {f.out.width, f.out.height};
        d.enableSharpening = f.sharpness > 0.0f;
        d.sharpness = f.sharpness;
        d.frameTimeDelta = f.frame_ms;
        d.preExposure = 1.0f;
        d.reset = f.reset || force_reset;
        d.cameraNear = f.near_z;   // FSR orders near / far itself from the inverted-depth flag
        d.cameraFar = f.far_z;
        d.cameraFovAngleVertical = f.fov_y;
        d.viewSpaceToMetersFactor = 1.0f;
        d.flags = non_linear_ ? FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB : 0u;
        static uint64_t told[4] = {0, 0, 0, 0};
        const uint64_t now[4] = {reinterpret_cast<uint64_t>(f.color.image), reinterpret_cast<uint64_t>(f.depth.image),
                                 reinterpret_cast<uint64_t>(f.motion.image), reinterpret_cast<uint64_t>(f.output.image)};
        if (told[0] != now[0] || told[1] != now[1] || told[2] != now[2] || told[3] != now[3]) {
            for (int i = 0; i < 4; ++i) told[i] = now[i];
            NXLOG_I(kTag, "resources handed to the upscaler: colour 0x%llx depth 0x%llx motion 0x%llx output 0x%llx (%ux%u -> %ux%u)",
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(f.color.image)),
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(f.depth.image)),
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(f.motion.image)),
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(f.output.image)), f.render.width,
                    f.render.height, f.out.width, f.out.height);
        }
        const ffxReturnCode_t r = dispatch_(active_ ? active_ : &ctx_, &d.header);
        if (r != FFX_API_RETURN_OK) {
            NXLOG_E(kTag, "ffxDispatch failed (%u): FSR 3 off", r);
            failed_ = true;
            return false;
        }
        return true;
    }

private:
    bool make(VkExtent2D render, VkExtent2D out) {
        if (ctx_) {
            dev_->wait_idle();
            destroy_(&ctx_, nullptr);
            ctx_ = nullptr;
        }
        ffxCreateBackendVKDesc be = {};
        be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        be.deviceContext.vkDevice = dev_->device();
        be.deviceContext.vkPhysicalDevice = dev_->physical();
        be.deviceContext.vkDeviceProcAddr = fx_device_proc;
        // [upscale] fsr_version picks the provider: the 4.x DLL carries both implementations (its strings list
        // fsr3upscaler and fsr4_*; the provider names are "3.1.6" and "4.1.2"), and without an override the DLL hands
        // out the newest one. FFX selects a version with ffxQuery(GET_VERSIONS) + ffxOverrideVersion chained into the
        // create desc. The FSR 4 model used to be unusable on RDNA2 (0.1 fps, HANDOVER 92); that was the cost of the
        // model before the optimisations, and it is no longer the reason to prefer 3.1.
        // read per context, not once per process: switching [dlss] upscaler away from fsr3 and back re-reads it, so the
        // ini can change without a restart (the version list itself is only logged when it changes)
        const std::string want_version = effective_want_version();
        ffxOverrideVersion ov = {};
        // FSR 4.x refuses the newer API entry points unless the create desc carries the upscale-specific version desc
        // as well: the 4.1.2 DLL answers "An instance of ffxCreateContextDescUpscaleVersion must be attached to
        // ffxCreateContextDescUpscale and specify a valid version to access new API functions" and falls back to the
        // 3.x behaviour (issue #96, log at 439 s). ffxOverrideVersion alone selects the provider but does not open
        // that path, so both go into the chain when a 4.x provider is chosen.
        ffxCreateContextDescUpscaleVersion ver = {};
        bool api4 = false;
        bool override_ready = false;
        if (!want_version.empty()) {
            ffxQueryDescGetVersions qv = {};
            qv.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
            qv.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
            qv.device = &be.deviceContext;
            uint64_t count = 8, ids[8] = {};
            const char* names[8] = {};
            qv.outputCount = &count;
            qv.versionIds = ids;
            qv.versionNames = names;
            const ffxReturnCode_t qr = query_(nullptr, &qv.header);
            if (qr == FFX_API_RETURN_OK && count > 0) {
                std::string list;
                for (uint64_t i = 0; i < count && i < 8; ++i) {
                    if (!list.empty()) list += ", ";
                    list += names[i] ? names[i] : "?";
                }
                static std::string told_versions;
                if (told_versions != list) { told_versions = list; NXLOG_I(kTag, "versions this DLL offers: %s (want %s)", list.c_str(), want_version.c_str()); }
                for (uint64_t i = 0; i < count && i < 8; ++i) {
                    if (names[i] && std::string(names[i]).rfind(want_version, 0) == 0) {
                        ov.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
                        ov.versionId = ids[i];
                        ov.header.pNext = &be.header;
                        override_ready = true;
                        api4 = names[i][0] == '4';
                        NXLOG_I(kTag, "version override: %s (id %llu) instead of the DLL default", names[i],
                                static_cast<unsigned long long>(ids[i]));
                        break;
                    }
                }
            } else {
                NXLOG_W(kTag, "GET_VERSIONS failed (%u): the fsr_version setting is ignored", static_cast<unsigned>(qr));
            }
        }
        if (api4) {
            ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
            ver.version = FFX_UPSCALER_VERSION;   // the API this build was compiled against (4.1.2)
            ver.header.pNext = &be.header;
            ov.header.pNext = &ver.header;        // ov -> ver -> backend
        }
        ffxCreateContextDescUpscale cd = {};
        cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        cd.header.pNext = override_ready ? &ov.header : (api4 ? &ver.header : &be.header);
        // HANDOVER 95.9: the flag set is switchable at run time (work\fsr_flags.txt) so the HDR / auto exposure
        // path can be A/B'd against the plain non-linear one the NS1 recomp's FSR 4.1.2 runs on this same driver.
        const FsrFlags fl = read_fsr_flags();
        non_linear_ = fl.non_linear;
        uint32_t create_flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
        if (fl.hdr) create_flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
        if (fl.auto_exposure) create_flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        if (fl.non_linear) create_flags |= FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
        cd.flags = create_flags;
        NXLOG_I(kTag, "create flags: hdr %d, auto exposure %d, non linear %d (0x%x)", fl.hdr ? 1 : 0, fl.auto_exposure ? 1 : 0,
                fl.non_linear ? 1 : 0, static_cast<unsigned>(create_flags));
        cd.maxRenderSize = {render.width, render.height};
        cd.maxUpscaleSize = {out.width, out.height};
        cd.fpMessage = &on_message;
        ffxReturnCode_t r = create_(&ctx_, &cd.header, nullptr);
        if (r != FFX_API_RETURN_OK) {   // the v1.1.4 DLL knows only the older backend descriptor
            ffxCreateBackendVKDesc_DEPRECATED old = {};
            old.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_DEPRECATED;
            old.vkDevice = dev_->device();
            old.vkPhysicalDevice = dev_->physical();
            old.vkDeviceProcAddr = fx_device_proc;
            if (api4) ver.header.pNext = &old.header;
            if (override_ready) ov.header.pNext = api4 ? &ver.header : &old.header;
            cd.header.pNext = override_ready ? &ov.header : (api4 ? &ver.header : &old.header);
            ctx_ = nullptr;
            r = create_(&ctx_, &cd.header, nullptr);
        }
        if (r != FFX_API_RETURN_OK) {
            NXLOG_E(kTag, "ffxCreateContext (upscale %ux%u -> %ux%u) failed (%u): FSR 3 off", render.width, render.height, out.width, out.height, r);
            ctx_ = nullptr;
            return false;
        }
        max_render_ = render;
        max_out_ = out;
        ffxQueryGetProviderVersion pv = {};
        pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        if (query_(&ctx_, &pv.header) == FFX_API_RETURN_OK && pv.versionName) version_ = std::string("FSR ") + pv.versionName;
        {
            std::lock_guard<std::mutex> lk(g_ver_mu);
            g_active_version = version_.empty() ? "FSR 3" : version_;
        }
        NXLOG_I(kTag, "context made: %s, %ux%u -> %ux%u, inverted depth, hdr %d, auto exposure %d", name(), render.width, render.height,
                out.width, out.height, (cd.flags & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE) ? 1 : 0,
                (cd.flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) ? 1 : 0);
        if (active_slot_) {   // the context belongs to its slot from here on; ctx_ must not own (and later destroy) it
            active_slot_->ctx = ctx_;
            active_slot_->render = render;
            active_slot_->out = out;
            ctx_ = nullptr;
        }
        return true;
    }

    gfx::Device* dev_ = nullptr;
    HMODULE mod_ = nullptr;
    PfnFfxCreateContext create_ = nullptr;
    PfnFfxDestroyContext destroy_ = nullptr;
    PfnFfxQuery query_ = nullptr;
    PfnFfxDispatch dispatch_ = nullptr;
    struct Slot {
        ffxContext ctx = nullptr;
        VkExtent2D render = {}, out = {};
        uint64_t used = 0;
        bool fresh = true;   // no history yet: the first dispatch into a new context must reset
    };
    // Which slot the dispatch is about to use, as an index because slots_ can reallocate. -1 = the single ctx_.
    size_t cur_slot_ = static_cast<size_t>(-1);
    Slot* active_slot_ = nullptr;      // the slot make() is filling in
    ffxContext* active_ = nullptr;     // its context, for the dispatch
    std::vector<Slot> slots_;          // up to kSlots contexts, one per size
    // six covers the whole dynamic-resolution ladder (960x540, 1024x576, 1120x632, 1216x684, 1280x720) plus the one-off
    // 1920x1080 evaluation the game does before it settles, so no size has to be rebuilt once it has been seen
    static constexpr size_t kSlots = 6;

    Slot* slot_for(VkExtent2D render, VkExtent2D out) {
        for (Slot& s : slots_) {
            if (s.render.width == render.width && s.render.height == render.height && s.out.width == out.width && s.out.height == out.height) {
                s.used = ++slot_tick_;
                active_ = &s.ctx;
                cur_slot_ = static_cast<size_t>(&s - slots_.data());
                return &s;
            }
        }
        if (slots_.size() >= kSlots) {   // evict the least recently used
            size_t victim = 0;
            for (size_t i = 1; i < slots_.size(); ++i)
                if (slots_[i].used < slots_[victim].used) victim = i;
            if (slots_[victim].ctx) destroy_(&slots_[victim].ctx, nullptr);
            slots_.erase(slots_.begin() + static_cast<long>(victim));
        }
        slots_.push_back(Slot{});
        active_slot_ = &slots_.back();
        cur_slot_ = slots_.size() - 1;
        active_slot_->used = ++slot_tick_;
        active_ = &active_slot_->ctx;
        ctx_ = nullptr;   // make() would otherwise destroy the previous slot's context
        if (!make(render, out)) {
            active_ = nullptr;
            if (!slots_.empty()) slots_.pop_back();
            cur_slot_ = static_cast<size_t>(-1);
            return nullptr;
        }
        NXLOG_I(kTag, "contexts alive: %zu (one per size, FSR follows the game's dynamic resolution)", slots_.size());
        return active_slot_;
    }
    void reset_contexts() {
        if (!ctx_ && slots_.empty()) return;
        cur_slot_ = static_cast<size_t>(-1);
        if (dev_) dev_->wait_idle();
        if (ctx_) {
            destroy_(&ctx_, nullptr);
            ctx_ = nullptr;
        }
        for (Slot& s : slots_) {
            if (s.ctx) destroy_(&s.ctx, nullptr);
        }
        slots_.clear();
        active_slot_ = nullptr;
        active_ = nullptr;
        max_render_ = {};
        max_out_ = {};
    }

    uint64_t slot_tick_ = 0;
    uint32_t seen_gen_ = 0;
    bool follow_size_active_ = false;

    ffxContext ctx_ = nullptr;
    VkExtent2D max_render_ = {}, max_out_ = {};
    std::string version_;
    bool non_linear_ = false;   // HANDOVER 95.9: set from work\fsr_flags.txt when the context is made
    bool failed_ = false;
};

}  // namespace

void set_ffx_version_live(const char* version) {
    const std::string next = version ? version : "";
    if (effective_want_version() == next) return;
    {
        std::lock_guard<std::mutex> lk(g_ver_mu);
        g_live_want_version = next;
    }
    g_ffx_gen.fetch_add(1, std::memory_order_release);
}

std::string temporal_ffx_active_version() {
    std::lock_guard<std::mutex> lk(g_ver_mu);
    return g_active_version;
}

bool temporal_ffx_available() { return !find_dll().empty(); }

TemporalBackend* temporal_ffx(void* gfx_device) {
    static Ffx* inst = nullptr;
    static bool tried = false;
    if (!tried && gfx_device) {
        tried = true;
        auto* f = new Ffx();
        if (f->load(*static_cast<gfx::Device*>(gfx_device))) inst = f;
        else delete f;
    }
    return inst;
}

}  // namespace nxrt::extras
