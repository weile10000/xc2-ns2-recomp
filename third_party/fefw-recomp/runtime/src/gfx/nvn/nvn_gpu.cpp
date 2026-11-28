// SPDX-License-Identifier: GPL-3.0-or-later
// The Vulkan side of the NVN back end: the render thread, submission slots (command buffer, descriptor pools, upload memory),
// the timeline semaphore, deferred destruction, barriers and the public backend API (include/nxrt/nvn_backend.h).
// Design follows smo-recomp (record on game threads, replay on one render thread) and SuperMarioWonderRecomp docs/nvn (a submit
// is replayed under one lock and never overlaps the next); the code is new.
#include "nxrt/nvn_backend.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/frame_limiter.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nxrt/lowlat.h"
#include "vk_gauges.h"
#include "nvn_gpu_internal.h"
#include "nvn_impl.h"
#include "nvn_pipeline.h"
#include "nvn_present_pass.h"
#include "extras/output_pass.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.gpu";
constexpr uint32_t kSlots = 72;                  // about six frames of submissions (the game submits ~11 times a frame): with fewer, recording waits on
                                                 // the GPU for a submission of a recent frame and the CPU cannot get frames ahead. 36 wrapped on XC2 NS2's
                                                 // heavy frames (texture streaming): slot.acquire waited 22 ms for the GPU in the middle of a replay
constexpr uint32_t kGpMax = 1024;      // GPU pass profiler: timestamps per submission
constexpr VkDeviceSize kUploadBytes = 16ull << 20;  // per slot; larger uploads get their own staging buffer (made and freed per submission: the
                                                    // replay-time snapshots of [nvn] mirror_cpu_snapshot / snapshot_instances need 10-25 MiB a frame)

// Live-count gauges of this file (leak monitor, nxrt/gauge.h).
nxrt::GaugeCounter g_gauge_deferred("nvn.deferred_destroy");     // closures waiting for their submission to finish (defer_destroy, Frame::defer)
nxrt::GaugeCounter g_gauge_render_queue("nvn.render_queue");     // jobs queued for the render thread
nxrt::GaugeCounter g_gauge_barriers_mem("nvn.cnt.barriers_mem");   // full memory barriers recorded (cumulative; per frame = delta / frames)
nxrt::GaugeCounter g_gauge_barriers_img("nvn.cnt.barriers_img");   // image layout barriers recorded (cumulative)

struct Slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> dpools;
    size_t dpool_cur = 0;
    gfx::Buffer upload;
    VkDeviceSize upload_off = 0;
    std::vector<gfx::Buffer> upload_overflow;  // reusable 16 MiB chunks borrowed from g->upload_pool (returned on slot reset, never freed per frame!)
    VkDeviceSize overflow_off = 0;
    std::vector<gfx::Buffer> extra;       // one-off staging buffers (> 16 MiB) / adopted texprep buffers of the submission
    uint64_t serial = 0;                  // last submission that used the slot
    bool timed = false;                   // frame trace: the submission wrote GPU timestamps (query pair index = slot index * 2)
    uint64_t cpu_t0 = 0;
    uint64_t sets_live = 0;               // descriptor sets allocated from this slot's pools since the last reset (vk.descriptor_sets)
    uint32_t gp_n = 0;                    // GPU pass profiler: timestamps written by the submission
    std::vector<std::pair<std::string, uint32_t>> gp_labels;   // label and argument of each interval (stamp i closes interval i)
    // [nvn] tex_prologue (Frame::prologue): command buffers allocated from `pool` and reused after its reset; the open one, and the ended one
    // that goes right before `cb` in the batch
    std::vector<VkCommandBuffer> pros;
    size_t pro_next = 0;
    VkCommandBuffer pro_cur = VK_NULL_HANDLE, pro_final = VK_NULL_HANDLE;
};

struct Deferred {
    uint64_t serial;
    std::function<void()> fn;
};

struct Gpu {
    gfx::Device* dev = nullptr;
    gfx::Renderer* rend = nullptr;
    FrameLimiter* limiter = nullptr;
    std::atomic<bool> keep_aspect{false};
    bool installed_entry_points = false;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<uint64_t> completed{0};
    uint64_t submitted = 0;               // render thread: last serial recorded (handed to vkQueueSubmit2, or waiting in `batch`)
    uint64_t flushed = 0;                 // render thread: last serial handed to vkQueueSubmit2
    // [nvn] submit_batch: recorded submissions wait here and go to the GPU in one vkQueueSubmit2 (flush_batch). Render thread.
    std::vector<Slot*> batch;
    uint32_t batch_max = 1;
    // submit_batch = frame: the batch is not flushed when the render thread runs out of work; it goes with the present, in the present's
    // own vkQueueSubmit2 (take_batch_for_present), or earlier when a thread waits for one of its serials (flush_wanted).
    bool frame_batch = false;
    std::atomic<uint64_t> flushed_pub{0};          // `flushed`, readable from any thread
    std::atomic<bool> flush_wanted{false};         // a thread waits for (or polls) a serial still in the batch
    std::atomic<uint64_t> flush_want_serial{0};    // the highest serial a thread asked to be flushed (want_flush): a flush that does not reach it keeps flush_wanted
    uint64_t recording = 0;               // render thread: serial of the submission being recorded (0 outside)
    std::atomic<uint64_t> last_submitted{0};
    // Serials are handed out when a submission is queued (the game thread gets its serial at once and goes on while the render thread
    // records), in queue order, so the timeline is signalled in increasing order. Under qmu.
    std::atomic<uint64_t> reserved{0};
    // Presents queued and not yet done; the game thread waits while [nvn] present_ahead of them are pending (back pressure).
    std::mutex pmu;
    std::condition_variable pcv;
    int presents_pending = 0;
    // The game's last present (under pmu): present_again shows it once more while the game presents nothing (it waits for a library applet).
    Ref<Window> last_window;
    int last_index = -1;
    std::chrono::steady_clock::time_point last_present{};
    Slot slots[kSlots];
    std::vector<gfx::Buffer> upload_pool;   // render-thread pool of reusable kUploadBytes chunks (eliminates per-frame vkAllocateMemory/vkFreeMemory)
    VkQueryPool qpool = VK_NULL_HANDLE;   // frame trace: two timestamps per slot
    VkQueryPool gpool = VK_NULL_HANDLE;   // GPU pass profiler ([prof] gpu_passes): kGpMax timestamps per slot
    uint32_t next_slot = 0;
    Counters ctr;

    // render thread
    std::thread th;
    std::mutex qmu;
    std::condition_variable qcv;
    std::deque<std::function<void()>> queue;
    bool stop = false;
    std::thread::id tid;

    // deferred destruction
    std::mutex dmu;
    std::vector<std::function<void()>> pending_destroy;   // from any thread, assigned a serial by the render thread
    std::deque<Deferred> deferred;                        // render thread

    PresentPass present;
    bool present_ok = false;
    extras::OutputPass output;                 // HDR encode and FSR 1 (S14); used instead of `present` when the extras are on
    bool output_ok = false;
    std::mutex unsupported_mu;
    std::map<std::string, uint64_t> unsupported;
};

Gpu* g = nullptr;
void flush_batch();   // [nvn] submit_batch, below
std::mutex g_install_mu;

uint64_t read_completed() {
    if (!g) return 0;
    if (g->dev->device_lost()) return g->completed.load();
    uint64_t v = 0;
    if (g->timeline) {
        const VkResult r = vkGetSemaphoreCounterValue(g->dev->device(), g->timeline, &v);
        if (r == VK_ERROR_DEVICE_LOST) {
            g->dev->note_device_lost(r, "vkGetSemaphoreCounterValue(nvn timeline)");
            return g->completed.load();
        }
        if (r != VK_SUCCESS) return g->completed.load();
        // Drivers have been observed returning VK_SUCCESS with UINT64_MAX on a lost/poisoned timeline: never let that corrupt g->completed.
        if (v == UINT64_MAX && g->last_submitted.load() < UINT64_MAX - 1024) {
            g->dev->note_device_lost(VK_ERROR_DEVICE_LOST, "vkGetSemaphoreCounterValue returned UINT64_MAX");
            return g->completed.load();
        }
    }
    if (v > g->completed.load()) g->completed.store(v);
    return g->completed.load();
}

void drain_pending_destroy() {
    std::vector<std::function<void()>> pend;
    {
        std::lock_guard<std::mutex> lk(g->dmu);
        pend.swap(g->pending_destroy);
    }
    const uint64_t s = g->recording ? g->recording : g->submitted;
    for (auto& fn : pend) g->deferred.push_back({s, std::move(fn)});
}

void retire(int64_t budget_ns = 0) {
    drain_pending_destroy();
    const bool lost = g->dev->device_lost();
    // Per the Vulkan spec, once VK_ERROR_DEVICE_LOST is observed all previously submitted commands are terminated and
    // destroying child objects of the lost device remains valid. Without draining here, defer_destroy closures (such as
    // mirror_capture CapArenas, textures, and pools) leak unboundedly after device loss (#81).
    const uint64_t done = lost ? UINT64_MAX : read_completed();
    const auto t0 = budget_ns > 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    size_t ran = 0;
    while (!g->deferred.empty() && g->deferred.front().serial <= done) {
        auto fn = std::move(g->deferred.front().fn);
        g->deferred.pop_front();
        g_gauge_deferred.sub(1);
        fn();
        ++ran;
        if (!lost && budget_ns > 0 && ran >= 4 &&
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count() >= budget_ns)
            break;
    }
    // Reclaim completed slots' upload_overflow and extra staging buffers immediately (~2 frames) instead of waiting 72 slots (~2s)
    // for the ring to wrap, preventing multi-GB host-visible memory accumulation during map switches.
    constexpr size_t kUploadPoolMax = 8;
    for (Slot& s : g->slots) {
        if (!s.serial || s.serial > done) continue;
        if (!s.upload_overflow.empty()) {
            for (auto& b : s.upload_overflow) {
                if (!lost && g->upload_pool.size() < kUploadPoolMax) g->upload_pool.push_back(b);
                else g->dev->destroy_buffer(b);
            }
            s.upload_overflow.clear();
            s.overflow_off = 0;
        }
        if (!s.extra.empty()) {
            for (auto& b : s.extra) g->dev->destroy_buffer(b);
            s.extra.clear();
        }
    }
    if (lost && !g->upload_pool.empty()) {
        for (auto& b : g->upload_pool) g->dev->destroy_buffer(b);
        g->upload_pool.clear();
    }
}

thread_local bool t_render_thread = false;
// [nvn] test_throw_every = N (tests only): every Nth recorded submission and every Nth present throws std::bad_alloc, to check that
// a failed render job costs a frame and nothing else.
void test_throw(std::atomic<uint64_t>& n) {
    static const int64_t every = nxrt::config_get_int("nvn", "test_throw_every", 0);
    if (every > 0 && (n.fetch_add(1) + 1) % static_cast<uint64_t>(every) == 0) throw std::bad_alloc();
}
std::atomic<uint64_t> g_test_records{0}, g_test_presents{0};
template <class Fn> struct OnExit { Fn fn; ~OnExit() { fn(); } };
template <class Fn> OnExit<Fn> on_exit(Fn fn) { return OnExit<Fn>{std::move(fn)}; }
std::atomic<int> g_render_throw_reports{0};

// A C++ exception thrown on the render thread is caught by render_loop, which aborts that job; the first few are logged with
// the throwing stack (the catch alone cannot say where it came from).
LONG WINAPI render_throw_report(EXCEPTION_POINTERS* ep) {
    if (t_render_thread && ep && ep->ExceptionRecord && ep->ExceptionRecord->ExceptionCode == 0xE06D7363u &&
        g_render_throw_reports.fetch_add(1) < 3) {
        MEMORYSTATUSEX ms = {sizeof(ms)};
        GlobalMemoryStatusEx(&ms);
        NXLOG_E(kTag, "render thread C++ exception (system commit free %.1f GB of %.1f GB): %s", ms.ullAvailPageFile / 1073741824.0,
                ms.ullTotalPageFile / 1073741824.0, nxrt::crash_build_report(ep).c_str());
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void render_loop() {
    t_render_thread = true;
    static void* veh = AddVectoredExceptionHandler(0, render_throw_report);
    (void)veh;
    g->tid = std::this_thread::get_id();
    log_set_thread_name("nvn-render");
    diag_register_thread("nvn-render", nullptr);   // so the freeze watchdog and the stall sampler can see its stack
    nxrt::hitch_set_render_thread();
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(g->qmu);
            if (!g->batch.empty() && (g->flush_wanted.load() || (!g->frame_batch && !g->stop && g->queue.empty()))) {
                lk.unlock();
                flush_batch();      // nothing more to record for now (or a thread waits for it): the GPU gets what there is
                lk.lock();
            }
            if (!g->stop && g->queue.empty()) {
                nxrt::HitchTimer ht(nxrt::kHpRenderIdle);      // the hitch line: the render thread waited for the game
                g->qcv.wait(lk, [] { return g->stop || !g->queue.empty() || (g->flush_wanted.load() && !g->batch.empty()); });
                if (g->queue.empty() && !g->stop) continue;    // woken to flush
            }
            if (g->queue.empty()) {   // stop requested and drained
                lk.unlock();
                flush_batch();
                return;
            }
            job = std::move(g->queue.front());
            g->queue.pop_front();
            g_gauge_render_queue.sub(1);
        }
        try {
            job();
        } catch (const std::exception& e) {
            NXLOG_E(kTag, "render job threw: %s", e.what());
        }
    }
}

bool create_slot(Slot& s) {
    gfx::Device& d = *g->dev;
    VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    ci.queueFamilyIndex = d.queue_family();
    if (vkCreateCommandPool(d.device(), &ci, nullptr, &s.pool) != VK_SUCCESS) return false;
    vk_gauge_command_pools().add(1);
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = s.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d.device(), &ai, &s.cb) != VK_SUCCESS) return false;
    vk_gauge_command_buffers().add(1);
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (d.create_buffer(kUploadBytes, usage, gfx::MemoryUse::Upload, &s.upload, "nvn upload") != VK_SUCCESS) return false;
    return true;
}

VkDescriptorPool new_dpool() {
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u << 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u << 13},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u << 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u << 12},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1u << 10},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u << 10},
    };
    VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = 1u << 14;
    ci.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
    ci.pPoolSizes = sizes;
    VkDescriptorPool p = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(g->dev->device(), &ci, nullptr, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
    vk_gauge_descriptor_pools().add(1);
    return p;
}

// VK_EXT_device_fault (enabled in vk_device.cpp): delegated to Device::query_device_fault so both NVN and Renderer share the latch/cache.
static std::string device_fault_text(gfx::Device* dev) {
    return dev ? dev->query_device_fault() : std::string{};
}

// Hands the recorded submissions of the batch to the GPU in one vkQueueSubmit2, in recording order, signalling the timeline with the last
// serial (the earlier ones of the batch complete with it). Every kernel submission costs a pass over all memory the device holds: on Linux
// (Proton, RADV) the multi-gigabyte host-memory pool imports are user pages the kernel looks up again on each one, about 2 ms a
// submission on a Legion Go S, so 11 submissions a frame took 20 of its 25 ms. Render thread.
// A flush reached `serial`: the request flag goes only when no thread asked for a later serial. It used to be cleared unconditionally, which lost
// the request of a thread that asked, during or just before a flush, for a serial the render thread had not recorded yet: that serial then sat in
// the batch until the next present, and the thread waiting for it was usually the one that queues that present (stalls of 50-150 ms in XC2 NS2,
// [prof] trace: guest-83 sync.wait 136 ms while the GPU ran 11 ms). Order: clear, then read the wanted serial, then set again (want_flush
// raises the serial before it sets the flag), so a request is never lost.
void after_flush(uint64_t serial) {
    g->flush_wanted.store(false);
    if (g->flush_want_serial.load() > serial) g->flush_wanted.store(true);
}

void flush_batch() {
    if (g->batch.empty()) return;
    std::vector<VkCommandBufferSubmitInfo> cbi;
    cbi.reserve(g->batch.size() * 2);
    for (size_t i = 0; i < g->batch.size(); i++) {
        if (g->batch[i]->pro_final) { cbi.push_back({VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}); cbi.back().commandBuffer = g->batch[i]->pro_final; }
        cbi.push_back({VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO});
        cbi.back().commandBuffer = g->batch[i]->cb;
    }
    const uint64_t serial = g->batch.back()->serial;
    VkSemaphoreSubmitInfo sig = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = g->timeline;
    sig.value = serial;
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = static_cast<uint32_t>(cbi.size());
    si.pCommandBufferInfos = cbi.data();
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &sig;
    VkResult r = g->dev->device_lost() ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
    if (!g->dev->device_lost()) {
        nxrt::HitchTimer ht(nxrt::kHpRenderSubmit);
        const uint64_t tl = nxrt::hitch_ticks();
        std::lock_guard<std::mutex> lk(g->dev->queue_mutex());
        nxrt::hitch_phase_add(nxrt::kHpRenderLockWait, nxrt::hitch_ticks() - tl);
        r = vkQueueSubmit2(g->dev->queue(), 1, &si, VK_NULL_HANDLE);
    }
    if (r != VK_SUCCESS) {
        g->dev->note_device_lost(r, "nvn flush_batch(vkQueueSubmit2)");
        // ONCE, not once per frame. A lost device makes every submit fail identically, and the retail log of one
        // session with a lost device held 21,872 copies of this line over 556 seconds - which buries the frames that
        // came BEFORE the loss, the ones that would say why it happened.
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            NXLOG_E(kTag, "vkQueueSubmit2 failed (%d): the device is lost or out of memory", static_cast<int>(r));
            if (const std::string why = device_fault_text(g->dev); !why.empty())
                NXLOG_E(kTag, "device fault: %s", why.c_str());
            NXLOG_E(kTag, "the GPU timeline can no longer be signalled: the device is gone. Restart the game.");
            NXLOG_E(kTag, "nothing will be drawn again; the game keeps running (sound and logic are unaffected), so "
                          "this is not a hang. Further submit failures are not logged.");
        }
    }
    g->flushed = serial;
    g->flushed_pub.store(serial);
    after_flush(serial);
    g->batch.clear();
    g->ctr.submits.fetch_add(1);
}

// submit_batch = frame: a thread outside the render thread waits for a serial that may still sit in the batch; the render thread flushes
// it at once instead of with the next present (which that thread may be the one to queue).
void want_flush(uint64_t serial) {
    if (!g->frame_batch || serial <= g->flushed_pub.load()) return;
    for (uint64_t w = g->flush_want_serial.load(); serial > w && !g->flush_want_serial.compare_exchange_weak(w, serial);) {}
    if (g->flush_wanted.exchange(true)) return;
    std::lock_guard<std::mutex> lk(g->qmu);
    g->qcv.notify_one();
}

void wait_serial_blocking(uint64_t serial) {
    if (!serial || !g || g->dev->device_lost() || read_completed() >= serial) return;
    if (serial > g->flushed && on_render_thread()) flush_batch();     // a serial still in the batch would never signal
    if (g->dev->device_lost()) return;
    NXFT("gpu.wait");
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &g->timeline;
    wi.pValues = &serial;
    for (;;) {
        if (g->dev->device_lost() || read_completed() >= serial) break;
        const VkResult r = vkWaitSemaphores(g->dev->device(), &wi, 2000000000ull);
        if (r == VK_SUCCESS) break;
        if (r == VK_TIMEOUT) continue;
        if (!g->dev->note_device_lost(r, "nvn wait_serial_blocking(vkWaitSemaphores)"))
            NXLOG_E(kTag, "vkWaitSemaphores failed (%d)", static_cast<int>(r));
        break;
    }
    read_completed();
}

Slot& acquire_slot() {
    NXFT_MIN("slot.acquire", 20000);
    Slot& s = g->slots[g->next_slot];
    g->next_slot = (g->next_slot + 1) % kSlots;
    {
        nxrt::HitchTimer ht(nxrt::kHpRenderGpuWait);
        wait_serial_blocking(s.serial);
    }
    if (s.timed) {
        s.timed = false;
        uint64_t ts[2] = {0, 0};
        const uint32_t idx = static_cast<uint32_t>(&s - g->slots) * 2;
        if (vkGetQueryPoolResults(g->dev->device(), g->qpool, idx, 2, sizeof ts, ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] >= ts[0]) {
            const double ns = static_cast<double>(ts[1] - ts[0]) * g->dev->properties().limits.timestampPeriod;
            g->ctr.gpu_busy_ns.fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
            if (ft::on()) {
                static const uint16_t z = ft::zone_id("gpu.exec");
                ft::event(z, s.cpu_t0, s.cpu_t0 + static_cast<uint64_t>(ns), static_cast<uint32_t>(s.serial));
            }
        }
    }
    if (s.gp_n >= 2 && g->gpool) {      // GPU pass profiler: one event per interval between two timestamps, named gp:<label>, arg = draws
        uint64_t ts[kGpMax];
        const uint32_t n = std::min(s.gp_n, kGpMax);
        const uint32_t base = static_cast<uint32_t>(&s - g->slots) * kGpMax;
        if (vkGetQueryPoolResults(g->dev->device(), g->gpool, base, n, sizeof(uint64_t) * n, ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            static std::map<std::string, uint16_t> zones;
            const double period = g->dev->properties().limits.timestampPeriod;
            uint64_t t = s.cpu_t0;
            if (ft::on()) {
                for (uint32_t i = 1; i < n && i < s.gp_labels.size(); i++) {
                    const uint64_t d = ts[i] >= ts[i - 1] ? static_cast<uint64_t>(static_cast<double>(ts[i] - ts[i - 1]) * period) : 0;
                    auto it = zones.find(s.gp_labels[i].first);
                    if (it == zones.end()) it = zones.emplace(s.gp_labels[i].first, ft::zone_id(("gp:" + s.gp_labels[i].first).c_str())).first;
                    ft::event(it->second, t, t + d, s.gp_labels[i].second);
                    t += d;
                }
            } else {   // [prof] gpu_passes without the frame trace: one summary line every 60 submissions
                static std::map<std::string, double> sum;
                static uint64_t frames = 0;
                for (uint32_t i = 1; i < n && i < s.gp_labels.size(); i++) {
                    const double d = ts[i] >= ts[i - 1] ? static_cast<double>(ts[i] - ts[i - 1]) * period / 1e6 : 0.0;
                    sum[s.gp_labels[i].first] += d;
                }
                if (++frames >= 60) {
                    std::string line;
                    for (const auto& kv : sum) {
                        char b[96];
                        snprintf(b, sizeof b, "%s %.2f | ", kv.first.c_str(), kv.second / static_cast<double>(frames));
                        line += b;
                    }
                    NXLOG_I("gpu", "gpu pass ms avg over %llu submissions: %s", static_cast<unsigned long long>(frames), line.c_str());
                    sum.clear();
                    frames = 0;
                }
            }
        }
    }
    s.gp_n = 0;
    s.gp_labels.clear();
    retire(1500000);   // 1.5 ms per-frame teardown budget: spreads scene-switch resource frees smoothly across frames
    vkResetCommandPool(g->dev->device(), s.pool, 0);
    s.pro_next = 0;
    s.pro_cur = s.pro_final = VK_NULL_HANDLE;
    for (VkDescriptorPool p : s.dpools) vkResetDescriptorPool(g->dev->device(), p, 0);
    vk_gauge_descriptor_sets().sub(static_cast<int64_t>(s.sets_live));   // the reset freed them
    s.sets_live = 0;
    s.dpool_cur = 0;
    s.upload_off = 0;
    s.overflow_off = 0;
    constexpr size_t kUploadPoolMax = 8;
    for (auto& b : s.upload_overflow) {
        if (g->upload_pool.size() < kUploadPoolMax) g->upload_pool.push_back(b);
        else g->dev->destroy_buffer(b);
    }
    s.upload_overflow.clear();
    for (auto& b : s.extra) g->dev->destroy_buffer(b);
    s.extra.clear();
    s.serial = 0;   // cleared while recording so retire() never touches the active slot
    return s;
}

}  // namespace

// ---- Frame ---------------------------------------------------------------------------------------------------------------
VkCommandBuffer Frame::prologue() {
    Slot& s = *static_cast<Slot*>(slot);
    if (s.pro_cur) return s.pro_cur;
    if (s.pro_next >= s.pros.size()) {
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = s.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer c = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(g->dev->device(), &ai, &c) != VK_SUCCESS) return VK_NULL_HANDLE;
        s.pros.push_back(c);
    }
    VkCommandBuffer c = s.pros[s.pro_next++];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(c, &bi) != VK_SUCCESS) return VK_NULL_HANDLE;
    s.pro_cur = c;
    return c;
}

std::atomic<uint64_t> g_staging_overflows{0};
uint64_t staging_overflow_count() { return g_staging_overflows.load(std::memory_order_relaxed); }

UploadAlloc Frame::upload(VkDeviceSize size, VkDeviceSize align) {
    Slot& s = *static_cast<Slot*>(slot);
    if (align == 0) align = 16;
    VkDeviceSize off = (s.upload_off + align - 1) & ~(align - 1);
    if (off + size <= kUploadBytes) {
        s.upload_off = off + size;
        UploadAlloc a;
        a.buffer = s.upload.buffer;
        a.offset = off;
        a.ptr = static_cast<uint8_t*>(s.upload.mapped) + off;
        a.size = size;
        return a;
    }
    // Pooled 16 MiB overflow chunk arena: avoids calling vkCreateBuffer/vkAllocateMemory per allocation when >16 MiB/submission!
    if (size <= kUploadBytes) {
        if (!s.upload_overflow.empty()) {
            gfx::Buffer& cur = s.upload_overflow.back();
            const VkDeviceSize o2 = (s.overflow_off + align - 1) & ~(align - 1);
            if (o2 + size <= cur.size) {
                s.overflow_off = o2 + size;
                UploadAlloc a;
                a.buffer = cur.buffer;
                a.offset = o2;
                a.ptr = static_cast<uint8_t*>(cur.mapped) + o2;
                a.size = size;
                return a;
            }
        }
        gfx::Buffer chunk;
        if (g->upload_pool.empty()) retire();   // reclaim any just-finished slots before allocating a new 16 MiB chunk
        if (!g->upload_pool.empty()) {
            chunk = g->upload_pool.back();
            g->upload_pool.pop_back();
        } else {
            g_staging_overflows.fetch_add(1, std::memory_order_relaxed);
            const VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            if (g->dev->create_buffer(kUploadBytes, usage, gfx::MemoryUse::Upload, &chunk, "nvn upload chunk") != VK_SUCCESS) {
                NXLOG_E(kTag, "upload chunk of %llu bytes failed", static_cast<unsigned long long>(kUploadBytes));
                return {};
            }
        }
        s.upload_overflow.push_back(chunk);
        s.overflow_off = size;
        UploadAlloc a;
        a.buffer = chunk.buffer;
        a.offset = 0;
        a.ptr = static_cast<uint8_t*>(chunk.mapped);
        a.size = size;
        return a;
    }
    return staging(size + align);
}

UploadAlloc Frame::staging(VkDeviceSize size) {
    if (size <= kUploadBytes) return upload(size, 256);
    g_staging_overflows.fetch_add(1, std::memory_order_relaxed);
    Slot& s = *static_cast<Slot*>(slot);
    gfx::Buffer b;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (g->dev->create_buffer(std::max<VkDeviceSize>(size, 256), usage, gfx::MemoryUse::Upload, &b, "nvn staging") != VK_SUCCESS) {
        NXLOG_E(kTag, "staging buffer of %llu bytes failed", static_cast<unsigned long long>(size));
        return {};
    }
    UploadAlloc a;
    a.buffer = b.buffer;
    a.offset = 0;
    a.ptr = static_cast<uint8_t*>(b.mapped);
    a.size = size;
    s.extra.push_back(b);
    return a;
}

UploadAlloc Frame::adopt_staging(gfx::Buffer b) {
    Slot& s = *static_cast<Slot*>(slot);
    UploadAlloc a;
    a.buffer = b.buffer;
    a.offset = 0;
    a.ptr = static_cast<uint8_t*>(b.mapped);
    a.size = b.size;
    s.extra.push_back(b);
    return a;
}

void Frame::defer(std::function<void()> fn) {
    g->deferred.push_back({serial, std::move(fn)});
    g_gauge_deferred.add(1);
}

VkDescriptorSet Frame::alloc_set(VkDescriptorSetLayout layout) {
    if (g->dev->device_lost()) return VK_NULL_HANDLE;
    Slot& s = *static_cast<Slot*>(slot);
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (s.dpool_cur >= s.dpools.size()) {
            VkDescriptorPool p = new_dpool();
            if (!p) return VK_NULL_HANDLE;
            s.dpools.push_back(p);
        }
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = s.dpools[s.dpool_cur];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult r = vkAllocateDescriptorSets(g->dev->device(), &ai, &set);
        if (r == VK_SUCCESS) {
            g->ctr.descriptor_sets.fetch_add(1, std::memory_order_relaxed);
            s.sets_live++;
            vk_gauge_descriptor_sets().add(1);
            return set;
        }
        if (r == VK_ERROR_DEVICE_LOST) {
            g->dev->note_device_lost(r, "nvn Frame::alloc_set(vkAllocateDescriptorSets)");
            return VK_NULL_HANDLE;
        }
        // Only grow a fresh descriptor pool when the current one is actually full/fragmented; other errors (e.g.
        // VK_ERROR_OUT_OF_DEVICE_MEMORY) must not spin creating hundreds of multi-megabyte pools (#41).
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) return VK_NULL_HANDLE;
        ++s.dpool_cur;   // exhausted or fragmented: move on to the next pool (created on demand)
    }
    return VK_NULL_HANDLE;
}

// ---- general helpers ---------------------------------------------------------------------------------------------------
bool on_render_thread() { return g && std::this_thread::get_id() == g->tid; }
gfx::Device& device() { return *g->dev; }
VkDevice vkdev() { return g->dev->device(); }
Counters& counters() { return g->ctr; }
Counters* counters_if() { return g ? &g->ctr : nullptr; }
uint64_t completed_serial() { return g ? g->completed.load() : 0; }
uint64_t poll_completed() { return g ? read_completed() : 0; }
uint64_t current_serial() { return g ? (g->recording ? g->recording : g->submitted) : 0; }

void note_unsupported(const char* what) {
    if (!g) return;
    g->ctr.unsupported_ops.fetch_add(1);
    std::lock_guard<std::mutex> lk(g->unsupported_mu);
    uint64_t& n = g->unsupported[what];
    if (n++ == 0) NXLOG_W(kTag, "not implemented, skipped: %s", what);
}

void defer_destroy(std::function<void()> fn) {
    if (!g) {
        fn();
        return;
    }
    std::lock_guard<std::mutex> lk(g->dmu);
    g->pending_destroy.push_back(std::move(fn));
    g_gauge_deferred.add(1);
}

void run_on_render_thread(const std::function<void()>& fn) {
    if (!g) return;
    if (on_render_thread()) {
        fn();
        return;
    }
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        g->queue.push_back([&] {
            auto signal = on_exit([&] {
                std::lock_guard<std::mutex> l2(m);
                done = true;
                cv.notify_all();
            });
            fn();
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return done; });
}

bool gpass_on() { return g && g->gpool != VK_NULL_HANDLE; }

void gpass_stamp(Frame& f, const char* label, uint32_t arg) {
    if (!g || !g->gpool || !f.slot) return;
    Slot& s = *static_cast<Slot*>(f.slot);
    if (s.gp_n >= kGpMax) return;
    vkCmdWriteTimestamp2(f.cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g->gpool, static_cast<uint32_t>(&s - g->slots) * kGpMax + s.gp_n);
    s.gp_labels.emplace_back(label, arg);
    s.gp_n++;
}

namespace {
// [nvn] pass_barriers = tracked (S38): a layout change waits only for the stages that can have used the image in its old layout and blocks
// only the stages that use it in the new one. The layout says how the image was used: the replay keeps every image in the layout of its
// last use (attachments in the attachment layouts, sampled images in the read-only layouts, copies in the transfer layouts); GENERAL and
// anything else keep the full barrier.
std::atomic<bool> g_narrow_barriers{false};

constexpr VkPipelineStageFlags2 kShaderStages = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags2 kDepthStages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

// Stages and accesses of an image's use in `layout`; writes_only: the access part names the writes only (the source half of a barrier).
void layout_scope(VkImageLayout layout, bool src, VkPipelineStageFlags2* stage, VkAccessFlags2* access) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            // the old contents do not matter, but the memory may have belonged to something still in flight: wait for all earlier commands
            *stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            *access = src ? VK_ACCESS_2_NONE : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            return;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            *stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            *access = src ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            return;
        case VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT:
            *stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | kShaderStages;
            *access = src ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                          : (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            return;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            *stage = kDepthStages;
            *access = src ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                          : (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            return;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            *stage = kDepthStages | kShaderStages;
            *access = src ? VK_ACCESS_2_NONE : (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            return;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            *stage = kShaderStages;
            *access = src ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            return;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            *stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            *access = src ? VK_ACCESS_2_NONE : VK_ACCESS_2_TRANSFER_READ_BIT;
            return;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            *stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            *access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            return;
        default:
            *stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            *access = src ? VK_ACCESS_2_MEMORY_WRITE_BIT : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            return;
    }
}
}  // namespace

void set_narrow_barriers(bool on) { g_narrow_barriers.store(on, std::memory_order_relaxed); }

void image_barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkImageAspectFlags aspect, uint32_t base_level,
                   uint32_t level_count, uint32_t base_layer, uint32_t layer_count) {
    g_gauge_barriers_img.add(1);
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    if (g_narrow_barriers.load(std::memory_order_relaxed)) {
        layout_scope(from, true, &b.srcStageMask, &b.srcAccessMask);
        layout_scope(to, false, &b.dstStageMask, &b.dstAccessMask);
    }
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, base_level, level_count, base_layer, layer_count};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

void memory_barrier(VkCommandBuffer cb) {
    g_gauge_barriers_mem.add(1);
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

void memory_barrier_in_pass(VkCommandBuffer cb) {
    g_gauge_barriers_mem.add(1);
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

void transfer_pre_barrier(VkCommandBuffer cb) {
    g_gauge_barriers_mem.add(1);
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

void transfer_post_barrier(VkCommandBuffer cb) {
    g_gauge_barriers_mem.add(1);
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                      VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                      VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

// ---- submission --------------------------------------------------------------------------------------------------------
namespace {

// `serial`: the one reserved when the job was queued (queue_submit). 0: a submission made inside a render job (texture dumps of the trace):
// it takes the next serial when nothing queued holds one, else it signals nothing and the queue is drained after it.
// [nvn] tex_prologue: ends the slot's open prologue (null when none was opened).
VkCommandBuffer end_prologue(Slot& s) {
    VkCommandBuffer c = s.pro_cur;
    if (!c) return VK_NULL_HANDLE;
    s.pro_cur = VK_NULL_HANDLE;
    return vkEndCommandBuffer(c) == VK_SUCCESS ? c : VK_NULL_HANDLE;
}

uint64_t submit_on_render_thread(const std::function<void(Frame&)>& record, uint64_t serial = 0) {
    if (g->dev->device_lost()) {
        if (!serial) {
            std::lock_guard<std::mutex> lk(g->qmu);
            serial = g->reserved.fetch_add(1) + 1;
        }
        if (serial > g->submitted) {
            g->submitted = serial;
            g->last_submitted.store(serial);
        }
        if (serial > g->flushed) {
            g->flushed = serial;
            g->flushed_pub.store(serial);
            after_flush(serial);
        }
        retire();
        return serial;
    }
    NXFT("replay");
    nxrt::HitchTimer ht_replay(nxrt::kHpRenderReplay);
    const auto t0 = std::chrono::steady_clock::now();
    Slot& s = acquire_slot();
    bool unsignalled = false;
    if (!serial) {
        std::lock_guard<std::mutex> lk(g->qmu);
        if (g->reserved.load() == g->submitted) serial = g->reserved.fetch_add(1) + 1;
        else { serial = g->submitted; unsignalled = true; }
    }
    g->recording = serial;
    Frame f;
    f.cb = s.cb;
    f.serial = serial;
    f.slot = &s;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s.cb, &bi);
    const uint32_t qi = static_cast<uint32_t>(&s - g->slots) * 2;
    if (g->gpool) {
        vkCmdResetQueryPool(s.cb, g->gpool, static_cast<uint32_t>(&s - g->slots) * kGpMax, kGpMax);
        s.gp_n = 0;
        s.gp_labels.clear();
        gpass_stamp(f, "(start)", 0);
    }
    if (g->qpool) {
        vkCmdResetQueryPool(s.cb, g->qpool, qi, 2);
        vkCmdWriteTimestamp(s.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g->qpool, qi);
        s.timed = true;
        s.cpu_t0 = ft::now_ns();
    }
    bool aborted = false;
    try {
        test_throw(g_test_records);
        record(f);
    } catch (const std::exception& e) {
        // An empty command buffer still signals the serial, so every wait for it (fences, the game's syncs, slot reuse) returns.
        NXLOG_E(kTag, "submission %llu: recording threw (%s): its commands are dropped, an empty submission signals it",
                static_cast<unsigned long long>(serial), e.what());
        replay_abort_submit();
        vkResetCommandPool(g->dev->device(), s.pool, 0);
        s.pro_next = 0;
        s.pro_cur = s.pro_final = VK_NULL_HANDLE;
        vkBeginCommandBuffer(s.cb, &bi);
        s.timed = false;
        s.gp_n = 0;
        s.gp_labels.clear();
        aborted = true;
    }
    if (g->gpool && !aborted) gpass_stamp(f, "tail", 0);
    if (g->qpool && !aborted) vkCmdWriteTimestamp(s.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g->qpool, qi + 1);
    vkEndCommandBuffer(s.cb);
    s.pro_final = end_prologue(s);
    if (!unsignalled) {
        s.serial = serial;
        g->submitted = serial;
        g->last_submitted.store(serial);
        g->recording = 0;
        g->batch.push_back(&s);
        if (g->batch.size() >= g->batch_max) flush_batch();
        g->ctr.submit_replay_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
        return serial;
    }
    flush_batch();      // debug path: submitted on its own, after everything recorded before it, signalling nothing
    VkCommandBufferSubmitInfo cbi[2] = {{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}};
    uint32_t ncb = 0;
    if (s.pro_final) cbi[ncb++].commandBuffer = s.pro_final;
    cbi[ncb++].commandBuffer = s.cb;
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = ncb;
    si.pCommandBufferInfos = cbi;
    VkResult r;
    {
        nxrt::HitchTimer ht(nxrt::kHpRenderSubmit);
        const uint64_t tl = nxrt::hitch_ticks();
        std::unique_lock<std::mutex> lk(g->dev->queue_mutex());
        nxrt::hitch_phase_add(nxrt::kHpRenderLockWait, nxrt::hitch_ticks() - tl);
        r = g->dev->device_lost() ? VK_ERROR_DEVICE_LOST : vkQueueSubmit2(g->dev->queue(), 1, &si, VK_NULL_HANDLE);
        if (r == VK_SUCCESS) vkQueueWaitIdle(g->dev->queue());   // nothing else tells when it is done
        else {
            lk.unlock();
            g->dev->note_device_lost(r, "nvn sync submit(vkQueueSubmit2)");
        }
    }
    g->recording = 0;
    s.serial = 0;
    return serial;
}

}  // namespace

// [nvn] split_draws: the replay's end_pass calls this at a pass boundary (no render pass open, the state dirty) once enough draws are
// recorded. The command buffer recorded so far goes to the queue on its own (no signal: the frame's serial is signalled by its last part),
// and the recording continues in a fresh one that starts with a full memory barrier, so everything the earlier part wrote is visible to it.
bool split_flush(Frame& f) {
    if (!g || !f.slot || f.cb == VK_NULL_HANDLE || g->dev->device_lost()) return false;
    Slot* s = static_cast<Slot*>(f.slot);
    flush_batch();                          // anything queued before this goes first (one queue: submission order is the order here)
    if (g->dev->device_lost()) return false;
    VkCommandBuffer old = f.cb;
    if (vkEndCommandBuffer(old) != VK_SUCCESS) return false;
    const VkCommandBuffer pro = end_prologue(*s);   // [nvn] tex_prologue: the uploads recorded for this part go right before it
    VkCommandBufferSubmitInfo cbis[2] = {{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}};
    uint32_t ncb = 0;
    if (pro) cbis[ncb++].commandBuffer = pro;
    cbis[ncb++].commandBuffer = old;
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = ncb;
    si.pCommandBufferInfos = cbis;
    VkResult r;
    {
        nxrt::HitchTimer ht(nxrt::kHpRenderSubmit);
        std::lock_guard<std::mutex> lk(g->dev->queue_mutex());
        r = vkQueueSubmit2(g->dev->queue(), 1, &si, VK_NULL_HANDLE);
    }
    if (r != VK_SUCCESS) {
        g->dev->note_device_lost(r, "nvn split_flush(vkQueueSubmit2)");
        NXLOG_E(kTag, "split submit failed (%d): the frame stays in one command buffer", static_cast<int>(r));
        return false;
    }
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = s->pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(g->dev->device(), &ai, &cb) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return false;
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
    s->cb = cb;
    f.cb = cb;
    counters().submits.fetch_add(1);
    return true;
}

void submit_recorded() {
    if (g) flush_batch();
}

bool take_batch_for_present(std::vector<VkCommandBuffer>* cmds, gfx::Renderer::SubmitPrefix* p) {
    if (!g || !g->frame_batch || g->batch.empty()) {
        // The frame's draws go to the GPU on their own submission here. The present then copies the window texture they
        // write, and nothing on the queue orders the two: hand the timeline value the flush signalled to the present, so
        // its submit waits for the draws (issue: N card stripes/flicker, the copy overtaking the frame).
        submit_recorded();
        if (g) { p->wait = g->timeline; p->wait_value = g->flushed; }
        return false;
    }
    cmds->clear();
    for (Slot* s : g->batch) {
        if (s->pro_final) cmds->push_back(s->pro_final);
        cmds->push_back(s->cb);
    }
    p->cmds = cmds->data();
    p->count = static_cast<uint32_t>(cmds->size());
    p->timeline = g->timeline;
    p->value = g->batch.back()->serial;
    return true;
}

void present_batch_done(const gfx::Renderer::SubmitPrefix& p) {
    if (!p.submitted && !g->dev->device_lost()) {   // keep the timeline moving if submit failed transiently without device loss
        VkSemaphoreSignalInfo ssi = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        ssi.semaphore = g->timeline;
        ssi.value = p.value;
        const VkResult sr = vkSignalSemaphore(g->dev->device(), &ssi);
        g->dev->note_device_lost(sr, "nvn present_batch_done(vkSignalSemaphore)");
    }
    g->flushed = p.value;
    g->flushed_pub.store(p.value);
    after_flush(p.value);
    g->batch.clear();
    g->ctr.submits.fetch_add(1);
}

// Queues a submission with the next serial (in queue order). wait: until the render thread has run it (and, with gpu_done, the GPU too).
uint64_t queue_submit(std::function<void(Frame&)> record, bool wait, bool gpu_done) {
    if (on_render_thread()) {
        const uint64_t serial = submit_on_render_thread(record);
        if (gpu_done) { wait_serial_blocking(serial); retire(); }
        return serial;
    }
    struct Done { std::mutex m; std::condition_variable cv; bool done = false; };
    auto done = wait ? std::make_shared<Done>() : nullptr;
    uint64_t serial = 0;
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        serial = g->reserved.fetch_add(1) + 1;
        g->queue.push_back([record = std::move(record), serial, done, gpu_done] {
            auto signal = on_exit([&] {
                if (!done) return;
                std::lock_guard<std::mutex> l2(done->m);
                done->done = true;
                done->cv.notify_all();
            });
            submit_on_render_thread(record, serial);
            if (gpu_done) { wait_serial_blocking(serial); retire(); }
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
    if (done) {
        std::unique_lock<std::mutex> lk(done->m);
        done->cv.wait(lk, [&] { return done->done; });
    }
    return serial;
}

uint64_t run_frame_sync(const std::function<void(Frame&)>& fn) {
    return queue_submit(fn, true, true);
}

// ---- gpu.h ---------------------------------------------------------------------------------------------------------------
bool active() { return g != nullptr; }
void cpu_will_read(uint64_t addr, uint64_t size) {
    if (g && out_any()) out_fetch(addr, size, false);
}
bool debug_labels() { return g && g->dev && g->dev->features().debug_labels; }

namespace {
std::atomic<uint64_t> g_idle_serial{0};   // without a back end submissions still get serials and complete at once (tests, tools)
}

// [nvn] async_submit (default on): the game thread queues its command lists and goes on; the render thread records and submits them in
// order. The serial is known at once (reserved in queue order), so sync objects and waits work as before: a wait for a serial the render
// thread has not submitted yet waits on the timeline until it has (wait-before-signal).
bool async_submit() {
    static const bool on = nxrt::config_get_bool("nvn", "async_submit", true);
    return on;
}

uint64_t submit(const std::vector<Ref<CmdList>>& lists) {
    if (!g) return ++g_idle_serial;
    if (g->dev->device_lost()) return g->reserved.fetch_add(1) + 1;
    NXFT("submit");
    // [nvn] mirror_capture: the CPU's writes since the last submit are copied now, before the game can write its next frame; capture and
    // queueing under one lock, so captures stay in queue order
    static std::mutex cap_order;
    std::unique_lock<std::mutex> cap_lk(cap_order, std::defer_lock);
    std::shared_ptr<void> cap;
    if (mirror_capture_on()) { cap_lk.lock(); cap = mirror_capture_take(); }
    return queue_submit([lists, cap](Frame& f) {
        mirror_capture_set(cap);
        replay_begin_submit(f);
        replay_lists(f, lists);
        replay_end_submit(f);
    }, !async_submit(), false);
}

uint64_t flush() { return g ? g->reserved.load() : g_idle_serial.load(); }
uint64_t last_serial() { return g ? g->reserved.load() : g_idle_serial.load(); }

void finish() {
    if (!g) return;
    run_on_render_thread([] {
        wait_serial_blocking(g->submitted);
        retire();
    });
}

bool serial_done(uint64_t serial) {
    if (!g || !serial) return true;
    if (g->dev->device_lost()) return true;
    if (g->completed.load() >= serial) return true;
    if (read_completed() >= serial) return true;
    want_flush(serial);     // polled: it must reach the GPU to ever complete
    return false;
}

namespace {
std::atomic<uint64_t> g_waits_begun{0}, g_waits_done{0};   // sync waits entered and left: a live graphics thread moves both
std::atomic<int> g_waiters{0};                 // game threads inside wait_serial (freeze report)
std::atomic<uint64_t> g_wait_serial{0}, g_wait_timeout{0};   // the latest wait: serial and timeout

std::string gpu_report() {
    if (!g) return {};
    uint64_t v = 0;
    VkResult sr = VK_SUCCESS;
    if (g->timeline) sr = vkGetSemaphoreCounterValue(g->dev->device(), g->timeline, &v);
    const bool poisoned = (sr == VK_ERROR_DEVICE_LOST) || (sr == VK_SUCCESS && v == UINT64_MAX && g->last_submitted.load() < UINT64_MAX - 1024);
    const bool lost = g->dev->device_lost() || poisoned;
    std::string fault;
    if (g->dev->features().device_fault) fault = g->dev->query_device_fault();
    char b[512];
    size_t q = 0;
    bool got = false;
    if (g->qmu.try_lock()) { q = g->queue.size(); got = true; g->qmu.unlock(); }
    std::snprintf(b, sizeof b, "NVN back end: %slast submitted serial %llu, timeline value %llu (rc %d), %d game thread(s) in a sync wait (latest wait: serial %llu, timeout %llu ns), sync waits begun %llu done %llu, submits %llu, presents %llu, render queue %s%zu%s%s\n",
                  lost ? "[DEVICE LOST] " : "",
                  static_cast<unsigned long long>(g->last_submitted.load()), static_cast<unsigned long long>(v), static_cast<int>(sr),
                  g_waiters.load(),
                  static_cast<unsigned long long>(g_wait_serial.load()), static_cast<unsigned long long>(g_wait_timeout.load()),
                  static_cast<unsigned long long>(g_waits_begun.load()), static_cast<unsigned long long>(g_waits_done.load()),
                  static_cast<unsigned long long>(g->ctr.submits.load()), static_cast<unsigned long long>(g->ctr.presents.load()), got ? "" : "(busy) ", q,
                  fault.empty() ? "" : "; device fault: ", fault.empty() ? "" : fault.c_str());
    // Video memory next to the fault: a device loss at 4K is either a bad submission or an exhausted heap, and the two
    // look identical from the draw ring alone (issues #10, #63, #89, #97). vram_live_info() only queries the physical
    // device and reads atomics, so it is safe on a lost device.
    const VramLiveInfo vi = vram_live_info();
    char vb[256];
    std::snprintf(vb, sizeof vb, "VRAM: %s%llu / %llu MiB, pool copies %llu / %llu MiB, capture %s, scene scale %.3f, output scale %.3f\n",
                  vi.budget_live ? "" : "heap ", static_cast<unsigned long long>(vi.vram_used_mb),
                  static_cast<unsigned long long>(vi.vram_total_mb), static_cast<unsigned long long>(vi.pool_copy_mb),
                  static_cast<unsigned long long>(vi.pool_cap_mb), vi.capture_on ? "on" : "off",
                  static_cast<double>(render_scale_live()), static_cast<double>(output_scale_live()));
    return std::string(b) + vb;
}
}  // namespace

int presents_since(uint64_t serial);   // below, beside the frame history

bool wait_serial(uint64_t serial, uint64_t timeout_ns) {
    if (!g || !serial || g->dev->device_lost() || serial_done(serial)) return true;
    if (timeout_ns == 0) return false;
    want_flush(serial);
    NXFT("sync.wait");
    nxrt::HitchTimer ht(nxrt::kHpGameSyncWait);
    g_wait_serial.store(serial);
    g_wait_timeout.store(timeout_ns);
    struct Counted { Counted() { g_waiters++; g_waits_begun++; } ~Counted() { g_waiters--; g_waits_done++; } } counted;
    // Where a game sync wait spends its time (log line every 10 s): entered while the render thread had not recorded the fenced work yet
    // ("replay": the wait covers replay + GPU), or after ("gpu": the GPU alone). Submissions behind = how far the fence is past what is recorded.
    static std::atomic<uint64_t> w_n[2], w_us[2], w_behind{0}, w_last_ms{0}, w_age_n[4], w_age_us[4];
    const uint64_t rec = g->last_submitted.load(std::memory_order_relaxed);
    const int kind = serial > rec ? 0 : 1;
    const int age = presents_since(serial);
    if (kind == 0) w_behind.fetch_add(serial - rec, std::memory_order_relaxed);
    const auto tw0 = std::chrono::steady_clock::now();
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &g->timeline;
    wi.pValues = &serial;
    const VkResult r = vkWaitSemaphores(g->dev->device(), &wi, timeout_ns);
    if (r == VK_ERROR_DEVICE_LOST) g->dev->note_device_lost(r, "nvn wait_serial(vkWaitSemaphores)");
    read_completed();
    const auto tw1 = std::chrono::steady_clock::now();
    w_n[kind].fetch_add(1, std::memory_order_relaxed);
    const uint64_t us_w = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(tw1 - tw0).count());
    w_us[kind].fetch_add(us_w, std::memory_order_relaxed);
    w_age_n[age].fetch_add(1, std::memory_order_relaxed);
    w_age_us[age].fetch_add(us_w, std::memory_order_relaxed);
    const uint64_t now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tw1.time_since_epoch()).count());
    uint64_t last = w_last_ms.load(std::memory_order_relaxed);
    if (!last) w_last_ms.compare_exchange_strong(last, now_ms);
    else if (now_ms - last >= 10000 && w_last_ms.compare_exchange_strong(last, now_ms)) {
        const uint64_t n0 = w_n[0].exchange(0), n1 = w_n[1].exchange(0), u0 = w_us[0].exchange(0), u1 = w_us[1].exchange(0), b = w_behind.exchange(0);
        NXLOG_I(kTag, "sync waits, last %.1f s: behind the render thread %llu (%.1f ms total, %.2f submissions behind on average), on the GPU only %llu (%.1f ms total)",
                (now_ms - last) / 1000.0, static_cast<unsigned long long>(n0), u0 / 1000.0, n0 ? static_cast<double>(b) / n0 : 0.0,
                static_cast<unsigned long long>(n1), u1 / 1000.0);
        uint64_t an[4], au[4];
        for (int i = 0; i < 4; ++i) { an[i] = w_age_n[i].exchange(0); au[i] = w_age_us[i].exchange(0); }
        NXLOG_I(kTag, "sync waits by fence age (presents since it was queued): 0: %llu (%.1f ms), 1: %llu (%.1f ms), 2: %llu (%.1f ms), 3+: %llu (%.1f ms)",
                static_cast<unsigned long long>(an[0]), au[0] / 1000.0, static_cast<unsigned long long>(an[1]), au[1] / 1000.0,
                static_cast<unsigned long long>(an[2]), au[2] / 1000.0, static_cast<unsigned long long>(an[3]), au[3] / 1000.0);
    }
    return r == VK_SUCCESS;
}

namespace {
// The last serial queued before each of the game's presents (the end of that frame's GPU work), for the low-latency GPU wait (lowlat.h).
// Written by the game thread in present(), read by the main thread at the frame boundary (the same thread in Pokopia).
std::atomic<uint64_t> g_frame_work[4];
std::atomic<uint64_t> g_frame_count{0};

bool gpu_wait_frames(int frames_back, uint64_t timeout_ns) {
    const uint64_t n = g_frame_count.load(std::memory_order_acquire);
    if (!g || n <= static_cast<uint64_t>(frames_back)) return true;
    const uint64_t serial = g_frame_work[(n - 1 - static_cast<uint64_t>(frames_back)) & 3].load(std::memory_order_relaxed);
    NXFT("lowlat.gpu_wait");
    return wait_serial(serial, timeout_ns);
}
}  // namespace

// Presents the game made after `serial` was queued (0: the fence is in the frame being built; 3: three or more frames back).
int presents_since(uint64_t serial) {
    const uint64_t n = g_frame_count.load(std::memory_order_acquire);
    int k = 0;
    while (k < 3 && static_cast<uint64_t>(k) < n && g_frame_work[(n - 1 - static_cast<uint64_t>(k)) & 3].load(std::memory_order_relaxed) >= serial) ++k;
    return k;
}

// [nvn] present_ahead (default 1; 0 = wait for each present): the game thread queues the present and goes on while fewer than this many
// presents are still pending on the render thread (the frame limiter and the swapchain's frame slots pace the render thread). 1 keeps the
// render thread one present behind the game at most (2 let a second finished frame wait in the queue: one more frame between input and screen).
void present(Window* window, int index) {
    if (!g || !window) return;
    NXFT("present.call");
    nxrt::hitch_note_game_present();
    gpu::vram_log_periodic();
    {
        std::lock_guard<std::mutex> lk(g->pmu);
        g->last_window = Ref<Window>::share(window);
        g->last_index = index;
        g->last_present = std::chrono::steady_clock::now();
    }
    {
        const uint64_t n = g_frame_count.load(std::memory_order_relaxed);
        g_frame_work[n & 3].store(g->reserved.load(), std::memory_order_relaxed);
        g_frame_count.store(n + 1, std::memory_order_release);
    }
    // auto (XC2 NS2's default, xc2_boot.cpp): lead by one frame only while [nvn] mirror_capture is on, which is what makes a guest that
    // rewrites buffers for the next frame safe; without it 0, so the GPU never reads what the guest writes ahead (torn parts of the picture)
    static const int ahead = [] {
        if (nxrt::config_get_str("nvn", "present_ahead", "") == "auto") {
            const int a = mirror_capture_on() ? 1 : 0;
            NXLOG_I(kTag, "present_ahead: %d (auto: mirror_capture is %s)", a, a ? "on" : "off");
            return a;
        }
        return static_cast<int>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "present_ahead", 1), 0, 3));
    }();
    if (ahead == 0 || on_render_thread()) {
        Ref<Window> keep = Ref<Window>::share(window);
        run_on_render_thread([&] {
            present_on_render_thread(window, index);
            if (g->dev->device_lost()) retire();
        });
        return;
    }
    {
        NXFT("present.backpressure");
        nxrt::HitchTimer ht(nxrt::kHpGameBackpressure);
        std::unique_lock<std::mutex> lk(g->pmu);
        g->pcv.wait(lk, [] { return g->presents_pending < ahead; });
        ++g->presents_pending;
    }
    Ref<Window> keep = Ref<Window>::share(window);
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        g->queue.push_back([keep, index] {
            auto release = on_exit([] {
                std::lock_guard<std::mutex> l2(g->pmu);
                --g->presents_pending;
                g->pcv.notify_all();
            });
            test_throw(g_test_presents);
            present_on_render_thread(keep.get(), index);
            if (g->dev->device_lost()) retire();
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
}

// The game's last frame once more, queued like a present (the overlay is drawn on top as on every frame). Only when the game has not
// presented for 50 ms and nothing is queued, so it never competes with the game's own frames.
void present_again() {
    if (!g) return;
    Ref<Window> keep;
    int index = -1;
    {
        std::lock_guard<std::mutex> lk(g->pmu);
        if (!g->last_window || g->presents_pending > 0) return;
        if (std::chrono::steady_clock::now() - g->last_present < std::chrono::milliseconds(50)) return;
        keep = g->last_window;
        index = g->last_index;
        ++g->presents_pending;
    }
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        g->queue.push_back([keep, index] {
            auto release = on_exit([] {
                std::lock_guard<std::mutex> l2(g->pmu);
                --g->presents_pending;
                g->pcv.notify_all();
            });
            present_on_render_thread(keep.get(), index, true);
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
}

void destroy_texture_gpu(TexGpu* t) {
    if (!t) return;
    if (!g) { tex_destroy(t); return; }
    defer_destroy([t] { tex_destroy(t); });
}

void release_texture_gpu(Texture* t) {
    if (!g || !t) return;
    Ref<Texture> keep = Ref<Texture>::share(t);
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        if (g->stop) return;
        g->queue.push_back([keep] {
            Texture* x = keep.get();
            if (!x->destroyed.load() || !x->gpu) return;   // initialized again meanwhile, or never got an image
            TexGpu* img = x->gpu;
            x->gpu = nullptr;
            pool_texture_remade(x);
            destroy_texture_gpu(img);
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
}

void destroy_pool_gpu(PoolGpu* p) {
    if (!p) return;
    if (!g) { pool_gpu_destroy(p); return; }
    defer_destroy([p] { pool_gpu_destroy(p); });
}

namespace {
std::atomic<int> g_origin{NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT}, g_depth_mode{NVN_DEPTH_MODE_NEAR_IS_MINUS_W};
}
void set_window_modes(int origin_mode, int depth_mode) { g_origin.store(origin_mode); g_depth_mode.store(depth_mode); }
int window_origin_mode() { return g_origin.load(); }
int depth_mode() { return g_depth_mode.load(); }
gfx::Renderer* renderer() { return g ? g->rend : nullptr; }
FrameLimiter* limiter() { return g ? g->limiter : nullptr; }
PresentPass* present_pass() { return g && g->present_ok ? &g->present : nullptr; }
extras::OutputPass* output_pass() { return g && g->output_ok ? &g->output : nullptr; }
bool keep_aspect() { return g && g->keep_aspect.load(std::memory_order_relaxed); }
void set_keep_aspect_live(bool on) {
    if (g) g->keep_aspect.store(on, std::memory_order_relaxed);
}

uint64_t timestamp_ns() {
    static const auto t0 = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()) + 1000000000ull;
}

std::string unsupported_report() {
    std::string out;
    if (!g) return out;
    std::lock_guard<std::mutex> lk(g->unsupported_mu);
    for (auto& [name, n] : g->unsupported) out += name + ": " + std::to_string(n) + "\n";
    return out;
}

// ---- lifecycle -----------------------------------------------------------------------------------------------------------
namespace {

void register_counter_gauges();

bool bring_up(gfx::Device& dev, gfx::Renderer* rend, FrameLimiter* limiter, bool keep_aspect) {
    std::lock_guard<std::mutex> lk(g_install_mu);
    if (g) return g->dev == &dev;
    auto* gp = new Gpu();
    gp->dev = &dev;
    gp->rend = rend;
    gp->limiter = limiter;
    gp->keep_aspect.store(keep_aspect, std::memory_order_relaxed);
    {   // [nvn] submit_batch: auto = frame under Wine (each kernel submission walks the imported pools again, flush_batch), 1 elsewhere.
        // frame: one kernel submission per frame, the present's; N: flush every N submissions and whenever the render thread idles.
        std::string b = nxrt::config_get_str("nvn", "submit_batch", "auto");
        HMODULE nt = GetModuleHandleW(L"ntdll.dll");
        const bool wine = nt && GetProcAddress(nt, "wine_get_version");
        const std::string how = b == "auto" ? (wine ? "auto, Wine" : "auto, not Wine") : b;
        if (b == "auto") b = wine ? "frame" : "1";
        gp->frame_batch = b == "frame";
        gp->batch_max = gp->frame_batch ? kSlots / 2 : static_cast<uint32_t>(std::clamp<int64_t>(std::atoll(b.c_str()), 1, kSlots / 2));
        NXLOG_I(kTag, "submit_batch %s (%s)", gp->frame_batch ? "frame" : std::to_string(gp->batch_max).c_str(), how.c_str());
    }
    g = gp;
    if (!dev.features().timeline_semaphore) {
        NXLOG_E(kTag, "the GPU has no timeline semaphores: the NVN back end cannot run");
        g = nullptr;
        delete gp;
        return false;
    }
    VkSemaphoreTypeCreateInfo tci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &tci};
    if (vkCreateSemaphore(dev.device(), &sci, nullptr, &gp->timeline) != VK_SUCCESS) {
        g = nullptr;
        delete gp;
        return false;
    }
    vk_gauge_semaphores().add(1);
    for (Slot& s : gp->slots)
        if (!create_slot(s)) {
            NXLOG_E(kTag, "creating a submission slot failed");
            g = nullptr;
            delete gp;
            return false;
        }
    if (ft::on() || dev.properties().limits.timestampComputeAndGraphics) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kSlots * 2;
        if (vkCreateQueryPool(dev.device(), &qi, nullptr, &gp->qpool) != VK_SUCCESS) gp->qpool = VK_NULL_HANDLE;
        else vk_gauge_query_pools().add(1);
    }
    // [prof] gpu_passes alone is enough: without the frame trace the intervals are summarised per second instead of
    // going into ft::event (2026-11-16: the open plain is GPU-bound, so the pass times are what we need to see).
    if (config_get_bool("prof", "gpu_passes", false)) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kSlots * kGpMax;
        if (vkCreateQueryPool(dev.device(), &qi, nullptr, &gp->gpool) != VK_SUCCESS) gp->gpool = VK_NULL_HANDLE;
        else vk_gauge_query_pools().add(1);
    }
    if (ft::on()) extras::lowlat_set_backlog_fn([]() -> uint64_t {      // latency probe: submissions the GPU has not finished
        if (!g || !g->timeline) return 0;
        uint64_t v = 0;
        vkGetSemaphoreCounterValue(g->dev->device(), g->timeline, &v);
        const uint64_t sub = g->last_submitted.load();
        return sub > v ? sub - v : 0;
    });
    extras::lowlat_set_gpu_wait_fn(&gpu_wait_frames);
    static const bool report_registered = (diag_add_report(gpu_report), true);   // the freeze dump shows the GPU state
    (void)report_registered;
    register_counter_gauges();
    gp->th = std::thread(render_loop);
    // wait for the thread id to be published
    for (int i = 0; i < 1000 && gp->tid == std::thread::id(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (rend) {
        run_on_render_thread([&] { gp->present_ok = gp->present.init(dev, rend->format()); });
        if (!gp->present_ok) NXLOG_E(kTag, "present pass failed to initialise: frames will not be shown");
        run_on_render_thread([&] { gp->output_ok = gp->output.init(dev); });
        if (!gp->output_ok) NXLOG_W(kTag, "output pass (HDR, FSR 1) failed to initialise: those extras are off");
    }
    NXLOG_I(kTag, "NVN back end up on %s (%u submission slots, timeline semaphore, host-memory import %s)", dev.properties().deviceName, kSlots,
            dev.features().external_memory_host ? "available" : "NOT available");
    return true;
}

const char* const kCounterGauges[] = {"nvn.cnt.draws", "nvn.cnt.passes", "nvn.cnt.pass_breaks", "nvn.cnt.dispatches", "nvn.cnt.clears", "nvn.cnt.copies",
                                      "nvn.cnt.submits", "nvn.cnt.descriptor_sets"};

void register_counter_gauges() {
    auto reg = [](const char* name, std::atomic<uint64_t> Counters::*m) {
        nxrt::gauge_register(name, [m]() -> int64_t { Gpu* gp = g; return gp ? static_cast<int64_t>((gp->ctr.*m).load(std::memory_order_relaxed)) : 0; });
    };
    reg("nvn.cnt.draws", &Counters::draws);
    reg("nvn.cnt.passes", &Counters::passes);
    reg("nvn.cnt.pass_breaks", &Counters::pass_breaks);
    reg("nvn.cnt.dispatches", &Counters::dispatches);
    reg("nvn.cnt.clears", &Counters::clears);
    reg("nvn.cnt.copies", &Counters::copies);
    reg("nvn.cnt.submits", &Counters::submits);
    reg("nvn.cnt.descriptor_sets", &Counters::descriptor_sets);
    // What the hitch log (nxrt/gauge.h) shows for a slow frame: growth of these during the frame, read lock-free on the presenting thread.
    static const bool hitch_registered = [] {
        nxrt::hitch_add_counter("pipelines_created", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.pipelines_created.load(std::memory_order_relaxed) : 0; });
        nxrt::hitch_add_counter("texture_uploads", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.texture_uploads.load(std::memory_order_relaxed) : 0; });
        nxrt::hitch_add_counter("texture_upload_kb", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.texture_upload_bytes.load(std::memory_order_relaxed) >> 10 : 0; });
        nxrt::hitch_add_counter("bc7_transcodes", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.bc7_transcodes.load(std::memory_order_relaxed) : 0; });
        nxrt::hitch_add_level("pipeline_compiles_in_flight", []() -> int64_t { return compiles_in_flight().load(std::memory_order_relaxed); });
        return true;
    }();
    (void)hitch_registered;
}

void tear_down() {
    std::lock_guard<std::mutex> lk(g_install_mu);
    if (!g) return;
    for (const char* n : kCounterGauges) nxrt::gauge_unregister(n);
    Gpu* gp = g;
    run_on_render_thread([&] {
        {
            std::lock_guard<std::mutex> l2(gp->pmu);
            gp->last_window = nullptr;
        }
        wait_serial_blocking(gp->submitted);
        retire();
        replay_shutdown();
        tex_shutdown();
        pool_shutdown();
        while (!gp->deferred.empty()) {
            auto fn = std::move(gp->deferred.front().fn);
            gp->deferred.pop_front();
            g_gauge_deferred.sub(1);
            fn();
        }
        drain_pending_destroy();
        for (auto& d : gp->deferred) { d.fn(); g_gauge_deferred.sub(1); }
        gp->deferred.clear();
        gp->present.shutdown();
        gp->output.shutdown();
    });
    {
        std::lock_guard<std::mutex> ql(gp->qmu);
        gp->stop = true;
    }
    gp->qcv.notify_all();
    if (gp->th.joinable()) gp->th.join();
    gfx::Device& d = *gp->dev;
    vkDeviceWaitIdle(d.device());
    for (Slot& s : gp->slots) {
        for (auto& b : s.upload_overflow) d.destroy_buffer(b);
        s.upload_overflow.clear();
        for (auto& b : s.extra) d.destroy_buffer(b);
        vk_gauge_descriptor_sets().sub(static_cast<int64_t>(s.sets_live));
        s.sets_live = 0;
        for (VkDescriptorPool p : s.dpools) {
            vkDestroyDescriptorPool(d.device(), p, nullptr);
            vk_gauge_descriptor_pools().sub(1);
        }
        d.destroy_buffer(s.upload);
        if (s.pool) {
            vkDestroyCommandPool(d.device(), s.pool, nullptr);
            vk_gauge_command_pools().sub(1);
            if (s.cb) vk_gauge_command_buffers().sub(1);
        }
    }
    for (auto& b : gp->upload_pool) d.destroy_buffer(b);
    gp->upload_pool.clear();
    if (gp->qpool) { vkDestroyQueryPool(d.device(), gp->qpool, nullptr); vk_gauge_query_pools().sub(1); }
    if (gp->gpool) { vkDestroyQueryPool(d.device(), gp->gpool, nullptr); vk_gauge_query_pools().sub(1); }
    if (gp->timeline) { vkDestroySemaphore(d.device(), gp->timeline, nullptr); vk_gauge_semaphores().sub(1); }
    g = nullptr;
    delete gp;
}

}  // namespace

bool init_for_test(gfx::Device& dev, gfx::Renderer* renderer) { return bring_up(dev, renderer, nullptr, false); }
void shutdown_for_test() { tear_down(); }

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn {

bool install_backend(const BackendDesc& desc) {
    if (!desc.device) return false;
    if (!be::gpu::bring_up(*desc.device, desc.renderer, desc.limiter, desc.keep_aspect)) return false;
    if (desc.register_entry_points) {
        install();
        register_all_impls();
        NXLOG_I("nvn", "NVN back end installed: %zu of %zu entry points implemented", implemented_count(), entry_count());
    }
    return true;
}

void shutdown_backend() { be::gpu::tear_down(); }
void present_last_frame_again() {
    if (be::gpu::active()) be::gpu::present_again();
}

bool prepare_shaders(const ShaderPrepareOptions& o, ShaderPrepareResult* r) {
    ShaderPrepareResult local;
    ShaderPrepareResult& out = r ? *r : local;
    if (!be::gpu::active()) { out = ShaderPrepareResult{}; out.error = "the graphics back end is not running"; return false; }
    return be::gpu::gpl_prepare(o, &out);
}
bool backend_installed() { return be::gpu::active(); }
void backend_flush_caches() {
    if (be::gpu::active()) be::gpu::flush_caches();
}

void backend_trace_next_frame() {
    if (be::gpu::active()) be::gpu::replay_request_trace();
}

void backend_trace_frames(int count) {
    if (be::gpu::active()) be::gpu::replay_request_trace_span(count);
}

void set_keep_aspect(bool on) {
    be::gpu::set_keep_aspect_live(on);
}

namespace {
std::atomic<uint32_t> g_vp_rw{0}, g_vp_rh{0}, g_vp_bw{0}, g_vp_bh{0};
std::atomic<bool> g_vp_crop{false}, g_vp_bypass{false};
}  // namespace

void set_game_viewport_info(uint32_t render_w, uint32_t render_h, uint32_t base_w, uint32_t base_h,
                            bool apply_present_crop, bool dlss_bypassed) {
    g_vp_rw.store(render_w, std::memory_order_relaxed);
    g_vp_rh.store(render_h, std::memory_order_relaxed);
    g_vp_bw.store(base_w, std::memory_order_relaxed);
    g_vp_bh.store(base_h, std::memory_order_relaxed);
    g_vp_crop.store(apply_present_crop, std::memory_order_relaxed);
    g_vp_bypass.store(dlss_bypassed, std::memory_order_relaxed);
}

GameViewportInfo game_viewport_info() {
    GameViewportInfo info;
    info.render_w = g_vp_rw.load(std::memory_order_relaxed);
    info.render_h = g_vp_rh.load(std::memory_order_relaxed);
    info.base_w = g_vp_bw.load(std::memory_order_relaxed);
    info.base_h = g_vp_bh.load(std::memory_order_relaxed);
    info.apply_present_crop = g_vp_crop.load(std::memory_order_relaxed);
    info.dlss_bypassed = g_vp_bypass.load(std::memory_order_relaxed);
    return info;
}

GameViewportInfo scale_viewport_info(const GameViewportInfo& gvi, float scene_scale, float out_scale, uint32_t quality_out_h,
                                     int out_min) {
    GameViewportInfo out = gvi;
    if (gvi.render_w > 0 && gvi.render_h > 0) {
        const float r_sc = be::gpu::compute_attachment_scale(
            static_cast<int>(gvi.render_w), static_cast<int>(gvi.render_h),
            NVN_TEXTURE_TARGET_2D, 0, false, 1, 1,
            scene_scale, out_scale, out_min, 0, quality_out_h > 0, static_cast<int>(quality_out_h));
        out.render_w = static_cast<uint32_t>(std::max(1L, std::lround(static_cast<double>(gvi.render_w) * r_sc)));
        out.render_h = static_cast<uint32_t>(std::max(1L, std::lround(static_cast<double>(gvi.render_h) * r_sc)));
    }
    if (gvi.base_w > 0 && gvi.base_h > 0) {
        const float b_sc = be::gpu::compute_attachment_scale(
            static_cast<int>(gvi.base_w), static_cast<int>(gvi.base_h),
            NVN_TEXTURE_TARGET_2D, 0, false, 1, 1,
            scene_scale, out_scale, out_min, 0, quality_out_h > 0, static_cast<int>(quality_out_h));
        out.base_w = static_cast<uint32_t>(std::max(1L, std::lround(static_cast<double>(gvi.base_w) * b_sc)));
        out.base_h = static_cast<uint32_t>(std::max(1L, std::lround(static_cast<double>(gvi.base_h) * b_sc)));
    }
    return out;
}

GameViewportInfo scaled_game_viewport_info() {
    const GameViewportInfo gvi = game_viewport_info();
    if (!be::gpu::active()) return gvi;
    static const int out_min = static_cast<int>(nxrt::config_get_int("gfx", "output_min_width", 1920));
    // Call render_scale() first to ensure lazy init of scales, then read out_scale and quality_out_h.
    (void)be::gpu::render_scale();
    // Use direct render/output scale via compute_attachment_scale behavior:
    // Note: when scene == 1.0f, be::gpu::render_scale() returns output_scale_live(), so expose exact render_scale_live too or query via config:
    // Better: compute_attachment_scale uses exact scene & out; let's check both.
    const float out_sc = be::gpu::output_scale_live();
    const uint32_t q_h = be::gpu::quality_output_height_live();
    // Wait: if scene != out, be::gpu::render_scale_now() != render_scale() when scene==1.0f! Let's read scene via render_scale_live().
    return scale_viewport_info(gvi, be::gpu::render_scale_live(), out_sc, q_h, out_min);
}

BackendStats backend_stats() {
    BackendStats s;
    if (!be::gpu::active()) return s;
    auto& c = be::gpu::counters();
    s.submits = c.submits; s.lists_replayed = c.lists_replayed; s.draws = c.draws; s.draws_skipped = c.draws_skipped;
    s.dispatches = c.dispatches; s.clears = c.clears; s.copies = c.copies; s.presents = c.presents;
    s.passes = c.passes; s.pass_breaks = c.pass_breaks; s.pipelines_created = c.pipelines_created;
    s.pipeline_cache_hits = c.pipeline_cache_hits; s.descriptor_sets = c.descriptor_sets;
    s.textures_created = c.textures_created; s.texture_uploads = c.texture_uploads; s.texture_upload_bytes = c.texture_upload_bytes;
    s.bc7_transcodes = c.bc7_transcodes; s.textures_failed = c.textures_failed; s.pools_imported = c.pools_imported;
    s.unresolved_addresses = c.unresolved_addresses; s.missing_handles = be::handles_missing(); s.unsupported_ops = c.unsupported_ops;
    s.submit_replay_ns = c.submit_replay_ns;
    s.gpu_busy_ns = c.gpu_busy_ns;
    return s;
}

std::string backend_unsupported_report() { return be::gpu::unsupported_report(); }

}  // namespace nxrt::nvn
