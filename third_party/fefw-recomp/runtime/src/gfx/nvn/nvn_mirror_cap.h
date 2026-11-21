// SPDX-License-Identifier: GPL-3.0-or-later
// S38: how much video memory the device-local pool copies ([nvn] pool_mirror) may take. Pure, so the tests can call it.
#pragma once

#include <algorithm>
#include <cstdint>

namespace nxrt::nvn {

struct MirrorCap {
    uint64_t bytes = 0;     // most bytes the copies may take in all (0: no copies)
    const char* why = "";   // for the log line
};

constexpr uint64_t kMirrorCapMax = 8192ull << 20;      // the default cap on a large card
constexpr uint64_t kMirrorMinVram = 4ull << 30;        // below this much video memory there are no copies by default
// [nvn] mirror_capture = auto: the capture needs a whole-pool copy of every pool it serves (XC2 NS2: ~2.6 GiB, mesh pools of 500-640
// MiB); a pool without one is read live, and with the guest a frame ahead that is the flicker the capture exists to remove. Below this
// budget auto leaves the capture off. 3.5 GiB: a 16 GB card reports a ~15.6 GiB heap (budget ~3.9 GiB), a 12 GB card ~2.9 GiB.
constexpr uint64_t kCaptureMinBudget = 3584ull << 20;

// vram_bytes: the largest device-local heap (0 when unknown). explicit_mb >= 0: [nvn] pool_mirror_mb was set in the ini or the
// environment, and that value wins; -1: not set, the cap follows the card: min(kMirrorCapMax, a quarter of the heap), none below 4 GiB.
// shared_memory: an integrated GPU or APU, whose "video memory" is system RAM; the copies would only cost RAM there, so none by default.
inline MirrorCap mirror_cap(uint64_t vram_bytes, int64_t explicit_mb, bool shared_memory = false) {
    if (explicit_mb >= 0) return {static_cast<uint64_t>(explicit_mb) << 20, "[nvn] pool_mirror_mb"};
    if (shared_memory) return {0, "integrated GPU (shares system memory)"};
    if (vram_bytes < kMirrorMinVram) return {0, "under 4 GiB of video memory"};
    return {std::min(kMirrorCapMax, vram_bytes / 4), vram_bytes / 4 < kMirrorCapMax ? "a quarter of the video memory" : "8192 MiB"};
}

}  // namespace nxrt::nvn
