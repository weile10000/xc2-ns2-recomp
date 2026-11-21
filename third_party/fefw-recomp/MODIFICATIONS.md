# Modifications to fefw-recomp

This directory is a **modified fork** of [Goldoire/fefw-recomp](https://github.com/Goldoire/fefw-recomp),
vendored into the Xenoblade Chronicles 2 NS2E recomp project. Upstream base and provenance: [UPSTREAM.md](UPSTREAM.md).

| | |
|---|---|
| upstream | https://github.com/Goldoire/fefw-recomp |
| licence | GPL-3.0-or-later (see [LICENSE](LICENSE) and [LEGAL.md](LEGAL.md)) |
| fork base commit in this repository | `6f1d964` (2026-11-12, baseline: 26 rounds of XC2 NS2 recomp) |
| files changed since the base | 60 |
| generated from | `git diff --name-only 6f1d964..HEAD -- third_party/fefw-recomp` |

> **Why this file and not per-file headers.** GPL-3.0 section 5(a) requires the work to carry prominent notices
> that it was modified, with a relevant date. This file is that notice, and it is authoritative. Per-file
> `// Modified ...` headers are deliberately not added: the project's internal notes cite line numbers of these files
> (for example `nvn_texture_registry.cpp:305`), and a one-line header shifts every reference. If a file is
> later copied out of this fork on its own, add the header to that copy.

## Changed files (date = last change)

| file | date | change |
|---|---|---|
| `runtime/docs/CONFIG.md` | 2026-11-21 | the settings the NVIDIA fixes and the render_scale fixes added (mirror_capture, sync_scope, render_scale_min_width, ...); pool_mirror_mb: at most 8192 MiB |
| `runtime/include/nxrt/write_watch.h` | 2026-11-21 | write-watch: per-64 KiB-reservation queries (capture_poll) for submit-time captures; VAMM pages tracked by GetWriteWatch |
| `runtime/src/mem/write_watch.cpp` | 2026-11-21 | write-watch: VAMM (MapPhysicalMemory) pages tracked through their write-watch record; capture_poll for submit-time captures |
| `runtime/tests/test_svc_memory.cpp` | 2026-11-21 | test: a large UnmapPhysicalMemory followed by a map of the same range succeeds |
| `runtime/src/gfx/nvn/api_device.cpp` | 2026-11-21 | [nvn] sync_scope switch: window-acquire syncs can wait for all GPU work (NVIDIA fallback) |
| `runtime/src/gfx/nvn/api_memory.cpp` | 2026-11-21 | [nvn] pool_prefetch_init: every GPU-reachable pool is imported in the background when it is made, not at its first draw |
| `runtime/src/gfx/nvn/nvn_mirror_cap.h` | 2026-11-21 | mirror cap 2048 -> 8192 MiB (still at most a quarter of the video memory); kCaptureMinBudget for mirror_capture = auto |
| `runtime/src/gfx/nvn/nvn_texprep.cpp` | 2026-11-21 | ready_textures(): finished background preparations, uploaded at the start of a submission ([nvn] tex_preupload) |
| `runtime/src/gfx/nvn/nvn_texprep.h` | 2026-11-21 | ready_textures(): finished background preparations, uploaded at the start of a submission ([nvn] tex_preupload) |
| `runtime/src/gfx/nvn/api_dlss.cpp` | 2026-11-19 | `XC2_DLSS_MIN_SCALE`: an env-gated floor for the render range `dlss_optimal` reports, so the dynamic-resolution experiment can ask what the game does when its DLSS input must be larger (HANDOVER 217). Off by default; the code is otherwise unchanged |
| `UPSTREAM.md` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/extras.h` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/include/nxrt/gfx.h` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/include/nxrt/memory.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/platform.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/services_api.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/svc.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/scripts/gen_earlyz_mask_shaders.py` | 2026-11-13 | nvn: the light list registration mask is derived from the list after every list draw - the black rectangle (HA |
| `runtime/src/core/switch2.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/diag/crash.cpp` | 2026-11-14 | pkg7 assert: the data is newer than the exefs (HANDOVER 79) |
| `runtime/src/gfx/extras/temporal.cpp` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/src/gfx/extras/temporal_ffx.cpp` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/src/gfx/nvn/api_dlss_replay.cpp` | 2026-11-21 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s; the DLSS/TAAU output follows [gfx] render_scale |
| `runtime/src/gfx/nvn/api_program.cpp` | 2026-11-14 | nvn: [nvn] shader_dump writes the code and control of every unresolved block (HANDOVER 74) |
| `runtime/src/gfx/nvn/api_texture.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/earlyz_mask_shaders.inc` | 2026-11-13 | nvn: the light list registration mask is derived from the list after every list draw - the black rectangle (HA |
| `runtime/src/gfx/nvn/nvn_core.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/nvn_enums.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/nvn_gpu.cpp` | 2026-11-21 | nvn: [nvn] split_draws hands the recorded commands to the GPU at pass boundaries - the GPU stops waiting for t; never lose a flush request; 72 submission slots; [nvn] mirror_capture at submit; [nvn] present_ahead = auto |
| `runtime/src/gfx/nvn/nvn_gpu_internal.h` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override; mirror capture and prologue interfaces (Frame::prologue, mirror_capture_*) |
| `runtime/src/gfx/nvn/nvn_gpu_pool.cpp` | 2026-11-21 | nvn: pool import/mirror granule is a config (default 64 MiB unchanged); 16 MiB does not help; movement overrun; device-local mirrors for VAMM and SHADER_CODE pools; CPU snapshots; chunked pool imports; [nvn] mirror_capture (= auto) |
| `runtime/src/gfx/nvn/nvn_gpu_texture.cpp` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override; [gfx] render_scale_min_width: textures narrower than it keep their size under render_scale |
| `runtime/src/gfx/nvn/nvn_pipeline.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/nvn_pipeline.h` | 2026-11-14 | nvn: push the uniform buffers as push descriptors ([nvn] push_uniforms, ubo_only_rebind) |
| `runtime/src/gfx/nvn/nvn_replay.cpp` | 2026-11-21 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s; snapshots of instance streams; first uploads in a submission prologue; storage images under [gfx] render_scale |
| `runtime/src/gfx/nvn/nvn_shaders.cpp` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override; reflection marks storage images a shader writes (OpImageWrite, image atomics) |
| `runtime/src/gfx/nvn/nvn_spirv_patch.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/nvn_spirv_patch.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/platform.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/vk_device.cpp` | 2026-11-19 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s; start-up failures name the VkResult and the driver's own Vulkan version (a 1.0 probe instance) instead of a bare `-9` |
| `runtime/src/ipc/ipc.cpp` | 2026-11-14 | dlc: the title never asks aoc, so the Torna entry is save driven (HANDOVER 66) |
| `runtime/src/kernel/handles.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/kernel/svc_info.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/kernel/svc_memory.cpp` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override; MapPhysicalMemory keeps a running page count (no walk of the whole map per call); large unmaps release on a background thread |
| `runtime/src/kernel/svc_thread.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/loader/module.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/apps/video_codec_probe.cpp` | 2026-11-18 | probe: AV1 on the SDK 3.0.1 nn::codec entry points (video_codec_probe <movie.webm> <out.png> <frame>) |
| `runtime/src/media/movie_trace.cpp` | 2026-11-19 | movie: [media] player_trace wrappers over the SDK's os calls, and the host-side poke that makes the player's per-frame timer due (one ticker for every timer, not one thread per timer) - without it the 4K CG froze for 7-8 s at a time (handover 58) |
| `runtime/src/media/movie_vdec.cpp` | 2026-11-19 | movie: [media] video_decoder = library leaves movie::VideoDecoder to the library (it drives nn::codec AV1) and keeps the host AudioDecoder; same decoder-thread default as video_codec.cpp |
| `runtime/src/media/video_codec.cpp` | 2026-11-19 | vcodec: nn::codec AV1 via dav1d and the SDK 3.0.1 names (ByTransferMemory, ImageTilingMode preference, event-returning submission event); decoder threads default to half the cores (4-8) and the film delay to 110 ms - the movie library's VideoDecoder path |
| `runtime/src/mem/memory.cpp` | 2026-11-14 | heap: lazy commit is implemented and measured, but it stays off (crashes at 0.82 s) |
| `runtime/src/services/audren/Server/AudioRenderSystem.h` | 2026-11-12 | nvn: trace the depth/blend state a draw is submitted with, summarize the frame; audren: log every voice the ga |
| `runtime/src/services/common.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/services/fs.cpp` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/src/services/hid.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/services/misc.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/decoder/SassDecoder.Tensor.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/decoder/SassDecoder.Tex.cs` | 2026-11-14 | shaders: SUATOM (0x593/0x595) and the TLD4 PTP modifier - 1529 of 1529 blocks the game asks for translate (HAN |
| `tools/shaders/ampere/decoder/SassDecoder.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/frontend/AmpereEmit.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/frontend/AmpereTranslator.cs` | 2026-11-14 | shaders: take the NVN2 stage from the control section - 518 to 2813 of 4222 blocks translate (HANDOVER 73) |
| `tools/shaders/harness/Program.cs` | 2026-11-14 | shaders: take the NVN2 stage from the control section - 518 to 2813 of 4222 blocks translate (HANDOVER 73) |
| `tools/shaders/harness/StaticGpuAccessor.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/harness/translator/Ryujinx.Graphics.Shader.csproj` | 2026-11-19 | shaders: take the NVN2 stage from the control section - 518 to 2813 of 4222 blocks translate (HANDOVER 73); `spirv-call-args.patch` back in the series - without it 563 of 12,132 blocks throw in `GenerateCall` and only translate through the DebugMode retry (handover 56 section 179) |

## Reproduce

```sh
git diff --name-only 6f1d964..HEAD -- third_party/fefw-recomp
git log --oneline 6f1d964..HEAD -- third_party/fefw-recomp
```

## Areas of change (summary)

- **runtime/src/gfx/nvn** — NVN/NVN2 replay: program keys, pipeline cache, texture registry and uploads, pools,
  SPIR-V patching, the early-Z mask shaders, shader dumping.
- **runtime/src/gfx/extras** — the upscaler chain: TAAU, FSR 3.1/4.1 through the FidelityFX API, per-size FFX contexts.
- **runtime/include/nxrt** — the interfaces those changes expose (extras.h, gfx.h, memory.h, svc.h, ...).
- **runtime/src/{kernel,loader,services,mem,ipc,core,diag}** — Switch 2 bring-up: 128 GiB alias region, svcInfo
  stubs, heap model, nn::audio/opus on the retail 3.0.1 build, fs read tracing, crash dumps.
- **tools/shaders** — the Ampere (NVN2) SASS path: stage from the control section, SUATOM/TLD4, tensor decoder.

- **runtime/src/media/video_codec.cpp** — nn::codec on the host: VP8/VP9 with libvpx, AV1 with dav1d, and the SDK 3.0.1
  entry-point names the movie library in subsdk0 uses (Initialize*DecoderByTransferMemory/ByRemoteMemory, ImageTilingMode
  preference, a Get*DecoderSubmissionReadyEvent that returns the event, SubmitAv1DecoderInputBuffer without the out size).
  This is what the library's own movie::VideoDecoder needs; it never goes through movie::Decoder.
- **runtime/games/xc2ns2** — the title facts registered for this game (program id, Switch 2 platform, save sizes),
  and ``movie_video_decoder``: the DLC openings are AV1 and play through the host movie::VideoDecoder (dav1d).
- **runtime/src/core/runtime.cpp, runtime/src/services/common.cpp** — the startup log names the config file it
  loaded and the language value it read, so a launcher that starts the game elsewhere can be diagnosed.
- **runtime/src/gfx/nvn/nvn_gpu_texture.cpp** - `[nvn] ds_stencil_bits` defaults to 0 on NVIDIA: a float-declared sampler over the stencil plane is undefined there and took the device down (issue #4).
- **runtime/src/services/misc.cpp** - add-on content is discovered from the package folder (and its ids may use either base); before this, a dump with the six add-ons reported zero.
- **runtime/src/gfx/nvn/nvn_replay.cpp** - `[nvn] draw_ring` (default 32): the freeze dump now ends with the last draws the replay submitted, which is what a GPU hang leaves to go on.
- **runtime/src/kernel/svc_memory.cpp (2026-11-19)** - `MapPhysicalMemory` refusal now names the block that holds the range
  (state, base, size): the title's two 60 GiB VAMM setups can collide with themselves and the old line could not tell that
  from a real guest mapping.
- **runtime/src/gfx/extras/temporal_ffx.cpp (2026-11-19)** - `amd_fidelityfx_upscaler_vk.dll` is accepted as the FSR entry point when the
  39 KB loader is absent: both export the same five `ffx*` symbols, and a player who only has the one file should not fall back to TAAU.
- **runtime/src/gfx/play/play.cpp (2026-11-20)** - `[gfx] present_mode` wins over `[display] vsync`. The renderer read the key
  when it was created, and `play::start` replaced it 115 ms later with the mode derived from vsync, so a player who picked
  FIFO in the launcher presented in Immediate (issue #12).
- **runtime/src/gfx/play/play_config.cpp (2026-11-20)** - a controls.ini written before revision 1 takes `[display] vsync = 1`:
  the factory file said `0` while the launcher's "FIFO (vsync)" entry wrote nothing, so existing installs presented in
  Immediate and the swapchain change above could not reach them. The runtime's own factory revision (1) is applied through
  the same table the game uses, before the .bak/save, and is never reported below 1 (issue #12).
- **runtime/src/gfx/vk_device.cpp, runtime/src/gfx/nvn/nvn_gpu.cpp (2026-11-20)** - `VK_EXT_device_fault` is enabled when
  the driver offers it, and the first failed submit logs the driver's own fault description and address instead of only the
  `-4` vkQueueSubmit2 returned (issue #10).
- **runtime/src/gfx/nvn/nvn_replay.cpp (2026-11-19)** - the "float sampler over the stencil plane" report (`ds_debug: fs <hash> ...`)
  is now written unconditionally, once per (shader, binding), instead of only under `[nvn] ds_debug`. The patch tool can only
  rewrite modules a log names, and the run that lost the device was the one with nothing recorded in it (issue #10).
- **runtime/src/services/audio_xaudio2.cpp, runtime/src/services/audio_out.cpp (2026-11-21)** - the one host sink is
  opened again only when the format changes, because rebuilding the XAudio2 voice threw away everything queued in it; and
  audout stops submitting while the voice already holds about a second, so a burst cannot be dropped. Both audio paths
  (audout:u and the SDK's OutputStream, xc2/src/x2_audio301.cpp) open the same sink, and a movie start used to cost
  1.9M frames here.
- **runtime/cmake/media_deps.cmake (2026-11-21)** - libvorbis is detected the way libvpx is, so a development build decodes the
  films' sound as well (`NXRT_HAVE_VORBIS=1`). Only `xc2/release` used to define it, so every dev `xc2_boot.exe` played the
  picture without sound.

- **runtime/src/media/video_ffmpeg.cpp, runtime/src/media/video_codec.cpp (2026-11-21)** - one D3D11VA device for every film
  stream instead of one per stream: it is built by a background thread when the video codec module installs, so the cost lands in
  the loading screen and not in the first frame of a cutscene (0.1.4: the presenting thread waited 183 ms of a 213 ms frame).
  The decoder logs the device's creation time and the time from a stream's first packet to its first picture.
- **runtime/src/diag/gauge.cpp, runtime/include/nxrt/gauge.h (2026-11-21)** - `hitch_note_film_end()`: for two minutes after a
  film ends, the `hitch:` summaries carry `[film ended N s before this window]`, so a slow stretch that follows a cutscene is
  visible in a player's log without asking them to time anything. Each summary also carries a `per frame:` split of the
  window (replay, idle, present, gpu wait, upload, pipe, pool import, guest service and wait, ms per frame): the slow-frame
  lines describe the tail, and a steady 20-30 fps is made of the average frame, which nothing reported before.

- **NVIDIA fixes, runtime/src/gfx/nvn, runtime/src/mem, runtime/src/kernel (2026-11-21)** - XC2 NS2 on NVIDIA GPUs
  froze or lost the device at start-up and showed flashes and flicker: the game rewrites per-frame buffers with no fence while
  the GPU may still read them. `[nvn] sync_scope` (default unchanged) is the slow fallback. VAMM pages are write-watched and
  get device-local mirrors with CPU snapshots at replay; flush requests are never lost; 72 submission slots, 16 MiB upload
  arenas, prefetched and chunk-imported big pools, first texture uploads in a submission prologue. `[nvn] mirror_capture`
  (the model of the Switch 1 recomp, AksharDP/xc2-recomp `runtime/vk_memory.c`, MIT): every buffer read goes through a
  device-local copy whose CPU writes are captured at the submit call, so the GPU reads the bytes of that moment. XC2 NS2
  sets it, `snapshot_instances` and `present_ahead` to `auto`: the capture runs when the copies' budget holds whole pools
  (>= 3.5 GiB, 16 GB cards), and only then does the guest lead by one frame (locked 60 on an RTX 5090).
- **runtime/src/kernel/svc_memory.cpp (2026-11-21)** - MapPhysicalMemory kept summing every mapped granule for its limit
  check (~1 ms per call at 2.5 GiB); it keeps a running count. UnmapPhysicalMemory of a whole mesh pool (10241 VirtualFree,
  64.6 of 69.2 ms on the presenting thread) hands the releases to a thread when 4 MiB or more; maps and alias windows wait
  for pending releases first. Smaller unmaps release at once, as before.
- **runtime/src/gfx/nvn: [gfx] render_scale (2026-11-21)** - scaling works for XC2 NS2 (render_scale = 3: the 3D scene at
  3840x2160 into DLSS): a fragment shader's read-only storage image keeps its scale and gets it in the support buffer
  (TexelFetchScale), a draw that writes an unscaled storage image keeps its targets unscaled (the light list and its
  registration mask stay texel for texel: no dot grid), the DLSS/TAAU output follows the scale, optional
  `[gfx] render_scale_min_width`. Nothing changes unless render_scale is set.
