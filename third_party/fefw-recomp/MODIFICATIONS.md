# Modifications to fefw-recomp

This directory is a **modified fork** of `Goldoire/fefw-recomp` (the original GitHub repository now returns **404 / removed**; this vendored copy is preserved and maintained self-contained inside the Xenoblade Chronicles 2 NS2E recomp project). Upstream base and provenance: [UPSTREAM.md](UPSTREAM.md).

| | |
|---|---|
| upstream | `Goldoire/fefw-recomp` *(GitHub repo removed / 404; vendored fork preserved here)* |
| licence | GPL-3.0-or-later (see [LICENSE](LICENSE) and [LEGAL.md](LEGAL.md)) |
| fork base commit in this repository | `6f1d964` (2026-11-12, baseline: 26 rounds of XC2 NS2 recomp) |
| unused upstream directory | `third_party/fefw-recomp/launcher/` is kept strictly as an upstream reference snapshot and is **deprecated/unused** — the active XC2 launcher lives in the repository root `launcher/` |
| files changed since the base | 179 |
| generated from | `git diff --name-only 6f1d964..HEAD -- third_party/fefw-recomp` |

> **Why this file and not per-file headers.** GPL-3.0 section 5(a) requires the work to carry prominent notices
> that it was modified, with a relevant date. This file is that notice, and it is authoritative. Per-file
> `// Modified ...` headers are deliberately not added: the project's internal notes cite line numbers of these files
> (for example `nvn_texture_registry.cpp:305`), and a one-line header shifts every reference. If a file is
> later copied out of this fork on its own, add the header to that copy.

## Changed files (date = last change)

| `file` | date | change |
|---|---|---|
| `runtime/src/gfx/shaders/nav_ribbon.{vert,frag}` | 2026-11-27 | new: GLSL shaders for native Vulkan 3D translucent conveyor-belt navigation ribbon with read-only scene depth occlusion |
| `runtime/src/gfx/extras/nav_ribbon.{h,cpp}` + `nav_ribbon_shaders.inc` | 2026-11-27 | new: `NavRibbonPass` dynamic-rendering Vulkan 3D ribbon pass (per-slot upload VB, depth sampler, SDR + HDR safe) |
| `runtime/include/nxrt/gfx.h` + `runtime/src/gfx/vk_renderer.cpp` | 2026-11-27 | pass optional per-frame read-only 3D scene depth view/layout/flip_y/depth_mode into `FrameInfo` for overlay depth occlusion |
| `runtime/src/gfx/nvn/nvn_replay.cpp` + `nvn_present.cpp` + `nvn_gpu_internal.h` | 2026-11-27 | capture latest 3D G-buffer depth texture (`take_scene_depth`) and hand its sampled depth-aspect view to `Renderer::set_scene_depth()` before overlay draw |
| `runtime/include/nxrt/overlay.h` | 2026-11-28 | add game-agnostic `HudCanvas` 2D foreground drawing facade (including `rect_filled_hgrad`, `convex_poly_filled`, `convex_poly_hgrad`, `polyline`, `image`, `image_rounded`), `Ribbon3DVertex` / `Ribbon3DParams` / `set_ribbon_3d()`, `PickerItem` / `PickerCallbacks` / `set_picker()` / `toggle_picker()`, `create_hud_texture_bc3()`, and `Overlay::set_hud_callback()` |
| `runtime/src/gfx/overlay.cpp` | 2026-11-28 | implement `HudCanvas` primitives (including horizontal-gradient rectangles/convex polygons, polylines, and `image`/`image_rounded`), runtime BC3 HUD texture upload (`create_hud_texture_bc3`), `NavRibbonPass` 3D ribbon draw before ImGui, and searchable `F4` destination picker window with full gamepad navigation (`D-pad/L-stick/A/B/L/R/X`) and XC2 sky-blue styling inside `Overlay::draw()` (SDR + HDR safe) |
| `runtime/include/nxrt/play.h` + `runtime/src/gfx/play/play.cpp` | 2026-11-28 | expose `play::set_hud_callback()`, `play::create_hud_texture_bc3()`, `play::set_picker()`, and `play::toggle_picker()` |
| `runtime/include/nxrt/{overlay.h,play.h}` + `runtime/src/gfx/overlay.cpp` + `runtime/src/gfx/play/play.cpp` | 2026-11-28 | add `create_hud_texture_bc7()`: `Overlay::create_hud_texture_bc3/bc7` now share one `create_hud_texture_bc()` body (format + label are arguments), so the BC7 menu face atlas `/menu/image/mnu_face.wilay` can be uploaded as a HUD texture for the voice-card portraits of the Expansion Pass / Switch 2 Edition blades |
| `runtime/src/gfx/nvn/nvn_prewarm.cpp` | 2026-11-27 | record and prewarm `uint_samplers` (`uint_tex_patch`) pipelines in player-local `pipeline_keys.bin` (`NXPKEYS2`), eliminating repeated draw-time fragment shader compiles for stencil/integer-sampler shaders |
| `runtime/src/gfx/nvn/nvn_gpl.cpp` | 2026-11-27 | widen `frames_flowing()` window from 100 ms to 1500 ms and offload uncached `resolve_stage_modules()` (SPIR-V patching + `vkCreateShaderModule`) onto `gpl_async` background workers so cold scene loads do not stall the render thread |
| `runtime/include/nxrt/gfx_tuning.h` | 2026-11-27 | declare `nxrt::gfx::prewarm_hwdec_async()` for post-device-init background Vulkan Video / D3D11VA prewarm |
| `runtime/src/media/video_ffmpeg.cpp` | 2026-11-27 | implement `prewarm_hwdec_async()` (`std::call_once` detached background `ffdec_prewarm()`) |
| `runtime/src/gfx/vk_device.cpp` | 2026-11-27 | trigger `prewarm_hwdec_async()` after windowed `Device::create()` completes, eliminating the ~100 ms first-cutscene Vulkan Video cold-start hitch without racing driver init (#42) |
| `runtime/src/gfx/nvn/nvn_gpu_texture.cpp` | 2026-11-27 | remove legacy 2026-11-21 `4k target` / `film texture` hot-path mutex+clock+map logging probe from `tex_mark_gpu_written()` |
| `runtime/src/services/hid.cpp` | 2026-11-27 | downgrade periodic normal `hid: pad 0 state` log from `NXLOG_W` to `NXLOG_D` (keep `OUT OF RANGE` at `NXLOG_W`) |
| `runtime/src/gfx/nvn/nvn_gpu_pool.cpp` | 2026-11-27 | remove unused `#include "nvn_capture_spans.h"` and dead `nvn_capture_spans.h` / `test_nvn_capture_spans.cpp` superseded by PR #84 |
| `runtime/src/gfx/nvn/nvn_gpu.cpp` | 2026-11-27 | stop post-`VK_ERROR_DEVICE_LOST` memory explosion (#81, #41): drain `deferred` destroys and free cached upload chunks in `retire()` on `device_lost()`, short-circuit `submit()` / `submit_on_render_thread()` / `serial_done()` on `device_lost()`, drain deferred destroys on `present()` when lost, and only grow a new `VkDescriptorPool` in `Frame::alloc_set()` on `VK_ERROR_OUT_OF_POOL_MEMORY` / `VK_ERROR_FRAGMENTED_POOL` |
| `runtime/src/gfx/nvn/nvn_gpu_pool.cpp` | 2026-11-27 | skip `mirror_capture_take()` on `device_lost()`, destroy returned `CapArena` buffers immediately when lost instead of caching, and add `~MirrorCapture()` so dropped captures never leak arenas (#81) |
| `runtime/src/gfx/nvn/nvn_texprep.cpp` | 2026-11-27 | skip VMA staging buffer creation in background `nvn-texprep` workers once `device_lost()` is latched (#81) |
| `runtime/tests/gfx/test_nvn_texgpu.cpp` | 2026-11-27 | add `device_lost_drains_deferred_and_stops_recording` regression test for post-`DEVICE_LOST` retirement and recording short-circuit (#81, #41) |
| `runtime/include/nxrt/play.h` | 2026-11-27 | add `LivePadConfig::stick_max` (#76) and `LivePadConfig::mouse_camera_capture` (#78); new `play::publish_raw_left_stick()` / `publish_raw_right_stick()` for the F1 calibration readout; new `play::binding_output()` and `play::set_external_mouse_capture()` |
| `runtime/src/gfx/play/play.cpp` | 2026-11-27 | publish `pad.stick_max` (#76) and `mouse.camera_capture` (#78) live alongside centre calibration and `mouse.stick_scale`; external mouse capture coordination |
| `runtime/src/gfx/play/play_types.h` | 2026-11-27 | `MouseSettings::camera_capture` (#78); `PadSettings::ls_center_x/y` and `rs_center_x/y` (stick calibration, issue #76); `MouseSettings::stick_scale` |
| `runtime/src/gfx/play/play_config.cpp` | 2026-11-27 | read and write `[mouse] camera_capture` (#78), `[pad] ls_center_x/ls_center_y` / `rs_center_x/rs_center_y` and `[mouse] stick_scale` (1..200, default 8) |
| `runtime/src/gfx/play/play_ui.cpp` | 2026-11-27 | F1 controls page: left-stick outer range slider (`stick_max`, #76), live mouse camera capture toggle (#78), keep unbound actions visible with `(unbound)` + `+` (#80), Mouse sensitivity slider, bilingual `GameVocabulary::labels`, stick calibration block, in-place key rebinding |
| `runtime/include/nxrt/gfx.h` | 2026-11-27 | expose `RenderGpuInfo` / `render_gpu_info()` so FFmpeg Vulkan Video binds to the selected render GPU (#87) |
| `runtime/src/gfx/vk_device.cpp` | 2026-11-27 | record selected Vulkan GPU index, vendor ID, and name in `render_gpu_info()` (#87) |
| `runtime/src/media/video_ffmpeg.cpp` | 2026-11-27 | pass `render_gpu_info().vulkan_index` to `av_hwdevice_ctx_create(AV_HWDEVICE_TYPE_VULKAN)` so dual-GPU laptops never silently pick GPU 0 (Intel iGPU); skip Intel Vulkan Video (`igvkMedia64.dll` crash, #63/#87) on `auto` in favour of D3D11VA |
| `runtime/include/nxrt/gfx_tuning.h` | 2026-11-27 | add `upscale_lod_bias` for automatic negative mip bias when temporal upscaling is active (#79) |
| `runtime/src/gfx/nvn/api_dlss_replay.cpp` | 2026-11-27 | compute quantized `log2(render_w / output_w) - 1` mip bias (`[upscale] auto_mip_bias`, default on) for DLSS/FSR3/TAAU (#79) |
| `runtime/src/gfx/nvn/nvn_gpu_texture.cpp` | 2026-11-27 | apply `upscale_lod_bias` in `tuned_sampler()` only to material samplers (`in.max_anisotropy > 1` with linear mips) and include it in `SamplerLast` (#79) |
| `runtime/src/services/fs_romfs.cpp` | 2026-11-27 | fix Atmosphère DLC title-ID layout fallback (`0100e9500403*` / `04007ef01f8a*` / sole title-ID under `atmosphere/contents`), fix `.ard/*.rec` Windows backslash normalization + `bf2.arh` trie full-path pre-binding + unmatched `.rec` warning, and add automatic loose `romfs/` -> `bf2.ard` tail override via streaming stored-zlib `xbc1` / raw slices (issue #71) |
| `runtime/include/nxrt/cheats.h` | 2026-11-23 | new (M4c): Atmosphère DMNT cheat VM interface (`parse_txt_filtered`, `load_from_mods_dir`, `init_from_config`, `tick`) |
| `runtime/src/core/cheats.cpp` | 2026-11-23 | new (M4c): DMNT cheat VM (Opcodes `0`–`9`, `VirtualQuery`/`mem::is_mapped` safe bounds, skips M3 `.text` static cheats via `nso.cpp` / `[fs] exefs` info & respects `[mods] pending_rebuild`) |
| `runtime/src/gfx/nvn/nvn_present.cpp` | 2026-11-23 | auto-crop presentation to `(0, 0, base_w, base_h)` for true full-screen native 1080p/1440p output; M4c `nxrt::cheats::tick()` per presented frame |
| `runtime/src/gfx/play/play.cpp` | 2026-11-23 | F11 fullscreen toggle, live `[upscale] render_quality` / `[am] resolution` / `[dlss] preset` / `[nvn] mirror_capture` sync to `nxrt.ini`, external pad bridge, bilingual HUD hint, `UiHost::request_exit()`, and M4c `nxrt::cheats::init_from_config/clear` hook |
| `runtime/src/loader/nso.cpp` | 2026-11-23 | record mapped `main` NSO `build_id` and `.text` size (`nxrt::cheats::register_main_nso`) for M4c `--exefs` launches |
| `runtime/tests/test_cheats.cpp` | 2026-11-23 | new (M4c): unit tests for DMNT opcodes, pointer chains, keypress conditionals, bounds safety, and `.cheats_enabled.tsv` filtering |
| `runtime/tests/gfx/test_play_config.cpp` | 2026-11-23 | add `render_quality_tuning_contract` regression test covering `[upscale] render_quality` ratios and `game_handles_render_quality` |
| `runtime/tests/test_svc_fs_romfs.cpp` | 2026-11-23 | add `romfs_mods_m2_and_m2b_archive_override`, `romfs_mods_m2b_multi_mod_tails_and_arh_compose`, and `romfs_mods_m4_option_layers_and_pending_rebuild` regression tests |
| `UPSTREAM.md` / `README.md` / `MODIFICATIONS.md` | 2026-11-23 | note removed upstream GitHub repo (`Goldoire/fefw-recomp`, 404) and mark `third_party/fefw-recomp/launcher/` as unused upstream snapshot |
| `runtime/include/nxrt/render_quality.h` | 2026-11-23 | new: `[upscale] render_quality` modes (`native`, `quality`, `balanced`, `performance`, `ultra_performance`, `console`) and their implied scale ratios (shared by runtime and F1 menu) |
| `runtime/include/nxrt/symbols.h` | 2026-11-22 | guest names: symbols_load_guest_names[_text], symbols_guest_count, symbols_describe_guest (the map's f_<guest VA> entries named from the lift's dynsym) |
| `runtime/src/diag/symbols.cpp` | 2026-11-22 | guest names: the map's f_<guest VA> entries take the lift's name; symbols_describe_guest matches the exact function start, not the nearest preceding export |
| `runtime/src/diag/watchdog.cpp` | 2026-11-22 | thread dump prints guest frames through symbols_describe_guest instead of a literal f_%llx |
| `runtime/src/diag/crash.cpp` | 2026-11-22 | the crash report's guest call stack uses the same names pkg7 assert: the data is newer than the exefs (HANDOVER 79) |
| `runtime/src/core/runtime.cpp` | 2026-11-22 | the start-up symbol line reports how many guest names were loaded |
| `runtime/tests/test_symbols.cpp` | 2026-11-22 | test: the sidecar renames map entries, and only an exact start is named |
| `runtime/src/diag/stackwalk.cpp` | 2026-11-24 | check stack readability via `VirtualQuery` and wrap `capture_stack()` in SEH (`__try / __except`) so unwinding a suspended thread inside a Wine/Proton syscall transition never crashes the host process (issue #35) |
| `runtime/src/media/video_nvdec.cpp` | 2026-11-24 | copy `cuvidMapVideoFrame` NV12 planes from `CUdeviceptr` device memory to host memory via `cuMemcpy2D` (`cuCtxPushCurrent` / `cuCtxPopCurrent`) instead of host `std::memcpy` (PR #48) |
| `runtime/src/media/video_codec.cpp` | 2026-11-24 | defer `ffdec_prewarm()` to `worker_loop` when a film decoder is created so `av_hwdevice_ctx_create(VULKAN)` at `t=0.001s` never races `gfx::Device::create()` in `amdvlk64.dll` (issue #42); add `media_decoder_has_backlog()` for CPU-decode clock re-anchoring (issue #35) |
| `runtime/src/media/movie_adec.cpp` | 2026-11-22 | a zero-byte packet is the end of the placeholder audio track (it used to be fed as a full buffer); the last frame gets the eos byte the SDK's AudioFrame::SetEndOfStreamState writes |
| `runtime/src/media/movie_trace.cpp` | 2026-11-24 | re-anchor `RenderThread` movie clock (`player[+644]=1` at `lr 0x7100722bfc`) when frames are dropped with no decoded backlog in `d->ready` (or after 6 consecutive drops), preventing permanent 100%-frame-drop movie freezes on CPU `dav1d` without pause/unpause (issue #35); pace `stall_watch` to 1 dump/s only while a film is active |
| `runtime/docs/CONFIG.md` | 2026-11-23 | generated configuration key table (`output_scale`, `output_min_width`, `acquire_wait`, `render_quality`, `dlss.preset`, `mirror_capture` 1536 MiB, `view_reinterpret`) |
| `runtime/THIRD_PARTY_NOTICES.md` | 2026-11-23 | clarify unbundled AMD FidelityFX (v2.4.0 binary EULA & v1.1.4 MIT) and NVIDIA Streamline/DLSS binaries plus optional user-confirmed direct launcher downloads |
| `runtime/include/nxrt/write_watch.h` | 2026-11-21 | write-watch: per-64 KiB-reservation queries (capture_poll) for submit-time captures; VAMM pages tracked by GetWriteWatch |
| `runtime/src/mem/write_watch.cpp` | 2026-11-21 | write-watch: VAMM (MapPhysicalMemory) pages tracked through their write-watch record; capture_poll for submit-time captures |
| `runtime/tests/test_svc_memory.cpp` | 2026-11-21 | test: a large UnmapPhysicalMemory followed by a map of the same range succeeds |
| `runtime/src/gfx/nvn/api_device.cpp` | 2026-11-23 | [nvn] sync_scope switch: window-acquire syncs can wait for all GPU work (NVIDIA fallback) handle `nvnWindowGetCrop` out-of-order `(int*, int*, int*, int*)` call and sanitize `R11G11B10F` clear colors; `[nvn] acquire_wait` (`auto` skips slot wait while `mirror_capture` is on) |
| `runtime/src/gfx/nvn/api_memory.cpp` | 2026-11-21 | [nvn] pool_prefetch_init: every GPU-reachable pool is imported in the background when it is made, not at its first draw |
| `runtime/src/gfx/nvn/nvn_mirror_cap.h` | 2026-11-21 | mirror cap 2048 -> 8192 MiB (still at most a quarter of the video memory); kCaptureMinBudget for mirror_capture = auto |
| `runtime/src/gfx/nvn/nvn_texprep.cpp` | 2026-11-21 | ready_textures(): finished background preparations, uploaded at the start of a submission ([nvn] tex_preupload) |
| `runtime/src/gfx/nvn/nvn_texprep.h` | 2026-11-21 | ready_textures(): finished background preparations, uploaded at the start of a submission ([nvn] tex_preupload) |
| `runtime/src/gfx/nvn/api_dlss.cpp` | 2026-11-19 | `XC2_DLSS_MIN_SCALE`: an env-gated floor for the render range `dlss_optimal` reports, so the dynamic-resolution experiment can ask what the game does when its DLSS input must be larger (HANDOVER 217). Off by default; the code is otherwise unchanged |
| `UPSTREAM.md` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/extras.h` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/include/nxrt/gfx.h` | 2026-11-24 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; add `DeviceFeatures::depth_bias_control_force_unorm` (`VK_EXT_depth_bias_control`, issue #44) |
| `runtime/include/nxrt/memory.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/platform.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/services_api.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/svc.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/scripts/gen_earlyz_mask_shaders.py` | 2026-11-13 | nvn: the light list registration mask is derived from the list after every list draw - the black rectangle (HA |
| `runtime/src/core/switch2.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/extras/temporal.cpp` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/src/gfx/extras/temporal_dlss.cpp` | 2026-11-23 | `[dlss] preset` (`default`, `J`, `K`, `L`/`M` for Performance / Ultra Performance only), live switch (`set_dlss_preset_live`) |
| `runtime/src/gfx/extras/temporal_ffx.cpp` | 2026-11-21 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s live FSR version override & context rebuild; auto-enable `fsr_follow_size` for FSR 4.x on amdvlk |
| `runtime/src/gfx/nvn/api_dlss_replay.cpp` | 2026-11-21 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s; the DLSS/TAAU output follows [gfx] render_scale live 3D upscaler/FSR/sharpness hot-switch (`set_dlss_live`, `dlss_live_info`) for F1 menu |
| `runtime/src/gfx/extras/temporal.h` | 2026-11-23 | live FSR version hot-switch (`set_ffx_version_live`, `temporal_ffx_active_version`); `set_dlss_preset_live` / `dlss_preset_current` (`[dlss] preset`) |
| `runtime/src/gfx/nvn/api_program.cpp` | 2026-11-14 | nvn: [nvn] shader_dump writes the code and control of every unresolved block (HANDOVER 74) |
| `runtime/src/gfx/nvn/api_texture.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/earlyz_mask_shaders.inc` | 2026-11-13 | nvn: the light list registration mask is derived from the list after every list draw - the black rectangle (HA |
| `runtime/src/gfx/nvn/nvn_core.h` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/nvn_enums.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/nvn/nvn_gpu.cpp` | 2026-11-22 | nvn: [nvn] split_draws hands the recorded commands to the GPU at pass boundaries - the GPU stops waiting for t; never lose a flush request; 72 submission slots; [nvn] mirror_capture at submit; [nvn] present_ahead = auto atomic live `keep_aspect` toggle (`set_keep_aspect`) always-on lightweight Vulkan timestamp query pool for `gpu_busy_ns` and live viewport info reporting |
| `runtime/src/gfx/nvn/nvn_gpu_internal.h` | 2026-11-26 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override; mirror capture and prologue interfaces (Frame::prologue, mirror_capture_*); add `kReadVertexInst` (`ReadUse`) and `gfx_no_wb` / `is_vertex` parameters to `mirror_capture_bind` |
| `runtime/src/gfx/nvn/nvn_gpu_pool.cpp` | 2026-11-26 | nvn: pool import/mirror granule is a config (default 64 MiB unchanged); 16 MiB does not help; movement overrun; device-local mirrors for VAMM and SHADER_CODE pools; CPU snapshots; chunked pool imports; [nvn] mirror_capture (= auto); [nvn] mirror_gfx_subpage: write back unaligned `< 4 KiB` head/tail boundary sub-pages of graphics-written SSBOs overlapping live per-vertex vertex buffers (`kReadVertex`, `divisor == 0`) with no actively CPU-written interior full pages (`cpu_wr_at`) via `cap_note_write` |
| `runtime/include/nxrt/gfx_tuning.h` | 2026-11-23 | add `game_handles_render_quality` flag so game modules that handle `[upscale] render_quality` natively keep NVN texture scaling at `1.0x` |
| `runtime/src/gfx/nvn/nvn_gpu_texture.cpp` | 2026-11-23 | `[gfx] render_scale_min_width`; opt-in `[nvn] view_reinterpret` (issue #15); `[gfx] output_scale` / `output_min_width`, `[upscale] render_quality` derived scales at start-up (skipped when `game_handles_render_quality` is set, fixing issue #39), and clamping tall >1080p output-side targets to output height |
| `runtime/src/gfx/nvn/nvn_pipeline.cpp` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override bump pipeline cache salt for SPIR-V depth-compare sampler auto-upgrading and `R11G11B10F` NaN scrub |
| `runtime/src/gfx/nvn/nvn_pipeline.h` | 2026-11-14 | nvn: push the uniform buffers as push descriptors ([nvn] push_uniforms, ubo_only_rebind) |
| `runtime/src/gfx/nvn/nvn_replay.cpp` | 2026-11-26 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s; snapshots of instance streams; first uploads in a submission prologue; storage images under [gfx] render_scale; occlusion query results go into the mirror capture copy (the cloud sea and fog were not drawn); under mirror_capture a query report is assembled in the copy and reaches the pool in one piece (flames, smoke and reflections flickered with present_ahead); [nvn] mirror_gfx_storage_writeback (graphics-stage storage writes stay in the copy; XC2's waterfall mist blinked) subdivide 1-quad 3D Cloud Sea sheets (`quad_split` 1..16 with varying Z); pool-offset aliasing barrier; sampler compare auto-upgrade; rewrite float samplers on uint color textures (#15); use `vkCmdSetDepthBias2EXT` with `VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT` when supported (#44); classify instanced VB streams (`divisor > 0`) as `kReadVertexInst` so `mirror_gfx_subpage` excludes CPU+GPU particle billboard streams |
| `runtime/src/gfx/nvn/nvn_shaders.cpp` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override; reflection marks storage images a shader writes (OpImageWrite, image atomics) |
| `runtime/src/gfx/nvn/nvn_spirv_patch.cpp` | 2026-11-24 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override sanitize NaN/Inf on fragment shader outputs targeting `R11G11B10F` color attachments; fix `spirv_uint_samplers` type/variable declaration ordering so patched `OpVariable` and new uint image/pointer types never precede existing `OpTypeInt`/`OpTypeImage`/`OpTypeSampledImage`/`OpTypePointer` dependencies |
| `runtime/src/gfx/nvn/nvn_spirv_patch.h` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override declare `patch_sanitize_f16_outputs` |
| `runtime/src/gfx/platform.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/include/nxrt/nvn_backend.h` | 2026-11-22 | expose `set_keep_aspect`, `set_dlss_live`, and `dlss_live_info` expose `set_game_viewport_info()` / `game_viewport_info()` and `BackendStats::gpu_busy_ns` |
| `runtime/include/nxrt/play.h` | 2026-11-21 | expose `overlay_blocks_game()`, `live_pad_config()`, and `note_external_pad()` |
| `runtime/src/gfx/play/play_types.h` | 2026-11-23 | default FPS counter hotkey F10, persistent `menu_lang`, `DisplaySettings::width/height = 0` (`auto`), `Game3dSettings` (`render_quality`, `dlss`, `dlss_preset`, `out_height`, `capture`), `render_output_height()` |
| `runtime/src/gfx/play/play_config.cpp` | 2026-11-21 | parse and persist `[fps] menu_lang` in `controls.ini` |
| `runtime/src/gfx/play/play_ui.h` | 2026-11-20 | add draw_settings_menu() wrapper replacing Diagnostics overlay |
| `runtime/src/gfx/overlay.cpp` | 2026-11-23 | XC2 Aegis theme + CJK font, `ImGuiCol_ModalWindowDimBg`, 4-level FPS counter with Detailed Performance HUD (AMD ADL2 / NVIDIA NVML / Vulkan VRAM / CPU / RAM / 120-frame graph / pipeline timings / resolution & scaling) |
| `runtime/src/gfx/play/play_ui.cpp` | 2026-11-23 | bilingual two-pane F1 menu with Display output resolution + Auto windowed size (up to 1440p), Graphics upscaler (DLSS/FSR/TAAU/Off), per-upscaler render quality mode, DLSS model (`default`/`K`/`J`), Experimental `[nvn] mirror_capture`, D-pad focus & top-right confirmed Quit Game (`host.request_exit()`), 4-level performance HUD selector |
| `runtime/src/gfx/vk_renderer.cpp` | 2026-11-21 | avoid `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` during swapchain rebuild to prevent Win32 fullscreen deadlock |
| `runtime/src/services/misc.cpp` | 2026-11-20 | auto-discover DLC folders in [aoc] dir when indices.txt is absent 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/gfx/vk_device.cpp` | 2026-11-24 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s; start-up failures name the VkResult and the driver's own Vulkan version (a 1.0 probe instance) instead of a bare `-9`; query and enable `VK_EXT_depth_bias_control` (`depthBiasControl` + `leastRepresentableValueForceUnormRepresentation`, issue #44) |
| `runtime/src/ipc/ipc.cpp` | 2026-11-14 | dlc: the title never asks aoc, so the Torna entry is save driven (HANDOVER 66) |
| `runtime/src/kernel/handles.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/kernel/svc_info.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/kernel/svc_memory.cpp` | 2026-11-21 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override; MapPhysicalMemory keeps a running page count (no walk of the whole map per call); large unmaps release on a background thread |
| `runtime/src/kernel/svc_thread.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/loader/module.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/apps/video_codec_probe.cpp` | 2026-11-18 | probe: AV1 on the SDK 3.0.1 nn::codec entry points (video_codec_probe <movie.webm> <out.png> <frame>) |
| `runtime/src/media/movie_vdec.cpp` | 2026-11-19 | movie: [media] video_decoder = library leaves movie::VideoDecoder to the library (it drives nn::codec AV1) and keeps the host AudioDecoder; same decoder-thread default as video_codec.cpp |
| `runtime/src/mem/memory.cpp` | 2026-11-14 | heap: lazy commit is implemented and measured, but it stays off (crashes at 0.82 s) |
| `runtime/src/services/audren/Server/AudioRenderSystem.h` | 2026-11-12 | nvn: trace the depth/blend state a draw is submitted with, summarize the frame; audren: log every voice the ga |
| `runtime/src/services/common.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `runtime/src/services/fs.cpp` | 2026-11-16 | replay: pipe/fb hot-path audit (round 45, 9.7-15.4 -> 5.3-6.8 ms); fsr_follow_size; fs trace_stack_file off; s |
| `runtime/src/services/hid.cpp` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/decoder/SassDecoder.Tensor.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/decoder/SassDecoder.Tex.cs` | 2026-11-14 | shaders: SUATOM (0x593/0x595) and the TLD4 PTP modifier - 1529 of 1529 blocks the game asks for translate (HAN |
| `tools/shaders/ampere/decoder/SassDecoder.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/frontend/AmpereEmit.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/ampere/frontend/AmpereTranslator.cs` | 2026-11-14 | shaders: take the NVN2 stage from the control section - 518 to 2813 of 4222 blocks translate (HANDOVER 73) |
| `tools/shaders/harness/Program.cs` | 2026-11-14 | shaders: take the NVN2 stage from the control section - 518 to 2813 of 4222 blocks translate (HANDOVER 73) |
| `tools/shaders/harness/StaticGpuAccessor.cs` | 2026-11-16 | 3.0.1 bring-up through 2026-11-16: rim audit, cloud shimmer, upscaler chain, FSR version override |
| `tools/shaders/harness/SpirvHarness.csproj` | 2026-11-24 | set `<CETCompat>false</CETCompat>` so older Windows 10 builds without full CET shadow-stack support do not abort at apphost startup with exit `2148734214` (issue #45) |
| `tools/shaders/harness/translator/Ryujinx.Graphics.Shader.csproj` | 2026-11-19 | shaders: take the NVN2 stage from the control section - 518 to 2813 of 4222 blocks translate (HANDOVER 73); `spirv-call-args.patch` back in the series - without it 563 of 12,132 blocks throw in `GenerateCall` and only translate through the DebugMode retry (handover 56 section 179) |
| `tools/shaders/harness/translator/spirv-call-args.patch` | 2026-11-24 | shaders (`AstOptimizer`): add `CanPropagate` recursive guard stopping single-use mutable memory loads (`StorageBuffer`/`GlobalMemory`/`SharedMemory`/`LocalMemory`/`Output`/`ImageLoad`) from folding past later stores/barriers (fixes XC2 morph-target prev-pos clobber, issue #15), and add `ImageAtomic` to `HasSideEffects` |

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

- **runtime/src/media/video_codec.cpp, movie_adec.cpp, movie_trace.cpp (2026-11-21)** - the film's picture stopped after a few
  frames. Three faults, each one the opposite of what the reference library does (`dm/host/hle_codec_video.c` in the
  nxnative profile, and the upstream audit of the reference emulator): `Acquire` handed out a buffer with nothing in it and a
  fence nothing would complete, so the client waited ~380 ms for the first picture and gave up (it now decodes one packet
  inline, as `hle_codec_video.c`'s comment requires); a zero-byte end-of-stream packet was turned into a picture of its own,
  telling the client the film ended at frame 62 with hundreds of pictures still queued (the marker now goes on the last real
  picture, and only an empty buffer is handed out once the queue is empty); and `Submit` answered 3603 past eight queued
  packets while the client ignores that result and moves on, so the whole stream was lost after 71 packets (the queue now
  keeps every submitted packet, bounded at 8192 for memory, and eight is only the watermark of the submission-ready event).
  The rest is diagnostic: a per-second film-audio heartbeat, a stall dump that names the fence the client is armed on and the
  decoder's queues, a counter of GPU writes to the film's picture texture, and a texture registry that reports rewrites of it.

- **runtime/src/media/video_codec.cpp, movie_adec.cpp (2026-11-21, film-model-10..12)** - the opening cutscene stalled 3-5 s and
  then jumped into its middle. Three causes, all on the same axis - the client is starved while the game loads: the decoder
  worker ran at below-normal priority from its first frame, so the first picture arrived 1089 ms after the client asked (it now
  runs at normal priority until one picture exists, then drops back); the film's audio track, which is only the player's clock,
  kept being handed out while the picture stood still, so the player skipped the picture forward to catch up with its own clock
  (delivery now stops at the last picture's time plus half a second); and the worker held only four decoded pictures, so a
  client returning from the loading screen could read one picture per ~40 ms decode and could never catch up (it now holds 24 for
  the first ten seconds of a film, then drops back to four). Measured after: 23 pictures deep during the window, 3 after it,
  no stall, one read 60 ms after its acquire.
- **runtime/src/gfx/nvn/nvn_replay.cpp: [nvn] mirror_gfx_storage_writeback (2026-11-21)** - default on (unchanged); off keeps
  what a graphics stage writes into storage in the mirror_capture copy: no write-back to the pool, and the CPU's pages there
  are no longer left out of the next capture. XC2 NS2 (xc2_boot.cpp sets it off) updates its effects' particles from vertex
  shaders over ~300 KiB ranges the game also writes on the CPU; with present_ahead the whole-range write-back and the
  in-flight exclusion dropped the game's next-frame words, and the waterfall mist blinked at render_scale 3. Write-back
  ~65 -> ~1 MB per second.
- **runtime/src/mem/memory.cpp, include/nxrt/memory.h, src/kernel/svc_info.cpp, src/kernel/svc_memory.cpp, src/mem/write_watch.cpp, src/gfx/nvn/nvn_gpu_pool.cpp (2026-11-23)** - `mem::alias_base()` probes a host-free 128 GiB aligned window when `kAliasBase` (`0x2000000000`) is already occupied on the host (`VammManager` `0xD401`, issue #33), and `MapPhysicalMemory` logs the blocking host `MEMORY_BASIC_INFORMATION`.
- **runtime/src/services/misc.cpp (2026-11-23)** - `aoc_indices()` uses UTF-8 (`u8string()` and `std::filesystem::path`-based `std::ifstream` / `is_directory`) instead of `path::string()`, avoiding Windows ANSI code-page conversion exceptions on non-ASCII DLC folders (issue #25).
- **PR #30 (2026-11-23): runtime/src/gfx/{vk_device.cpp,play/play.cpp,nvn/{nvn_gpl.cpp,nvn_gpu.cpp,nvn_gpu_internal.h,nvn_gpu_pool.cpp,nvn_gpu_texture.cpp,nvn_pipeline.cpp,nvn_replay.cpp,nvn_spirv_patch.cpp,nvn_texprep.cpp}}, include/nxrt/gfx.h, docs/CONFIG.md** - fix `nvn_pipeline.cpp` `with_early` UAF; add `VK_AMD_shader_early_and_late_fragment_tests` (`early_fragment_tests = auto`) and `OpImageTexelPointer` matching; skip `earlyz_mask_rebuild` when hardware-patched and narrow the fallback barrier; add `[nvn] fb_reuse` and `VK_EXT_attachment_feedback_loop_layout` zero-copy colour feedback (with dynamic rendering pass split & inter-pass narrow barrier preserving feedback layout); cool GPU-written hot mirror pages when `!mirror_capture_on()`; lower `nvn-texprep` thread priority; auto-hide idle cursor in fullscreen/borderless modes.
- **P2 (#15 / #18 / 0.2.1, 2026-11-23):**
  - `runtime/src/gfx/nvn/nvn_gpu_texture.cpp`, `runtime/docs/CONFIG.md`: add opt-in `[nvn] view_reinterpret` (default `0`) for Vulkan-legal same-width `UNORM` view reinterpretation of 8/16-bit integer colour textures (issue #15 A/B).
  - `runtime/src/gfx/nvn/nvn_mirror_cap.h`, `runtime/src/gfx/nvn/nvn_gpu_pool.cpp`, `runtime/src/gfx/nvn/nvn_gpu_internal.h`, `runtime/src/gfx/nvn/nvn_gpu.cpp`, `runtime/include/nxrt/nvn_backend.h`: lower `kCaptureMinBudget` (`3584 -> 1536 MiB`) so 8 GB cards use `mirror_capture = auto`; add `vram_live_info()` and 60 s `vram:` periodic log line.
  - `runtime/include/nxrt/gfx.h`, `runtime/src/gfx/vk_device.cpp`, `runtime/src/gfx/overlay.cpp`: enable `VK_EXT_memory_budget` when present and gate `VkPhysicalDeviceMemoryBudgetPropertiesEXT` queries cleanly.
  - `runtime/src/gfx/play/{play_ui.h,play_ui.cpp,play.cpp,play_types.h,play_config.cpp}`: focus first option card on F1 open/tab switch and lock Left/Right D-pad ownership on focused cards; top-right Quit Game button with confirmation popup wired via `UiHost::request_exit()` (issue #18); show live VRAM/pool copies in F1 status card; default windowed size to `0` (`auto`, display resolution up to 1440p).
- **PR #36 integration & Option B official upscaler downloaders (2026-11-23):**
  - `runtime/src/gfx/nvn/api_device.cpp`: add `[nvn] acquire_wait` (`auto`: skip slot wait while `mirror_capture` is on).
  - `runtime/src/gfx/extras/{temporal.h,temporal_dlss.cpp}`: add `[dlss] preset` (`default`, `J`, `K`, plus `L`/`M` for Performance/Ultra Performance) with live switch.
  - `runtime/include/nxrt/render_quality.h`, `runtime/src/gfx/nvn/nvn_gpu_texture.cpp`, `runtime/src/gfx/play/{play_types.h,play.cpp,play_ui.cpp}`, `runtime/src/gfx/overlay.cpp`: add `[gfx] output_scale` / `output_min_width`, intent-based `[upscale] render_quality`, DLSS in the F1 upscaler list & model selector, Display output resolution selector, Experimental `mirror_capture` toggle, and `ImGuiCol_ModalWindowDimBg` theme dim.
  - `runtime/THIRD_PARTY_NOTICES.md`: update AMD FidelityFX (v2.4.0 binary EULA & v1.1.4 MIT) and NVIDIA Streamline/DLSS notices to reflect unbundled distribution plus optional user-confirmed direct launcher downloads from official vendor releases.
- **2026-11-24 (#45, #44, PR #48, #42, #35):**
  - `tools/shaders/harness/SpirvHarness.csproj`: set `<CETCompat>false</CETCompat>` for older Windows 10 builds without full CET shadow-stack support (#45).
  - `runtime/include/nxrt/gfx.h`, `runtime/src/gfx/vk_device.cpp`, `runtime/src/gfx/nvn/nvn_replay.cpp`: enable `VK_EXT_depth_bias_control` (`leastRepresentableValueForceUnormRepresentation`) and use `vkCmdSetDepthBias2EXT` with `VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT` on `D32_SFLOAT_S8_UINT` shadow maps to eliminate RTX 40-series 3D shadow stripes (#44, #39).
  - `runtime/src/media/video_nvdec.cpp`: copy `cuvidMapVideoFrame` NV12 planes from `CUdeviceptr` device memory to host memory via `cuMemcpy2D` (`cuCtxPushCurrent` / `cuCtxPopCurrent`, PR #48).
  - `runtime/src/media/video_codec.cpp`: defer `ffdec_prewarm()` from `install_video_codec_hle()` (`t=0.001s`) to `worker_loop` when a film decoder is created so `av_hwdevice_ctx_create(AV_HWDEVICE_TYPE_VULKAN)` never races `gfx::Device::create()` inside `amdvlk64.dll` (#42); add `media_decoder_has_backlog()`.
  - `runtime/src/media/movie_trace.cpp`, `runtime/src/diag/stackwalk.cpp`: re-anchor `RenderThread`'s movie clock (`player[+644]=1` at `lr 0x7100722bfc`) when frames are dropped with no decoded backlog in `d->ready` (or after 6 consecutive drops) so CPU `dav1d` films never enter the 100%-frame-drop freeze spiral (#35); guard `capture_stack()` with `VirtualQuery` stack readability checks and SEH (`__try / __except`) so `diag_thread_dump()` never faults in Wine/Proton `ntdll!RtlVirtualUnwind` (#35).
  - `runtime/src/gfx/nvn/nvn_spirv_patch.cpp`: in `spirv_uint_samplers`, anchor newly created `OpTypeImage` / `OpTypeSampledImage` / `OpTypePointer` after `std::max(iat, u32_at)` and any reused existing type declarations, and relocate patched `UniformConstant` `OpVariable` instructions when their original offset precedes `ptr_def_at` (fixes `spirv-val` forward-reference error and `amdvlk64.dll` pipeline compilation crash on `d2767845f04b8ef2`).
- **2026-11-24 (post-v0.2.3: #51/#54, #50, #25, #57, #52, #55/#56):**
  - `runtime/src/gfx/vk_device.cpp`, `runtime/src/gfx/nvn/nvn_replay.cpp`, `runtime/docs/CONFIG.md`: gate `VK_EXT_depth_bias_control` (`FORCE_UNORM` via `vkCmdSetDepthBias2EXT`) behind opt-in `[nvn] depth_bias_force_unorm` (`false` by default) to resolve RTX 40/50 shadow-acne band regressions (#51, #54).
  - `runtime/src/gfx/play/play_ui.cpp`, `runtime/tests/gfx/test_play_ui.cpp`, `runtime/src/gfx/nvn/api_dlss_replay.cpp`: isolate `##rows_pane` nav child window with `ImGuiWindowFlags_AlwaysVerticalScrollbar`, `ImGuiButtonFlags_EnableNav`, `NavMoveRequestCancel()` on value steps, `ImGuiItemFlags_NoNav` on the live pad monitor, ZR Apply & Save, and `gamepad_dpad_nav_stays_in_rows_pane` (#50); log `upscaler_name(active)` instead of stale `u` in `dlss_replay`.
  - `runtime/src/services/misc.cpp`, `runtime/tests/test_svc_misc.cpp`: add `aoc_path_from_str()` and inspect wide `filename().native()` in `aoc_indices()` so non-ASCII or unpaired-surrogate DLC directory names do not throw `std::filesystem::system_error` (#25).
  - `runtime/scripts/gen_extras_shaders.py`, `runtime/src/gfx/extras/{extras_shaders.inc,output_pass.h,output_pass.cpp}`, `runtime/src/gfx/nvn/nvn_present.cpp`, `runtime/src/gfx/play/play_ui.cpp`, `runtime/tests/gfx/test_extras_output.cpp`: add display-referred HDR highlight expansion (`OutputParams::expand_ratio` via `pc.src.w`), enable `live_preview` on `hdr_boost`, and add `hdr10_highlight_expansion_lifts_display_white` (#57).
  - `runtime/src/media/{nv12_convert.h,movie_vdec.cpp,video_codec.cpp,movie_adec.cpp}`, `runtime/tests/test_media_vcodec.cpp`: enable `dav1d` `max_frame_delay = 0` with EOS flush/drain, shared SSE2 `i420_to_nv12()`, `dec_mu`-serialized `decode_one()`, worker `join()` before codec teardown in `finalize_decoder()`, waiter wakeups in `adec_finalize()`, and media unit tests (#52).
  - `runtime/src/gfx/nvn/{nvn_replay.cpp,nvn_gpu_texture.cpp,nvn_gpu_internal.h,nvn_gpu.cpp}`, `runtime/include/nxrt/nvn_backend.h`, `runtime/src/gfx/overlay.cpp`, `runtime/tests/gfx/test_nvn_texgpu.cpp`: add `try_bypass_upscaled_output_compute()` for XC2 compute upscale variants `0794aaf86ae7fd58`/`6139946183e76e3d`/`9d2efe26d94e3ac0`, extract `compute_attachment_scale()` with `quality_set` & `1536x864`/`960x540` caps, add `scaled_game_viewport_info()` for the HUD overlay while keeping `game_viewport_info()` for present crop, and add `attachment_scale_xc2_targets` / `scale_viewport_info_for_overlay` (#55, #56).
  - `runtime/include/nxrt/game.h`, `runtime/games/xc2ns2/xc2ns2_data.cpp`, `runtime/src/gfx/nvn/{nvn_framegen.h,nvn_framegen.cpp}`, `runtime/src/gfx/extras/framegen_ffx.cpp`, `runtime/docs/CONFIG.md`, `runtime/tests/gfx/test_nvn_texgpu.cpp`: add `game::Info::frame_gen_hudless_default` (enabled for `xc2ns2`) so `[dlss] frame_gen_hudless` passes `HUDLessColor` cleanly out of the box, match `nvn_present`'s sRGB vs UNORM view choice via `hudless_copy_format()` for HDR10/scRGB + SDR consistency, decode `VK_FORMAT_A2B10G10R10_UNORM_PACK32` in `frame_gen_dump_every`, and add `hudless_copy_format_matches_hdr_and_sdr` (#53).
  - `runtime/src/gfx/nvn/{nvn_pipeline.h,nvn_pipeline.cpp,nvn_gpl.cpp,nvn_replay.cpp}`, `runtime/docs/CONFIG.md`, `runtime/tests/gfx/test_nvn_prewarm.cpp`: guard `VkPipelineCache` compile vs `vkGetPipelineCacheData` / destruction with `std::shared_mutex` (`create_graphics_pipeline_tracked` & `create_compute_pipeline_tracked`) and `g_save_mu`, separate runtime GPL parts & linked pipelines into a non-persisted `pipeline_library_cache_handle()`, add `[nvn] pipeline_cache_save` (`auto`: off by default on Intel `0x8086`, keeping in-memory cache and driver disk cache) to prevent `igvk64.dll+0x79ddf3` `vkGetPipelineCacheData` crashes, and add `pipeline_cache_save_policy_and_opt_out` (#49).
  - `runtime/src/gfx/extras/temporal_ffx.cpp`: default `[upscale] fsr_version` to `"3.1"` (matching `play.cpp` and avoiding accidental FSR 4.1.2 selection on RDNA2/amdvlk out of the box), and auto-enable per-size `slots_` isolation whenever the active provider resolves to `FSR 4.*` to prevent `amdvlk64.dll` dynamic-resolution resize crashes.
  - `runtime/src/gfx/extras/{framegen.h,framegen.cpp}`, `runtime/src/gfx/overlay.cpp`: reset the FSR 3 present baseline safely across swapchain recreations (fixing the `shown fps` counter wrap overflow), publish lock-free `framegen_telemetry()`, and show dual `game -> shown FPS (FG)` plus a `Frame gen` status row in the F10 overlay/HUD whenever frame generation is active.
- **2026-11-24 (HDR UI split):**
  - `runtime/src/gfx/nvn/{nvn_hudless.h,nvn_hudless.cpp,nvn_framegen.h,nvn_framegen.cpp,nvn_replay.cpp,nvn_gpu_internal.h,nvn_gpu_texture.cpp}`, `runtime/tests/gfx/test_nvn_texgpu.cpp`: move the scene-before-UI copy out of frame generation into `nvn_hudless` (double-buffered, shared by frame generation per DLSS evaluation and by HDR every frame), add `tex_copy_source_ok()`, and add `hudless_copy_rules`.
  - `runtime/src/gfx/nvn/{nvn_hdr_ui.h,nvn_hdr_ui.cpp,nvn_present.cpp}`, `runtime/src/gfx/extras/{hdr_math.h,hdr_math.cpp,output_pass.h,output_pass.cpp,extras_shaders.inc,extras_state.cpp}`, `runtime/scripts/gen_extras_shaders.py`, `runtime/include/nxrt/extras.h`, `runtime/docs/CONFIG.md`, `runtime/tests/gfx/{test_extras_output.cpp,test_hdr_math.cpp}`: keep the game's UI out of the HDR highlight expansion (`[hdr] ui_split`, default on) by comparing the presented window texture with its copy before the UI (`u_scene` / `u_final`, push constants 80 to 112 bytes), add `[hdr] ui_white_nits` and `[hdr] deband` (one-step triangular noise on the sRGB code before the HDR encode), and add the UI split, UI brightness and deband tests.
  - `runtime/src/gfx/extras/{hdr_pattern.h,hdr_pattern.cpp}`, `runtime/src/gfx/play/{play_ui.cpp,play_hints.h,play_hints.cpp,play.cpp,play_config.cpp,play_types.h}`, `runtime/tests/gfx/{test_extras_output.cpp,test_play_hints.cpp,test_play_config.cpp}`: add HDR calibration test patterns (peak, staircase, black level) to the F1 Graphics page, add value tags and "what to look for" guidance to the picture and HDR settings, save and load `ui_white_nits` / `deband` in controls.ini, and add `hdr10_test_patterns`, `sdr_ignores_test_patterns`, `hdr_pattern_lasts_while_asked` and the `play_hints` tests.
- **2026-11-25 (runtime hot-path perf, dual-clock 90/120 fps, boot shader prepare, prepared GPL sharing):**
  - `runtime/src/core/{dispatch.cpp,hooks.cpp}`, `runtime/src/kernel/{host_place.cpp,sched.cpp}`, `runtime/src/mem/write_watch.cpp`, `runtime/src/gfx/extras/temporal.cpp`: optimize dispatcher lookup, hook slot dispatch, fiber scheduling, write-watch capture polling, and temporal resize checks.
  - `runtime/src/gfx/nvn/{api_cmd_state.cpp,api_cmd_work.cpp,api_cmdbuf.cpp,api_device.cpp,nvn_abi.h,nvn_cmd.h,nvn_cmd_handles.cpp,nvn_core.h,nvn_gpu.cpp,nvn_gpu_internal.h,nvn_gpu_pool.cpp,nvn_gpu_texture.cpp,nvn_loader.cpp,nvn_pipeline.cpp,nvn_pipeline.h,nvn_replay.cpp,nvn_texprep.cpp,nvn_texture_registry.cpp}`: optimize NVN command replay hot paths, mirror pool dirty-range tracking, texture/resource registry lookups, and pipeline cache lookup.
  - `runtime/include/nxrt/{frame_limiter.h,movie_vdec.h,services_api.h}`, `runtime/src/gfx/frame_limiter.cpp`, `runtime/src/media/movie_vdec.cpp`, `runtime/src/services/vi_nv.cpp`, `runtime/src/gfx/play/{play.cpp,play_types.h}`, `runtime/docs/CONFIG.md`: expose live dynamic-FPS / present-interval & `vi` vsync-rate updates for dual-clock 90/120 fps gameplay with 60 fps cutscene cap, default F1 upscaler to TAAU 100% native (`auto` fps_limit), and document new keys in `CONFIG.md`.
  - `runtime/src/gfx/nvn/{nvn_prewarm.cpp,nvn_pipeline.cpp,nvn_gpl.cpp}`, `runtime/docs/CONFIG.md`: add boot shader preparation (`[nvn] shader_prepare`), share prepared Vulkan GPL libraries with runtime `GraphicsPipeline::add_variant()` cache keys, clear per-module prepared caches cleanly on reset, bump `kPrepareScheme` to `2`, and preserve FS outputs when `key.color_count == 0` so prewarmed depth/shadow FS variants match draw-time keys (`0` draw-time stalls).
- **2026-11-25 (3D texture block depth):**
  - `runtime/src/gfx/nvn/nvn_texture_layout.cpp`, `runtime/tests/gfx/test_nvn_texture.cpp`: size the level-0 block depth of non-sparse 3D textures to the slice count (smallest power of two covering it, at most 16) instead of always 16, and take the 3D block shape from a packaged layout's byte 0 (low half height, high half depth); fixes XC2's 256x128x8 BC4 lightning flipbook drawing as purple stripes. Adds `layout_3d_block_depth_fits_the_slices`.
- **2026-11-24 (scene HDR, HDR style presets, live HDR toggle):**
  - `runtime/src/gfx/extras/{hdr_tonemap_scene.cpp,hdr_tonemap_patch.h,hdr_nvn.h,hdr_nvn.cpp}`, `runtime/tests/gfx/test_hdr_tonemap_scene.cpp`: new scene-based tone-map patch for XC2's title and field tone maps (exposure, shaper, 16^3 grading LUT): above mid grey the brightness follows the game's own scene values at the LUT's slope instead of the SDR shoulder; `hdr_tonemap_patched()` lists the patched shaders.
  - `runtime/src/gfx/nvn/{nvn_gpu_texture.cpp,nvn_gpu_internal.h,nvn_replay.cpp,api_dlss_replay.cpp}`: `tex_promote_float()` (shared `swap_image()` with `tex_rescale()`) turns the window and the upscaler's output into RGBA16F while the patched tone map runs, so values above 1 reach the output pass.
  - `runtime/src/gfx/nvn/{nvn_pipeline.h,nvn_pipeline.cpp}`, `runtime/src/gfx/extras/{hdr_live.h,hdr_live.cpp}`, `runtime/src/gfx/vk_renderer.cpp`: live `[hdr] mode` / `format` (swapchain rebuilt without a restart); `GraphicsKey::sdr_tone_map` runs the game's own tone map while the output is SDR.
  - `runtime/src/gfx/extras/{output_pass.h,output_pass.cpp,overlay_composite.cpp}`, `runtime/scripts/gen_extras_shaders.py`, `runtime/src/gfx/extras/extras_shaders.inc`, `runtime/src/gfx/nvn/nvn_present.cpp`, `runtime/src/gfx/overlay.cpp`, `runtime/tests/gfx/test_extras_output.cpp`: output pass fits the kept highlights under the display peak, Vivid clips and expands as before, debanding keeps values above 1; the F1 overlay composites in SDR too.
  - `runtime/include/nxrt/extras.h`, `runtime/src/gfx/extras/extras_state.cpp`, `runtime/src/gfx/play/{play.cpp,play_config.cpp,play_types.h,play_ui.cpp}`, `runtime/tests/gfx/test_play_config.cpp`, `runtime/docs/CONFIG.md`: `[hdr] style` (`accurate` / `vivid`), live HDR mode and format in F1, Native offered only when the game declares HDR levels.
- **2026-11-26 (mirror_capture: graphics SSBO sub-pages, PR #72):**
  - `runtime/src/gfx/nvn/{nvn_gpu_pool.cpp,nvn_gpu_internal.h,nvn_replay.cpp}`: preserve unchanged GPU-written graphics SSBO sub-page boundary bytes when `mirror_gfx_storage_writeback = 0`, so neighbouring CPU UBO writes on the same 4 KiB page do not overwrite GPU facial morph vertices with stale host memory; `mirror_capture_bind` gains `gfx_no_wb`.
  - `runtime/src/gfx/nvn/nvn_capture_spans.h`, `runtime/tests/gfx/test_nvn_capture_spans.cpp`: the span arithmetic (which bytes of a captured page range the CPU side may copy) moved into a pure header with its own test group, so a mistake there fails the suite instead of silently dropping GPU-written bytes.
- **2026-11-26 (mirror_capture: replace memcmp sub-page skipping with VB-boundary write-back):**
  - `runtime/src/gfx/nvn/{nvn_gpu_internal.h,nvn_gpu_pool.cpp}`, `runtime/docs/CONFIG.md`: replace PR #72's `memcmp`/shadow capture skipping (which swallowed CPU same-value `0.0f` resets of XC2's 16 B movie draw-once gate `1c5c6103b8077dba` / `e8d2e97fefaacd4b` after the GPU wrote `100000.0f`, freezing DLC movie playback) with render-thread tracking of live vertex buffers (`kReadVertex`, 1 MiB buckets) and targeted `cap_note_write` write-back + in-flight `g_cap_wr` exclusion ONLY for unaligned `< 4 KiB` head/tail boundary sub-pages of graphics-written SSBOs overlapping a live VB (~0.25-0.77 MiB/s vs ~47 MiB/s full write-back); re-enable `[nvn] mirror_gfx_subpage` by default.
- **2026-11-28 (revert startup Vulkan Video prewarm race, issue #91):**
  - `runtime/src/gfx/vk_device.cpp`: remove `prewarm_hwdec_async()` right after `gfx::Device::create()` — creating a second FFmpeg Vulkan device concurrently with initial swapchain / NVN pipeline bring-up deadlocked `nvoglv64.dll` / `amdvlk64.dll` on boot (#91, #42); hardware video decode initializes lazily when the first cutscene opens.
