# Working notes for agents

Pipeline that aligns a session-length in-car video to ERDP `.arrow` telemetry,
then renders a lap overlay. See `README.md` for user-facing usage. This file is
context for code agents.

## Alignment: telemetry ↔ video offset

Two independent estimators live behind the `audio_alignment` argument
(`_resolve_alignment` in `src/pipeline.jl`). **Sign convention (both):**
`offset_s` means `telemetry_time = video_time + offset_s`.

- `:auto` — `align_audio_rpm` (`src/alignment.jl`): RPM-from-audio (firing-tone
  STFT) cross-correlated against `EngineRotVel`. The *sharp* estimator — use it
  when the clip has real **engine** audio. Has a lap-aliasing defense (top-K
  peaks + a session-unique "is-racing" boolean seed). The xcorr peak is parabola-
  refined to sub-frame (`_parabolic_peak`, shared with the visual aligner) —
  `offset_s` vs `coarse_offset_s`/`subsample_shift_s` in the result. The peak is
  broad (RPM is slow + ~0.5 s STFT window) so don't read ms precision into it.
- `:visual` — `align_visual_rotation` (`src/visual_align.jl`): camera yaw/pitch
  RATE from phase correlation on a far-field horizon crop, cross-correlated vs
  `ChassisRotVelYawIDR`/`ChassisRotVelPitchIDR`. For clips with **no usable
  engine audio** (e.g. radio-only feeds). Two-stage coarse→fine with parabolic
  sub-sample; `channel_spread_s` (yaw vs pitch) is a built-in self-check.
- `:seed` (default), `:none`, or a manual `Float64`.

**There is no ground truth — only estimators.** Validate by *convergence* of
methods that don't share a failure mode (audio↔RPM, visual rotation, and the
planned forward-flow↔GPS-speed axis). On Watkins Glen car 16, audio gave
−594.0 and visual joint −594.5 (0.5 s apart) — that agreement is the signal.
Don't trust a filename wall-clock as truth: re-encoded clips have
`creation_time` stripped, and even raw `IC<car>_<YYYYMMDD>_<HHMMSS>.mpg` names
proved ~170 s off (tz/PTS).

### Lap aliasing
Corners repeat every lap, so the visual correlation is a *comb* with a tooth per
lap — any tooth aligns the traces. Pick the right one with a coarse seed
(`seed`/`seed_tol_s` kwargs) or a session-unique signal. On road courses the
distinct corners usually make one tooth dominate unseeded (San Diego car 16: 3
windows converged on −208.5); ovals need a seed.

**Forward-flow channel (experimental, `src/visual_align.jl`, not wired into
`:visual`):** `align_forward_speed` / `video_forward_track` — a speed PROXY
(inter-frame change in a foreground crop) correlated, UNFILTERED (only smoothed,
never band-passed — the band-pass would delete the slow session envelope that
makes it unique), against `VectorGPS_Speed`. It's the COARSE lap-fixer, not a
precise locker: on San Diego it landed within ~25 s of the −208.5 truth (broad
peak, picked a near-tied neighbour) — enough to seed the yaw/pitch fine refine,
not enough to trust standalone. Slow (decodes a long window); 4 fps is a
slideshow — higher fps / a better window is the obvious next iteration.

## Telemetry channels (2026 format)
`CHANNEL_BINDING` in `src/telemetry.jl`. 2026 arrows renamed some channels:
speed is **`VectorGPS_Speed`** (not `OTD_Conv_Speed`), lap fraction is
**`VectorGPS_LapFrac`**. Rate gyros (`ChassisRotVel{Yaw,Pitch,Roll}IDR`) are NOT
in the binding — `visual_align` reads them directly. Rate channels are NaN
before the first line-crossing (pre-green); drop non-finite samples. `Time` is
0-based seconds @100 Hz (green = 0); `timestamp` is absolute epoch-ms.

## ffmpeg / GPU
`detect_backend()` (`src/runtime.jl`) prefers a system ffmpeg with NVENC
(install the gyan.dev build) → `h264_nvenc` encode + `-hwaccel cuda` decode;
else falls back to bundled `FFMPEG_jll` (CPU, dxva2/d3d11va only — no NVDEC).
Always route ffmpeg through `with_backend(backend)`. `_which_ffmpeg` now swallows
a non-zero `where`/`which` exit and falls back to known install dirs
(`_ffmpeg_fallback_dirs`: `~/ffmpeg/bin`, the winget `Gyan.FFmpeg` link shim,
`C:\ffmpeg\bin`, choco) before returning `nothing` — so an ffmpeg that isn't on
PATH is still found instead of crashing.

## Performance conventions
- Run Julia with `-t auto`; the hot correlation kernel is threaded.
- `src/visual_align.jl` `_vs_ncc_kernel!` is a deliberate **function barrier** —
  keep the threaded hot loop in a separately-typed function so nothing boxes
  under `Threads.@threads` (a naive closure cost 400 GiB/101 s; the barrier made
  it 30 MiB/0.12 s). Inner reduction is `@simd` (emits `vfmadd` — verified).
  Windowed sums use zero-padded prefix sums (branch-free).
- STFT loop in `audio_rpm_trace` (`src/alignment.jl`) is threaded
  (`_stft_rpm_kernel!` function barrier). One buf+plan per chunk, built SERIALLY
  first (FFTW *planning* is not thread-safe; *execution* on separate buffers is),
  chunks capped at `nthreads-2`. The real win was killing per-frame allocation
  with in-place `mul!` (3.70→0.78 s serial; threaded 0.126 s ≈ 29× total, output
  bit-identical). Note: ffmpeg audio-decode now dominates `:auto` wall-clock, not
  the STFT.
- Next perf target (un-done): thread the per-frame phase-correlation loop in
  `video_rotation_track` (`src/visual_align.jl`) — same per-thread-plan pattern,
  but 2D `fft` plan thread-safety + the prev→cur frame dependency + a whole-
  session memory ceiling make it more involved than the STFT.
- The per-frame render loop (`src/pipeline.jl`) is already O(1) per frame with a
  baked static surface; don't add per-frame searches.
