# Pre-Race Film Generator

Julia pipeline that takes a session-length in-car video (`.mpg`/`.mp4`) + ERDP
telemetry (`.arrow`), aligns them — by audio↔RPM correlation **or**, for feeds
with no usable engine audio, by camera-rotation↔gyro correlation from the video
itself — trims to a chosen lap, and renders a 1280×720 H.264 overlay video
showing the track map, six telemetry traces (MPH, RPM, GEAR, THROTTLE, BRAKE,
STEERING) and their current values.

## Weekly workflow

Race session files live **outside the repo** on an external drive. Point
the package at the current week's folder with one environment variable:

```powershell
# Per-session (PowerShell)
$env:PRERACEFILM_DATA_DIR = "D:\Race_Videos\25POC1"

# Or persistently for your user account
[Environment]::SetEnvironmentVariable(
    "PRERACEFILM_DATA_DIR", "D:\Race_Videos\25POC1", "User")
```

Next week, change the path. Everything that reads files — the CLI, the
Pluto notebook, and `list_session_files()` — picks it up automatically.

If your `.arrow` files live in a different folder, also set
`PRERACEFILM_ARROW_DIR`.

## Three ways to use it

### 1. Pluto notebook (interactive prototype)

```powershell
julia --project=notebooks
julia> using Pluto; Pluto.run()
```

Open `notebooks/Lap_Picker.jl`. Lists all sessions in your data dir, picks
a lap, renders.

### 2. CLI (scripts, cron, agent)

```powershell
julia --project=. bin/pre_race_film.jl render `
    --video "D:\Race_Videos\25POC1\19_POCONO_car9_sessionID2.mpg" `
    --arrow "D:\Race_Videos\25POC1\19_POCONO_car9_sessionID2.arrow" `
    --lap 50 --out "out\car9_lap50.mp4"
```

Other subcommands:
- `backend` — show resolved ffmpeg / NVENC info
- `laps --arrow PATH` — list detected race laps as JSON

### 3. Julia API (other tools, agent integration)

```julia
using PreRaceFilm

# Find available sessions in the configured data dir
list_session_files()

# Detect laps
detect_laps("D:/Race_Videos/25POC1/19_POCONO_car9_sessionID2.arrow")

# Render
generate_lap_video(
    "D:/Race_Videos/25POC1/19_POCONO_car9_sessionID2.mpg",
    "D:/Race_Videos/25POC1/19_POCONO_car9_sessionID2.arrow",
    50;
    output_path = "out/car9_lap50.mp4",
    driver_label = "Car #9",
    event_label  = "Pocono",
)
```

For JSON-driven workflows (web service, agent tool calls):
- `list_laps_json(arrow_path)` → `Vector{Dict}`
- `generate_lap_video_json(config::Dict)` → result `Dict`

## GPU acceleration

The package auto-detects a system `ffmpeg` with NVENC. On a machine with an
Nvidia GPU and a hardware-encoder ffmpeg build (e.g. the
[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) Windows binaries), encoding
uses `h264_nvenc` and source video decode uses NVDEC (`-hwaccel cuda`).
Falls back to `libx264` on the bundled `FFMPEG_jll` otherwise.

Check current backend:
```powershell
julia --project=. bin/pre_race_film.jl backend
```

## Alignment

Telemetry and video clocks rarely start together. The `audio_alignment` argument
(`generate_lap_video(...; audio_alignment = …)`) picks the estimator. Sign
convention: `offset_s` means `telemetry_time = video_time + offset_s`.

| Mode | What it does | When to use |
|---|---|---|
| `:seed` *(default)* | `find_race_start(rpm) − find_audio_active_start(video)` | Almost always — fast and robust |
| `:none` | No shift | Streams are already aligned |
| `:auto` | FFT cross-correlation of RPM vs firing-band audio envelope | Clip has clean **engine** audio. Sub-sample refined (parabolic). |
| `:visual` | Camera yaw/pitch **rate** (phase-correlation on a horizon crop) ↔ chassis gyros `ChassisRotVel{Yaw,Pitch}IDR` | Clip has **no usable engine audio** (e.g. broadcast/radio in-car feeds) |
| `Float64` | Manual override in seconds | Reuse a known offset, tuning, or fallback |

There is no ground truth — validate by **convergence** of methods that don't
share a failure mode. On Watkins Glen car 16, `:auto` gave −594.0 s and `:visual`
−594.5 s independently. `:visual` self-checks via `channel_spread_s` (yaw vs
pitch agreement; ≲0.05 s = a clean lock) and reports a `candidate_peaks` comb —
on lap-repeating ovals pass a coarse `seed` to pick the right lap; on road
courses the distinct corners usually make one tooth dominate on their own.

```julia
# No engine audio (e.g. a dolby/broadcast in-car feed): use the video itself
generate_lap_video(video, arrow, lap; output_path = out, audio_alignment = :visual)

# Reuse a known offset (skips re-alignment entirely)
generate_lap_video(video, arrow, lap; output_path = out, audio_alignment = -208.49)
```

## Project layout

```
src/
  PreRaceFilm.jl       top-level module
  runtime.jl           ffmpeg/NVENC backend detection
  telemetry.jl         Arrow load, lap detection, channel binding
  datadir.jl           env-var session library resolution
  track_map.jl         track_map_db.json + position lookup
  alignment.jl         audio extract, RPM xcorr (threaded STFT), seed/auto offset
  visual_align.jl      camera-rotation↔gyro alignment (the :visual mode)
  render.jl            Cairo overlay compositor
  pipeline.jl          generate_lap_video + JSON helpers
bin/
  pre_race_film.jl     CLI
notebooks/
  Lap_Picker.jl        Pluto notebook UI
Track Maps/
  track_map_db.json    track outlines + arc-length tables
```

## Adding a new track

Drop the outline into `Track Maps/track_map_db.json` under a new key with
`x`, `y`, `s` (arc length, ft) arrays and `total_dist_ft`. Then add an
alias to `TRACK_KEY_MAP` in `src/track_map.jl` so the filename
auto-detector recognises it.
