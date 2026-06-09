using Cairo
using DataFrames
using Printf

"""
    generate_lap_video(video_path, arrow_path, lap_number;
                       output_path,
                       track            = nothing,
                       track_map_db     = default_db_path(),
                       driver_label     = "",
                       event_label      = "",
                       fps              = 25,
                       resolution       = (1280, 720),
                       audio_alignment  = :auto,
                       ranges           = default_ranges(),
                       progress         = nothing) -> NamedTuple

Top-level entry point. Detects the requested lap in the arrow file, aligns
the video to telemetry via audio↔RPM cross-correlation (unless overridden),
trims the source clip, renders overlay frames with Cairo, and pipes
everything through a single ffmpeg invocation to produce `output_path`.
"""
function generate_lap_video(video_path::AbstractString,
                            arrow_path::AbstractString,
                            lap_number::Integer;
                            output_path::AbstractString,
                            track::Union{Nothing,Symbol,AbstractString} = :auto,
                            track_map_db::AbstractString = default_db_path(),
                            driver_label::AbstractString = "",
                            event_label::AbstractString  = "",
                            fps::Int = 25,
                            resolution::Tuple{Int,Int} = (1280, 720),
                            audio_alignment::Union{Symbol,Real} = _config_alignment(),
                            ranges = default_ranges(),
                            encoder::Symbol = :auto,
                            progress::Union{Nothing,Function} = nothing)
    _require_file(video_path, "video", "PRERACEFILM_DATA_DIR")
    _require_file(arrow_path, "arrow", "PRERACEFILM_ARROW_DIR")
    backend = detect_backend(; prefer = encoder)
    tel = load_telemetry(arrow_path)
    laps = detect_laps(tel; drop_partial = false)
    lap_row = findfirst(==(Int(lap_number)), laps.lap)
    lap_row === nothing && error("Lap $lap_number not found in $arrow_path")
    lap = laps[lap_row, :]
    lap_rows = lap.row_start:lap.row_end

    t_tel_start = lap.t_start
    lap_dur     = lap.duration

    # Audio alignment
    offset_s, align_meta = _resolve_alignment(audio_alignment, video_path, arrow_path, backend)

    # Map telemetry window → video window
    video_lap_start = t_tel_start - offset_s
    video_lap_dur   = lap_dur
    if video_lap_start < 0
        @warn "Aligned video start clipped to 0" requested=video_lap_start
        video_lap_dur += video_lap_start
        video_lap_start = 0.0
    end

    # Track map (auto-detect from filename if track == :auto / "auto")
    tm = nothing
    track_key = nothing
    if track === :auto || (track isa AbstractString && lowercase(String(track)) == "auto")
        track_key = auto_detect_track(arrow_path)
        track_key === nothing &&
            @warn "Could not auto-detect track from '$arrow_path' — rendering without map."
    elseif track isa AbstractString
        track_key = String(track)
    end
    if track_key !== nothing
        tm = load_track_map(track_map_db, track_key)
        tm === nothing && @warn "Track map not found for '$track_key' — rendering without map."
    end

    layout = OverlayLayout(W = resolution[1], H = resolution[2])
    track_surface = tm === nothing ? nothing :
        bake_track_background(tm, layout.map_w - 20, layout.top_h - 20)

    # Build channels + normalised time
    channels = build_channels(tel, lap_rows, ranges)
    t_raw    = Float64.(view(tel.time, lap_rows))
    t_norm   = (t_raw .- t_raw[1]) ./ (t_raw[end] - t_raw[1])
    lap_fracs = Float64.(view(tel.lap_frac, lap_rows))
    # OTD_Conv_LapFraction is cumulative (lap_int + frac), so subtract floor
    # to get [0,1) and multiply by total distance for arc length.
    track_dist = tm === nothing ? zeros(Float64, length(lap_fracs)) :
        ((lap_fracs .- floor.(lap_fracs)) .* tm.total_dist_ft)

    static_surface = bake_static_surface(layout, channels, t_norm,
                                          track_surface, driver_label, event_label)

    # Prepare frame buffer
    frame_surf = CairoARGBSurface(layout.W, layout.H)
    cr = CairoContext(frame_surf)

    total_frames = max(1, round(Int, lap_dur * fps))
    raw_rgba = argbuffer(frame_surf)

    # ── Launch ffmpeg ────────────────────────────────────────────────────
    output_dir = dirname(abspath(output_path))
    isdir(output_dir) || mkpath(output_dir)

    proc = with_backend(backend) do exe
        cmd = String[exe, "-y", "-hide_banner", "-loglevel", "error"]
        # Hardware decode of source if available
        append!(cmd, backend.hwaccel_args)
        append!(cmd, ["-ss", string(video_lap_start), "-t", string(video_lap_dur),
                      "-i", String(video_path),
                      "-f", "rawvideo", "-pix_fmt", "bgra",
                      "-s", "$(layout.W)x$(layout.H)", "-r", string(fps),
                      "-i", "pipe:0",
                      "-filter_complex",
                      "[0:v]scale=$(layout.vid_w):$(layout.top_h)[vid];" *
                      "color=black:$(layout.W)x$(layout.H):r=$(fps)[bg];" *
                      "[bg][vid]overlay=0:0[bgv];" *
                      "[bgv][1:v]overlay=0:0[v]",
                      "-map", "[v]", "-map", "0:a?"])
        append!(cmd, backend.encoder_args)
        append!(cmd, ["-c:a", "aac", "-b:a", "192k", "-shortest", String(output_path)])
        open(Cmd(cmd), "w")
    end

    try
        cur_vals = Vector{Float64}(undef, length(channels))
        cur_norms = Vector{Float64}(undef, length(channels))
        nrows = length(lap_rows)

        for i in 0:(total_frames - 1)
            tq = i / (total_frames - 1 + eps())
            # locate the telemetry sample by normalised time
            idx_f = tq * (nrows - 1) + 1
            i0 = clamp(floor(Int, idx_f), 1, nrows - 1)
            frac = idx_f - i0
            for (k, ch) in enumerate(channels)
                cur_vals[k]  = ch.data[i0] * (1 - frac) + ch.data[i0 + 1] * frac
                cur_norms[k] = ch.norm[i0] * (1 - frac) + ch.norm[i0 + 1] * frac
            end
            cur_dist = tm === nothing ? 0.0 :
                track_dist[i0] * (1 - frac) + track_dist[i0 + 1] * frac
            lap_t = tq * lap_dur

            blit_surface!(frame_surf, static_surface)
            draw_dynamic!(cr, layout, channels, tq, cur_vals, cur_norms,
                          tm, cur_dist, lap_t)
            Cairo.flush(frame_surf)
            write(proc, raw_rgba)

            if progress !== nothing && i % 25 == 0
                progress((frame = i + 1, total = total_frames))
            end
        end
    finally
        close(proc.in)
        wait(proc)
    end

    size_mb = filesize(output_path) / 1e6
    return (
        output_path     = String(output_path),
        file_size_mb    = size_mb,
        total_frames    = total_frames,
        lap_number      = Int(lap_number),
        lap_time_s      = lap_dur,
        audio_offset_s  = offset_s,
        track_map_used  = tm !== nothing,
        track_key       = track_key,
        encoder         = backend.encoder,
        ffmpeg_backend  = backend.use_system ? :system : :bundled,
        alignment       = align_meta,
    )
end

"""
    _config_alignment() -> Union{Symbol,Float64}

Read `[defaults].audio_alignment` from the TOML config. Numbers come back
as Float64, strings are parsed (numeric strings → Float64, other strings →
Symbol). Falls back to `:seed`.
"""
function _config_alignment()
    v = config_get("defaults", "audio_alignment", :seed)
    v isa Symbol     && return v
    v isa Real       && return Float64(v)
    if v isa AbstractString
        n = tryparse(Float64, v)
        return n === nothing ? Symbol(v) : n
    end
    return :seed
end

function _require_file(path::AbstractString, kind::AbstractString, env_var::AbstractString)
    isfile(path) && return
    hint = haskey(ENV, env_var) ?
        "Currently $env_var = $(ENV[env_var])" :
        "Tip: set \$env:$env_var to your session library path " *
        "(e.g. \"D:\\\\Race_Videos\\\\25POC1\") so the CLI / Pluto picker " *
        "auto-find files."
    error("""
        $kind file not found:
            $path
        $hint
        """)
end

"""
    _resolve_alignment(spec, video_path, arrow_path, backend) -> (offset_s, meta)

Translate the `audio_alignment` argument into a concrete offset in seconds.

- `:none`     — no shift (telemetry and video already aligned)
- `:seed`     — fast: race_t_tel − audio_active_t, no FFT correlation
- `:auto`     — full FFT cross-correlation (slower, lower confidence on
                short / noisy sessions)
- `Float64 n` — manual override, use as-is
"""
function _resolve_alignment(spec, video_path, arrow_path, backend)
    if spec === :none
        return 0.0, (mode = :none,)
    elseif spec isa Real
        return Float64(spec), (mode = :override, value = Float64(spec))
    elseif spec === :seed
        tel = load_telemetry(arrow_path)
        race_idx     = find_race_start(tel.rpm)
        race_t_tel   = Float64(tel.time[race_idx])
        audio_active = find_audio_active_start(video_path; backend = backend)
        offset = race_t_tel - audio_active
        return offset, (mode = :seed,
                        race_t_tel = race_t_tel,
                        audio_active_vid_s = audio_active,
                        offset_s = offset)
    elseif spec === :auto
        m = align_audio_rpm(video_path, arrow_path)
        return m.offset_s, merge((mode = :auto,), m)
    else
        error("Unknown audio_alignment: $spec")
    end
end


# ─── Integration surface: JSON-friendly wrappers ─────────────────────────────
#
# The functions below give non-Julia callers (CLI args, HTTP bodies, agent
# tool calls, batch scripts) a flat config-dict interface to the same core
# pipeline. They return plain `Dict`s so the result serialises with JSON3
# without further conversion.

"""
    list_laps_json(arrow_path; drop_partial=true) -> Vector{Dict{String,Any}}

JSON-serialisable version of `detect_laps`. Each row becomes a Dict.
"""
function list_laps_json(arrow_path::AbstractString;
                        min_seconds::Real = 20.0,
                        drop_partial::Bool = true)
    df = detect_laps(arrow_path; min_seconds = min_seconds, drop_partial = drop_partial)
    return [Dict(string(k) => _to_json_value(getproperty(row, k))
                 for k in propertynames(row))
            for row in eachrow(df)]
end

"""
    generate_lap_video_json(config::AbstractDict) -> Dict{String,Any}

JSON-friendly wrapper around `generate_lap_video`. Required keys:
`video_path`, `arrow_path`, `lap_number`, `output_path`. All other keys are
optional and map to keyword arguments. Returns a result Dict.
"""
function generate_lap_video_json(config::AbstractDict)
    video_path   = config["video_path"]
    arrow_path   = config["arrow_path"]
    lap_number   = Int(config["lap_number"])
    output_path  = config["output_path"]

    kwargs = Dict{Symbol,Any}()
    haskey(config, "track")           && (kwargs[:track]           = _maybe_symbol(config["track"]))
    haskey(config, "track_map_db")    && (kwargs[:track_map_db]    = config["track_map_db"])
    haskey(config, "driver_label")    && (kwargs[:driver_label]    = config["driver_label"])
    haskey(config, "event_label")     && (kwargs[:event_label]     = config["event_label"])
    haskey(config, "fps")             && (kwargs[:fps]             = Int(config["fps"]))
    haskey(config, "resolution")      && (kwargs[:resolution]      = Tuple{Int,Int}(config["resolution"]))
    haskey(config, "audio_alignment") && (kwargs[:audio_alignment] = _maybe_symbol(config["audio_alignment"]))
    haskey(config, "encoder")         && (kwargs[:encoder]         = Symbol(config["encoder"]))

    result = generate_lap_video(video_path, arrow_path, lap_number;
                                output_path = output_path, kwargs...)
    return Dict(string(k) => _to_json_value(v) for (k, v) in pairs(result))
end

function _maybe_symbol(v::AbstractString)
    startswith(v, ":") && return Symbol(v[2:end])
    n = tryparse(Float64, v)
    return n === nothing ? v : n
end
_maybe_symbol(v::Symbol) = v
_maybe_symbol(v::Real) = Float64(v)
_maybe_symbol(v) = v

_to_json_value(v::Symbol) = string(v)
_to_json_value(v::NamedTuple) = Dict(string(k) => _to_json_value(getproperty(v, k))
                                     for k in propertynames(v))
_to_json_value(v::Tuple) = collect(_to_json_value(x) for x in v)
_to_json_value(v::AbstractDict) = Dict(string(k) => _to_json_value(val) for (k, val) in v)
_to_json_value(v::AbstractVector) = [_to_json_value(x) for x in v]
_to_json_value(v) = v
