using Cairo
using DataFrames
using Printf

"""
    generate_lap_video(cfg::RaceConfig, car::Integer, lap::Integer;
                       fps=25, resolution=(1280,720), template=:full,
                       track=:auto, track_map_db=default_db_path(),
                       ranges=default_ranges(), alignment_method=nothing,
                       fine_tune_s=nothing, overwrite=false, progress=nothing)
        -> NamedTuple

The entry point. Resolves the session files, driver/event labels, per-car
overrides, and output path from `cfg`, then detects the lap, aligns the video to
telemetry by the chosen `alignment_method`, trims the source clip, renders Cairo
overlay frames, and pipes everything through a single ffmpeg invocation.

`alignment_method` is `:seed | :audio | :visual | <offset_s>`, resolved as
explicit kwarg → per-car `race.toml` → race-wide `race.toml`. It is **required** —
if set nowhere, this errors rather than guessing. `process` resolves it once per
car and passes the offset back in.

Returns `(; output_path, skipped=true)` without rendering if the output already
exists and `overwrite=false`.

    cfg = getConfig("25POC1")            # or getConfig() for [current].race
    generate_lap_video(cfg, 9, 119; alignment_method = :audio, template = :minimal)
    process(cfg; cars = [9], laps = :all, alignment_method = :seed)   # batch
"""
function generate_lap_video(cfg::RaceConfig, car::Integer, lap::Integer;
                            fps::Int = 25,
                            resolution::Tuple{Int,Int} = (1280, 720),
                            template::Symbol = :full,
                            track::Union{Nothing,Symbol,AbstractString} = :auto,
                            track_map_db::AbstractString = default_db_path(),
                            ranges = default_ranges(),
                            alignment_method = nothing,
                            fine_tune_s = nothing,
                            start_loop::Union{Nothing,AbstractString} = nothing,
                            end_loop::Union{Nothing,AbstractString}   = nothing,
                            overwrite::Bool = false,
                            progress::Union{Nothing,Function} = nothing)
    # ── Resolve from the config ──────────────────────────────────────────────
    session     = find_car_session(cfg, car)
    video_path  = session.video
    arrow_path  = session.arrow
    lap_number  = lap
    car_number  = Int(car)
    loop_sfx    = _loop_suffix(start_loop, end_loop)
    tmpl_sfx    = template === :raw ? "_raw" : ""
    output_path = joinpath(cfg.output_dir,
                           "$(cfg.race)_car$(car)_lap$(lap)$(loop_sfx)$(tmpl_sfx).mp4")
    isdir(dirname(output_path)) || mkpath(dirname(output_path))
    if !overwrite && isfile(output_path)
        @info "Already rendered, skipping: $output_path  (pass overwrite=true to redo)"
        return (output_path = output_path, skipped = true)
    end
    driver_label = driver_for(cfg, car)
    event_label  = let e = event_label_default(cfg)
        isempty(e) ? something(auto_detect_track(arrow_path), "") : e
    end
    method = something(alignment_method,                              # explicit kwarg wins
                       car_override(cfg, car, "alignment_method"),    # then per-car race.toml
                       cfg.alignment_method,                          # then race-wide race.toml
                       Some(nothing))
    method === nothing && error(
        "No alignment_method for car #$car. Choose one explicitly — " *
        "pass alignment_method = :seed | :audio | :visual | <offset_s>, or set " *
        "`alignment_method` in race.toml (race-wide or under [cars.$car]).")
    ft_ov = car_override(cfg, car, "fine_tune_s")
    fine_tune_s = fine_tune_s !== nothing ? Float64(fine_tune_s) :
                  ft_ov       !== nothing ? Float64(ft_ov) : 1.0
    @info "Rendering $driver_label car #$car lap $lap → $output_path"


"""
use this after every fine tune adjustment

using Revise           
using PreRaceFilm
cfg = getConfig("25SON1")

"""

    # ── Render ───────────────────────────────────────────────────────────────
    _require_file(video_path, "video")
    _require_file(arrow_path, "arrow")
    tel = load_telemetry(arrow_path)
    laps = detect_laps(tel; drop_partial = false)
    lap_row = findfirst(==(Int(lap_number)), laps.lap)
    lap_row === nothing && error("Lap $lap_number not found in $arrow_path")
    lap = laps[lap_row, :]
    full_lap_rows = lap.row_start:lap.row_end
    # Optionally narrow the window to between two timing loops within the lap.
    # No-op when both kwargs are nothing.
    lap_rows, t_tel_start, t_tel_end =
        _narrow_lap_to_loops(tel, full_lap_rows, start_loop, end_loop,
                             Float64(lap.t_start), Float64(lap.t_end))
    lap_dur = t_tel_end - t_tel_start

    # Resolve the chosen method to a concrete offset + manual fine-tune
    est = _resolve_alignment(method, video_path, arrow_path)
    raw_offset_s = est.offset_s
    offset_s = raw_offset_s + Float64(fine_tune_s)
    align_meta = merge((mode = est.method, confidence = est.confidence,
                        raw_offset_s   = raw_offset_s,
                        fine_tune_s    = Float64(fine_tune_s),
                        final_offset_s = offset_s),
                       est.detail)

    # Map telemetry window → video window
    video_lap_start = t_tel_start - offset_s
    video_lap_dur   = lap_dur
    if video_lap_start < 0
        @warn "Aligned video start clipped to 0" requested=video_lap_start
        video_lap_dur += video_lap_start
        video_lap_start = 0.0
    end

    # `:raw` is a bare clip — skip all the Cairo bake/draw + filter-graph work
    # below and hand off to render_raw_clip. Still uses the same alignment +
    # fine_tune resolution so the cut lands on the right lap boundary.
    if template === :raw
        raw = render_raw_clip(video_path, video_lap_start, video_lap_dur, output_path)
        return (
            output_path     = raw.output_path,
            file_size_mb    = raw.file_size_mb,
            lap_number      = Int(lap_number),
            lap_time_s      = lap_dur,
            audio_offset_s  = offset_s,
            track_map_used  = false,
            track_key       = nothing,
            encoder         = raw.encoder,
            template        = :raw,
            alignment       = align_meta,
        )
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
    channels = template === :minimal ?
        build_channels_minimal(tel, lap_rows, ranges) :
        template === :full ?
            build_channels(tel, lap_rows, ranges) :
            error("Unknown template: $template (expected :full | :minimal | :raw)")
    stats = template === :minimal ?
        build_stats_minimal(tel, lap_rows, ranges) :
        ChannelTrace[]
    t_raw    = Float64.(view(tel.time, lap_rows))
    t_norm   = (t_raw .- t_raw[1]) ./ (t_raw[end] - t_raw[1])
    lap_fracs = Float64.(view(tel.lap_frac, lap_rows))
    # Marker position is integrated GPS speed (mph → ft/s, cumulative). Using
    # lap_frac × total_dist_ft directly puts the dot on the vendor's reference
    # line, which doesn't match OUR polygon's arc-length parameterization
    # through corners — the dot drifts ahead/behind in twisty sections. The
    # integrated-speed track_dist advances at the car's actual physical rate
    # (slows under braking, accelerates on exit) so it lines up with our
    # polygon's arc-length table directly. Lap-start position is still
    # anchored to lap_frac so the dot begins at the right S/F offset.
    track_dist = if tm === nothing
        zeros(Float64, length(lap_fracs))
    else
        speed_fts = Float64.(view(tel.speed, lap_rows)) .* (5280.0 / 3600.0)
        dt        = vcat(0.0, diff(t_raw))
        anchor    = (lap_fracs[1] - floor(lap_fracs[1])) * tm.total_dist_ft
        anchor .+ cumsum(speed_fts .* dt)
    end

    static_surface = template === :minimal ?
        bake_static_surface_minimal(layout, channels, t_norm, track_surface,
                                    driver_label, event_label, stats) :
        bake_static_surface(layout, channels, t_norm,
                            track_surface, driver_label, event_label)

    # Car-number graphic for the track-map marker (minimal template only).
    car_graphic = nothing
    if template === :minimal && car_number !== nothing && tm !== nothing
        car_graphic = load_car_number_graphic(car_number, CAR_NUMBER_GRAPHIC_H)
        car_graphic === nothing &&
            @warn "No car-number graphic found for car #$car_number — falling back to dot."
    end

    # Prepare frame buffer
    frame_surf = CairoARGBSurface(layout.W, layout.H)
    cr = CairoContext(frame_surf)

    total_frames = max(1, round(Int, lap_dur * fps))
    raw_rgba = argbuffer(frame_surf)

    # ── Launch ffmpeg ────────────────────────────────────────────────────
    output_dir = dirname(abspath(output_path))
    isdir(output_dir) || mkpath(output_dir)

    cmd = String[ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    append!(cmd, hwaccel_args())                 # hardware decode of source if available
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
    append!(cmd, encode_args())
    append!(cmd, ["-c:a", "aac", "-b:a", "192k", "-shortest", String(output_path)])
    proc = open(Cmd(cmd), "w")

    try
        cur_vals = Vector{Float64}(undef, length(channels))
        cur_norms = Vector{Float64}(undef, length(channels))
        cur_stat_vals = Vector{Float64}(undef, length(stats))
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
            for (k, st) in enumerate(stats)
                cur_stat_vals[k] = st.data[i0] * (1 - frac) + st.data[i0 + 1] * frac
            end
            cur_dist = tm === nothing ? 0.0 :
                track_dist[i0] * (1 - frac) + track_dist[i0 + 1] * frac
            lap_t = tq * lap_dur

            blit_surface!(frame_surf, static_surface)
            if template === :minimal
                draw_dynamic_minimal!(cr, layout, channels, tq, cur_vals, cur_norms,
                                      tm, cur_dist, lap_t, stats, cur_stat_vals,
                                      car_graphic)
            else
                draw_dynamic!(cr, layout, channels, tq, cur_vals, cur_norms,
                              tm, cur_dist, lap_t)
            end
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
        encoder         = _caps().nvenc ? :h264_nvenc : :libx264,
        template        = template,
        alignment       = align_meta,
    )
end

"""
    generate_comparison_video(cfg, carA, lapA, carB, lapB;
                              alignment_method=nothing, fine_tune_s=nothing,
                              fps=25, resolution=(1280,720),
                              ranges=default_ranges(), overwrite=false)
        -> NamedTuple

Render two laps side-by-side with overlaid telemetry. Each driver gets their
own alignment + fine-tune resolution (same precedence as `generate_lap_video`:
kwarg → per-car race.toml → race-wide). The faster lap (shorter duration)
becomes the reference: clip length = faster lap's duration, source audio =
faster driver, trace colors = green (faster) / red (slower).

Output path: `<output_dir>/<race>_carAvsB_lapL1vsL2_comparison.mp4`.
"""
function generate_comparison_video(cfg::RaceConfig,
                                   carA::Integer, lapA::Integer,
                                   carB::Integer, lapB::Integer;
                                   alignment_method = nothing,
                                   fine_tune_s = nothing,
                                   start_loop::Union{Nothing,AbstractString} = nothing,
                                   end_loop::Union{Nothing,AbstractString}   = nothing,
                                   fps::Int = 25,
                                   resolution::Tuple{Int,Int} = (1280, 720),
                                   ranges = default_ranges(),
                                   overwrite::Bool = false)
    sess_A = find_car_session(cfg, carA)
    sess_B = find_car_session(cfg, carB)
    loop_sfx    = _loop_suffix(start_loop, end_loop)
    output_path = joinpath(cfg.output_dir,
        "$(cfg.race)_car$(carA)vs$(carB)_lap$(lapA)vs$(lapB)$(loop_sfx)_comparison.mp4")
    isdir(dirname(output_path)) || mkpath(dirname(output_path))
    if !overwrite && isfile(output_path)
        @info "Already rendered, skipping: $output_path  (pass overwrite=true to redo)"
        return (output_path = output_path, skipped = true)
    end

    _require_file(sess_A.video, "video"); _require_file(sess_A.arrow, "arrow")
    _require_file(sess_B.video, "video"); _require_file(sess_B.arrow, "arrow")

    # Per-driver alignment resolution. Same precedence as the single-lap entry
    # point; the explicit kwarg (if given) applies to BOTH drivers.
    method_A = something(alignment_method,
                         car_override(cfg, carA, "alignment_method"),
                         cfg.alignment_method, Some(nothing))
    method_B = something(alignment_method,
                         car_override(cfg, carB, "alignment_method"),
                         cfg.alignment_method, Some(nothing))
    (method_A === nothing || method_B === nothing) && error(
        "No alignment_method for one of the cars (#$carA or #$carB). " *
        "Pass alignment_method explicitly or set it in race.toml.")
    ft_A_ov = car_override(cfg, carA, "fine_tune_s")
    ft_B_ov = car_override(cfg, carB, "fine_tune_s")
    ft_A = fine_tune_s !== nothing ? Float64(fine_tune_s) :
           ft_A_ov     !== nothing ? Float64(ft_A_ov)     : 0.0
    ft_B = fine_tune_s !== nothing ? Float64(fine_tune_s) :
           ft_B_ov     !== nothing ? Float64(ft_B_ov)     : 0.0

    @info "Comparison: car #$carA lap $lapA  vs  car #$carB lap $lapB → $output_path"

    # Per-driver telemetry, lap window, alignment offset.
    tel_A  = load_telemetry(sess_A.arrow)
    tel_B  = load_telemetry(sess_B.arrow)
    laps_A = detect_laps(tel_A; drop_partial = false)
    laps_B = detect_laps(tel_B; drop_partial = false)
    row_A  = findfirst(==(Int(lapA)), laps_A.lap)
    row_B  = findfirst(==(Int(lapB)), laps_B.lap)
    row_A === nothing && error("Lap $lapA not found in $(sess_A.arrow)")
    row_B === nothing && error("Lap $lapB not found in $(sess_B.arrow)")
    info_A = laps_A[row_A, :]; info_B = laps_B[row_B, :]

    # Optionally narrow each driver's window to between the requested loops.
    # Same loop names apply to both drivers; each finds their own crossing.
    full_rows_A = info_A.row_start:info_A.row_end
    full_rows_B = info_B.row_start:info_B.row_end
    lap_rows_A_full, t_a_start, t_a_end =
        _narrow_lap_to_loops(tel_A, full_rows_A, start_loop, end_loop,
                             Float64(info_A.t_start), Float64(info_A.t_end))
    lap_rows_B_full, t_b_start, t_b_end =
        _narrow_lap_to_loops(tel_B, full_rows_B, start_loop, end_loop,
                             Float64(info_B.t_start), Float64(info_B.t_end))
    lap_dur_A = t_a_end - t_a_start
    lap_dur_B = t_b_end - t_b_start

    est_A = _resolve_alignment(method_A, sess_A.video, sess_A.arrow)
    est_B = _resolve_alignment(method_B, sess_B.video, sess_B.arrow)
    offset_A = est_A.offset_s + ft_A
    offset_B = est_B.offset_s + ft_B

    video_lap_start_A = max(0.0, t_a_start - offset_A)
    video_lap_start_B = max(0.0, t_b_start - offset_B)

    # Sync: faster driver sets the clip length. Slower driver's tail is cut.
    faster_id  = faster_driver(lap_dur_A, lap_dur_B)
    faster_dur = min(lap_dur_A, lap_dur_B)
    @info "Faster: driver $(faster_id)  (durations  A=$(round(lap_dur_A, digits=2))s  B=$(round(lap_dur_B, digits=2))s  →  clip=$(round(faster_dur, digits=2))s)"

    # Clip each driver's lap rows to ≤ faster_dur of elapsed time, so trace
    # x-axis represents real seconds and both lines align at every x.
    lap_rows_A = clip_lap_rows(lap_rows_A_full, lap_dur_A, faster_dur)
    lap_rows_B = clip_lap_rows(lap_rows_B_full, lap_dur_B, faster_dur)

    channels_A = build_comparison_channels(tel_A, lap_rows_A, ranges)
    channels_B = build_comparison_channels(tel_B, lap_rows_B, ranges)
    stats_A    = build_comparison_stats(tel_A, lap_rows_A, ranges)
    stats_B    = build_comparison_stats(tel_B, lap_rows_B, ranges)

    # Per-driver normalised time within the kept window.  Each goes 0→1 across
    # their kept rows (which span 0→faster_dur of real time), so at any x both
    # lines reflect the same number of seconds into the lap.
    t_raw_A  = Float64.(view(tel_A.time, lap_rows_A))
    t_raw_B  = Float64.(view(tel_B.time, lap_rows_B))
    t_norm_A = (t_raw_A .- t_raw_A[1]) ./ faster_dur
    t_norm_B = (t_raw_B .- t_raw_B[1]) ./ faster_dur

    layout = comparison_layout(resolution[1], resolution[2])
    # driver_for already returns a "Car #N" string when no driver name is set,
    # so don't prefix with car number again — that's where the duplicate came from.
    driver_A_label = driver_for(cfg, carA)
    driver_B_label = driver_for(cfg, carB)
    event_label    = let e = event_label_default(cfg)
        isempty(e) ? something(auto_detect_track(sess_A.arrow), "") : e
    end

    static_surface = bake_static_surface_comparison(layout, channels_A, channels_B,
                                                    t_norm_A, t_norm_B, faster_id,
                                                    Int(carA), Int(carB),
                                                    driver_A_label, driver_B_label,
                                                    event_label)

    frame_surf = CairoARGBSurface(layout.W, layout.H)
    cr = CairoContext(frame_surf)
    total_frames = max(1, round(Int, faster_dur * fps))
    raw_rgba = argbuffer(frame_surf)

    # ── ffmpeg: 3 inputs (videoA, videoB, overlay frames) ───────────────────
    cmd = String[ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    append!(cmd, hwaccel_args())
    append!(cmd, ["-ss", string(video_lap_start_A), "-t", string(faster_dur),
                  "-i", String(sess_A.video)])
    append!(cmd, hwaccel_args())
    append!(cmd, ["-ss", string(video_lap_start_B), "-t", string(faster_dur),
                  "-i", String(sess_B.video)])
    append!(cmd, ["-f", "rawvideo", "-pix_fmt", "bgra",
                  "-s", "$(layout.W)x$(layout.H)", "-r", string(fps),
                  "-i", "pipe:0",
                  "-filter_complex",
                  "[0:v]scale=$(layout.vid_w):$(layout.top_h)[a];" *
                  "[1:v]scale=$(layout.vid_w):$(layout.top_h)[b];" *
                  "color=black:$(layout.W)x$(layout.H):r=$(fps)[bg];" *
                  "[bg][a]overlay=0:0[s1];" *
                  "[s1][b]overlay=$(layout.vid_w):0[s2];" *
                  "[s2][2:v]overlay=0:0[v]",
                  "-map", "[v]",
                  "-map", faster_id === :A ? "0:a?" : "1:a?"])
    append!(cmd, encode_args())
    append!(cmd, ["-c:a", "aac", "-b:a", "192k", "-shortest", String(output_path)])
    proc = open(Cmd(cmd), "w")

    try
        nA = length(lap_rows_A); nB = length(lap_rows_B)
        vals_A  = Vector{Float64}(undef, length(channels_A))
        vals_B  = Vector{Float64}(undef, length(channels_B))
        norms_A = Vector{Float64}(undef, length(channels_A))
        norms_B = Vector{Float64}(undef, length(channels_B))
        statv_A = Vector{Float64}(undef, length(stats_A))
        statv_B = Vector{Float64}(undef, length(stats_B))

        for i in 0:(total_frames - 1)
            tq = i / (total_frames - 1 + eps())
            # Each driver's index runs over their kept rows independently
            idxA = tq * (nA - 1) + 1
            idxB = tq * (nB - 1) + 1
            iA0 = clamp(floor(Int, idxA), 1, nA - 1); fA = idxA - iA0
            iB0 = clamp(floor(Int, idxB), 1, nB - 1); fB = idxB - iB0

            for (k, ch) in enumerate(channels_A)
                vals_A[k]  = ch.data[iA0] * (1 - fA) + ch.data[iA0 + 1] * fA
                norms_A[k] = ch.norm[iA0] * (1 - fA) + ch.norm[iA0 + 1] * fA
            end
            for (k, ch) in enumerate(channels_B)
                vals_B[k]  = ch.data[iB0] * (1 - fB) + ch.data[iB0 + 1] * fB
                norms_B[k] = ch.norm[iB0] * (1 - fB) + ch.norm[iB0 + 1] * fB
            end
            for (k, st) in enumerate(stats_A)
                statv_A[k] = st.data[iA0] * (1 - fA) + st.data[iA0 + 1] * fA
            end
            for (k, st) in enumerate(stats_B)
                statv_B[k] = st.data[iB0] * (1 - fB) + st.data[iB0 + 1] * fB
            end
            lap_t = tq * faster_dur

            blit_surface!(frame_surf, static_surface)
            draw_dynamic_comparison!(cr, layout, channels_A, channels_B,
                                     stats_A, stats_B, faster_id, tq,
                                     vals_A, vals_B, norms_A, norms_B,
                                     statv_A, statv_B, lap_t)
            write(proc.in, reinterpret(UInt8, raw_rgba))
        end
    finally
        close(proc.in)
        wait(proc)
    end

    size_mb = isfile(output_path) ? filesize(output_path) / 1e6 : 0.0
    return (
        output_path     = String(output_path),
        file_size_mb    = size_mb,
        total_frames    = total_frames,
        A               = (car = Int(carA), lap = Int(lapA), lap_dur_s = lap_dur_A, offset_s = offset_A),
        B               = (car = Int(carB), lap = Int(lapB), lap_dur_s = lap_dur_B, offset_s = offset_B),
        faster          = faster_id,
        clip_dur_s      = faster_dur,
        encoder         = _caps().nvenc ? :h264_nvenc : :libx264,
        template        = :comparison,
    )
end

# Fail early with a clear message instead of a cryptic Arrow/ffmpeg error downstream.
_require_file(path::AbstractString, kind::AbstractString) =
    isfile(path) || error("$kind file not found: $path")    #TODO move to config, doesn't belong here

# Narrow `lap_rows` (and the corresponding [t_start, t_end] window) to the
# segment between two timing loops. Either bound can be `nothing` to leave that
# end of the window at the lap boundary.
#
# Important subtlety: a NASCAR lap's `lap`-channel value increments AT the S/F
# crossing, so row 1 of lap N is an "SF" sample (the one that flipped the
# counter), and the SF that ENDS lap N is actually the row 1 of lap N+1.
# To handle "from L10 to next SF" (the end of the lap), we always search
# `end_loop` strictly AFTER `start_loop`'s row, and we allow that search to
# extend into the following lap by up to one full lap's worth of rows.
function _narrow_lap_to_loops(tel, lap_rows::UnitRange{Int},
                              start_loop::Union{Nothing,AbstractString},
                              end_loop::Union{Nothing,AbstractString},
                              default_t_start::Float64, default_t_end::Float64)
    (start_loop === nothing && end_loop === nothing) &&
        return lap_rows, default_t_start, default_t_end

    n_total = length(tel.time)

    # 1) Start: first occurrence of start_loop within the lap.
    start_row = first(lap_rows)
    t_start   = default_t_start
    if start_loop !== nothing
        loops  = String.(view(tel.loop, lap_rows))
        idx    = findfirst(==(String(start_loop)), loops)
        if idx === nothing
            present = sort(unique(loops))
            error("start_loop '$start_loop' not crossed during this lap. " *
                  "Loops present in this lap: $(join(present, ", "))")
        end
        start_row = lap_rows[idx]
        t_start   = Float64(tel.time[start_row])
    end

    # 2) End: first occurrence of end_loop AFTER start_row, scanning up to one
    # full extra lap of samples so the SF that ends this lap (which belongs to
    # lap N+1's row range) is reachable.
    end_row = last(lap_rows)
    t_end   = default_t_end
    if end_loop !== nothing
        search_from = start_row + 1
        lookahead   = min(n_total, last(lap_rows) + length(lap_rows))
        range       = search_from:lookahead
        loops       = String.(view(tel.loop, range))
        idx         = findfirst(==(String(end_loop)), loops)
        if idx === nothing
            present = sort(unique(String.(view(tel.loop, lap_rows))))
            error("end_loop '$end_loop' not crossed after start_loop in this lap " *
                  "or the following lap. Loops present in this lap: $(join(present, ", "))")
        end
        end_row = search_from + idx - 1
        t_end   = Float64(tel.time[end_row])
    end

    return start_row:end_row, t_start, t_end
end

# Filename-safe suffix carrying the chosen loop window. Empty when both args
# are nothing (no rename, no path collision with full-lap renders).
function _loop_suffix(start_loop::Union{Nothing,AbstractString},
                     end_loop::Union{Nothing,AbstractString})
    _safe(s) = replace(String(s), "/" => "", " " => "")
    parts = String[]
    start_loop !== nothing && push!(parts, "from$(_safe(start_loop))")
    end_loop   !== nothing && push!(parts, "to$(_safe(end_loop))")
    isempty(parts) && return ""
    return "_" * join(parts, "_")
end

"""
    _resolve_alignment(spec, video_path, arrow_path) -> AlignEstimate

Translate the `alignment_method` into a concrete offset in seconds.

- `:none`     — no shift (telemetry and video already aligned)
- `:seed`     — fast: race_t_tel − audio_active_t, no correlation
- `:audio`    — audio firing-tone ↔ EngineRotVel cross-correlation
- `:visual`   — streaming aligner (visual_align2): yaw/pitch/roll vs gyros + forward
                zoom vs GPS speed; returns the gyro lock, the rest in `meta`
- `Float64 n` — manual override, use as-is
"""
function _resolve_alignment(spec, video_path, arrow_path)
    # TODO: migrate this if/elseif chain to multiple dispatch — an abstract `AlignMode`
    # with concrete `Audio`/`Visual`/`Seed`/`None`/`Manual(x)` types and a
    # `_resolve_alignment(::Audio, …)` method each, instead of this branch ladder.
    #TODO: more unified input struct. perhaps raise the config thingy of the visual
    # allignment thing to a global thing, the audio version has a lot of similarities,
    # should be more common
    if spec === :none
        return AlignEstimate(0.0, 1.0, :none, (;))
    elseif spec isa Real
        return AlignEstimate(Float64(spec), 1.0, :override, (;))
    elseif spec === :seed
        time, rpm    = load_channels(arrow_path, CHANNEL_BINDING.time, CHANNEL_BINDING.rpm)
        race_idx     = find_race_start(rpm)
        race_t_tel   = time[race_idx]
        audio_active = find_audio_active_start(video_path)
        return AlignEstimate(race_t_tel - audio_active, NaN, :seed,
                             (race_t_tel = race_t_tel, audio_active_vid_s = audio_active))
    elseif spec === :audio || spec === :auto      # :auto kept as a legacy alias
        return align_audio_rpm(video_path, arrow_path)        # already an AlignEstimate
    elseif spec === :visual
        # streaming aligner (visual_align2): yaw is the video↔telemetry lock and already
        # comes back as an AlignEstimate — pass it straight through. pitch/roll/forward
        # are independent cross-checks, available in the full align() result if needed.
        return align(video_path, arrow_path).yaw
    else
        error("Unknown alignment_method: $spec")
    end
end


# ─── Integration surface: JSON-friendly wrappers ─────────────────────────────
#
# The functions below give non-Julia callers (CLI args, HTTP bodies, agent
# tool calls, batch scripts) a flat config-dict interface to the same core
# pipeline. They return plain `Dict`s so the result serialises with JSON3
# without further conversion. #TODO: is this necessary? also, #TODO, use JSON.jl


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

JSON-friendly wrapper around `generate_lap_video`. Required keys: `car`, `lap`
(and `race`, or omit it to use `[current].race`). Optional keys map to keyword
arguments (`fps`, `resolution`, `template`, `track`, `alignment_method`,
`overwrite`). Returns a result Dict.
"""
function generate_lap_video_json(config::AbstractDict)
    cfg = getConfig(String(get(config, "race", "")))
    car = Int(config["car"])
    lap = Int(config["lap"])

    kwargs = Dict{Symbol,Any}()
    haskey(config, "track")           && (kwargs[:track]           = _maybe_symbol(config["track"]))
    haskey(config, "track_map_db")    && (kwargs[:track_map_db]    = config["track_map_db"])
    haskey(config, "fps")             && (kwargs[:fps]             = Int(config["fps"]))
    haskey(config, "resolution")      && (kwargs[:resolution]      = Tuple{Int,Int}(config["resolution"]))
    haskey(config, "alignment_method") && (kwargs[:alignment_method] = _maybe_symbol(config["alignment_method"]))
    haskey(config, "template")        && (kwargs[:template]        = Symbol(config["template"]))
    haskey(config, "overwrite")       && (kwargs[:overwrite]       = Bool(config["overwrite"]))

    result = generate_lap_video(cfg, car, lap; kwargs...)
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
