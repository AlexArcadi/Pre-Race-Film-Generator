# Minimal 3-channel overlay template: THROTTLE / BRAKE / STEERING.
#
# Pairs with render.jl. Reuses the shared helpers there (OverlayLayout,
# ChannelTrace, _mk, CH_COLORS, bake_track_background, blit_surface!,
# argbuffer). Provides its own bake/draw functions so the bottom strip
# can carve out a right-side stat column showing boxed MPH / RPM / GEAR
# values, without disturbing the :full template.
#
# Select via `template = :minimal` on generate_lap_video / process. The full
# 6-channel layout remains the default.

const CH_ORDER_MINIMAL = (:THROTTLE, :BRAKE, :STEERING)
const STAT_ORDER_MINIMAL = (:MPH, :RPM, :GEAR)
const STAT_W_MINIMAL = 130    # right-side stat column width, px
const VAL_W_MINIMAL  = 95     # value column width (overrides layout.val_w for minimal)
const CAR_NUMBER_GRAPHIC_H = 36   # car-number graphic height on the track map, px

# Cache decoded car-number surfaces so multi-lap renders don't re-decode.
const _CAR_NUMBER_CACHE = Dict{Tuple{Int,Int}, Cairo.CairoSurface}()
_car_number_dir() = joinpath(dirname(@__DIR__), "NCS Car Number Graphics")

function _find_car_number_file(num::Integer)
    folder = joinpath(_car_number_dir(), string(Int(num)))
    isdir(folder) || return nothing
    for f in readdir(folder)
        lf = lowercase(f)
        (endswith(lf, ".jpg") || endswith(lf, ".jpeg") || endswith(lf, ".png")) &&
            return joinpath(folder, f)
    end
    return nothing
end

"""
    load_car_number_graphic(num, height) -> Union{Nothing, CairoSurface}

Decode `NCS Car Number Graphics/<num>/<file>` via ffmpeg at the requested
height (width auto from aspect ratio), then flood-fill near-white pixels
that are border-connected to transparent. Interior white (the number's
own fill) is preserved because the flood fill seeds only on the edges.
Returns nothing if the folder or file is missing. Cached by (num, height).
"""
function load_car_number_graphic(num::Integer,
                                 height::Int = CAR_NUMBER_GRAPHIC_H)
    key = (Int(num), height)
    haskey(_CAR_NUMBER_CACHE, key) && return _CAR_NUMBER_CACHE[key]
    path = _find_car_number_file(num)
    path === nothing && return nothing

    args = String[ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                  "-i", String(path),
                  "-vf", "scale=-2:$height",
                  "-f", "rawvideo", "-pix_fmt", "bgra", "pipe:1"]
    bytes = read(Cmd(args))
    isempty(bytes) && return nothing
    px_total = length(bytes) ÷ 4
    W = px_total ÷ height
    W * height == px_total || return nothing

    surf = Cairo.CairoARGBSurface(W, height)
    buf  = argbuffer(surf)
    copyto!(buf, reinterpret(UInt32, bytes))
    _floodfill_white_to_transparent!(buf, W, height)
    Cairo.mark_dirty(surf)   # we wrote pixels directly; force Cairo to re-read on next source use
    _CAR_NUMBER_CACHE[key] = surf
    return surf
end

# Cairo ARGB32 on little-endian: byte 0 = B, 1 = G, 2 = R, 3 = A.
# Threshold 0xF0 (240) tolerates JPG compression noise around the white bg.
function _floodfill_white_to_transparent!(px::AbstractVector{UInt32}, W::Int, H::Int;
                                          threshold::UInt8 = 0xF0)
    is_near_white(p::UInt32) =
        UInt8(p          & 0xFF) >= threshold &&
        UInt8((p >> 8)   & 0xFF) >= threshold &&
        UInt8((p >> 16)  & 0xFF) >= threshold

    n = W * H
    visited = falses(n)
    stack = Int[]
    sizehint!(stack, max(64, n ÷ 4))

    @inline function seed!(i::Int)
        @inbounds if !visited[i] && is_near_white(px[i])
            visited[i] = true
            push!(stack, i)
        end
    end

    for x in 1:W
        seed!(x)
        seed!((H - 1) * W + x)
    end
    for y in 1:H
        seed!((y - 1) * W + 1)
        seed!((y - 1) * W + W)
    end

    while !isempty(stack)
        i = pop!(stack)
        x = (i - 1) % W + 1
        y = (i - 1) ÷ W + 1
        x > 1 && seed!(i - 1)
        x < W && seed!(i + 1)
        y > 1 && seed!(i - W)
        y < H && seed!(i + W)
    end

    @inbounds for i in 1:n
        visited[i] && (px[i] = UInt32(0))
    end
    return px
end

"""
    build_channels_minimal(tel, lap_rows, ranges) -> Vector{ChannelTrace}

Throttle/brake/steering only, in that order. Same axis ranges and
formatters as the corresponding rows in `build_channels`.
"""
function build_channels_minimal(tel, lap_rows::UnitRange{Int}, ranges)
    throttle = Float64.(view(tel.throttle, lap_rows))
    brake    = Float64.(view(tel.brake,    lap_rows))
    steering = Float64.(view(tel.steering, lap_rows))

    fmt_int(v)   = (@sprintf("%d", round(Int, v)))
    fmt_steer(v) = (@sprintf("%+.1f", v))

    return ChannelTrace[
        _mk(:THROTTLE, throttle, ranges.throttle..., "%",   fmt_int),
        _mk(:BRAKE,    brake,    ranges.brake...,    "PSI", fmt_int),
        _mk(:STEERING, steering, ranges.steering...,  "°",  fmt_steer),
    ]
end

"""
    build_stats_minimal(tel, lap_rows, ranges) -> Vector{ChannelTrace}

MPH / RPM / GEAR for the right-hand stat column. Same formatters as in
the full template. Uses `ChannelTrace` for storage but only `.data` and
`.fmt` are read — `.norm` is unused for stat-only display.
"""
function build_stats_minimal(tel, lap_rows::UnitRange{Int}, ranges)
    speed = Float64.(view(tel.speed, lap_rows))
    rpm   = Float64.(view(tel.rpm,   lap_rows))
    gear  = Float64.(view(tel.gear,  lap_rows))

    mph_range = ranges.mph === :auto ?
        (max(0.0, floor((minimum(speed) - 10) / 10) * 10),
                  ceil((maximum(speed) + 10) / 10) * 10) :
        ranges.mph

    fmt_int(v)  = (@sprintf("%d", round(Int, v)))
    fmt_gear(v) = string(clamp(round(Int, v), 1, 5))

    return ChannelTrace[
        _mk(:MPH,  speed, mph_range...,   "", fmt_int),
        _mk(:RPM,  rpm,   ranges.rpm...,  "", fmt_int),
        _mk(:GEAR, gear,  ranges.gear..., "", fmt_gear),
    ]
end

# Effective trace width when the stat column is on: traces end earlier
# so the value column + stat column fit on the right.
_minimal_trace_w(layout::OverlayLayout) = layout.W - VAL_W_MINIMAL - STAT_W_MINIMAL
_minimal_val_x(layout::OverlayLayout)   = _minimal_trace_w(layout)
_minimal_stat_x(layout::OverlayLayout)  = _minimal_trace_w(layout) + VAL_W_MINIMAL

"""
    bake_static_surface_minimal(layout, channels, t_norm, track_bg,
                                driver, event, stats) -> CairoSurface

Like `bake_static_surface`, but the bottom strip is split into three
columns left→right: traces, THROTTLE/BRAKE/STEERING values, and a
boxed MPH/RPM/GEAR stat column on the right.
"""
function bake_static_surface_minimal(layout::OverlayLayout,
                                     channels::Vector{ChannelTrace},
                                     t_norm::Vector{Float64},
                                     track_surface::Union{Nothing,CairoSurface},
                                     driver_label::AbstractString,
                                     event_label::AbstractString,
                                     stats::Vector{ChannelTrace})
    surf = CairoARGBSurface(layout.W, layout.H)
    cr = CairoContext(surf)

    trace_w = _minimal_trace_w(layout)
    val_x   = _minimal_val_x(layout)
    stat_x  = _minimal_stat_x(layout)

    # leave video panel transparent: ARGB initialised to 0
    paint_rect!(cr, layout.vid_w, 0, layout.map_w, layout.top_h, colorant"#000000")
    # Telemetry strip background
    paint_rect!(cr, 0, layout.top_h, layout.W, layout.bot_h, colorant"#111111")
    # Value column (re-uses #111111, kept for intent)
    paint_rect!(cr, val_x, layout.top_h, VAL_W_MINIMAL, layout.bot_h, colorant"#111111")
    # Stat column slightly darker
    paint_rect!(cr, stat_x, layout.top_h, STAT_W_MINIMAL, layout.bot_h, colorant"#0a0a0a")

    # Track map
    if track_surface !== nothing
        margin = 10
        tw = layout.map_w - 2 * margin
        th = layout.top_h - 2 * margin
        sw = Float64(width(track_surface))
        sh = Float64(height(track_surface))
        sx = tw / sw; sy = th / sh
        save(cr)
        translate(cr, layout.vid_w + margin, margin)
        scale(cr, sx, sy)
        set_source_surface(cr, track_surface, 0, 0)
        paint(cr)
        restore(cr)
    end

    # Driver / event text (top-left of map panel)
    select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
    set_font_size(cr, 14)
    set_rgb!(cr, colorant"white")
    move_to(cr, layout.vid_w + 8, 18); show_text(cr, String(driver_label))
    set_font_size(cr, 11)
    set_rgb!(cr, colorant"#888888")
    move_to(cr, layout.vid_w + 8, 34); show_text(cr, String(event_label))

    # Traces
    nch = length(channels)
    row_h  = layout.bot_h / nch
    pad    = row_h * 0.075
    plot_h = row_h * 0.85

    npts = length(t_norm)
    xs = Vector{Float64}(undef, npts)
    ys = Vector{Float64}(undef, npts)

    for (i, ch) in enumerate(channels)
        y_bot_screen = layout.top_h + (i - 1) * row_h
        color = getproperty(CH_COLORS, ch.name)

        if i > 1
            set_rgb!(cr, colorant"#2a2a2a")
            set_line_width(cr, 0.8)
            move_to(cr, 0, y_bot_screen); line_to(cr, trace_w, y_bot_screen)
            stroke(cr)
        end

        @inbounds for k in 1:npts
            xs[k] = t_norm[k] * trace_w
            ys[k] = y_bot_screen + pad + (1 - ch.norm[k]) * plot_h
        end
        stroke_polyline!(cr, xs, ys, color, 1.2)

        set_font_size(cr, 10)
        set_rgb!(cr, color)
        select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
        label = ch.unit == "" ? String(ch.name) : "$(ch.name) $(ch.unit)"
        move_to(cr, 6, y_bot_screen + row_h / 2 + 3); show_text(cr, label)

        set_font_size(cr, 9)
        set_rgb!(cr, colorant"#aaaaaa")
        move_to(cr, val_x + 10, y_bot_screen + 12); show_text(cr, String(ch.name))
    end

    # Stat boxes + labels (right column)
    nst = length(stats)
    if nst > 0
        cell_h = layout.bot_h / nst
        pad_x  = 8.0
        pad_y  = 8.0
        for (i, st) in enumerate(stats)
            bx = stat_x + pad_x
            by = layout.top_h + (i - 1) * cell_h + pad_y
            bw = STAT_W_MINIMAL - 2 * pad_x
            bh = cell_h - 2 * pad_y
            color = getproperty(CH_COLORS, st.name)

            set_rgb!(cr, color)
            set_line_width(cr, 1.5)
            rectangle(cr, bx, by, bw, bh)
            stroke(cr)

            select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
            set_font_size(cr, 11)
            set_rgb!(cr, color)
            move_to(cr, bx + 6, by + 14)
            show_text(cr, String(st.name))
        end
    end

    return surf
end

"""
    draw_dynamic_minimal!(cr, layout, channels, t_q, cur_vals, cur_norms,
                          tm, cur_dist, lap_time_s, stats, cur_stat_vals)

Per-frame draw for the minimal template. Adds the current MPH/RPM/GEAR
numbers centered inside the stat-column boxes, plus the same cursor,
trace dots, value-column numbers, track-map dot, and lap timer the full
template renders.
"""
function draw_dynamic_minimal!(cr, layout::OverlayLayout,
                               channels::Vector{ChannelTrace},
                               t_q::Float64,
                               cur_vals::Vector{Float64},
                               cur_norms::Vector{Float64},
                               tm,
                               cur_dist::Float64,
                               lap_time_s::Float64,
                               stats::Vector{ChannelTrace},
                               cur_stat_vals::Vector{Float64},
                               car_graphic::Union{Nothing,Cairo.CairoSurface} = nothing)
    trace_w = _minimal_trace_w(layout)
    val_x   = _minimal_val_x(layout)
    stat_x  = _minimal_stat_x(layout)

    nch = length(channels)
    row_h  = layout.bot_h / nch
    pad    = row_h * 0.075
    plot_h = row_h * 0.85
    cursor_x = t_q * trace_w

    set_rgba!(cr, colorant"#ff2222", 0.9)
    set_line_width(cr, 1.5)
    move_to(cr, cursor_x, layout.top_h)
    line_to(cr, cursor_x, layout.H)
    stroke(cr)

    for (i, ch) in enumerate(channels)
        y_bot = layout.top_h + (i - 1) * row_h
        color = getproperty(CH_COLORS, ch.name)
        cy = y_bot + pad + (1 - clamp(cur_norms[i], 0.0, 1.0)) * plot_h
        set_rgb!(cr, color)
        arc(cr, cursor_x, cy, 4.0, 0, 2π); fill(cr)

        set_rgb!(cr, color)
        select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
        set_font_size(cr, 22)
        move_to(cr, val_x + 14, y_bot + row_h * 0.65)
        show_text(cr, ch.fmt(cur_vals[i]))
    end

    # Stat box values
    nst = length(stats)
    if nst > 0 && length(cur_stat_vals) == nst
        cell_h = layout.bot_h / nst
        pad_x  = 8.0
        pad_y  = 8.0
        select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
        set_font_size(cr, 32)
        for (i, st) in enumerate(stats)
            bx = stat_x + pad_x
            by = layout.top_h + (i - 1) * cell_h + pad_y
            bw = STAT_W_MINIMAL - 2 * pad_x
            bh = cell_h - 2 * pad_y
            color = getproperty(CH_COLORS, st.name)
            text = st.fmt(cur_stat_vals[i])

            te = text_extents(cr, text)
            # te = [x_bearing, y_bearing, width, height, x_advance, y_advance]
            tx = bx + bw / 2 - te[3] / 2 - te[1]
            # vertically center below the small top label (label band ~ 20px)
            label_band = 22.0
            avail_h = bh - label_band
            ty = by + label_band + avail_h / 2 + te[4] / 2 + te[2] / 2
            set_rgb!(cr, color)
            move_to(cr, tx, ty)
            show_text(cr, text)
        end
    end

    if tm !== nothing
        margin = 10
        tw = layout.map_w - 2 * margin
        th = layout.top_h - 2 * margin
        inset = 0.05
        xn, yn = dist_to_map_norm(cur_dist, tm)
        px = layout.vid_w + margin + (inset + xn * (1 - 2 * inset)) * tw
        py = margin + (1 - (inset + yn * (1 - 2 * inset))) * th
        if car_graphic !== nothing
            gw = Float64(Cairo.width(car_graphic))
            gh = Float64(Cairo.height(car_graphic))
            save(cr)
            translate(cr, px - gw / 2, py - gh / 2)
            set_source_surface(cr, car_graphic, 0, 0)
            paint(cr)
            restore(cr)
        else
            set_rgb!(cr, colorant"#ffee00")
            arc(cr, px, py, 6.0, 0, 2π); fill(cr)
            set_rgb!(cr, colorant"white")
            set_line_width(cr, 1.0)
            arc(cr, px, py, 6.0, 0, 2π); stroke(cr)
        end
    end

    select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
    set_font_size(cr, 14)
    set_rgb!(cr, colorant"#ffee00")
    mins = floor(Int, lap_time_s / 60)
    secs = lap_time_s - 60 * mins
    move_to(cr, layout.vid_w + 8, 52)
    show_text(cr, @sprintf("%d:%05.2f", mins, secs))
end
