using Cairo
using Colors

const CH_COLORS = (
    MPH      = colorant"#00bfff",
    RPM      = colorant"#ffaa00",
    GEAR     = colorant"#cc88ff",
    THROTTLE = colorant"#00ee44",
    BRAKE    = colorant"#ff3333",
    STEERING = colorant"#ff88cc",
)
const CH_ORDER = (:MPH, :RPM, :GEAR, :THROTTLE, :BRAKE, :STEERING)

struct ChannelTrace
    name::Symbol
    data::Vector{Float64}    # raw values, one per telemetry sample within the lap
    norm::Vector{Float64}    # 0..1 mapped against (ymin, ymax)
    ymin::Float64
    ymax::Float64
    unit::String
    fmt::Function            # value -> String for the value column
end

struct OverlayLayout
    W::Int
    H::Int
    top_h::Int               # video / map row height
    bot_h::Int               # telemetry strip height
    vid_w::Int               # video panel width
    map_w::Int               # track-map panel width
    val_w::Int               # value column width
    trace_w::Int             # = W - val_w
end

function OverlayLayout(; W = 1280, H = 720, top_frac = 0.55,
                         vid_frac = 0.62, val_w = 160)
    top = round(Int, H * top_frac)
    bot = H - top
    vw  = round(Int, W * vid_frac)
    return OverlayLayout(W, H, top, bot, vw, W - vw, val_w, W - val_w)
end

set_rgb!(cr, c::Colorant) = set_source_rgb(cr, red(c), green(c), blue(c))
set_rgba!(cr, c::Colorant, a) = set_source_rgba(cr, red(c), green(c), blue(c), a)

function paint_rect!(cr, x, y, w, h, color::Colorant; alpha = 1.0)
    set_rgba!(cr, color, alpha)
    rectangle(cr, x, y, w, h)
    fill(cr)
end

function stroke_polyline!(cr, xs, ys, color::Colorant, lw)
    isempty(xs) && return
    set_rgb!(cr, color)
    set_line_width(cr, lw)
    move_to(cr, xs[1], ys[1])
    @inbounds for i in 2:length(xs)
        line_to(cr, xs[i], ys[i])
    end
    stroke(cr)
end

function build_channels(tel, lap_rows::UnitRange{Int}, ranges)
    speed    = Float64.(view(tel.speed,    lap_rows))
    rpm      = Float64.(view(tel.rpm,      lap_rows))
    gear     = Float64.(view(tel.gear,     lap_rows))
    throttle = Float64.(view(tel.throttle, lap_rows))
    brake    = Float64.(view(tel.brake,    lap_rows))
    steering = Float64.(view(tel.steering, lap_rows))

    mph_range = ranges.mph === :auto ?
        (max(0.0, floor((minimum(speed) - 10) / 10) * 10),
                  ceil((maximum(speed) + 10) / 10) * 10) :
        ranges.mph

    fmt_int(v) = (@sprintf("%d", round(Int, v)))
    fmt_gear(v) = string(clamp(round(Int, v), 1, 5))
    fmt_steer(v) = (@sprintf("%+.1f", v))

    channels = ChannelTrace[
        _mk(:MPH,      speed,    mph_range...,      "",   fmt_int),
        _mk(:RPM,      rpm,      ranges.rpm...,     "",   fmt_int),
        _mk(:GEAR,     gear,     ranges.gear...,    "",   fmt_gear),
        _mk(:THROTTLE, throttle, ranges.throttle..., "%",  fmt_int),
        _mk(:BRAKE,    brake,    ranges.brake...,   "PSI",fmt_int),
        _mk(:STEERING, steering, ranges.steering..., "°", fmt_steer),
    ]
    return channels
end

function _mk(name, data, ymin, ymax, unit, fmt)
    norm = clamp.((data .- ymin) ./ (ymax - ymin), 0.0, 1.0)
    ChannelTrace(name, data, norm, ymin, ymax, unit, fmt)
end

"""
    bake_static_surface(layout, channels, t_norm, track_bg, driver, event) -> CairoSurface

Pre-render the parts of the overlay that don't change between frames:
- black backdrop for the map and telemetry strip
- track-map background (if present)
- the six full traces
- channel labels + value-column captions
- driver/event strings
"""
function bake_static_surface(layout::OverlayLayout,
                             channels::Vector{ChannelTrace},
                             t_norm::Vector{Float64},
                             track_surface::Union{Nothing,CairoSurface},
                             driver_label::AbstractString,
                             event_label::AbstractString)
    surf = CairoARGBSurface(layout.W, layout.H)
    cr = CairoContext(surf)

    # leave video panel transparent: ARGB initialised to 0
    # Map panel background
    paint_rect!(cr, layout.vid_w, 0, layout.map_w, layout.top_h, colorant"#000000")
    # Telemetry strip background
    paint_rect!(cr, 0, layout.top_h, layout.W, layout.bot_h, colorant"#111111")
    # Value column slightly distinct
    paint_rect!(cr, layout.trace_w, layout.top_h, layout.val_w, layout.bot_h, colorant"#111111")

    # Track map
    if track_surface !== nothing
        # Fit inside the map panel with a small margin
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
    row_h = layout.bot_h / nch
    pad   = row_h * 0.075
    plot_h = row_h * 0.85

    npts = length(t_norm)
    xs = Vector{Float64}(undef, npts)
    ys = Vector{Float64}(undef, npts)

    for (i, ch) in enumerate(channels)
        y_bot_screen = layout.top_h + (i - 1) * row_h
        color = getproperty(CH_COLORS, ch.name)

        # Divider between rows
        if i > 1
            set_rgb!(cr, colorant"#2a2a2a")
            set_line_width(cr, 0.8)
            move_to(cr, 0, y_bot_screen); line_to(cr, layout.trace_w, y_bot_screen)
            stroke(cr)
        end

        @inbounds for k in 1:npts
            xs[k] = t_norm[k] * layout.trace_w
            ys[k] = y_bot_screen + pad + (1 - ch.norm[k]) * plot_h
        end
        stroke_polyline!(cr, xs, ys, color, 1.2)

        # Channel label, value caption
        set_font_size(cr, 10)
        set_rgb!(cr, color)
        select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
        label = ch.unit == "" ? String(ch.name) : "$(ch.name) $(ch.unit)"
        move_to(cr, 6, y_bot_screen + row_h / 2 + 3); show_text(cr, label)

        # value column caption
        set_font_size(cr, 9)
        set_rgb!(cr, colorant"#aaaaaa")
        move_to(cr, layout.trace_w + 10, y_bot_screen + 12); show_text(cr, String(ch.name))
    end

    return surf
end

"""
    bake_track_background(tm, w, h) -> CairoSurface

Pre-render the static track outline + S/F marker. The dynamic position
marker is drawn per-frame on top of this.
"""
function bake_track_background(tm, w::Int, h::Int)
    surf = CairoARGBSurface(w, h)
    cr = CairoContext(surf)
    paint_rect!(cr, 0, 0, w, h, colorant"black")

    # Map normalised x/y into the surface with a 5% inset
    inset = 0.05
    map_x(xn) = (inset + xn * (1 - 2 * inset)) * w
    map_y(yn) = (1 - (inset + yn * (1 - 2 * inset))) * h

    set_rgb!(cr, colorant"#666666")
    set_line_width(cr, 2.5)
    set_line_join(cr, Cairo.CAIRO_LINE_JOIN_ROUND)
    set_line_cap(cr, Cairo.CAIRO_LINE_CAP_ROUND)
    move_to(cr, map_x(tm.x_norm[1]), map_y(tm.y_norm[1]))
    for i in 2:length(tm.x_norm)
        line_to(cr, map_x(tm.x_norm[i]), map_y(tm.y_norm[i]))
    end
    stroke(cr)

    # S/F marker
    set_rgb!(cr, colorant"#ffee00")
    rectangle(cr, map_x(tm.x_norm[1]) - 4, map_y(tm.y_norm[1]) - 4, 8, 8)
    fill(cr)

    return surf
end

"""
    draw_dynamic!(cr, layout, channels, t_norm, t_q, cur_vals,
                  tm, cur_dist, lap_time_s)

Draw the per-frame elements on top of the baked static surface. `cr` is
expected to be writing into a fresh surface that was just initialised by
copying the static surface.
"""
function draw_dynamic!(cr, layout::OverlayLayout,
                       channels::Vector{ChannelTrace},
                       t_q::Float64,
                       cur_vals::Vector{Float64},
                       cur_norms::Vector{Float64},
                       tm,                       # ::Union{Nothing,TrackMap}
                       cur_dist::Float64,
                       lap_time_s::Float64)
    nch = length(channels)
    row_h = layout.bot_h / nch
    pad   = row_h * 0.075
    plot_h = row_h * 0.85
    cursor_x = t_q * layout.trace_w

    # Vertical red cursor across the whole telemetry strip
    set_rgba!(cr, colorant"#ff2222", 0.9)
    set_line_width(cr, 1.5)
    move_to(cr, cursor_x, layout.top_h)
    line_to(cr, cursor_x, layout.H)
    stroke(cr)

    for (i, ch) in enumerate(channels)
        y_bot = layout.top_h + (i - 1) * row_h
        color = getproperty(CH_COLORS, ch.name)
        cy = y_bot + pad + (1 - clamp(cur_norms[i], 0.0, 1.0)) * plot_h
        # Trace dot
        set_rgb!(cr, color)
        arc(cr, cursor_x, cy, 4.0, 0, 2π); fill(cr)

        # Value column number
        set_rgb!(cr, color)
        select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
        set_font_size(cr, 22)
        move_to(cr, layout.trace_w + 14, y_bot + row_h * 0.65)
        show_text(cr, ch.fmt(cur_vals[i]))
    end

    # Track-map dot
    if tm !== nothing
        margin = 10
        tw = layout.map_w - 2 * margin
        th = layout.top_h - 2 * margin
        inset = 0.05
        xn, yn = dist_to_map_norm(cur_dist, tm)
        px = layout.vid_w + margin + (inset + xn * (1 - 2 * inset)) * tw
        py = margin + (1 - (inset + yn * (1 - 2 * inset))) * th
        set_rgb!(cr, colorant"#ffee00")
        arc(cr, px, py, 6.0, 0, 2π); fill(cr)
        set_rgb!(cr, colorant"white")
        set_line_width(cr, 1.0)
        arc(cr, px, py, 6.0, 0, 2π); stroke(cr)
    end

    # Lap timer in top-right of the map panel
    select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
    set_font_size(cr, 14)
    set_rgb!(cr, colorant"#ffee00")
    mins = floor(Int, lap_time_s / 60)
    secs = lap_time_s - 60 * mins
    move_to(cr, layout.vid_w + 8, 52)
    show_text(cr, @sprintf("%d:%05.2f", mins, secs))
end

"""
    blit_surface!(dst::CairoSurface, src::CairoSurface)

Copy pixels from `src` to `dst` (same size). Equivalent to memcpy of the
backing buffer.
"""
function blit_surface!(dst::CairoSurface, src::CairoSurface)
    dst_buf = argbuffer(dst)
    src_buf = argbuffer(src)
    copyto!(dst_buf, src_buf)
    return dst
end

"""
    argbuffer(surf::CairoSurface) -> Vector{UInt32}

Wrap the surface's backing pixel buffer (one UInt32 per ARGB32 pixel) without
copying. Writing this vector to an IO writes raw little-endian BGRA bytes,
which matches ffmpeg's `-pix_fmt bgra` input format on little-endian hosts.
"""
function argbuffer(surf::CairoSurface)
    ptr = Cairo.image_surface_get_data(surf)
    n = Int(width(surf)) * Int(height(surf))
    return unsafe_wrap(Array, ptr, (n,); own = false)
end
