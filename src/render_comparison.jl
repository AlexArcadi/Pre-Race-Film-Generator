# Side-by-side comparison template: two videos top, 4-channel telemetry strip
# below with both drivers' traces overlaid per row (faster = green, slower =
# red). Pipeline.jl's `generate_comparison_video` sets up the lap windows and
# ffmpeg's 3-input filter graph (videoA, videoB, overlay-frames); this file
# owns the Cairo bake / draw pair and the per-driver data prep.
#
# Sync model: natural-time, clip ends when faster driver finishes. Both
# drivers' telemetry is clipped to `faster_dur` so the trace x-axis represents
# REAL elapsed seconds since each lap began — at every x the two drivers were
# the same number of seconds into their respective laps.

const CH_ORDER_COMPARISON  = (:THROTTLE, :BRAKE, :STEERING, :GEAR)
const COMPARISON_VAL_W     = 220     # right column total (split into two sub-cols)
const COMPARISON_VID_FRAC  = 0.5     # top region split exactly in half
const COMPARISON_TOP_FRAC  = 0.55    # video region vs telemetry strip
const STAT_BOX_W           = 220     # per-video MPH/RPM box width
const STAT_BOX_H           = 64      # per-video MPH/RPM box height
const COMPARISON_FASTER_C  = colorant"#22ee44"
const COMPARISON_SLOWER_C  = colorant"#ff3344"

# Comparison-specific layout. Keeps the same OverlayLayout fields the existing
# render helpers expect (so blit_surface!, paint_rect!, etc. work unchanged),
# but the `vid_w` field now means "per-side video width" — both halves of the
# top region — and `val_w` is the dual-driver value column.
function comparison_layout(W::Int, H::Int)
    top = round(Int, H * COMPARISON_TOP_FRAC)
    bot = H - top
    vw  = round(Int, W * COMPARISON_VID_FRAC)   # left video width = right video width
    return OverlayLayout(W, H, top, bot, vw, W - vw, COMPARISON_VAL_W,
                         W - COMPARISON_VAL_W)
end

"""
    faster_driver(lap_dur_A, lap_dur_B) -> :A or :B

The one with the shorter lap. Ties go to A.
"""
faster_driver(a::Real, b::Real) = a <= b ? :A : :B

"""
    clip_lap_rows(lap_rows, lap_dur, clip_dur) -> UnitRange

Keep the leading samples of `lap_rows` covering at most `clip_dur` seconds of
the lap. Used for the slower driver to drop their post-faster-finish tail.
"""
function clip_lap_rows(lap_rows::UnitRange{Int}, lap_dur::Real, clip_dur::Real)
    n = length(lap_rows)
    n <= 1 && return lap_rows
    keep = min(n, round(Int, clip_dur / lap_dur * (n - 1)) + 1)
    return lap_rows[1]:(lap_rows[1] + keep - 1)
end

"""
    build_comparison_channels(tel, lap_rows, ranges) -> Vector{ChannelTrace}

Throttle / Brake / Steering / Gear in CH_ORDER_COMPARISON. Same axis ranges
and formatters as the full template. Call once per driver with their
already-clipped `lap_rows`.
"""
function build_comparison_channels(tel, lap_rows::UnitRange{Int}, ranges)
    throttle = Float64.(view(tel.throttle, lap_rows))
    brake    = Float64.(view(tel.brake,    lap_rows))
    steering = Float64.(view(tel.steering, lap_rows))
    gear     = Float64.(view(tel.gear,     lap_rows))

    fmt_int(v)   = (@sprintf("%d", round(Int, v)))
    fmt_steer(v) = (@sprintf("%+.1f", v))
    fmt_gear(v)  = string(clamp(round(Int, v), 1, 5))

    return ChannelTrace[
        _mk(:THROTTLE, throttle, ranges.throttle..., "%",   fmt_int),
        _mk(:BRAKE,    brake,    ranges.brake...,    "PSI", fmt_int),
        _mk(:STEERING, steering, ranges.steering...,  "°",  fmt_steer),
        _mk(:GEAR,     gear,     ranges.gear...,     "",    fmt_gear),
    ]
end

"""
    build_comparison_stats(tel, lap_rows, ranges) -> Vector{ChannelTrace}

MPH / RPM for each video's overlaid stat box. Only `.data` and `.fmt` are
read; `.norm` is unused.
"""
function build_comparison_stats(tel, lap_rows::UnitRange{Int}, ranges)
    speed = Float64.(view(tel.speed, lap_rows))
    rpm   = Float64.(view(tel.rpm,   lap_rows))
    mph_range = ranges.mph === :auto ?
        (max(0.0, floor((minimum(speed) - 10) / 10) * 10),
                  ceil((maximum(speed) + 10) / 10) * 10) :
        ranges.mph
    fmt_int(v) = (@sprintf("%d", round(Int, v)))
    return ChannelTrace[
        _mk(:MPH, speed, mph_range...,  "", fmt_int),
        _mk(:RPM, rpm,   ranges.rpm..., "", fmt_int),
    ]
end

# Geometry helpers — left/right video panes and per-driver value sub-columns.
_comp_left_vid_x(layout::OverlayLayout)  = 0
_comp_right_vid_x(layout::OverlayLayout) = layout.vid_w
_comp_val_left_x(layout::OverlayLayout)  = layout.trace_w
_comp_val_right_x(layout::OverlayLayout) = layout.trace_w + COMPARISON_VAL_W ÷ 2

"""
    bake_static_surface_comparison(layout, channels_A, channels_B, t_norm_A,
                                   t_norm_B, faster_id, driver_A_label,
                                   driver_B_label, event_label) -> CairoSurface

Pre-render the parts of the comparison overlay that don't change frame to
frame: telemetry strip background, value-column captions, channel labels,
and both drivers' full lap traces overlaid per row (green = faster, red =
slower). The video panes themselves are left transparent so ffmpeg can
composite the two scaled source videos underneath.
"""
function bake_static_surface_comparison(layout::OverlayLayout,
                                        channels_A::Vector{ChannelTrace},
                                        channels_B::Vector{ChannelTrace},
                                        t_norm_A::Vector{Float64},
                                        t_norm_B::Vector{Float64},
                                        faster_id::Symbol,
                                        carA::Integer, carB::Integer,
                                        driver_A_label::AbstractString,
                                        driver_B_label::AbstractString,
                                        event_label::AbstractString)
    surf = CairoARGBSurface(layout.W, layout.H)
    cr   = CairoContext(surf)

    color_A = faster_id === :A ? COMPARISON_FASTER_C : COMPARISON_SLOWER_C
    color_B = faster_id === :B ? COMPARISON_FASTER_C : COMPARISON_SLOWER_C
    val_left_x  = _comp_val_left_x(layout)
    val_right_x = _comp_val_right_x(layout)

    # Telemetry strip background
    paint_rect!(cr, 0, layout.top_h, layout.W, layout.bot_h, colorant"#111111")
    # Value column slightly darker so the colored numbers pop
    paint_rect!(cr, val_left_x, layout.top_h, COMPARISON_VAL_W, layout.bot_h,
                colorant"#0a0a0a")
    # Thin divider between the two driver sub-columns
    set_rgb!(cr, colorant"#2a2a2a")
    set_line_width(cr, 1.0)
    move_to(cr, val_right_x, layout.top_h); line_to(cr, val_right_x, layout.H)
    stroke(cr)

    # Driver / event header — top-left of each video pane. Painted onto a
    # translucent dark backdrop so the text reads cleanly against bright sky,
    # asphalt, or whatever else the video is showing through.
    _draw_corner_header(cr, 0,            driver_A_label, event_label, color_A)
    _draw_corner_header(cr, layout.vid_w, driver_B_label, event_label, color_B)

    # Traces: both drivers, every row, color overridden to faster/slower
    nch    = length(channels_A)
    row_h  = layout.bot_h / nch
    pad    = row_h * 0.075
    plot_h = row_h * 0.85

    for (i, (ch_A, ch_B)) in enumerate(zip(channels_A, channels_B))
        y_bot = layout.top_h + (i - 1) * row_h

        # Row divider
        if i > 1
            set_rgb!(cr, colorant"#2a2a2a")
            set_line_width(cr, 0.8)
            move_to(cr, 0, y_bot); line_to(cr, layout.trace_w, y_bot)
            stroke(cr)
        end

        # Driver A trace
        nA = length(ch_A.norm)
        xs = Vector{Float64}(undef, nA); ys = Vector{Float64}(undef, nA)
        @inbounds for k in 1:nA
            xs[k] = t_norm_A[k] * layout.trace_w
            ys[k] = y_bot + pad + (1 - ch_A.norm[k]) * plot_h
        end
        stroke_polyline!(cr, xs, ys, color_A, 1.2)

        # Driver B trace
        nB = length(ch_B.norm)
        xs = Vector{Float64}(undef, nB); ys = Vector{Float64}(undef, nB)
        @inbounds for k in 1:nB
            xs[k] = t_norm_B[k] * layout.trace_w
            ys[k] = y_bot + pad + (1 - ch_B.norm[k]) * plot_h
        end
        stroke_polyline!(cr, xs, ys, color_B, 1.2)

        # Channel label (left edge of trace strip). Pinned to the BOTTOM of
        # the row — the previous mid-row placement put the text ascender close
        # enough to the divider above that it clipped on some fonts.
        select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
        set_font_size(cr, 10)
        set_rgb!(cr, colorant"#dddddd")
        label = ch_A.unit == "" ? String(ch_A.name) : "$(ch_A.name) $(ch_A.unit)"
        move_to(cr, 6, y_bot + row_h - 8); show_text(cr, label)

        # Value sub-column captions — small car-number labels above each
        # driver's number, color-coded the same as their trace.
        set_font_size(cr, 9)
        set_rgb!(cr, color_A)
        move_to(cr, val_left_x  + 8, y_bot + 18); show_text(cr, "#$(carA) Car")
        set_rgb!(cr, color_B)
        move_to(cr, val_right_x + 8, y_bot + 18); show_text(cr, "#$(carB) Car")
    end

    return surf
end

"""
    draw_dynamic_comparison!(cr, layout, channels_A, channels_B, stats_A,
                             stats_B, faster_id, t_q, vals_A, vals_B,
                             norms_A, norms_B, stat_vals_A, stat_vals_B,
                             lap_time_s)

Per-frame overlay: vertical cursor across the telemetry strip, current-value
dots on both traces in each row, color-coded value numbers in the right
sub-columns, and the MPH/RPM stat boxes on each video pane (bottom-left).
"""
function draw_dynamic_comparison!(cr, layout::OverlayLayout,
                                  channels_A::Vector{ChannelTrace},
                                  channels_B::Vector{ChannelTrace},
                                  stats_A::Vector{ChannelTrace},
                                  stats_B::Vector{ChannelTrace},
                                  faster_id::Symbol,
                                  t_q::Float64,
                                  vals_A::Vector{Float64},
                                  vals_B::Vector{Float64},
                                  norms_A::Vector{Float64},
                                  norms_B::Vector{Float64},
                                  stat_vals_A::Vector{Float64},
                                  stat_vals_B::Vector{Float64},
                                  lap_time_s::Float64)
    color_A = faster_id === :A ? COMPARISON_FASTER_C : COMPARISON_SLOWER_C
    color_B = faster_id === :B ? COMPARISON_FASTER_C : COMPARISON_SLOWER_C
    val_left_x  = _comp_val_left_x(layout)
    val_right_x = _comp_val_right_x(layout)
    sub_w       = COMPARISON_VAL_W ÷ 2

    nch    = length(channels_A)
    row_h  = layout.bot_h / nch
    pad    = row_h * 0.075
    plot_h = row_h * 0.85
    cursor_x = t_q * layout.trace_w

    # Cursor
    set_rgba!(cr, colorant"#ffffff", 0.7)
    set_line_width(cr, 1.5)
    move_to(cr, cursor_x, layout.top_h); line_to(cr, cursor_x, layout.H)
    stroke(cr)

    # Per-row dots + value numbers
    select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
    for i in 1:nch
        y_bot = layout.top_h + (i - 1) * row_h
        ch_A = channels_A[i]; ch_B = channels_B[i]

        # Trace dots
        cy_A = y_bot + pad + (1 - clamp(norms_A[i], 0.0, 1.0)) * plot_h
        cy_B = y_bot + pad + (1 - clamp(norms_B[i], 0.0, 1.0)) * plot_h
        set_rgb!(cr, color_A); arc(cr, cursor_x, cy_A, 4.0, 0, 2π); fill(cr)
        set_rgb!(cr, color_B); arc(cr, cursor_x, cy_B, 4.0, 0, 2π); fill(cr)

        # Value numbers — centered in each sub-column
        set_font_size(cr, 20)
        set_rgb!(cr, color_A)
        tA = ch_A.fmt(vals_A[i])
        tA_ext = text_extents(cr, tA)
        tx = val_left_x + sub_w / 2 - tA_ext[3] / 2 - tA_ext[1]
        move_to(cr, tx, y_bot + row_h * 0.7); show_text(cr, tA)

        set_rgb!(cr, color_B)
        tB = ch_B.fmt(vals_B[i])
        tB_ext = text_extents(cr, tB)
        tx = val_right_x + sub_w / 2 - tB_ext[3] / 2 - tB_ext[1]
        move_to(cr, tx, y_bot + row_h * 0.7); show_text(cr, tB)
    end

    # MPH/RPM stat boxes — one per video, bottom-left of each video pane
    _draw_stat_box(cr, 0,            layout.top_h - STAT_BOX_H, color_A, stats_A, stat_vals_A)
    _draw_stat_box(cr, layout.vid_w, layout.top_h - STAT_BOX_H, color_B, stats_B, stat_vals_B)

    # Lap timer — small, centered between the two videos at the bottom of the top region
    select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
    set_font_size(cr, 14)
    set_rgb!(cr, colorant"#ffee00")
    mins = floor(Int, lap_time_s / 60); secs = lap_time_s - 60 * mins
    txt = @sprintf("%d:%05.2f", mins, secs)
    te = text_extents(cr, txt)
    move_to(cr, layout.W / 2 - te[3] / 2 - te[1], layout.top_h - 8); show_text(cr, txt)
end

# Translucent rounded-corner box with MPH / RPM labels and big numbers, painted
# over the video so the engine instruments stay readable through the overlay.
function _draw_stat_box(cr, x::Real, y::Real, color::Colorant,
                        stats::Vector{ChannelTrace}, vals::Vector{Float64})
    paint_rect!(cr, x + 8, y + 6, STAT_BOX_W - 16, STAT_BOX_H - 12,
                colorant"#000000"; alpha = 0.62)
    select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)
    for (i, st) in enumerate(stats)
        col_x = x + 14 + (i - 1) * ((STAT_BOX_W - 28) / length(stats))
        col_w = (STAT_BOX_W - 28) / length(stats)
        set_font_size(cr, 10)
        set_rgb!(cr, colorant"#aaaaaa")
        move_to(cr, col_x, y + 22); show_text(cr, String(st.name))
        set_font_size(cr, 26)
        set_rgb!(cr, color)
        text = st.fmt(vals[i])
        te   = text_extents(cr, text)
        tx   = col_x + col_w / 2 - te[3] / 2 - te[1]
        move_to(cr, tx, y + STAT_BOX_H - 14); show_text(cr, text)
    end
end

# Driver / event header for the top-left of one video pane. Translucent dark
# backdrop sized to the longer of the two strings, so it reads cleanly against
# any video frame underneath. `x` is the left edge of the video pane.
function _draw_corner_header(cr, x::Real,
                             driver_label::AbstractString,
                             event_label::AbstractString,
                             driver_color::Colorant)
    pad_x = 8
    pad_y = 6
    select_font_face(cr, "monospace", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_BOLD)

    # Measure both strings at their final sizes so the backdrop fits exactly.
    set_font_size(cr, 18)
    de = text_extents(cr, String(driver_label))   # driver line
    set_font_size(cr, 12)
    ee = text_extents(cr, String(event_label))    # event line

    box_w = max(de[3], ee[3]) + 2 * pad_x
    box_h = de[4] + ee[4] + 3 * pad_y
    paint_rect!(cr, x + 6, 6, box_w, box_h, colorant"#000000"; alpha = 0.65)

    # Driver number/name — bigger, colored
    set_font_size(cr, 18)
    set_rgb!(cr, driver_color)
    move_to(cr, x + 6 + pad_x, 6 + pad_y + de[4]); show_text(cr, String(driver_label))

    # Event/track — smaller, light gray
    set_font_size(cr, 12)
    set_rgb!(cr, colorant"#eeeeee")
    move_to(cr, x + 6 + pad_x, 6 + pad_y + de[4] + pad_y + ee[4]); show_text(cr, String(event_label))
end
