# Minimal 3-channel overlay template: THROTTLE / BRAKE / STEERING.
#
# Pairs with render.jl. Every helper there (OverlayLayout, ChannelTrace,
# _mk, CH_COLORS, bake_static_surface, bake_track_background,
# draw_dynamic!, blit_surface!, argbuffer) is already channel-count
# agnostic — it iterates whichever channels you hand it. So this file
# only needs to supply a different `build_channels`.
#
# Select via `template = :minimal` on generate_lap_video / render_lap /
# process. The full 6-channel layout remains the default.

const CH_ORDER_MINIMAL = (:THROTTLE, :BRAKE, :STEERING)

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
