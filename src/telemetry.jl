using Arrow
using DataFrames
using Tables
using Statistics

const CHANNEL_BINDING = (
    time     = :Time,
    lap      = :lap,
    speed    = (:OTD_Conv_Speed, :VectorGPS_Speed, :ChassisVelGPS),
    rpm      = :EngineRotVel,
    gear     = :DriverGearNumber,
    throttle = :EngineThrottlePosition,
    brake    = :BrakePressFront,
    steering = :DriverSteeringAngle,
    lap_frac = (:OTD_Conv_LapFraction, :VectorGPS_LapFrac),
    loop     = :loop_currently_on,
)

_resolve_col(cols, sym::Symbol) = sym in cols ? sym : nothing
function _resolve_col(cols, syms::Tuple)
    for s in syms
        s in cols && return s
    end
    return nothing
end

default_ranges() = (
    mph      = :auto,
    rpm      = (6000.0, 9500.0),
    gear     = (0.5, 5.5),
    throttle = (0.0, 100.0),
    brake    = (0.0, 900.0),
    steering = (-185.0, 65.0),
)

"""
    load_telemetry(arrow_path) -> NamedTuple

Memory-map the arrow file and return the bound channels plus the full Arrow.Table
so callers can read additional columns if they want.
"""
function load_telemetry(arrow_path::AbstractString)
    tbl = Arrow.Table(arrow_path)
    cols = Set(Tables.columnnames(tbl))
    ks  = keys(CHANNEL_BINDING)
    syms = ntuple(length(ks)) do i
        k = ks[i]; v = CHANNEL_BINDING[k]
        r = _resolve_col(cols, v)
        r === nothing && error("Required column for $k not found in $arrow_path (tried $v)")
        r
    end
    resolved = NamedTuple{ks}(syms)
    return (
        table    = tbl,
        time     = getproperty(tbl, resolved.time),
        lap      = getproperty(tbl, resolved.lap),
        speed    = getproperty(tbl, resolved.speed),
        rpm      = getproperty(tbl, resolved.rpm),
        gear     = getproperty(tbl, resolved.gear),
        throttle = getproperty(tbl, resolved.throttle),
        brake    = getproperty(tbl, resolved.brake),
        steering = getproperty(tbl, resolved.steering),
        lap_frac = getproperty(tbl, resolved.lap_frac),
        loop     = getproperty(tbl, resolved.loop),
    )
end

"""
    load_channels(arrow_path, channels...) -> Tuple of Vector{Float64}

Read the named channels as `Float64` and drop every row where ANY of them is
non-finite (one shared keep-mask), so callers get clean, equal-length signals.
This is the single NaN guard for telemetry — downstream code assumes finite input.
"""
function load_channels(arrow_path::AbstractString, channels::Symbol...)
    tbl  = Arrow.Table(arrow_path)
    cols = [Float64.(getproperty(tbl, c)) for c in channels]
    keep = trues(length(first(cols)))
    for c in cols
        keep .&= isfinite.(c)
    end
    return Tuple(c[keep] for c in cols)
end

"""
    detect_laps(arrow_path; min_seconds=20.0, drop_partial=true) -> DataFrame

Detect contiguous runs per `lap` value and return a per-lap summary.

Columns: `lap`, `t_start`, `t_end`, `duration`, `row_start`, `row_end`,
         `loop_start`, `loop_end`, `max_rpm`, `max_mph`, `is_race_lap`.

`is_race_lap` is `false` for laps shorter than `min_seconds`, or with
duration outside the median±50% band — i.e. the pit-out / cool-down /
incomplete laps. When `drop_partial=true` (default) those rows are filtered.
"""
function detect_laps(arrow_path::AbstractString;
                     min_seconds::Real = 20.0,
                     drop_partial::Bool = true)
    t = load_telemetry(arrow_path)
    return detect_laps(t; min_seconds = min_seconds, drop_partial = drop_partial)
end

function detect_laps(t::NamedTuple;
                     min_seconds::Real = 20.0,
                     drop_partial::Bool = true)
    lap   = t.lap
    time  = t.time
    rpm   = t.rpm
    speed = t.speed
    loop  = t.loop

    n = length(lap)
    n > 1 || return DataFrame()

    rows = NamedTuple[]
    i = 1
    while i <= n
        j = i
        l = lap[j]
        while j < n && lap[j+1] == l
            j += 1
        end
        t_start = Float64(time[i])
        t_end   = Float64(time[j])
        dur     = t_end - t_start
        # max RPM / MPH in this segment
        max_r = -Inf; max_s = -Inf
        @inbounds for k in i:j
            r = Float64(rpm[k]);   r > max_r && (max_r = r)
            s = Float64(speed[k]); s > max_s && (max_s = s)
        end
        push!(rows, (
            lap        = Int(l),
            t_start    = t_start,
            t_end      = t_end,
            duration   = dur,
            row_start  = i,
            row_end    = j,
            loop_start = String(loop[i]),
            loop_end   = String(loop[j]),
            max_rpm    = max_r,
            max_mph    = max_s,
        ))
        i = j + 1
    end

    df = DataFrame(rows)

    # Classify race laps by duration band around the median
    median_dur = isempty(df) ? 0.0 : median(df.duration)
    lo = max(min_seconds, 0.5 * median_dur)
    hi = 1.5 * median_dur > 0 ? 1.5 * median_dur : Inf
    df.is_race_lap = (df.duration .>= lo) .& (df.duration .<= hi)

    return drop_partial ? df[df.is_race_lap, :] : df
end
