using JSON3

default_db_path() = joinpath(@__DIR__, "..", "Track Maps", "track_map_db.json")

const TRACK_KEY_MAP = Dict{String,String}(
    "pocono"         => "Pocono",
    "michigan"       => "MichiganInternationalSpeedway",
    "dover"          => "Dover",
    "charlotte"      => "Charlotte_Oval",
    "talladega"      => "Talladega",
    "kansas"         => "Kansas",
    "bristol"        => "Bristol",
    "martinsville"   => "Martinsville",
    "las vegas"      => "Las Vegas",
    "watkins glen"   => "WatkinsGlen",
    "nashville"      => "NashvilleSuperSpeedway",
    "indianapolis"   => "Indianapolis_Oval",
    "texas"          => "Texas",
    "sonoma"         => "Sonoma",
)

struct TrackMap
    name::String
    x::Vector{Float64}      # raw track x
    y::Vector{Float64}      # raw track y
    s::Vector{Float64}      # cumulative arc length (ft)
    total_dist_ft::Float64
    x_norm::Vector{Float64} # 0..1
    y_norm::Vector{Float64} # 0..1
end

"""
    auto_detect_track(arrow_path) -> String or nothing

Guess the track from the arrow filename by matching tokens against the
`TRACK_KEY_MAP` aliases. Returns the canonical DB key on a hit. Useful for
agent / batch workflows that don't want to require an explicit `track` arg.
"""
function auto_detect_track(arrow_path::AbstractString)
    name = lowercase(basename(String(arrow_path)))
    tokens = split(name, r"[_\-.\s]+")
    for tok in tokens
        haskey(TRACK_KEY_MAP, tok) && return TRACK_KEY_MAP[tok]
    end
    for (alias, key) in TRACK_KEY_MAP
        occursin(alias, name) && return key
    end
    return nothing
end

"""
    resolve_track_key(db, hint) -> String or nothing

Match a user-supplied track hint ("Pocono", "pocono", "POCONO", file slug)
against the keys present in the loaded track-map DB.
"""
function resolve_track_key(db, hint::AbstractString)
    h = lowercase(strip(hint))
    keys_strs = String.(string.(collect(keys(db))))
    for k in keys_strs
        lowercase(k) == h && return k
    end
    if haskey(TRACK_KEY_MAP, h)
        cand = TRACK_KEY_MAP[h]
        cand in keys_strs && return cand
    end
    for k in keys_strs
        occursin(h, lowercase(k)) && return k
    end
    return nothing
end

"""
    load_track_map(db_path, track_hint) -> TrackMap or nothing
"""
function load_track_map(db_path::AbstractString, track_hint::AbstractString)
    isfile(db_path) || return nothing
    db = JSON3.read(read(db_path, String))
    key = resolve_track_key(db, track_hint)
    key === nothing && return nothing
    entry = db[Symbol(key)]
    x = Float64.(collect(entry[:x]))
    y = Float64.(collect(entry[:y]))

    s = if haskey(entry, :s)
        Float64.(collect(entry[:s]))
    else
        _arc_length(x, y, haskey(entry, :total_dist_ft) ? Float64(entry[:total_dist_ft]) : nothing)
    end

    total = haskey(entry, :total_dist_ft) ? Float64(entry[:total_dist_ft]) : Float64(s[end])
    xmin, xmax = extrema(x); ymin, ymax = extrema(y)
    xn = (x .- xmin) ./ (xmax - xmin)
    yn = (y .- ymin) ./ (ymax - ymin)
    return TrackMap(String(key), x, y, s, total, xn, yn)
end

"""
    _arc_length(x, y, total_dist_ft=nothing) -> Vector{Float64}

Cumulative point-to-point Euclidean distance along (x, y). If `total_dist_ft`
is given, the result is scaled so the final value equals it (handles tracks
where x,y are in arbitrary units, not feet).
"""
function _arc_length(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                     total_dist_ft::Union{Nothing,Real} = nothing)
    n = length(x)
    s = Vector{Float64}(undef, n)
    s[1] = 0.0
    @inbounds for i in 2:n
        s[i] = s[i-1] + hypot(x[i] - x[i-1], y[i] - y[i-1])
    end
    if total_dist_ft !== nothing && s[end] > 0
        scale = Float64(total_dist_ft) / s[end]
        s .*= scale
    end
    return s
end

"""
    dist_to_map_norm(dist_ft, tm) -> (xn, yn)

Wrap `dist_ft` around `total_dist_ft` and look up the normalised (0..1) track
coordinates by linear interpolation.
"""
function dist_to_map_norm(dist_ft::Real, tm::TrackMap)
    d = mod(Float64(dist_ft), tm.total_dist_ft)
    return (linear_interp(tm.s, tm.x_norm, d), linear_interp(tm.s, tm.y_norm, d))
end

"""
    linear_interp(xs, ys, xq) -> y

Linear interpolation; clamps to endpoints. `xs` must be monotonically
non-decreasing.
"""
function linear_interp(xs::AbstractVector, ys::AbstractVector, xq::Real)
    n = length(xs)
    xq <= xs[1]  && return Float64(ys[1])
    xq >= xs[n]  && return Float64(ys[n])
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if xs[mid] <= xq
            lo = mid
        else
            hi = mid
        end
    end
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    frac = (xq - x0) / (x1 - x0)
    return Float64(y0) * (1 - frac) + Float64(y1) * frac
end
