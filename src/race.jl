using TOML

"""
Per-race configuration. Loaded via `getConfig(race_code)` — the race is
identified by the config you pass around, NOT by global state. Multiple
configs for different races can coexist in the same Julia session.

A `RaceConfig` carries:

- The race code, data directory, arrow directory (paths embedded — no
  more global "current data_dir")
- Race metadata: event, track, date
- A filename-stem template for mapping car number → files
- Driver names by car number
- Per-car overrides (e.g. baked-in audio_alignment, alternate stem)

The race.toml schema lives at `<data_root>/<race>/race.toml`:

    event     = "25POC1"
    track     = "Pocono"
    date      = "2025-06-01"
    file_stem = "19_POCONO_car{car}_sessionID2"

    [drivers]
    9 = "Chase Elliott"
    10 = "Aric Almirola"

    [cars.10]
    audio_alignment = -1100.0
    stem = "10_POCONO_alt_naming"

This schema is intentionally close to what an ERDP_DATA per-race config
could look like, so the two pipelines can share configs later.
"""

const RACE_CONFIG_FILENAME = "race.toml"

struct RaceConfig
    race::String                            # e.g. "25POC1"
    data_dir::String                        # where .mpg files live
    arrow_dir::String                       # where .arrow files live
    config_path::String                     # path of the race.toml that was read ("" if absent)
    event::String
    track::String
    date::String
    file_stem::String                       # template with `{car}` placeholder
    drivers::Dict{Int,String}
    car_overrides::Dict{Int,Dict{String,Any}}
end

# ── Loading ────────────────────────────────────────────────────────────────

"""
    getConfig(race::AbstractString = "") -> RaceConfig

Resolve and load configuration for a race weekend.

- `race` is the folder name under `[paths].data_root` (e.g. "25POC1").
  If omitted, falls back to `[current].race` from `config.local.toml`.
- Reads `<data_root>/<race>/race.toml` if present; otherwise returns a
  config with empty driver/track/template that still lets filename
  heuristics work.

Multiple `RaceConfig`s can coexist — pass them explicitly to every
function that uses one.
"""
function getConfig(race::AbstractString = ""; arrow_root::AbstractString = "")
    race = isempty(race) ? config_get("current", "race", "") : race
    isempty(race) && error(
        "No race specified. Either pass a code: `getConfig(\"25POC1\")`, " *
        "or set `[current].race` in `config.local.toml`.")

    root = String(config_get("paths", "data_root", ""))
    legacy_data = String(config_get("paths", "data_dir", ""))
    race_dir = if !isempty(root)
        joinpath(root, race)
    elseif !isempty(legacy_data)
        legacy_data
    else
        abspath(joinpath(@__DIR__, "..", "Sample Race Data"))
    end

    isdir(race_dir) || error(
        "Race folder not found: $race_dir\n" *
        "Set `[paths].data_root` in `config.local.toml` (preferred), or " *
        "use `[paths].data_dir` to point at one race folder directly.")

    arrow_dir = if !isempty(arrow_root)
        joinpath(arrow_root, race)
    else
        legacy_arrow = String(config_get("paths", "arrow_dir", ""))
        e = get(ENV, "PRERACEFILM_ARROW_DIR", "")
        !isempty(e) ? e : (!isempty(legacy_arrow) ? legacy_arrow : race_dir)
    end

    cfg_path = joinpath(race_dir, RACE_CONFIG_FILENAME)
    if isfile(cfg_path)
        raw = TOML.parsefile(cfg_path)
        event     = String(get(raw, "event",     race))
        track     = String(get(raw, "track",     ""))
        date      = String(get(raw, "date",      ""))
        file_stem = String(get(raw, "file_stem", ""))

        drivers = Dict{Int,String}()
        for (k, v) in get(raw, "drivers", Dict{String,Any}())
            n = tryparse(Int, String(k))
            n === nothing || (drivers[n] = String(v))
        end

        overrides = Dict{Int,Dict{String,Any}}()
        for (k, v) in get(raw, "cars", Dict{String,Any}())
            n = tryparse(Int, String(k))
            n === nothing && continue
            v isa AbstractDict || continue
            overrides[n] = Dict{String,Any}(String(kk) => vv for (kk, vv) in v)
        end

        return RaceConfig(race, race_dir, arrow_dir, cfg_path,
                          event, track, date, file_stem, drivers, overrides)
    else
        return RaceConfig(race, race_dir, arrow_dir, "",
                          race, "", "", "",
                          Dict{Int,String}(), Dict{Int,Dict{String,Any}}())
    end
end

# ── Helpers (cfg-explicit; no global state) ────────────────────────────────

"""
    stem_for(cfg::RaceConfig, car::Integer; kind::Symbol = :both) -> Union{String,Nothing}

Filename stem for a car. Resolution per kind (`:video`, `:arrow`, or
`:both`):

1. Per-car override `<kind>_stem` (e.g. `arrow_stem` for `:arrow`)
2. Per-car override `stem` (means "both video and arrow share this stem")
3. `file_stem` template with `{car}` substituted
4. `nothing` (caller falls back to a filename scan)

Use `stem_for(cfg, 11; kind=:arrow)` when the .arrow and .mpg have
different naming this week.
"""
function stem_for(cfg::RaceConfig, car::Integer; kind::Symbol = :both)
    n = Int(car)
    if haskey(cfg.car_overrides, n)
        ov = cfg.car_overrides[n]
        key = string(kind, "_stem")
        kind !== :both && haskey(ov, key) && return String(ov[key])
        haskey(ov, "stem")                && return String(ov["stem"])
    end
    isempty(cfg.file_stem) && return nothing
    return replace(cfg.file_stem, "{car}" => string(car))
end

video_stem_for(cfg::RaceConfig, car::Integer) = stem_for(cfg, car; kind = :video)
arrow_stem_for(cfg::RaceConfig, car::Integer) = stem_for(cfg, car; kind = :arrow)

"""
    driver_for(cfg::RaceConfig, car::Integer) -> String

Driver name from race.toml, or `"Car #N"` if not set.
"""
driver_for(cfg::RaceConfig, car::Integer) =
    get(cfg.drivers, Int(car), "Car #$(Int(car))")

"""
    car_override(cfg::RaceConfig, car::Integer, key::AbstractString; default=nothing)

Pull a per-car override value (e.g. `"audio_alignment"`).
"""
function car_override(cfg::RaceConfig, car::Integer, key::AbstractString;
                      default = nothing)
    n = Int(car)
    haskey(cfg.car_overrides, n) || return default
    return get(cfg.car_overrides[n], String(key), default)
end

"""
    event_label_default(cfg::RaceConfig) -> String

Human-readable label for the overlay. Combines track + event code.
"""
function event_label_default(cfg::RaceConfig)
    isempty(cfg.event) && return cfg.track
    isempty(cfg.track) && return cfg.event
    return "$(cfg.track) — $(cfg.event)"
end

"""
    list_cars(cfg::RaceConfig) -> Vector{Int}

Cars known for this race. Prefers `[drivers]` keys in race.toml; falls back
to a filename scan of `data_dir` when race.toml is absent.
"""
function list_cars(cfg::RaceConfig)
    isempty(cfg.drivers) || return sort(collect(keys(cfg.drivers)))
    cars = Int[]
    isdir(cfg.data_dir) || return cars
    for f in readdir(cfg.data_dir)
        endswith(lowercase(f), ".mpg") || continue
        m = match(r"car(\d+)"i, f)
        m === nothing && continue
        push!(cars, parse(Int, m.captures[1]))
    end
    return sort(unique(cars))
end

"""
    find_car_session(cfg::RaceConfig, car::Integer) -> NamedTuple

`(car, video, arrow, name)` for one car. Uses stem_for(cfg, car) when the
race.toml template is set; falls back to a filename scan otherwise.
"""
function find_car_session(cfg::RaceConfig, car::Integer)
    vs = video_stem_for(cfg, car)
    as = arrow_stem_for(cfg, car)
    if vs !== nothing && as !== nothing
        video = joinpath(cfg.data_dir,  vs * ".mpg")
        arrow = joinpath(cfg.arrow_dir, as * ".arrow")
        isfile(video) || error("Car #$car video not found: $video")
        isfile(arrow) || error("Car #$car .arrow not found: $arrow")
        return (car = Int(car), video = video, arrow = arrow, name = vs)
    end

    df = list_session_files(data = cfg.data_dir, arrow = cfg.arrow_dir)
    nrow(df) > 0 || error("No session files in $(cfg.data_dir)")
    pattern = Regex("car0*$(Int(car))(?:[^0-9]|\$)", "i")
    candidates = filter(r -> occursin(pattern, r.name), eachrow(df))
    isempty(candidates) && error("Car #$car not found in $(cfg.data_dir)")
    s = first(candidates)
    s.has_arrow || error("Car #$car has no matching .arrow file (stem=$(s.name))")
    return (car = Int(car), video = s.video, arrow = s.arrow, name = s.name)
end

# ── Entry points ───────────────────────────────────────────────────────────

"""
    render_lap(cfg::RaceConfig, car::Integer, lap::Integer;
               event_label = nothing,
               driver_label = nothing,
               overwrite::Bool = false,
               kwargs...) -> NamedTuple

Render ONE lap for ONE car. Driver name and event label default to values
from `cfg`. Other kwargs flow through to `generate_lap_video`.

Used internally by `process(cfg; ...)`; safe to call directly for one-off
work.
"""
function render_lap(cfg::RaceConfig, car::Integer, lap::Integer;
                    event_label = nothing,
                    driver_label = nothing,
                    overwrite::Bool = false,
                    kwargs...)
    session = find_car_session(cfg, car)

    out = joinpath(output_dir(), "$(cfg.race)_car$(car)_lap$(lap).mp4")
    isdir(dirname(out)) || mkpath(dirname(out))
    if !overwrite && isfile(out)
        @info "Already rendered, skipping: $out  (pass overwrite=true to redo)"
        return (output_path = out, skipped = true)
    end

    driver_label === nothing && (driver_label = driver_for(cfg, car))
    if event_label === nothing
        e = event_label_default(cfg)
        event_label = isempty(e) ? something(auto_detect_track(session.arrow), "") : e
    end

    # Per-car overrides flow in only when the caller didn't already pass them.
    align_override = car_override(cfg, car, "audio_alignment")
    if align_override !== nothing && !haskey(kwargs, :audio_alignment)
        kwargs = (; kwargs..., audio_alignment = align_override)
    end
    ft_override = car_override(cfg, car, "fine_tune_s")
    if ft_override !== nothing && !haskey(kwargs, :fine_tune_s)
        kwargs = (; kwargs..., fine_tune_s = Float64(ft_override))
    end

    @info "Rendering $driver_label car #$car lap $lap → $out"
    return generate_lap_video(session.video, session.arrow, lap;
        output_path  = out,
        driver_label = driver_label,
        event_label  = event_label,
        kwargs...)
end

"""
    process(cfg::RaceConfig; cars=:all, laps=:all, overwrite=false, kwargs...)

Top-level entry point. Render every (car, lap) combination described by
`cfg`. Audio alignment is computed once per car and reused across that
car's laps. Returns a vector of result NamedTuples (one per render).

- `cars`: `:all` (every car in `cfg.drivers`) or a vector of car numbers
- `laps`: `:all` (every race lap detected in each car's arrow) or a vector
- `overwrite`: re-render even if the output file exists
- Extra `kwargs` (e.g. `fps = 30`) flow through to `generate_lap_video`.
"""
function process(cfg::RaceConfig;
                 cars::Union{Symbol,AbstractVector{<:Integer}} = :all,
                 laps::Union{Symbol,AbstractVector{<:Integer}} = :all,
                 overwrite::Bool = false,
                 kwargs...)
    cars_list = cars === :all ? list_cars(cfg) : Int.(collect(cars))
    isempty(cars_list) && error(
        "No cars resolved for $(cfg.race). Either populate `[drivers]` in " *
        "$(cfg.config_path == "" ? joinpath(cfg.data_dir, RACE_CONFIG_FILENAME) : cfg.config_path) " *
        "or pass `cars = [...]` explicitly.")

    results = NamedTuple[]
    for car in cars_list
        session = find_car_session(cfg, car)

        # One alignment per car: bake-in override > FFT once > pass through
        align = car_override(cfg, car, "audio_alignment")
        if align === nothing
            @info "Computing audio alignment for $(driver_for(cfg, car)) (car #$car)…"
            align = align_audio_rpm(session.video, session.arrow).offset_s
            @info "  offset = $(round(align; digits=2)) s"
        end

        car_laps = laps === :all ?
            collect(detect_laps(session.arrow).lap) :
            Int.(collect(laps))

        for lap in car_laps
            push!(results, render_lap(cfg, car, lap;
                audio_alignment = align,
                overwrite = overwrite,
                kwargs...))
        end
    end
    return results
end
