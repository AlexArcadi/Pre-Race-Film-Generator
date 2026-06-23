# Per-race accessors + render entry points. `RaceConfig` / `getConfig` live in
# config.jl (the config edge); these consume the resolved config object.

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
            push!(results, generate_lap_video(cfg, car, lap;
                audio_alignment = align,
                overwrite = overwrite,
                kwargs...))
        end
    end
    return results
end
