using TOML

"""
TOML config loader. Looks for (in order):

1. `config.local.toml` in the repo root  (gitignored — your personal paths)
2. `config.toml`       in the repo root  (committed — team defaults)

Values from the chosen file are surfaced via `config_get(section, key)`. This
is the only place config is read; `getConfig` and the `src/datadir.jl` path
resolvers build on it.
"""

const _CONFIG_CACHE = Ref{Union{Nothing,Dict{String,Any}}}(nothing)

function _config_search_paths()
    root = abspath(joinpath(@__DIR__, ".."))
    return [joinpath(root, "config.local.toml"), joinpath(root, "config.toml")]
end

"""
    load_config(; force=false) -> Dict{String,Any}

Read the first config file that exists. Caches the result; pass
`force=true` to re-read (e.g. after editing the TOML).
"""
function load_config(; force::Bool = false)
    !force && _CONFIG_CACHE[] !== nothing && return _CONFIG_CACHE[]
    for p in _config_search_paths()
        isfile(p) && return _CONFIG_CACHE[] = TOML.parsefile(p)
    end
    return _CONFIG_CACHE[] = Dict{String,Any}()
end

"""
    config_get(section, key, default=nothing)

Pull `[section].key` from the loaded config. `default` is returned if the
section or key is missing.
"""
function config_get(section::AbstractString, key::AbstractString, default = nothing)
    sec = get(load_config(), section, nothing)
    sec isa AbstractDict || return default
    return get(sec, key, default)
end

# ── Session-library resolution ───────────────────────────────────────────────
# Race session files (.mpg / .arrow) live OUTSIDE the repo and the location
# changes weekly. Paths come from `[paths]` in `config.local.toml`:
#
#     [paths]
#     data_dir   = "D:\\Race_Videos\\25POC1"
#     arrow_dir  = "D:\\Race_Videos\\25POC1"   # optional; defaults to data_dir
#     output_dir = "out"                       # optional; defaults to repo out/
#
# These resolvers are the config edge — `getConfig` calls them once to build a
# `RaceConfig`; work code reads the resolved paths off that object.

# TODO: data_dir/arrow_dir + config_get survive only for the Pluto picker.
# When Pluto is gone, delete them and inline a single TOML.parsefile in
# getConfig — then config_get dies too ("bye config_get").
"""
    data_dir() -> String

Resolution order: `[paths].data_dir` in the TOML config → legacy
`Sample Race Data/` in the repo root → `""`.
"""
function data_dir()
    c = config_get("paths", "data_dir", "")
    c isa AbstractString && !isempty(c) && return String(c)
    legacy = abspath(joinpath(@__DIR__, "..", "Sample Race Data"))
    return isdir(legacy) ? legacy : ""
end

"""
    arrow_dir() -> String

Where to look for .arrow telemetry: `[paths].arrow_dir` in the TOML config →
`data_dir()`.
"""
function arrow_dir()
    c = config_get("paths", "arrow_dir", "")
    c isa AbstractString && !isempty(c) && return String(c)
    return data_dir()
end

"""
    list_session_files(; data = data_dir(), arrow = arrow_dir()) -> DataFrame

Return a table of `(name, video, arrow, video_size_mb, arrow_size_mb)` for
every video/arrow pair that share a stem. Useful for the Pluto picker and
for batch jobs.
"""
function list_session_files(; data::AbstractString = data_dir(),
                              arrow::AbstractString = arrow_dir())
    videos = isdir(data) ? sort(filter(f -> endswith(lowercase(f), ".mpg"),
                                       readdir(data; join = true))) : String[]
    arrows = isdir(arrow) ? sort(filter(f -> endswith(lowercase(f), ".arrow"),
                                        readdir(arrow; join = true))) : String[]

    arrow_by_stem = Dict(splitext(basename(a))[1] => a for a in arrows)
    rows = NamedTuple[]
    for v in videos
        stem = splitext(basename(v))[1]
        a = get(arrow_by_stem, stem, "")
        push!(rows, (
            name           = stem,
            video          = v,
            arrow          = a,
            video_size_mb  = round(filesize(v) / 1e6;  digits = 1),
            arrow_size_mb  = isempty(a) ? 0.0 : round(filesize(a) / 1e6; digits = 1),
            has_arrow      = !isempty(a),
        ))
    end
    return DataFrame(rows)
end

# ── Per-race configuration ───────────────────────────────────────────────────

const RACE_CONFIG_FILENAME = "race.toml"

"""
    RaceConfig

Per-race config: resolved paths + race metadata, loaded once by `getConfig`.
`<data_dir>/race.toml` supplies event/track/date, a `file_stem` template
(`{car}` → car number), `[drivers]`, and `[cars.N]` overrides (e.g.
`audio_alignment`, `stem`).
"""
struct RaceConfig
    race::String
    data_dir::String
    arrow_dir::String
    output_dir::String
    config_path::String
    event::String
    track::String
    date::String
    file_stem::String
    drivers::Dict{Int,String}
    car_overrides::Dict{Int,Dict{String,Any}}
end

"""
    getConfig(race=""; arrow_root="") -> RaceConfig

Resolve a race weekend from `config.local.toml` (`[paths]` + `[current].race`)
and its `race.toml`. Errors if no race or no data path is configured.
"""
function getConfig(race::AbstractString = ""; arrow_root::AbstractString = "")
    race = isempty(race) ? String(config_get("current", "race", "")) : race
    isempty(race) && error("No race: pass getConfig(\"25POC1\") or set [current].race")

    root = String(config_get("paths", "data_root", ""))
    dir  = String(config_get("paths", "data_dir", ""))
    data_dir = !isempty(root) ? joinpath(root, race) :
               !isempty(dir)  ? dir : error("Set [paths].data_root or data_dir in config.local.toml")
    isdir(data_dir) || error("Race folder not found: $data_dir")

    arrow_dir = !isempty(arrow_root) ? joinpath(arrow_root, race) :
                let a = String(config_get("paths", "arrow_dir", "")); isempty(a) ? data_dir : a end
    out = String(config_get("paths", "output_dir", "out"))
    output_dir = isabspath(out) ? out : abspath(joinpath(@__DIR__, "..", out))

    cfg_path = joinpath(data_dir, RACE_CONFIG_FILENAME)
    isfile(cfg_path) || return RaceConfig(race, data_dir, arrow_dir, output_dir, "",
        race, "", "", "", Dict{Int,String}(), Dict{Int,Dict{String,Any}}())

    raw  = TOML.parsefile(cfg_path)
    ints(d) = ((parse(Int, String(k)), v) for (k, v) in d if tryparse(Int, String(k)) !== nothing)
    drivers   = Dict{Int,String}(k => String(v) for (k, v) in ints(get(raw, "drivers", Dict())))
    overrides = Dict{Int,Dict{String,Any}}(k => Dict{String,Any}(String(kk) => vv for (kk, vv) in v)
                    for (k, v) in ints(get(raw, "cars", Dict())) if v isa AbstractDict)
    return RaceConfig(race, data_dir, arrow_dir, output_dir, cfg_path,
        String(get(raw, "event", race)), String(get(raw, "track", "")),
        String(get(raw, "date", "")), String(get(raw, "file_stem", "")), drivers, overrides)
end
