#!/usr/bin/env julia
#
# PreRaceFilm CLI.
#
# Examples:
#   julia --project=. bin/pre_race_film.jl laps --arrow path/to/session.arrow
#   julia --project=. bin/pre_race_film.jl render --race 25POC1 --car 9 --lap 50
#
# All commands print JSON to stdout on success. Errors go to stderr with a
# non-zero exit code. This is the integration surface used by the AI agent
# and any other downstream tool.

using Pkg
project = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(project)

using PreRaceFilm
using JSON3

const USAGE = """
PreRaceFilm CLI
===============

Commands:
  laps      List race laps detected in an .arrow file
  render    Render an overlay video for a chosen lap
  backend   Print the resolved ffmpeg backend (NVENC, libx264, ...)

Run `<command> --help` for command-specific options.
"""

const LAPS_USAGE = """
laps  --arrow PATH [--include-partial] [--min-seconds N]

Required:
  --arrow PATH               .arrow telemetry file

Options:
  --include-partial          Include pit-out / cool-down / caution laps
  --min-seconds N            Minimum lap duration to consider (default 20)
"""

const RENDER_USAGE = """
render  --car N --lap N [--race CODE] [options]

Required:
  --car N                    Car number
  --lap N                    Lap number to render

Options:
  --race CODE                Race folder (default: [current].race in config)
  --track NAME               Track key (e.g. "Pocono"). Default: auto-detect
  --no-track                 Skip the mini-map entirely
  --fps N                    Output fps (default 25)
  --width N                  Output width (default 1280)
  --height N                 Output height (default 720)
  --template NAME            full | minimal (default full)
  --align MODE               seed | auto | none | <offset_s>  (default: seed)
  --overwrite                Re-render even if the output exists
"""

function main(argv::Vector{String})
    if isempty(argv) || argv[1] in ("-h", "--help")
        println(USAGE); return 0
    end
    cmd = argv[1]
    rest = argv[2:end]
    try
        return if cmd == "laps"
            cmd_laps(rest)
        elseif cmd == "render"
            cmd_render(rest)
        elseif cmd == "backend"
            cmd_backend(rest)
        else
            println(stderr, "Unknown command: $cmd")
            println(stderr, USAGE)
            2
        end
    catch e
        println(stderr, "ERROR: ", sprint(showerror, e))
        for f in stacktrace(catch_backtrace())[1:min(6,end)]
            println(stderr, "  ", f)
        end
        1
    end
end

function cmd_laps(args::Vector{String})
    if isempty(args) || args[1] in ("-h", "--help")
        println(LAPS_USAGE); return 0
    end
    opts = parse_args(args)
    arrow = required(opts, "arrow")
    drop_partial = !get(opts, "include-partial", false)
    min_seconds = parse(Float64, get(opts, "min-seconds", "20"))
    JSON3.pretty(stdout, list_laps_json(arrow;
                                       min_seconds = min_seconds,
                                       drop_partial = drop_partial))
    println()
    return 0
end

function cmd_render(args::Vector{String})
    if isempty(args) || args[1] in ("-h", "--help")
        println(RENDER_USAGE); return 0
    end
    opts = parse_args(args)
    config = Dict{String,Any}(
        "car" => parse(Int, required(opts, "car")),
        "lap" => parse(Int, required(opts, "lap")),
    )
    haskey(opts, "race")      && (config["race"]      = opts["race"])
    haskey(opts, "track")     && (config["track"]     = opts["track"])
    haskey(opts, "no-track")  && (config["track"]     = nothing)
    haskey(opts, "fps")       && (config["fps"]       = parse(Int, opts["fps"]))
    haskey(opts, "width") && haskey(opts, "height") &&
        (config["resolution"] = [parse(Int, opts["width"]), parse(Int, opts["height"])])
    haskey(opts, "template")  && (config["template"]  = opts["template"])
    haskey(opts, "overwrite") && (config["overwrite"] = true)

    if haskey(opts, "align")
        v = opts["align"]
        config["audio_alignment"] = (v in ("seed","auto","none")) ? ":" * v : v
    end

    result = generate_lap_video_json(config)
    JSON3.pretty(stdout, result)
    println()
    return 0
end

function cmd_backend(_args::Vector{String})
    caps = PreRaceFilm._caps()
    JSON3.pretty(stdout, Dict(
        "ffmpeg" => ffmpeg_exe(),
        "nvenc"  => caps.nvenc,
        "nvdec"  => caps.nvdec,
    ))
    println()
    return 0
end

"""
    parse_args(args) -> Dict{String,String}

Tiny flag parser. Supports `--key value` and bare `--flag` (sets to `"true"`).
"""
function parse_args(args::Vector{String})
    out = Dict{String,Any}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("Unexpected positional arg: $a")
        key = a[3:end]
        if i == length(args) || startswith(args[i + 1], "--")
            out[key] = true
            i += 1
        else
            out[key] = args[i + 1]
            i += 2
        end
    end
    return out
end

function required(opts::Dict, k::String)
    haskey(opts, k) || error("Missing required argument: --$k")
    return opts[k]
end

isinteractive() || exit(main(ARGS))
