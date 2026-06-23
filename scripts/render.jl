# REPL / VS Code entry point. Open in VS Code, hit "Julia: Execute active
# file in REPL" (Ctrl+Shift+Enter with the Julia extension). The script
# self-activates the project, loads the package, and runs whatever is at
# the bottom.
#
# After the first include, call these directly in the REPL — no re-include
# needed:
#
#     cfg = getConfig("25POC1")           # or just getConfig() to use [current].race
#     process(cfg; cars=[9], laps=[119])
#     process(cfg; cars=:all, laps=[1, 50, 119])
#     process(cfg; cars=[9], laps=:all)   # every detected race lap for car 9
#     generate_lap_video(cfg, 9, 119)      # one-off, no batching
#     list_cars(cfg)                       # who's in this race?
#     detect_laps(find_car_session(cfg, 9).arrow)  # laps available for car 9
#
# The race is determined by what you pass to `getConfig`. Multiple races
# can be active in the same session at once.

using Pkg
let root = abspath(joinpath(@__DIR__, ".."))
    abspath(Pkg.project().path) != joinpath(root, "Project.toml") && Pkg.activate(root)
end

using PreRaceFilm

# ── Run on include ─────────────────────────────────────────────────────────
# Edit the two calls below to render what you want.

const CFG = getConfig()         # uses [current].race from config.local.toml

process(CFG; cars = [77], laps = [119])

# More examples — uncomment any:
# process(CFG; cars = [9, 10, 11], laps = [5, 50, 119])
# process(CFG; cars = [9], laps = :all)
# generate_lap_video(CFG, 9, 119; overwrite = true)
# generate_lap_video(CFG, 10, 50; fps = 30)
