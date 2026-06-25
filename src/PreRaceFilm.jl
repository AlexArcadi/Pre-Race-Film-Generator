module PreRaceFilm

using Printf
using Arrow
using Tables
using DSP
using FFTW
using Statistics
using LinearAlgebra: mul!



include("math.jl")
include("runtime.jl")
include("config.jl")
include("telemetry.jl")
include("race.jl")
include("track_map.jl")
include("alignment.jl")
include("visual_align2.jl")
include("render.jl")
include("render_minimal.jl")
include("render_raw.jl")
include("pipeline.jl")

export detect_laps,
       load_telemetry,
       load_track_map,
       align_audio_rpm,
       find_race_start,
       find_audio_active_start,
       generate_lap_video,
       generate_lap_video_json,
       list_laps_json,
       default_ranges,
       default_db_path,
       ffmpeg_exe,
       probe_video,
       auto_detect_track,
       list_session_files,
       RaceConfig,
       getConfig,
       process,
       stem_for,
       video_stem_for,
       arrow_stem_for,
       driver_for,
       car_override,
       event_label_default,
       list_cars,
       find_car_session,
       build_channels_minimal,
       build_stats_minimal,
       load_car_number_graphic

end # module
