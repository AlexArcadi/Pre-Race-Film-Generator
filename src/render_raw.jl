# Bare-clip template: no overlay, no channels, no map. Pipeline.jl branches to
# `render_raw_clip` when `template = :raw` is selected, skipping the Cairo bake
# / draw pair and the dual-input ffmpeg filter graph that `:full` / `:minimal`
# rely on. The output is just the source video trimmed to the lap's video-time
# window, re-encoded with the resolved backend so the container is a clean MP4
# regardless of what the source was (`.mpg`, `.mp4`, …).
#
# Re-encode rather than stream-copy: `-c copy` would snap the cut to the
# nearest keyframe before `-ss`, which can be several seconds off on MPEG-PS
# sources with sparse I-frames — wrong for a precise per-lap clip. NVENC is
# fast enough that frame-accurate seek is worth the extra cost.

"""
    render_raw_clip(video_path, video_lap_start, video_lap_dur, output_path)
        -> NamedTuple

Trim `video_path` to `[video_lap_start, video_lap_start + video_lap_dur]` (s)
and write it to `output_path` as an MP4 with the source audio. Returns
`(output_path, file_size_mb, encoder)`.
"""
function render_raw_clip(video_path::AbstractString,
                         video_lap_start::Real,
                         video_lap_dur::Real,
                         out_path::AbstractString)
    cmd = String[ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    append!(cmd, hwaccel_args())
    append!(cmd, ["-ss", string(video_lap_start),
                  "-t", string(video_lap_dur),
                  "-i", String(video_path)])
    append!(cmd, encode_args())
    append!(cmd, ["-c:a", "aac", "-b:a", "192k", String(out_path)])
    run(Cmd(cmd))

    size_mb = isfile(out_path) ? filesize(out_path) / 1e6 : 0.0
    return (output_path  = String(out_path),
            file_size_mb = size_mb,
            encoder      = _caps().nvenc ? :h264_nvenc : :libx264)
end
