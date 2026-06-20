using FFMPEG_jll

"""
    FfmpegBackend

Resolved ffmpeg runtime: which binary to use, whether NVENC/NVDEC are
available, and the encoder/decoder args to pass.

Build via `detect_backend(; prefer = :auto)`. The result caches detection
so subsequent calls don't re-probe.
"""
struct FfmpegBackend
    exe::String
    use_system::Bool
    has_nvenc::Bool
    has_nvdec::Bool
    encoder::String          # "h264_nvenc" or "libx264"
    encoder_args::Vector{String}
    hwaccel_args::Vector{String}
end

const _BACKEND_CACHE = Ref{Union{Nothing,FfmpegBackend}}(nothing)

"""
    detect_backend(; prefer=:auto, force=false) -> FfmpegBackend

`prefer` ∈ `(:auto, :gpu, :cpu, :system, :bundled)`:
- `:auto`  — system ffmpeg with NVENC > system ffmpeg + libx264 > FFMPEG_jll + libx264
- `:gpu`   — error unless NVENC is available
- `:cpu`   — force libx264 (and FFMPEG_jll, since it's always there)
- `:system` — force system ffmpeg
- `:bundled` — force FFMPEG_jll

Pass `force=true` to bypass the cache.
"""
function detect_backend(; prefer::Symbol = :auto, force::Bool = false)
    !force && _BACKEND_CACHE[] !== nothing && return _BACKEND_CACHE[]

    sys = _which_ffmpeg()
    sys_caps = sys === nothing ? (nvenc = false, nvdec = false) : _probe_capabilities(sys)

    backend = if prefer === :cpu
        _bundled_backend()
    elseif prefer === :bundled
        _bundled_backend()
    elseif prefer === :system
        sys === nothing && error("System ffmpeg not found on PATH")
        _system_backend(sys, sys_caps)
    elseif prefer === :gpu
        if sys !== nothing && sys_caps.nvenc
            _system_backend(sys, sys_caps)
        else
            error("NVENC not available — no system ffmpeg with h264_nvenc found")
        end
    else  # :auto
        if sys !== nothing && sys_caps.nvenc
            _system_backend(sys, sys_caps)
        elseif sys !== nothing
            _system_backend(sys, sys_caps)
        else
            _bundled_backend()
        end
    end

    _BACKEND_CACHE[] = backend
    return backend
end

# Well-known ffmpeg install locations to probe when it isn't on PATH: a manual
# unzip under the user profile, the winget `Gyan.FFmpeg` link shim, C:\ffmpeg,
# and chocolatey's shim dir.
function _ffmpeg_fallback_dirs()
    dirs = String[]
    home = get(ENV, "USERPROFILE", get(ENV, "HOME", ""))
    if !isempty(home)
        push!(dirs, joinpath(home, "ffmpeg", "bin"))
        push!(dirs, joinpath(home, "AppData", "Local", "Microsoft", "WinGet", "Links"))
    end
    push!(dirs, raw"C:\ffmpeg\bin")
    push!(dirs, joinpath(get(ENV, "ProgramData", raw"C:\ProgramData"), "chocolatey", "bin"))
    return dirs
end

"""
    _which_ffmpeg() -> Union{String,Nothing}

Locate a system ffmpeg. Asks the OS (`where`/`which`) first; if that finds
nothing it returns `nothing` instead of throwing (a non-zero exit from `where`
is "not found", not an error), then probes common install dirs so an ffmpeg
that simply isn't on PATH is still picked up. `detect_backend` treats `nothing`
as "fall back to the bundled FFMPEG_jll".
"""
function _which_ffmpeg()
    exe = Sys.iswindows() ? "ffmpeg.exe" : "ffmpeg"
    # 1. On PATH? Swallow a non-zero exit (means "not found") instead of throwing.
    onpath = try
        raw = Sys.iswindows() ? readchomp(`where ffmpeg`) : readchomp(`which ffmpeg`)
        p = first(eachline(IOBuffer(raw)))
        isempty(p) ? nothing : p
    catch
        nothing
    end
    onpath !== nothing && return onpath
    # 2. Not on PATH — probe known install locations.
    for d in _ffmpeg_fallback_dirs()
        cand = joinpath(d, exe)
        isfile(cand) && return cand
    end
    return nothing
end

function _probe_capabilities(exe::AbstractString)
    nvenc = nvdec = false
    try
        out = read(`$exe -hide_banner -hwaccels`, String)
        nvdec = occursin("cuda", lowercase(out))
    catch; end
    try
        out = read(`$exe -hide_banner -encoders`, String)
        nvenc = occursin("h264_nvenc", lowercase(out))
    catch; end
    return (nvenc = nvenc, nvdec = nvdec)
end

function _system_backend(exe::AbstractString, caps)
    if caps.nvenc
        encoder_args = ["-c:v", "h264_nvenc",
                        "-preset", "p4",
                        "-tune", "hq",
                        "-rc", "vbr",
                        "-cq", "20",
                        "-b:v", "0",
                        "-pix_fmt", "yuv420p"]
        hwaccel_args = caps.nvdec ?
            ["-hwaccel", "cuda"] : String[]
        return FfmpegBackend(String(exe), true, true, caps.nvdec,
                             "h264_nvenc", encoder_args, hwaccel_args)
    else
        encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-pix_fmt", "yuv420p"]
        return FfmpegBackend(String(exe), true, false, false,
                             "libx264", encoder_args, String[])
    end
end

function _bundled_backend()
    exe = FFMPEG_jll.get_ffmpeg_path()
    encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p"]
    return FfmpegBackend(String(exe), false, false, false,
                         "libx264", encoder_args, String[])
end

"""
    with_backend(f, backend::FfmpegBackend)

Run `f(exe::String)` with the FFMPEG_jll env applied if `backend` is the
bundled one (so its DLLs resolve). For system ffmpeg, just hand `f` the
path directly — system installations have their libs on the system PATH.
"""
function with_backend(f, backend::FfmpegBackend)
    if backend.use_system
        return f(backend.exe)
    else
        return FFMPEG_jll.ffmpeg(f)
    end
end
