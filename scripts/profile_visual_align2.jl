# Profiling harness for the visual_align2 hot path (per-frame stage compute).
#
# Runs each hot function (decoder-free) on a synthetic frame + primed state, so
# you can isolate per-step cost without the ffmpeg/channel pipeline around it.
#
#   non-interactive (prints the timing table):
#     julia --project -t 4 scripts/profile_visual_align2.jl
#
#   interactive (poke at the global `H` = frame + configs + primed states):
#     julia --project -t 4 -i scripts/profile_visual_align2.jl
#
#     @code_warntype P.magnitude!(H.sfwd)               # type stability
#     @code_llvm debuginfo=:none P.magnitude!(H.sfwd)   # look for vfmadd / vsqrtpd / vector ops
#     using BenchmarkTools;  @btime P.magnitude!($(H.sfwd))
#     using Profile;  Profile.@profile hot_forward(50_000);  Profile.print(mincount=50)
#     using ProfileView;  @profview hot_forward(200_000)   # flame graph (if installed)
#     summary()                                            # reprint the table
#
# NB: synthetic random frame — this measures SPEED, not alignment correctness.

using PreRaceFilm
using LinearAlgebra: mul!
const P = PreRaceFilm

function setup(; fw = 320, fh = 180)
    frame = P.Frame(fw, fh)
    frame.data .= rand(fw, fh) .* 255.0          # synthetic gray frame
    frame.index = 5
    dummy = Channel{P.Frame}(1); pool = Channel{P.Frame}(1)
    mkcfg(c) = P.StageConfig(c, P._hann(c.w, c.h), dummy, nothing, pool, [[0.0]], [[0.0]], 30.0, 0.0, 100)
    rot = P.Crop(0.25, 0.50, 0.22, 0.28, fw, fh); cfg_rot = mkcfg(rot)
    fwd = P.Crop(0.18, 0.64, 0.30, 0.34, fw, fh); cfg_fwd = mkcfg(fwd)
    srot = P.make_state(P.Rotation(), cfg_rot)
    sfwd = P.make_state(P.Forward(), cfg_fwd)
    for _ in 1:3                                  # prime prev-frame state
        P.shift!(srot, cfg_rot, frame); P.forward_zoom!(sfwd, cfg_fwd, frame)
    end
    return (; frame, cfg_rot, cfg_fwd, srot, sfwd, rot, fwd)
end

const H = setup()

# loop drivers — feed these to @profile / @profview / @time
hot_forward(n)   = (for _ in 1:n; P.forward_zoom!(H.sfwd, H.cfg_fwd, H.frame); end; nothing)
hot_rotation(n)  = (for _ in 1:n; P.shift!(H.srot, H.cfg_rot, H.frame); end; nothing)
hot_crop(n)      = (for _ in 1:n; P.crop!(H.sfwd, H.cfg_fwd, H.frame); end; nothing)
hot_magnitude(n) = (for _ in 1:n; P.magnitude!(H.sfwd); end; nothing)
hot_logpolar(n)  = (for _ in 1:n; P.logpolar!(H.sfwd); end; nothing)
hot_imgfft(n)    = (for _ in 1:n; mul!(H.sfwd.cur_freq, H.sfwd.img_plan, H.sfwd.cur); end; nothing)
hot_lpshift(n)   = (for _ in 1:n; P.lp_shift!(H.sfwd); end; nothing)

function tus(f, n = 5000)        # warm, GC, then time → µs per call
    f(2); GC.gc()
    return round(@elapsed(f(n)) / n * 1e6, digits=2)
end

function summary()
    println("rotation crop = $(H.rot.w)x$(H.rot.h)   forward crop = $(H.fwd.w)x$(H.fwd.h)\n")
    for (name, f) in (("forward zoom!", hot_forward), ("rotation shift!", hot_rotation),
                      ("  crop!", hot_crop), ("  image FFT", hot_imgfft),
                      ("  magnitude!", hot_magnitude), ("  logpolar!", hot_logpolar),
                      ("  lp_shift!", hot_lpshift))
        println(rpad(name, 18), tus(f), " us")
    end
end

if isinteractive()
    println("\nReady — $(Threads.nthreads()) threads. Globals: H, hot_forward/rotation/crop/" *
            "magnitude/logpolar/imgfft/lpshift(n), summary().")
else
    summary()
end
