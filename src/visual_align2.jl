# ─────────────────────────────────────────────────────────────────────────────
# Design (verbatim from the request that kicked this off):
#
#   mutable struct frame
#   height
#   width
#   raw::Vector(Uint)
#   index
#   Data
#   end
#
#   then set up channels that take frames, and link one processor to the next,
#   they can be their own async processes now.
#
#   We also need an updateFrame that takes an old frame from a pool and updates it
#   so we don't get stale data. this allows reuse.
#
#   Each type of process should have the same name. We need a type struct yaw end,
#   struct pitch end, struct forward end or so. We need a common interface. each
#   one should get an input struct at initialzation with their window, their
#   channels (in, out channels, daisy chained together).
#
#   Then they no longer need an ffmpeg backend, they only get a stream of frames.
#   THey initialize a vector to store their data in, when the sream ends, they
#   each kick off their correlation process, and return a result struct.
#
#   decoder ──► ch ──► [Yaw] ──► ch ──► [Pitch] ──► ch ──► [Forward] ──► pool
# ─────────────────────────────────────────────────────────────────────────────



# ── types ────────────────────────────────────────────────────────────────────
mutable struct Frame
    height::Int
    width::Int
    raw::Vector{UInt8}
    index::Int
    data::Matrix{Float64}
end

Frame(width::Int, height::Int) =
    Frame(height, width, Vector{UInt8}(undef, width * height), -1,
          Matrix{Float64}(undef, width, height))

abstract type Stage end
struct Rotation <: Stage end   # yaw + pitch, both read off one phase-correlation surface
struct Forward  <: Stage end

# Crop window as resolved, inclusive pixel bounds in the frame's [x, y] grid. The
# fractions → pixels conversion happens ONCE, in this constructor, when `align`
# builds each StageConfig — the worker never recomputes offsets.
struct Crop
    x0::Int
    x1::Int
    y0::Int
    y1::Int
    w::Int          # x1 - x0 + 1, stored so the worker never recomputes it
    h::Int          # y1 - y0 + 1
end
# Fraction args typed Float64 so this can't collide with the 6-Int field constructor.
function Crop(x0f::Float64, wf::Float64, y0f::Float64, hf::Float64, frame_w::Int, frame_h::Int)
    x0 = max(1, round(Int, x0f * frame_w) + 1)
    y0 = max(1, round(Int, y0f * frame_h) + 1)
    # Snap the crop extent up to a 5-smooth size: a prime dim (e.g. 61) forces
    # FFTW onto its Bluestein slow path — ~12× slower for the same pixel count.
    w = min(frame_w, nextprod([2, 3, 5], round(Int, wf * frame_w)))
    h = min(frame_h, nextprod([2, 3, 5], round(Int, hf * frame_h)))
    x0 = min(x0, frame_w - w + 1)        # shift in-bounds if the snap overran the edge
    y0 = min(y0, frame_h - h + 1)
    return Crop(x0, x0 + w - 1, y0, y0 + h - 1, w, h)
end

# A worker's init input: its (resolved) crop, its channels, its telemetry
# reference (preloaded by `align` from one arrow read), and fps. `out_ch ===
# nothing` ⇒ last in the chain ⇒ recycle frames back to `pool`.
struct StageConfig
    crop::Crop
    window::Matrix{Float64}       # Hann over the crop, built once by `align`
    in_ch::Channel{Frame}
    out_ch::Union{Channel{Frame}, Nothing}
    pool::Channel{Frame}
    ref_t::Vector{Vector{Float64}}   # telemetry time, one vector per channel this stage votes on
    ref_x::Vector{Vector{Float64}}   # matching telemetry values (gyro yaw/pitch, GPS speed, …)
    fps::Float64
    start_s::Float64              # decode start, so series times are absolute video time
    capacity::Int                 # expected frame count; the worker preallocates series to it
end

# State = per-stage MUTABLE scratch only (buffers, plans, series). Everything
# immutable (crop, window, fps, refs) lives in StageConfig and is read from there.
abstract type State end

# The one place crop math lives: window `config`'s crop out of `frame` into
# `state.cur` (complex, DC-removed). Shared by every stage's consume!.
function crop!(state::State, config::StageConfig, frame::Frame)
    crop = config.crop; cur = state.cur; win = config.window; data = frame.data
    x0 = crop.x0; cw = crop.w; ch = crop.h
    total = 0.0
    # copy + DC sum in one contiguous pass. Direct (non-view) 2D indexing strength-
    # reduces to a stride-1 column walk — unlike copyto!(@view …), which falls to
    # generic SubArray iteration (per-element sub2ind, no memcpy).
    @fastmath @inbounds for jy in 1:ch
        col = crop.y0 + jy - 1
        @simd for jx in 1:cw
            v = data[x0 - 1 + jx, col]
            cur[jx, jy] = v
            total += v
        end
    end
    mean = total / (cw * ch)
    @fastmath @inbounds @simd for i in eachindex(cur)      # demean before window, vectorized
        cur[i] = (cur[i] - mean) * win[i]
    end
    return cur
end

# Bump the sample count and stamp the (absolute video) time; return the new index.
# Each stage then writes its named series at that index.
function advance!(state::State, config::StageConfig, frame::Frame)
    state.n += 1
    state.times[state.n] = config.start_s + frame.index / config.fps
    return state.n
end

# ── interface (stubs — fill once inputs are settled) ─────────────────────────

# The decoder reads the next frame's bytes straight into `frame.raw`; this stamps
# the index and SIMD-widens raw (UInt8) → data (Float64). No copy, no temp buffer.
function update_frame!(frame::Frame, index::Int)
    frame.index = index
    raw = frame.raw; data = frame.data
    @inbounds @simd for i in eachindex(data)      
        data[i] = Float64(raw[i])
    end
    return frame
end


# One stage as an async task: allocate its own state, fold every frame, pass it
# on (or recycle if last), then correlate when the input channel closes.
function run_worker(stage::Stage, state::State, config::StageConfig)
    for frame in config.in_ch
        consume!(stage, state, config, frame)
        config.out_ch === nothing ? put!(config.pool, frame) : put!(config.out_ch, frame)
    end
    config.out_ch === nothing || close(config.out_ch)
    return correlate(stage, state, config)
end

# Decoder task: owns ffmpeg, streams full gray frames into `out_ch`. Pulls a
# recycled Frame from `pool`, reads the next frame straight into its `raw`, widens
# to `data`, and forwards it. Closes `out_ch` at end-of-stream to drain the chain.
function run_decoder(out_ch::Channel{Frame}, pool::Channel{Frame}, video_path::AbstractString,
                     vf::String, start_s::Real, dur_s::Real, fps::Real)
    targs = isfinite(dur_s) ? ["-t", string(dur_s)] : String[]
    exe = ffmpeg_exe()
    io = open(`$exe -hide_banner -loglevel error $(hwaccel_args()) -ss $start_s $targs -i $video_path -vf $vf -r $fps -f rawvideo pipe:1`, "r")
    index = 0
    while true
        frame = take!(pool)
        try
            read!(io, frame.raw)
        catch e
            e isa EOFError && (put!(pool, frame); break)
            rethrow()
        end
        update_frame!(frame, index)
        put!(out_ch, frame)
        index += 1
    end
    close(io)
    close(out_ch)
end

# Build the pool + daisy-chained channels (decoder → Rotation → Forward → pool),
# spawn the decoder and one worker per stage, and gather every channel's estimate.
# States are built here (serially) before spawning — FFTW planning isn't thread-safe.
function align(video_path::AbstractString, arrow_path::AbstractString;
               start_s::Real = 300.0, dur_s::Real = 900.0, fps::Real = 30.0,
               frame_w::Int = 320, frame_h::Int = 180,
               rotation_crop = (0.25, 0.50, 0.22, 0.28),
               forward_crop  = (0.18, 0.64, 0.30, 0.34))
    # one read; Float64 + drop rows where any channel is non-finite (guard at the load site).
    time, yaw, pitch, speed, roll = load_channels(arrow_path,
        :Time, :ChassisRotVelYawIDR, :ChassisRotVelPitchIDR, :VectorGPS_Speed, :ChassisRotVelRollIDR)

    # ffmpeg returns a bit more than dur·fps (its -ss seeks to a keyframe just before
    # start_s, decoding some pre-roll); over-allocate so the workers never overflow.
    slack_s = max(2.0, 0.01 * Float64(dur_s))
    capacity = ceil(Int, (Float64(dur_s) + slack_s) * fps) + 16
    rot = Crop(rotation_crop..., frame_w, frame_h)
    fwd = Crop(forward_crop...,  frame_w, frame_h)

    pool = Channel{Frame}(10)
    for _ in 1:10
        put!(pool, Frame(frame_w, frame_h))
    end
    ch_rot = Channel{Frame}(4)
    ch_fwd = Channel{Frame}(4)

    cfg_rot = StageConfig(rot, _hann(rot.w, rot.h), ch_rot, ch_fwd,  pool,
                          [time, time], [yaw, pitch], Float64(fps), Float64(start_s), capacity)
    cfg_fwd = StageConfig(fwd, _hann(fwd.w, fwd.h), ch_fwd, nothing, pool,
                          [time, time], [speed, roll], Float64(fps), Float64(start_s), capacity)

    # build states serially (FFTW planning), then run each stage on its own task
    state_rot = make_state(Rotation(), cfg_rot)
    state_fwd = make_state(Forward(),  cfg_fwd)

    vf = "scale=$(frame_w):$(frame_h),format=gray"
    decoder = Threads.@spawn run_decoder(ch_rot, pool, video_path, vf, start_s, dur_s, fps)
    w_rot = Threads.@spawn run_worker(Rotation(), state_rot, cfg_rot)
    w_fwd = Threads.@spawn run_worker(Forward(),  state_fwd, cfg_fwd)

    wait(decoder)
    rot_ests = fetch(w_rot)      # [yaw, pitch]
    fwd_ests = fetch(w_fwd)      # [forward, roll]
    return (yaw = rot_ests[1], pitch = rot_ests[2], forward = fwd_ests[1], roll = fwd_ests[2])
end

# ─────────────────────────────────────────────────────────────────────────────
# Yaw / Pitch — camera rotation by phase correlation on a far-field crop. Both
# share the machinery; Yaw keeps the horizontal shift, Pitch the vertical.
# ─────────────────────────────────────────────────────────────────────────────

# Separable Hann window, cw × ch.
function _hann(cw::Int, ch::Int)
    wx = [0.5 - 0.5cos(2π * (i - 1) / max(cw - 1, 1)) for i in 1:cw]
    wy = [0.5 - 0.5cos(2π * (i - 1) / max(ch - 1, 1)) for i in 1:ch]
    return wx * wy'
end

mutable struct RotationState{P, IP} <: State
    cur::Matrix{Float64}          # windowed crop (REAL → rfft), filled by crop!
    cur_freq::Matrix{ComplexF64}  # rfft half-spectrum (cw÷2+1) × ch
    prev_freq::Matrix{ComplexF64}
    cross::Matrix{ComplexF64}
    corr::Matrix{Float64}         # real correlation surface (brfft output)
    plan::P
    iplan::IP
    have_prev::Bool
    yaw::Vector{Float64}          # horizontal-shift series, one sample per frame
    pitch::Vector{Float64}        # vertical-shift series
    times::Vector{Float64}
    n::Int
end

function make_state(::Rotation, config::StageConfig)
    cw = config.crop.w; ch = config.crop.h
    cur = Matrix{Float64}(undef, cw, ch)
    cur_freq = Matrix{ComplexF64}(undef, cw ÷ 2 + 1, ch); cross = similar(cur_freq)
    corr = Matrix{Float64}(undef, cw, ch)
    cap = config.capacity
    return RotationState(cur, cur_freq, similar(cur_freq), cross, corr,
        plan_rfft(cur), plan_brfft(cross, cw), false,
        Vector{Float64}(undef, cap),    # yaw
        Vector{Float64}(undef, cap),    # pitch
        Vector{Float64}(undef, cap), 0) # times, n
end

# (dx, dy) sub-pixel shift between this crop and the previous; (NaN, NaN) priming.
function shift!(state::RotationState, config::StageConfig, frame::Frame)
    crop!(state, config, frame)
    mul!(state.cur_freq, state.plan, state.cur)
    if !state.have_prev
        state.prev_freq, state.cur_freq = state.cur_freq, state.prev_freq
        state.have_prev = true
        return (NaN, NaN)
    end
    cross = state.cross; prev = state.prev_freq; cur = state.cur_freq
    @fastmath @inbounds @simd for i in eachindex(cross)   # cross-power + phase-normalize, one pass
        z = prev[i] * conj(cur[i])
        cross[i] = z / (sqrt(abs2(z)) + eps())
    end
    mul!(state.corr, state.iplan, state.cross)
    state.prev_freq, state.cur_freq = state.cur_freq, state.prev_freq   # cur becomes next prev
    return peak_shift(state.corr)
end

# Sub-pixel peak of a (real) phase-correlation surface → (dx, dy), wraparound handled.
function peak_shift(corr)
    nx, ny = size(corr); bx = 1; by = 1; best = -Inf
    @inbounds for y in 1:ny, x in 1:nx
        v = corr[x, y]
        if v > best; best = v; bx = x; by = y; end
    end
    peak = corr[bx, by]
    dx = (bx - 1) + _parabolic_peak(corr[mod1(bx - 1, nx), by], peak, corr[mod1(bx + 1, nx), by])
    dy = (by - 1) + _parabolic_peak(corr[bx, mod1(by - 1, ny)], peak, corr[bx, mod1(by + 1, ny)])
    dx = dx >= nx / 2 ? dx - nx : dx
    dy = dy >= ny / 2 ? dy - ny : dy
    return (dx, dy)
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward — vehicle speed by Fourier-Mellin: FFT magnitude → log-polar →
# phase-correlate consecutive frames; the radial (log-scale) shift is the scene's
# zoom rate ∝ speed. Translation-invariant, so cornering doesn't contaminate it.
# ─────────────────────────────────────────────────────────────────────────────

# Sub-bin (angular, radial) shift of the log-polar phase-correlation peak.
# angular ⇒ roll, radial ⇒ zoom (∝ speed). Signed, wraparound on both axes.
function logpolar_peak(corr)
    n_angle, n_radius = size(corr)
    best = -Inf; pa = 1; pr = 1
    @inbounds for b in 1:n_radius, a in 1:n_angle
        v = corr[a, b]
        if v > best; best = v; pa = a; pr = b; end
    end
    peak = corr[pa, pr]
    angular = (pa - 1) + _parabolic_peak(corr[mod1(pa - 1, n_angle), pr], peak, corr[mod1(pa + 1, n_angle), pr])
    radial  = (pr - 1) + _parabolic_peak(corr[pa, mod1(pr - 1, n_radius)], peak, corr[pa, mod1(pr + 1, n_radius)])
    angular = angular > n_angle  / 2 ? angular - n_angle  : angular
    radial  = radial  > n_radius / 2 ? radial  - n_radius : radial
    return angular, radial
end

mutable struct ForwardState{P, LP, LIP} <: State
    cur::Matrix{Float64}             # windowed crop (REAL now → rfft), filled by crop!
    cur_freq::Matrix{ComplexF64}     # rfft half-spectrum: (cw÷2+1) × ch
    mag::Matrix{Float64}             # high-passed |half-spectrum|, y-fftshifted
    highpass::Matrix{Float64}        # FM high-pass over the half-spectrum
    shy::Vector{Int}                 # y-fftshift only (rfft puts DC at row 1 in x → no shx)
    sample_x::Matrix{Float64}
    sample_y::Matrix{Float64}
    lp::Matrix{Float64}              # this frame's log-polar magnitude (REAL → rfft)
    lp_freq::Matrix{ComplexF64}      # its rfft half-spectrum (n_angle÷2+1) × n_radius
    lp_prev_freq::Matrix{ComplexF64} # last frame's, carried over — no recompute
    lp_cross::Matrix{ComplexF64}
    lp_corr::Matrix{Float64}         # real correlation surface (brfft output)
    img_plan::P
    lp_plan::LP
    lp_iplan::LIP
    n_angle::Int
    n_radius::Int
    have_prev::Bool
    zoom::Vector{Float64}        # radial-shift series (∝ speed)
    roll::Vector{Float64}        # angular-shift series (∝ roll rate)
    times::Vector{Float64}
    n::Int
end

function make_state(::Forward, config::StageConfig; n_angle::Int = 128, n_radius::Int = 64)
    cw = config.crop.w; ch = config.crop.h
    rw = cw ÷ 2 + 1                                             # rfft half-spectrum rows (fx ≥ 0)
    half_y = ch ÷ 2
    # Reddy–Chatterji high-pass over the half-spectrum: fx = (row-1)≥0, fy centred at col half_y+1.
    highpass = Matrix{Float64}(undef, rw, ch)
    @inbounds for j in 1:ch, i in 1:rw
        fx = (i - 1) / cw                                       # 0 .. 0.5
        fy = (j - (half_y + 1)) / ch                            # -0.5 .. ~0.5
        cosine = cos(π * fx) * cos(π * fy)
        highpass[i, j] = (1 - cosine) * (2 - cosine)
    end
    shy = [((j - 1 + half_y) % ch) + 1 for j in 1:ch]          # y-fftshift only (DC already at row 1)
    # log-polar grid on DC = (row 1, col half_y+1), sampling the fx ≥ 0 half-plane
    # (angles -π/2 .. π/2) — the magnitude spectrum is point-symmetric, so half covers it.
    cx = 1.0; cy = half_y + 1.0
    rmax = min(cw ÷ 2, half_y) - 1.0
    log_radii = range(log(3.0), log(rmax); length = n_radius)
    sample_x = Matrix{Float64}(undef, n_angle, n_radius); sample_y = similar(sample_x)
    @inbounds for a in 1:n_angle
        angle = -π / 2 + π * (a - 1) / n_angle; cos_a = cos(angle); sin_a = sin(angle)
        for b in 1:n_radius
            radius = exp(log_radii[b])
            sample_x[a, b] = clamp(cx + radius * cos_a, 1.0, rw - 1e-3)
            sample_y[a, b] = clamp(cy + radius * sin_a, 1.0, ch - 1e-3)
        end
    end
    cur = Matrix{Float64}(undef, cw, ch)
    cur_freq = Matrix{ComplexF64}(undef, rw, ch)
    lp = Matrix{Float64}(undef, n_angle, n_radius)
    lp_freq = Matrix{ComplexF64}(undef, n_angle ÷ 2 + 1, n_radius); lp_cross = similar(lp_freq)
    lp_corr = Matrix{Float64}(undef, n_angle, n_radius)
    return ForwardState(
        cur, cur_freq, Matrix{Float64}(undef, rw, ch), highpass, shy, sample_x, sample_y,
        lp, lp_freq, similar(lp_freq), lp_cross, lp_corr,
        plan_rfft(cur), plan_rfft(lp), plan_brfft(lp_cross, n_angle), n_angle, n_radius, false,
        Vector{Float64}(undef, config.capacity),    # zoom
        Vector{Float64}(undef, config.capacity),    # roll
        Vector{Float64}(undef, config.capacity), 0) # times, n
end

# The FM pass: window the crop → image FFT → centred magnitude → log-polar →
# phase-correlate vs the previous → (zoom, roll). (NaN, NaN) while priming.
function forward_zoom!(state::ForwardState, config::StageConfig, frame::Frame)
    crop!(state, config, frame)
    mul!(state.cur_freq, state.img_plan, state.cur)
    magnitude!(state)
    logpolar!(state)                                # → state.lp (real)
    mul!(state.lp_freq, state.lp_plan, state.lp)    # the only log-polar FFT this frame
    @inbounds state.lp_freq[1, 1] = 0               # DC removal ≡ demean pre-FFT (no window → exact)
    if !state.have_prev
        state.lp_prev_freq, state.lp_freq = state.lp_freq, state.lp_prev_freq
        state.have_prev = true
        return (NaN, NaN)
    end
    roll, zoom = lp_shift!(state)
    state.lp_prev_freq, state.lp_freq = state.lp_freq, state.lp_prev_freq   # carry forward
    return (zoom, roll)
end

# rfft half-spectrum → high-passed |spectrum|, y-fftshifted. DC sits at row 1 (fx≥0)
# so there's no x-shift — each column is one contiguous run, abs·highpass goes wide (sqrt∘abs2 vectorizes
function magnitude!(state::ForwardState)
    cur_freq = state.cur_freq; mag = state.mag; hp = state.highpass; shy = state.shy
    rw, ch = size(mag)
    @fastmath @inbounds for j in 1:ch
        col = shy[j]                                   # y-fftshift; x needs none (DC at row 1)
        @simd for i in 1:rw                            # contiguous column read of the half-spectrum
            mag[i, j] = sqrt(abs2(cur_freq[i, col])) * hp[i, j]
        end
    end
    return mag
end

# mag → state.lp: bilinear log-polar sample (separable muladd lerp → FMAs), real.
# DC is NOT removed here — it's killed by zeroing the DC bin after the FFT
# (forward_zoom!), exactly equivalent (no window on the log-polar, so no leakage)
# and saves the sum + demean passes over `lp`.
function logpolar!(state::ForwardState)
    mag = state.mag; out = state.lp; sx = state.sample_x; sy = state.sample_y
    @inbounds for b in 1:state.n_radius, a in 1:state.n_angle
        x = sx[a, b]; y = sy[a, b]
        xi = floor(Int, x); yi = floor(Int, y); fx = x - xi; fy = y - yi
        m00 = mag[xi, yi];     m10 = mag[xi + 1, yi]
        m01 = mag[xi, yi + 1]; m11 = mag[xi + 1, yi + 1]
        top = muladd(fx, m10 - m00, m00)               # lerp in x at yi
        bot = muladd(fx, m11 - m01, m01)               # lerp in x at yi+1
        out[a, b] = muladd(fy, bot - top, top)         # lerp in y
    end
    return out
end

# Phase-correlate the carried previous log-polar FFT vs this frame's → (angular, radial)
# = (roll, zoom). Cross-power + phase-normalize fused into ONE pass (sqrt∘abs2, not hypot).
function lp_shift!(state::ForwardState)
    cross = state.lp_cross
    prev = state.lp_prev_freq
    cur = state.lp_freq
    @fastmath @inbounds @simd for i in eachindex(cross)
        z = prev[i] * conj(cur[i])
        cross[i] = z / (sqrt(abs2(z)) + eps())
    end
    mul!(state.lp_corr, state.lp_iplan, cross)
    return logpolar_peak(state.lp_corr)
end

# ── consume! — fold one frame into a stage's series (all stages, together) ────
function consume!(::Rotation, state::RotationState, config::StageConfig, frame::Frame)
    dx, dy = shift!(state, config, frame)
    if !isnan(dx)
        i = advance!(state, config, frame)
        # Sign convention (proven, see scratchpad/sign_{calib,telem}.jl): the proxy
        # measures world-in-frame motion, opposite to the car's rotation — dx>0 = yaw
        # RIGHT, dy>0 = pitch DOWN. The gyros are x-fwd/y-left/z-up: ChassisRotVelYawIDR>0
        # = LEFT (−0.87 vs GPS heading), pitch>0 = nose up. Negate so proxy shares the
        # telemetry sign (proxy+ ↔ telem+), letting the correlation lock the sign.
        state.yaw[i]   = -dx
        state.pitch[i] = -dy
    end
    return nothing
end

function consume!(::Forward, state::ForwardState, config::StageConfig, frame::Frame)
    zoom, roll = forward_zoom!(state, config, frame)
    if !isnan(zoom)
        i = advance!(state, config, frame)
        state.zoom[i] = zoom
        state.roll[i] = roll
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Correlation — at end-of-stream, slide each stage's series against its telemetry
# reference and return the offset (telemetry_time = video_time + offset) as an
# AlignEstimate. Rotation band-passes (sign-locked peak — proxy is sign-matched to the
# gyro); Forward smooths (signed peak — the speed proxy and speed share sign).
# ─────────────────────────────────────────────────────────────────────────────


_znorm(x) = (m = mean(x); s = std(x); s == 0 ? x .- m : (x .- m) ./ s)


# Normalized sliding cross-correlation: out[L+1] = Pearson(template, ref[L+1:L+m]).
# prefix/prefix2 are zero-padded prefix sums of ref → branch-free window stats.
# Serial on purpose: the three stages already run concurrently as pipeline tasks,
# so threading here would just oversubscribe the same pool.
function _ncc!(out::Vector{Float64}, template::Vector{Float64}, ref::Vector{Float64},
               prefix::Vector{Float64}, prefix2::Vector{Float64}, m::Int, Lmax::Int)
    @inbounds for L in 0:Lmax
        acc = 0.0
        @simd for i in 1:m
            acc += template[i] * ref[L + i]
        end
        μ = (prefix[L + m + 1] - prefix[L + 1]) / m
        σ2 = (prefix2[L + m + 1] - prefix2[L + 1]) / m - μ * μ
        out[L + 1] = σ2 > 1e-12 ? acc / (sqrt(σ2) * m) : 0.0
    end
    return out
end

# Slide the (conditioned, z-normed) series against the conditioned reference and
# return (offset_s, confidence). `condition` is the activity envelope (see _activity);
# its output is non-negative, so the lock is always a positive NCC peak.
function _crosscorr(state::State, series, ref_t, ref_x, fs::Float64, condition)
    vtimes = view(state.times, 1:state.n)
    _, template = _resample(vtimes, view(series, 1:state.n), fs)
    ref_t0, ref = _resample(ref_t, ref_x, fs)
    template = _znorm(condition(template, fs))
    ref = condition(ref, fs)
    m, n = length(template), length(ref)
    m < n || error("template ($m) longer than reference ($n) — shorten the window")
    prefix  = pushfirst!(cumsum(ref), 0.0)
    prefix2 = pushfirst!(cumsum(ref .^ 2), 0.0)
    Lmax = n - m
    ncc = Vector{Float64}(undef, Lmax + 1)
    _ncc!(ncc, template, ref, prefix, prefix2, m, Lmax)
    k = argmax(ncc)
    sub = 1 < k < length(ncc) ? _parabolic_peak(ncc[k - 1], ncc[k], ncc[k + 1]) : 0.0
    offset = (ref_t0 + (k - 1 + sub) / fs) - first(vtimes)
    return offset, ncc[k]
end

# Activity envelope: rectify + light low-pass — the conditioning that actually locks.
# Both the video proxy and the gyro share the on-track-vs-pit / corner-by-corner energy;
# band-passing it (the old approach) deleted the slow envelope that carries the lock.
# `abs` is sign-invariant, so the −dx/−dy sign bake doesn't matter for the correlation.
_activity(x, fs; smooth_s = 2.0) = _moving_average(x, max(1, round(Int, smooth_s * fs)); f = abs)

# Rotation: yaw and pitch activity envelopes, each vs its gyro.
function correlate(::Rotation, state::RotationState, config::StageConfig)
    yo, yc = _crosscorr(state, state.yaw,   config.ref_t[1], config.ref_x[1], 30.0, _activity)
    po, pc = _crosscorr(state, state.pitch, config.ref_t[2], config.ref_x[2], 30.0, _activity)
    return [AlignEstimate(yo, yc, :yaw,   (n_frames = state.n,)),
            AlignEstimate(po, pc, :pitch, (n_frames = state.n,))]
end

# Forward: zoom (∝ speed) and roll activity envelopes, vs GPS speed and the roll gyro.
function correlate(::Forward, state::ForwardState, config::StageConfig)
    fo, fc = _crosscorr(state, state.zoom, config.ref_t[1], config.ref_x[1], 30.0, _activity)
    ro, rc = _crosscorr(state, state.roll, config.ref_t[2], config.ref_x[2], 30.0, _activity)
    return [AlignEstimate(fo, fc, :forward, (n_frames = state.n,)),
            AlignEstimate(ro, rc, :roll,    (n_frames = state.n,))]
end
