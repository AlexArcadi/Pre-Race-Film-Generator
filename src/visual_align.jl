using FFTW
using DSP
using Statistics
using Arrow
using Tables

# ─────────────────────────────────────────────────────────────────────────────
# Visual ↔ telemetry alignment.
#
# Companion to the audio↔RPM aligner in `alignment.jl`, for clips with no usable
# engine audio (e.g. radio-only in-car feeds). It recovers the camera's yaw/pitch
# RATE from the video — phase correlation on consecutive frames, restricted to a
# far-field "horizon band" where image motion is ~pure rotation (depth-independent,
# parallax-free) — and cross-correlates that against the chassis rate gyros
# (`ChassisRotVelYawIDR` / `ChassisRotVelPitchIDR`).
#
# We only need a signal PROPORTIONAL to the gyro (sign + scale free) because we
# sync by cross-correlation: no camera intrinsics, no undistortion, no rad/s.
#
# Validated on Watkins Glen car 16: visual joint offset −594.5s vs the audio↔RPM
# offset −594.0s — two physically independent estimators agreeing to 0.5s.
# ─────────────────────────────────────────────────────────────────────────────

# Camera-dependent crop (fractions of frame) isolating distant content: grandstands,
# horizon, vanishing point. EXCLUDE near foreground (hood/road) and static cockpit
# (A-pillars, banners) — those pin the correlation to zero or to translation flow.
const VISUAL_CROP_DEFAULT = (x0 = 0.25, w = 0.50, y0 = 0.22, h = 0.28)

# ── frame preprocessing ─────────────────────────────────────────────────────
# raw gray frame (already downscaled by ffmpeg) -> windowed, DC-removed Float64
function _vs_preprocess(frame::AbstractMatrix{<:Real})
    g = Float64.(frame)
    g .-= mean(g)
    h, w = size(g)
    wy = 0.5 .* (1 .- cos.(2π .* (0:h-1) ./ max(h - 1, 1)))
    wx = 0.5 .* (1 .- cos.(2π .* (0:w-1) ./ max(w - 1, 1)))
    return g .* (wy * wx')
end

# Sub-sample peak interpolation — single shared implementation lives in
# alignment.jl (`_parabolic_peak`), so the audio and visual aligners localize
# their correlation peaks with identical math.
_vs_parabolic(l, c, r) = _parabolic_peak(l, c, r)
_vs_wrapget(r, i, j, h, w) = r[mod(i - 1, h) + 1, mod(j - 1, w) + 1]

"""
    _vs_phase_shift(a, b) -> (dx, dy, peak)

Sub-pixel global translation mapping preprocessed frame `a` onto `b` via the
normalized cross-power spectrum. `dx` = horizontal shift (yaw proxy), `dy` =
vertical (pitch proxy). `peak` is a peak-to-mean confidence.
"""
function _vs_phase_shift(a::AbstractMatrix, b::AbstractMatrix)
    A = fft(a); B = fft(b)
    R = A .* conj(B); R ./= abs.(R) .+ eps()
    r = real(ifft(R))
    py, px = Tuple(argmax(r)); h, w = size(r)
    dy = (py - 1) + _vs_parabolic(_vs_wrapget(r, py-1, px, h, w), r[py, px], _vs_wrapget(r, py+1, px, h, w))
    dx = (px - 1) + _vs_parabolic(_vs_wrapget(r, py, px-1, h, w), r[py, px], _vs_wrapget(r, py, px+1, h, w))
    dy = dy >= h/2 ? dy - h : dy
    dx = dx >= w/2 ? dx - w : dx
    peak = r[py, px] / (sum(abs, r) / length(r) + eps())
    return dx, dy, peak
end

# ── video → rotation track ──────────────────────────────────────────────────
"""
    video_rotation_track(video_path; start_s, dur_s, fps, crop, outw, outh, backend)
        -> (t, yaw, pitch, peak)

ffmpeg-downconverts a window of the video to a small grayscale horizon-band
stream (piped, no temp files) and phase-correlates consecutive frames. `t` is in
video clip-time (seconds). Uses the repo ffmpeg backend so the bundled binary's
libs resolve.
"""
function video_rotation_track(video_path::AbstractString;
                              start_s::Real, dur_s::Real, fps::Real = 30.0,
                              crop = VISUAL_CROP_DEFAULT, outw::Int = 256, outh::Int = 80,
                              backend::FfmpegBackend = detect_backend())
    vf = "crop=iw*$(crop.w):ih*$(crop.h):iw*$(crop.x0):ih*$(crop.y0),scale=$(outw):$(outh),format=gray"
    # GPU-decode when the backend has NVDEC (-hwaccel cuda); frames copy back to
    # system memory so the CPU crop/scale/gray filters still apply. No-op on the
    # bundled backend (empty hwaccel_args).
    bytes = with_backend(backend) do exe
        read(`$exe -hide_banner -loglevel error $(backend.hwaccel_args) -ss $start_s -t $dur_s -i $video_path -vf $vf -r $fps -f rawvideo pipe:1`)
    end
    fsz = outw * outh
    nframes = length(bytes) ÷ fsz
    nframes < 2 && error("video_rotation_track: got $nframes frames (need ≥2) at start=$start_s")
    t = Float64[]; yaw = Float64[]; pitch = Float64[]; pk = Float64[]
    prev = nothing
    @inbounds for k in 0:nframes-1
        off = k * fsz
        frame = permutedims(reshape(view(bytes, off+1:off+fsz), outw, outh))  # -> [y,x]
        cur = _vs_preprocess(frame)
        if prev !== nothing
            dx, dy, p = _vs_phase_shift(prev, cur)
            push!(yaw, dx); push!(pitch, dy); push!(pk, p)
            push!(t, start_s + (k - 0.5) / fps)
        end
        prev = cur
    end
    return (t = t, yaw = yaw, pitch = pitch, peak = pk)
end

# ── signal conditioning + global cross-correlation search ───────────────────
_vs_znorm(x) = (m = mean(x); s = std(x); s == 0 ? (x .- m) : (x .- m) ./ s)

function _vs_resample(t, x, fs; t0 = first(t), t1 = last(t))
    g = collect(t0:(1/fs):t1)
    out = similar(g)
    @inbounds for (i, q) in enumerate(g)
        if q <= first(t); out[i] = float(first(x))
        elseif q >= last(t); out[i] = float(last(x))
        else
            j = searchsortedlast(t, q); w = (q - t[j]) / (t[j+1] - t[j])
            out[i] = (1 - w) * x[j] + w * x[j+1]
        end
    end
    return g, out
end

function _vs_bandpass(x, fs; lo = 0.1, hi = 8.0, order = 4)
    nyq = fs / 2; hi = min(hi, 0.95 * nyq); lo = max(lo, 1e-3)
    flt = digitalfilter(Bandpass(lo, hi), Butterworth(order); fs = fs)
    return filtfilt(flt, float.(collect(x)))
end

"""
    _vs_ncc_kernel!(out, V, R, cs, cs2, m, Lmax)

Normalized sliding cross-correlation: `out[L+1]` = Pearson(`V`, `R[L+1:L+m]`) for
`L in 0:Lmax`. `cs`/`cs2` are ZERO-PADDED prefix sums of `R` (`cs[1]=0`) so the
window sum is `cs[L+m+1]-cs[L+1]` — no per-lag branch. Kept as its own typed
function (a "function barrier") so nothing boxes under `Threads.@threads`; the
inner reduction is `@simd` (emits `vfmadd`).
"""
function _vs_ncc_kernel!(out::Vector{Float64}, V::Vector{Float64}, R::Vector{Float64},
                         cs::Vector{Float64}, cs2::Vector{Float64}, m::Int, Lmax::Int)
    Threads.@threads for L in 0:Lmax
        d = 0.0
        @inbounds @simd for i in 1:m
            d += V[i] * R[L+i]
        end
        @inbounds begin
            μ  = (cs[L+m+1]  - cs[L+1])  / m
            σ2 = (cs2[L+m+1] - cs2[L+1]) / m - μ * μ
            out[L+1] = σ2 > 1e-12 ? d / (sqrt(σ2) * m) : 0.0
        end
    end
    return out
end

"""
    _vs_xcorr_search(vt, vx, rt, rx; fs, lo, hi) -> (Δgrid, ncc)

Sign-invariant normalized cross-correlation of a short video template against a
long telemetry reference. Returns the full curve: `Δgrid` (offset s, where
telemetry_time = video_time + Δ) and per-offset Pearson `ncc`.
"""
function _vs_xcorr_search(vt, vx, rt, rx; fs = 30.0, lo = 0.1, hi = 8.0)
    _, V = _vs_resample(vt, vx, fs)
    Rg, R = _vs_resample(rt, rx, fs)
    V = _vs_znorm(_vs_bandpass(V, fs; lo, hi))
    R = _vs_bandpass(R, fs; lo, hi)
    m, n = length(V), length(R)
    m < n || error("visual template ($m) must be shorter than telemetry ref ($n)")
    v_t0 = first(vt); r_t0 = first(Rg)
    cs  = pushfirst!(cumsum(R), 0.0)       # zero-padded prefix sums (branch-free windows)
    cs2 = pushfirst!(cumsum(R .^ 2), 0.0)
    K = n - m
    ncc = Vector{Float64}(undef, K + 1)
    _vs_ncc_kernel!(ncc, V, R, cs, cs2, m, K)
    Δgrid = [(r_t0 + k / fs) - v_t0 for k in 0:K]
    return Δgrid, ncc
end

"""
    _vs_refine_curve(vt, vx, rt, rx, Δ0; fs=240, halfwin=2.0, lo, hi) -> (δgrid, corr)

High-resolution local cross-correlation in a ±`halfwin` window around a coarse
offset `Δ0`, resampled to `fs` Hz. Because both signals are bandlimited, the
peak of this finely-sampled curve (+ parabolic interpolation) localizes the
offset far below the original frame spacing. SIMD inner / threaded outer.
"""
function _vs_refine_curve(vt, vx, rt, rx, Δ0; fs = 240.0, halfwin = 2.0, lo = 0.1, hi = 8.0)
    _, V = _vs_resample(vt, vx, fs; t0 = first(vt), t1 = last(vt))
    V = _vs_znorm(_vs_bandpass(V, fs; lo = lo, hi = hi))
    m = length(V)
    tlo = first(vt) + Δ0 - halfwin; thi = last(vt) + Δ0 + halfwin
    _, R = _vs_resample(rt, rx, fs; t0 = tlo, t1 = thi)
    R = _vs_bandpass(R, fs; lo = lo, hi = hi)
    nL = length(R) - m
    nL < 2 && error("refine window too small (got nL=$nL)")
    cs  = pushfirst!(cumsum(R), 0.0)
    cs2 = pushfirst!(cumsum(R .^ 2), 0.0)
    corr = Vector{Float64}(undef, nL + 1)
    _vs_ncc_kernel!(corr, V, R, cs, cs2, m, nL)
    δgrid = [Δ0 - halfwin + L / fs for L in 0:nL]
    return δgrid, corr
end

# top-K peaks of a curve with a minimum spacing (offset units = seconds)
function _vs_top_k(Δgrid, score, fs; k = 12, min_spacing_s = 30.0)
    order = sortperm(score; rev = true)
    sel = Tuple{Float64,Float64}[]
    sp = min_spacing_s
    for j in order
        d = Δgrid[j]
        if all(abs(d - s[1]) >= sp for s in sel)
            push!(sel, (d, score[j])); length(sel) >= k && break
        end
    end
    return sel
end

function _vs_load_rate(arrow_path, channel)
    tbl = Arrow.Table(arrow_path)
    T = Float64.(collect(Tables.getcolumn(tbl, :Time)))
    X = Float64.(collect(Tables.getcolumn(tbl, Symbol(channel))))
    m = isfinite.(T) .& isfinite.(X)   # drop pre-green NaNs
    return T[m], X[m]
end

"""
    align_visual_rotation(video_path, arrow_path; start_s=600, dur_s=300,
                          crop=VISUAL_CROP_DEFAULT, fs=30, band=(0.1,8.0),
                          seed=nothing, seed_tol_s=60, backend=detect_backend())
        -> NamedTuple

Estimate the video↔telemetry offset from camera rotation. Returns `offset_s`
(telemetry_time = video_time + offset_s — SAME sign convention as
`align_audio_rpm`), a `confidence` (joint |ncc| at the lock), the per-channel
locks, and `candidate_peaks` (the lap-aliased comb; pass `seed` — e.g. a coarse
offset from wall-clock or GPS-speed — to pick the right lap).
"""
function align_visual_rotation(video_path::AbstractString, arrow_path::AbstractString;
                               start_s::Real = 600.0, dur_s::Real = 300.0,
                               crop = VISUAL_CROP_DEFAULT, fs::Real = 30.0,
                               band::Tuple{Real,Real} = (0.1, 8.0),
                               seed::Union{Nothing,Real} = nothing, seed_tol_s::Real = 60.0,
                               fs_fine::Real = 240.0, refine_halfwin_s::Real = 2.0,
                               backend::FfmpegBackend = detect_backend())
    lo, hi = band
    track = video_rotation_track(video_path; start_s = start_s, dur_s = dur_s,
                                 fps = fs, crop = crop, backend = backend)
    Ty, Yaw   = _vs_load_rate(arrow_path, "ChassisRotVelYawIDR")
    Tp, Pitch = _vs_load_rate(arrow_path, "ChassisRotVelPitchIDR")
    Δg, ny = _vs_xcorr_search(track.t, track.yaw,   Ty, Yaw;   fs = fs, lo = lo, hi = hi)
    _,  np = _vs_xcorr_search(track.t, track.pitch, Tp, Pitch; fs = fs, lo = lo, hi = hi)
    joint = abs.(ny) .+ abs.(np)

    # ── coarse pick: within ±seed_tol of seed if given, else global max of joint
    pick = seed === nothing ? eachindex(joint) :
           findall(d -> abs(d - Float64(seed)) <= seed_tol_s, Δg)
    isempty(pick) && (pick = eachindex(joint))
    Δ0 = Δg[pick[argmax(joint[pick])]]

    # ── fine refine: re-correlate at fs_fine in a ±halfwin window, sub-sample peak
    dg, cy = _vs_refine_curve(track.t, track.yaw,   Ty, Yaw,   Δ0;
                              fs = fs_fine, halfwin = refine_halfwin_s, lo = lo, hi = hi)
    _,  cp = _vs_refine_curve(track.t, track.pitch, Tp, Pitch, Δ0;
                              fs = fs_fine, halfwin = refine_halfwin_s, lo = lo, hi = hi)
    jf = abs.(cy) .+ abs.(cp)
    # parabolic sub-sample peak of a curve -> (offset_s, peak_value)
    subpk(c) = (k = argmax(c);
                s = (1 < k < length(c)) ? _vs_parabolic(c[k-1], c[k], c[k+1]) : 0.0;
                (dg[k] + s / fs_fine, c[k]))
    off_j, pk_j = subpk(jf)
    off_y, pk_y = subpk(abs.(cy))
    off_p, pk_p = subpk(abs.(cp))

    return (
        offset_s        = off_j,
        confidence      = pk_j / 2,
        coarse_offset_s = Δ0,
        yaw_offset_s    = off_y, yaw_conf   = pk_y,
        pitch_offset_s  = off_p, pitch_conf = pk_p,
        channel_spread_s = abs(off_y - off_p),   # yaw/pitch agreement = a self-check
        window          = (start_s, dur_s),
        mean_phase_peak = mean(track.peak),
        seed            = seed,
        candidate_peaks = [(offset_s = d, conf = c) for (d, c) in _vs_top_k(Δg, joint, fs)],
        method          = :visual_rotation_xcorr,
    )
end

# ═════════════════════════════════════════════════════════════════════════════
# Forward optical-flow ↔ GPS-speed alignment (the COARSE, lap-fixing channel).
#
# The yaw/pitch rotation aligner is sharp but lap-aliased — corners repeat, so
# its correlation is a comb with one tooth per lap. This channel fixes WHICH
# tooth. It extracts a speed PROXY from the video — the inter-frame image change
# in a forward-looking foreground crop, which grows with how fast the scene
# streams past — and cross-correlates it against `VectorGPS_Speed`.
#
# The trick (per the design notes): the session SHAPE of speed — out-lap, racing
# pace, the dips at cautions/pits, the unique sequence of braking zones — is
# GLOBALLY unique over a session, not lap-periodic. So we DON'T band-pass it (a
# high-pass would delete exactly the slow envelope that makes it unique); we only
# lightly smooth. The result locks the absolute lap with no seed; yaw/pitch then
# refine the sub-second offset.
# ═════════════════════════════════════════════════════════════════════════════

# Foreground crop: lower-centre, where forward translation (road streaming past)
# dominates over far-field rotation. Excludes sky/horizon and the static cockpit.
const FORWARD_CROP_DEFAULT = (x0 = 0.30, w = 0.40, y0 = 0.55, h = 0.35)

"""
    video_forward_track(video_path; start_s=0, dur_s=Inf, fps=4, crop, outw, outh, backend)
        -> (t, forward)

Low-frame-rate speed proxy from the video: per-frame mean absolute inter-frame
pixel change in a forward foreground crop. `forward[k]` grows with vehicle speed
(scene streams past faster). Low fps is deliberate — we only need the slow
session envelope, so this stays cheap even over a whole session.
"""
function video_forward_track(video_path::AbstractString;
                             start_s::Real = 0.0, dur_s::Real = Inf, fps::Real = 4.0,
                             crop = FORWARD_CROP_DEFAULT, outw::Int = 160, outh::Int = 90,
                             backend::FfmpegBackend = detect_backend())
    vf = "crop=iw*$(crop.w):ih*$(crop.h):iw*$(crop.x0):ih*$(crop.y0),scale=$(outw):$(outh),format=gray"
    targs = isfinite(dur_s) ? ["-t", string(dur_s)] : String[]
    bytes = with_backend(backend) do exe
        read(`$exe -hide_banner -loglevel error $(backend.hwaccel_args) -ss $start_s $targs -i $video_path -vf $vf -r $fps -f rawvideo pipe:1`)
    end
    fsz = outw * outh
    nframes = length(bytes) ÷ fsz
    nframes < 2 && error("video_forward_track: got $nframes frames (need ≥2) at start=$start_s")
    t   = Vector{Float64}(undef, nframes - 1)
    fwd = Vector{Float64}(undef, nframes - 1)
    prev = Vector{UInt8}(undef, fsz)
    @inbounds for k in 0:nframes-1
        off = k * fsz
        if k > 0
            s = 0.0
            @simd for i in 1:fsz
                d = Int(bytes[off + i]) - Int(prev[i])
                s += abs(d)
            end
            fwd[k] = s / fsz
            t[k]   = start_s + (k - 0.5) / fps
        end
        copyto!(prev, 1, bytes, off + 1, fsz)
    end
    return (t = t, forward = fwd)
end

# Simple centred moving-average (zero-padded prefix sums). Used to denoise the
# speed proxy WITHOUT high-passing — we keep the slow session envelope.
function _fwd_smooth(x::AbstractVector{<:Real}, n::Int)
    n <= 1 && return Float64.(collect(x))
    m = length(x); cs = pushfirst!(cumsum(Float64.(collect(x))), 0.0)
    out = Vector{Float64}(undef, m); half = n ÷ 2
    @inbounds for i in 1:m
        lo = max(1, i - half); hi = min(m, i + half)
        out[i] = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)
    end
    return out
end

"""
    _fwd_xcorr_search(vt, vx, rt, rx; fs=4, smooth_s=3) -> (Δgrid, ncc)

Positive (sign-preserving) normalized cross-correlation of the video speed proxy
against telemetry speed. No band-pass — only light smoothing — so the slow
session envelope (the lap-fixing signal) survives. Returns the full curve;
`Δgrid` is the offset (telemetry_time = video_time + Δ).
"""
function _fwd_xcorr_search(vt, vx, rt, rx; fs = 4.0, smooth_s = 3.0)
    _, V = _vs_resample(vt, vx, fs)
    Rg, R = _vs_resample(rt, rx, fs)
    sn = max(1, round(Int, smooth_s * fs))
    V = _vs_znorm(_fwd_smooth(V, sn))
    R = _fwd_smooth(R, sn)
    m, n = length(V), length(R)
    m < n || error("forward template ($m) must be shorter than telemetry ref ($n) — shorten dur_s")
    cs  = pushfirst!(cumsum(R), 0.0)
    cs2 = pushfirst!(cumsum(R .^ 2), 0.0)
    K = n - m
    ncc = Vector{Float64}(undef, K + 1)
    _vs_ncc_kernel!(ncc, V, R, cs, cs2, m, K)        # signed Pearson; speed proxy ↔ speed is same-sign
    Δgrid = [(first(Rg) + k / fs) - first(vt) for k in 0:K]
    return Δgrid, ncc
end

"""
    align_forward_speed(video_path, arrow_path; start_s=300, dur_s=1800, fps=4,
                        smooth_s=3, backend) -> NamedTuple

Coarse, lap-unambiguous offset from forward-flow↔GPS-speed. `start_s` should be
past any pre-green/formation footage so the template lands inside the (green-
based) telemetry; `dur_s` must be shorter than the telemetry span. Returns
`offset_s` (telemetry_time = video_time + offset_s), `confidence` (peak Pearson),
and the `candidate_peaks` comb.
"""
function align_forward_speed(video_path::AbstractString, arrow_path::AbstractString;
                             start_s::Real = 300.0, dur_s::Real = 1800.0, fps::Real = 4.0,
                             smooth_s::Real = 3.0,
                             backend::FfmpegBackend = detect_backend())
    track = video_forward_track(video_path; start_s = start_s, dur_s = dur_s,
                                fps = fps, backend = backend)
    tel = load_telemetry(arrow_path)
    rt = Float64.(collect(tel.time)); rs = Float64.(collect(tel.speed))
    keep = isfinite.(rt) .& isfinite.(rs)
    rt = rt[keep]; rs = rs[keep]
    Δg, ncc = _fwd_xcorr_search(track.t, track.forward, rt, rs; fs = fps, smooth_s = smooth_s)
    k = argmax(ncc)
    return (
        offset_s        = Δg[k],
        confidence      = ncc[k],
        window          = (start_s, dur_s),
        n_video_frames  = length(track.forward),
        candidate_peaks = [(offset_s = d, conf = c) for (d, c) in _vs_top_k(Δg, ncc, fps; min_spacing_s = 20.0)],
        method          = :forward_flow_speed_xcorr,
    )
end
