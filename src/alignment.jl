using DSP
using FFTW
using FFMPEG_jll
using Statistics

"""
    extract_audio_mono(video_path; start_s, duration_s, sr=8000,
                       backend=detect_backend()) -> Vector{Float32}

Pull mono PCM samples from the video via ffmpeg. Default 8 kHz is plenty for
the firing-fundamental band (≤800 Hz). Returns Float32 in [-1, 1].
"""
function extract_audio_mono(video_path::AbstractString;
                            start_s::Real = 0.0,
                            duration_s::Union{Nothing,Real} = nothing,
                            sr::Int = 8000,
                            backend::FfmpegBackend = detect_backend())
    bytes = with_backend(backend) do exe
        args = String[exe, "-hide_banner", "-loglevel", "error",
                      "-ss", string(start_s), "-i", String(video_path)]
        duration_s !== nothing && append!(args, ["-t", string(duration_s)])
        append!(args, ["-ac", "1", "-ar", string(sr),
                       "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])
        read(Cmd(args))
    end
    isempty(bytes) && error("ffmpeg returned no audio bytes for $video_path at start=$start_s dur=$duration_s")
    n = length(bytes) - mod(length(bytes), 4)
    return collect(reinterpret(Float32, view(bytes, 1:n)))
end

"""
    rpm_to_firing_envelope(audio, sr; band=(200,800), env_hz=50) -> (env, env_sr)

Band-pass the audio around the V8 firing fundamental, take the analytic
envelope, then decimate to `env_hz`.
"""
function audio_firing_envelope(audio::AbstractVector{<:AbstractFloat}, sr::Int;
                               band::Tuple{Real,Real} = (200.0, 800.0),
                               env_hz::Int = 50)
    low, high = band
    bp_filter = digitalfilter(Bandpass(Float64(low), Float64(high)), Butterworth(4); fs = sr)
    bp = filtfilt(bp_filter, Float64.(audio))
    # Rectify + low-pass to get amplitude envelope (cheaper than analytic signal)
    rect = abs.(bp)
    lp = digitalfilter(Lowpass(Float64(env_hz) / 2), Butterworth(4); fs = sr)
    env = filtfilt(lp, rect)
    # Decimate to env_hz
    decim = max(1, fld(sr, env_hz))
    env_d = env[1:decim:end]
    env_sr = sr ÷ decim
    return env_d, env_sr
end

"""
    rpm_proxy_signal(time, rpm; env_sr=50) -> Vector{Float64}

Resample the RPM trace to the envelope's sample rate. RPM is already a
"power proxy" — when the engine is louder the RPM is higher — so we just
need it on a common time grid. Returns a vector covering [time[1], time[end]].
"""
function rpm_proxy_signal(time::AbstractVector, rpm::AbstractVector; env_sr::Int = 50)
    t0, t1 = Float64(time[1]), Float64(time[end])
    n = floor(Int, (t1 - t0) * env_sr) + 1
    out = Vector{Float64}(undef, n)
    j = 1
    @inbounds for i in 1:n
        tq = t0 + (i - 1) / env_sr
        while j < length(time) && Float64(time[j+1]) < tq
            j += 1
        end
        if j >= length(time)
            out[i] = Float64(rpm[end])
        else
            x0, x1 = Float64(time[j]), Float64(time[j+1])
            y0, y1 = Float64(rpm[j]),  Float64(rpm[j+1])
            frac = (tq - x0) / (x1 - x0)
            out[i] = y0 * (1 - frac) + y1 * frac
        end
    end
    # NaN in the source .arrow propagates through interpolation and then
    # poisons cumsum-based smoothing. Treat NaN samples as "engine off".
    @inbounds for i in eachindex(out)
        isnan(out[i]) && (out[i] = 0.0)
    end
    return out
end

zscore(v) = begin
    μ = mean(v); σ = std(v)
    σ == 0 ? v .- μ : (v .- μ) ./ σ
end

"""
    fft_xcorr_lag(ref, query, max_lag) -> (best_k, normalised_peak)

FFT-based cross-correlation: O(N log N) instead of O(N·max_lag). Returns the
integer lag `k` (in samples) maximising `Σ ref[i] * query[i+k]` over
`|k| <= max_lag`, plus a normalised confidence score in roughly [-1, 1].

`ref` and `query` must be equal-length, real, finite Float64 vectors.
Positive `k` means `query` is *ahead* of `ref` by `k` samples (i.e. `ref`
needs to shift right by `k` to align with `query`).
"""
function fft_xcorr_lag(ref::AbstractVector{<:Real},
                       query::AbstractVector{<:Real},
                       max_lag::Int;
                       seed_k::Int = 0)
    N = min(length(ref), length(query))
    r = Float64.(view(ref, 1:N))
    q = Float64.(view(query, 1:N))
    replace!(r, NaN => 0.0); replace!(q, NaN => 0.0)
    r .-= mean(r); q .-= mean(q)

    n  = nextpow(2, 2N)
    rp = vcat(r, zeros(n - N))
    qp = vcat(q, zeros(n - N))
    R  = rfft(rp); Q = rfft(qp)
    xc = irfft(conj.(R) .* Q, n)

    max_lag = clamp(max_lag, 1, n ÷ 2 - 1)
    klo = seed_k - max_lag
    khi = seed_k + max_lag
    best_k = seed_k; best_c = -Inf
    @inbounds for k in klo:khi
        idx = k >= 0 ? k + 1 : n + k + 1
        (idx < 1 || idx > n) && continue
        c = xc[idx]
        if c > best_c
            best_c = c; best_k = k
        end
    end

    norm = sqrt(sum(abs2, r) * sum(abs2, q))
    confidence = norm == 0 ? 0.0 : best_c / norm
    return best_k, confidence
end

"""
    fft_xcorr_top_k(ref, query, max_lag; k=10, min_spacing=0) -> Vector{Tuple{Int,Float64}}

Like `fft_xcorr_lag` but returns the K highest peaks within `|lag| ≤ max_lag`,
enforcing a minimum lag spacing between selected peaks (so we don't grab a
cluster of samples around the same peak). Sorted by correlation value
descending. Used by `align_audio_rpm` to short-list candidates that can be
disambiguated by session-level features.
"""
function fft_xcorr_top_k(ref::AbstractVector{<:Real},
                         query::AbstractVector{<:Real},
                         max_lag::Int;
                         k::Int = 10,
                         min_spacing::Int = 0)
    N = min(length(ref), length(query))
    r = Float64.(view(ref, 1:N))
    q = Float64.(view(query, 1:N))
    replace!(r, NaN => 0.0); replace!(q, NaN => 0.0)
    r .-= mean(r); q .-= mean(q)

    n  = nextpow(2, 2N)
    rp = vcat(r, zeros(n - N))
    qp = vcat(q, zeros(n - N))
    R  = rfft(rp); Q = rfft(qp)
    xc = irfft(conj.(R) .* Q, n)

    norm = sqrt(sum(abs2, r) * sum(abs2, q))
    max_lag = clamp(max_lag, 1, n ÷ 2 - 1)

    # Score every lag in the window
    lags = collect(-max_lag:max_lag)
    vals = Vector{Float64}(undef, length(lags))
    @inbounds for (i, lag) in enumerate(lags)
        idx = lag >= 0 ? lag + 1 : n + lag + 1
        vals[i] = norm == 0 ? 0.0 : xc[idx] / norm
    end

    # Greedy top-K with spacing constraint
    order = sortperm(vals; rev = true)
    selected = Tuple{Int,Float64}[]
    for j in order
        lag = lags[j]; val = vals[j]
        if all(abs(lag - sel[1]) >= min_spacing for sel in selected)
            push!(selected, (lag, val))
            length(selected) >= k && break
        end
    end
    return selected
end

"""
    rolling_mean(x, window) -> Vector{Float64}

Symmetric centred rolling mean, computed in one pass via cumulative sum.
Used to flatten lap-periodic structure (~54 s) before FFT cross-correlation
so the global offset peak isn't aliased onto lap-multiple sub-peaks.
"""
function rolling_mean(x::AbstractVector{<:Real}, window::Int)
    n = length(x)
    out = Vector{Float64}(undef, n)
    n == 0 && return out
    safe = [isnan(v) ? 0.0 : Float64(v) for v in x]
    cs = cumsum(safe)
    half = window ÷ 2
    @inbounds for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)
        s  = cs[hi] - (lo > 1 ? cs[lo - 1] : 0.0)
        out[i] = s / (hi - lo + 1)
    end
    return out
end

"""
    find_race_start(rpm; window_s=60, dt_s=0.01,
                    threshold=3000, margin_s=30, sustain_s=300) -> Int

Index of the first sample where the rolling-window-averaged RPM exceeds
`threshold` and stays above it for at least `sustain_s` seconds, minus a
`margin_s` safety pad. Useful as a coarse seed for alignment and for
trimming away the pre-race idle / staging period.
"""
function find_race_start(rpm::AbstractVector{<:Real};
                         window_s::Real  = 60.0,
                         dt_s::Real      = 0.01,
                         threshold::Real = 3000.0,
                         margin_s::Real  = 30.0,
                         sustain_s::Real = 300.0)
    w       = round(Int, window_s / dt_s)
    mar     = round(Int, margin_s / dt_s)
    sustain = round(Int, sustain_s / dt_s)
    rpm_f   = Float64.(rpm); replace!(rpm_f, NaN => 0.0)
    cs      = cumsum(rpm_f)
    n       = length(rpm_f)
    avg     = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        lo = i > w ? cs[i - w] : 0.0
        avg[i] = (cs[i] - lo) / min(i, w)
    end

    i = 1
    while i <= n
        rise = findnext(>=(threshold), avg, i)
        rise === nothing && break
        drop = findnext(<(threshold), avg, rise)
        duration = drop === nothing ? n - rise : drop - rise
        if duration >= sustain
            return max(1, rise - mar)
        end
        i = drop === nothing ? n + 1 : drop
    end
    return 1
end

"""
    find_audio_active_start(video_path; probe_s=10.0, step_s=60.0,
                            sr=8000, rms_threshold=0.005) -> Float64

Scan the video in `step_s` increments, extract a short `probe_s` audio
sample, and return the first video timestamp where the RMS exceeds
`rms_threshold`. Used to skip muted intros / commentary-only stretches.
Returns 0.0 if loud audio is found in the very first probe.
"""
function find_audio_active_start(video_path::AbstractString;
                                 probe_s::Real = 10.0,
                                 step_s::Real = 60.0,
                                 sr::Int = 8000,
                                 rms_threshold::Real = 0.005,
                                 search_limit_s::Real = 7200.0,
                                 backend::FfmpegBackend = detect_backend())
    t = 0.0
    while t < search_limit_s
        a = try
            extract_audio_mono(video_path; start_s = t, duration_s = probe_s,
                               sr = sr, backend = backend)
        catch
            return t
        end
        rms = sqrt(mean(abs2, a))
        rms >= rms_threshold && return t
        t += step_s
    end
    return 0.0
end

"""
    audio_rpm_trace(audio, sr; window_size=2048, hop=200,
                    freq_band_hz=(300,800), cylinders=8) -> NamedTuple

Extract an RPM-from-audio trace by tracking the firing fundamental in a
short-time Fourier transform. For a 4-stroke engine:

    firing_freq_hz = RPM / 60 × cylinders / 2
    → RPM = freq × 120 / cylinders

For NASCAR Cup V8 (cylinders=8): RPM = freq × 15. Racing RPM 6000–9000 maps
to fundamental 400–600 Hz; the default band `(300, 800)` covers that with
margin for caution-lap RPMs without slipping into 2nd-harmonic territory.

Returns `(rpm, energy, frame_hz)` — energy per frame is the in-band power,
used to mask quiet (silent or out-of-band) frames before correlation.
"""
function audio_rpm_trace(audio::AbstractVector{<:Real}, sr::Int;
                         window_size::Int = 2048,
                         hop::Int = 200,
                         freq_band_hz::Tuple{Real,Real} = (300.0, 800.0),
                         cylinders::Int = 8)
    rpm_per_hz = 120.0 / cylinders
    bin_hz     = sr / window_size
    lo_bin     = max(2, round(Int, freq_band_hz[1] / bin_hz) + 1)
    hi_bin     = min(window_size ÷ 2, round(Int, freq_band_hz[2] / bin_hz) + 1)

    n = length(audio)
    n_frames = max(0, (n - window_size) ÷ hop + 1)
    n_frames == 0 && return (rpm = Float64[], energy = Float64[], frame_hz = sr / hop)

    # Hann window
    win = [0.5 - 0.5 * cos(2π * (i - 1) / (window_size - 1)) for i in 1:window_size]
    buf  = Vector{Float64}(undef, window_size)
    plan = plan_rfft(buf)

    rpm_trace = Vector{Float64}(undef, n_frames)
    energy    = Vector{Float64}(undef, n_frames)

    @inbounds for k in 1:n_frames
        s = (k - 1) * hop + 1
        for i in 1:window_size
            buf[i] = Float64(audio[s + i - 1]) * win[i]
        end
        S = plan * buf

        peak_bin = lo_bin
        peak_mag = abs2(S[lo_bin])
        e = 0.0
        for b in lo_bin:hi_bin
            m = abs2(S[b])
            e += m
            if m > peak_mag
                peak_mag = m
                peak_bin = b
            end
        end

        # Parabolic interpolation around the peak for sub-bin precision
        if peak_bin > lo_bin && peak_bin < hi_bin
            y0 = log(abs2(S[peak_bin - 1]) + 1e-30)
            y1 = log(peak_mag + 1e-30)
            y2 = log(abs2(S[peak_bin + 1]) + 1e-30)
            denom = (y0 - 2 * y1 + y2)
            delta = abs(denom) < 1e-12 ? 0.0 : 0.5 * (y0 - y2) / denom
            peak_freq = (peak_bin - 1 + clamp(delta, -1.0, 1.0)) * bin_hz
        else
            peak_freq = (peak_bin - 1) * bin_hz
        end
        rpm_trace[k] = peak_freq * rpm_per_hz
        energy[k]    = e
    end

    return (rpm = rpm_trace, energy = energy, frame_hz = sr / hop)
end

"""
    active_boundaries(signal, smooth_window_frames, threshold) -> (start, end)

Find the indices where a signal first rises above and last falls below
`threshold`, after smoothing with a centred rolling mean. Smoothing kills
per-lap dips so we recover the SESSION-level active span (engine started →
engine stopped). Returns `(0, 0)` if nothing exceeds threshold.
"""
function active_boundaries(signal::AbstractVector{<:Real},
                           smooth_window_frames::Int,
                           threshold::Real)
    sm = rolling_mean(signal, smooth_window_frames)
    s  = findfirst(>(threshold), sm)
    e  = findlast(>(threshold), sm)
    s === nothing && return (0, 0)
    return (s, e)
end

function align_audio_rpm(video_path::AbstractString,
                         arrow_path::AbstractString;
                         band::Tuple{Real,Real} = (300.0, 800.0),
                         audio_sr::Int = 4000,
                         window_size::Int = 2048,
                         hop::Int = 200,
                         cylinders::Int = 8,
                         max_lag_s::Real = 1800.0,
                         energy_pctile::Real = 0.4,
                         k_candidates::Int = 12,
                         disambiguation_tol_s::Real = 60.0,
                         backend::FfmpegBackend = detect_backend())
    tel    = load_telemetry(arrow_path)
    t0_tel = Float64(tel.time[1])

    audio = extract_audio_mono(video_path; sr = audio_sr, backend = backend)
    a     = audio_rpm_trace(audio, audio_sr;
                            window_size = window_size, hop = hop,
                            freq_band_hz = band, cylinders = cylinders)
    frame_hz = a.frame_hz

    # Mask low-energy frames (silent / muted / out-of-band).
    e_thresh   = quantile(a.energy, energy_pctile)
    audio_rpm  = Float64[a.energy[i] >= e_thresh ? a.rpm[i] : 0.0
                         for i in eachindex(a.rpm)]

    tel_rpm = rpm_proxy_signal(tel.time, tel.rpm; env_sr = round(Int, frame_hz))

    # ─── Stage 1: short-list candidates from FFT cross-correlation ───
    # Top-K peaks, spaced ≥ 30 s apart so we don't grab a cluster of samples
    # around the same lobe. The TRUE peak should be in this list; lap-period
    # aliased copies of it will be in there too.
    max_lag     = round(Int, max_lag_s * frame_hz)
    min_spacing = round(Int, 30.0 * frame_hz)
    peaks = fft_xcorr_top_k(tel_rpm, audio_rpm, max_lag;
                            k = k_candidates, min_spacing = min_spacing)

    # ─── Stage 2: session-level seed via boolean-signal xcorr ───
    # Convert both signals to a binary "is-racing" indicator (smoothed RPM
    # crosses 5000). The resulting signal has a unique global SHAPE — silent
    # → racing → quiet — with cautions that don't repeat at a fixed lap
    # period, so the FFT cross-correlation of the booleans peaks sharply at
    # the true offset, free of lap-period aliasing.
    smooth_win   = round(Int, 10.0 * frame_hz)
    tel_racing   = Float64[v > 5000.0 ? 1.0 : 0.0 for v in rolling_mean(tel_rpm,   smooth_win)]
    audio_racing = Float64[v > 5000.0 ? 1.0 : 0.0 for v in rolling_mean(audio_rpm, smooth_win)]
    seed_k, seed_conf = fft_xcorr_lag(tel_racing, audio_racing, max_lag)
    session_offset_s = t0_tel - seed_k / frame_hz
    have_session_seed = seed_conf > 0.1

    tol_frames = round(Int, disambiguation_tol_s * frame_hz)
    session_lag_frames = Float64(seed_k)

    # ─── Stage 3: pick the highest-xcorr peak within tolerance of seed ───
    selected = nothing
    if have_session_seed
        for (lag, val) in peaks
            if abs(lag - session_lag_frames) <= tol_frames
                selected = (lag, val); break
            end
        end
    end
    fallback = selected === nothing
    fallback && (selected = peaks[1])

    best_k = selected[1]
    conf   = selected[2]
    offset_s = t0_tel - best_k / frame_hz

    return (
        offset_s         = offset_s,
        confidence       = conf,
        lag_samples      = best_k,
        frame_hz         = frame_hz,
        active_frames    = count(>(e_thresh), a.energy),
        total_frames     = length(a.energy),
        session_offset_s = session_offset_s,
        session_seed_ok  = have_session_seed,
        used_fallback    = fallback,
        candidate_peaks  = [(lag = lag, offset_s = t0_tel - lag / frame_hz, conf = val)
                            for (lag, val) in peaks],
        method           = :spectral_xcorr_session_disambiguated,
    )
end
