# Per-axis alignment diagnostic. Run in VS Code (Julia: "Execute File in REPL").
# Per axis (5 rows): normalized trace overlay at the locked offset, FFT magnitude,
# and the cross-correlation (abs + abs+smooth, FFT and direct) zoomed to the lock.
using Serialization, FFTW, Statistics, Plots
plotly()

CACHE = raw"C:\Users\BenModel\AppData\Local\Temp\claude\c--Users-BenModel-Documents-GitHub-Pre-Race-Film-Generator\f62059d2-c923-40a4-8860-aeace36069b7\scratchpad\cache_car16_race.jls"
d = deserialize(CACHE)
fps = d.fps

SWEEP_S  = 3600.0    # ± xcorr compute range (s) — must cover SEED_S
SEED_S   = -2330.0   # located lock; the cross-correlation view centers here
WINDOW_S = 60.0      # ± view + peak-search window around SEED_S
SMOOTH_S = 2.0       # low-pass window (s) for the abs+smooth step
TELEM_SMOOTH = 5     # moving-average (samples, native telem rate) before resampling — anti-alias + denoise

# ── numeric primitives (inlined from PreRaceFilm math/alignment/visual_align2) ──
function resample(t, x, fs)                    # (t,x) onto a uniform fs grid → (t0, values)
    t0 = first(t); t1 = last(t); n = length(t)
    ng = floor(Int, (t1 - t0) * fs) + 1
    out = Vector{Float64}(undef, ng); invfs = 1.0 / fs; k = 1
    @inbounds for i in 1:ng
        q = muladd(i - 1, invfs, t0)
        while k < n - 1 && t[k+1] < q; k += 1; end
        w = t[k+1] == t[k] ? 0.0 : (q - t[k]) / (t[k+1] - t[k])
        out[i] = muladd(x[k+1] - x[k], w, x[k])
    end
    return t0, out
end

function smooth(x, n)                          # centred moving average, window n
    n <= 1 && return copy(x)
    cs = pushfirst!(cumsum(x), 0.0); m = length(x); out = Vector{Float64}(undef, m); half = n ÷ 2
    @inbounds for i in 1:m
        lo = max(1, i - half); hi = min(m, i + half)
        out[i] = (cs[hi+1] - cs[lo]) / (hi - lo + 1)
    end
    return out
end

function fft_xcorr_curve(ref, query, max_lag)  # FFT cross-correlation, global-energy normalized
    N = min(length(ref), length(query))
    r = ref[1:N];   r .-= mean(r)
    q = query[1:N]; q .-= mean(q)
    n = nextpow(2, 2N)
    R = rfft(vcat(r, zeros(n - N))); Q = rfft(vcat(q, zeros(n - N)))
    xc = irfft(conj.(R) .* Q, n)
    nrm = sqrt(sum(abs2, r) * sum(abs2, q))
    max_lag = clamp(max_lag, 1, n ÷ 2 - 1)
    lags = collect(-max_lag:max_lag)
    vals = [nrm == 0 ? 0.0 : xc[lag >= 0 ? lag + 1 : n + lag + 1] / nrm for lag in lags]
    return lags, vals
end

# ── helpers ──────────────────────────────────────────────────────────────────
znorm(x) = (μ = mean(x); σ = std(x); σ == 0 ? x .- μ : (x .- μ) ./ σ)

# `x` read `k` samples ahead on its own index grid; out-of-range → NaN (won't plot).
function shifted(x, k)
    out = fill(NaN, length(x))
    @inbounds for i in eachindex(x)
        j = i + k
        1 <= j <= length(x) && (out[i] = x[j])
    end
    return out
end

# FFT cross-correlation curve, returned as (offset_s, correlation).
function fft_xcorr(a, b)
    lags, corr = fft_xcorr_curve(a, b, round(Int, SWEEP_S * fps))
    return lags ./ fps, corr
end

# Direct (time-domain) cross-correlation over the ±WINDOW_S view, normalized like
# the FFT one so the two overlay — a method check at the zoomed scale.
function direct_xcorr(a, b; ds = 6)
    ra = (a .- mean(a))[1:ds:end]; rb = (b .- mean(b))[1:ds:end]; fd = fps / ds
    nrm = sqrt(sum(abs2, ra) * sum(abs2, rb))
    offs = collect((SEED_S - WINDOW_S):(1/fd):(SEED_S + WINDOW_S))
    cor = map(offs) do o
        k = round(Int, o * fd); i0 = max(1, 1 - k); i1 = min(length(ra), length(rb) - k)
        s = 0.0
        @inbounds @simd for i in i0:i1
            s = muladd(ra[i], rb[i + k], s)
        end
        s / nrm
    end
    return offs, cor
end

# Peak offset within the ±WINDOW_S view around SEED_S (edge artifacts can't win).
function peak_in_window(offs, corr)
    m = abs.(offs .- SEED_S) .<= WINDOW_S
    return offs[m][argmax(corr[m])]
end

# Subsample indices to ~`target` points — keeps plotly fast on long curves.
thin(len; target = 4000) = 1:max(1, cld(len, target)):len
AXES = [(:yaw,     d.rot_t,    d.yaw,         d.yt, d.yx),
        (:pitch,   d.rot_t,    d.pitch,       d.pt, d.px),
        (:roll,    d.fwd_t,    d.roll,        d.rt, d.rx),
        (:forward, d.fwd_t,    d.zoom,        d.st, d.sx),
        (:audio,   d.audio_pt, d.audio_power, d.et, d.ex)]

plts = Any[]
for (name, vt, vx, tt, tx) in AXES
    _, video = resample(vt, vx, fps)
    _, telem = resample(tt, smooth(tx, TELEM_SMOOTH), fps)   # denoise gyro BEFORE resampling (anti-alias)
    n = min(length(video), length(telem))
    video = video[1:n]; telem = telem[1:n]
    seconds = (0:n-1) ./ fps

    # conditioning steps: abs = activity envelope (on-track vs pits); abs+smooth low-passes it
    w = max(1, round(Int, SMOOTH_S * fps))
    abs_v,    abs_t    = abs.(video),         abs.(telem)
    smooth_v, smooth_t = smooth(abs_v, w), smooth(abs_t, w)

    offs, corr_abs = fft_xcorr(abs_v, abs_t)            # activity envelope (rectified)
    _,    corr_sm  = fft_xcorr(smooth_v, smooth_t)      # + light low-pass (the production recipe)
    doff, dcor     = direct_xcorr(smooth_v, smooth_t)   # same, time-domain — checks the FFT curve
    offset = peak_in_window(offs, corr_sm)

    # panel 1 — normalized traces, telem shifted onto video time by the lock
    telem_at_lock = shifted(znorm(telem), round(Int, offset * fps))
    r = thin(n; target = 8000)
    traces = plot(seconds[r], znorm(video)[r]; label = "video", xlabel = "s",
                  title = "$name @ $(round(offset, digits=1)) s")
    plot!(traces, seconds[r], telem_at_lock[r]; label = "telem")

    # panel 2 — FFT magnitude (log)
    freqs = (0:n÷2) .* (fps / n); keep = freqs .<= 12
    spectrum = plot(freqs[keep], abs.(rfft(video))[keep]; label = "video",
                    yscale = :log10, title = "FFT magnitude", xlabel = "Hz")
    plot!(spectrum, freqs[keep], abs.(rfft(telem))[keep]; label = "telem")

    # panel 3 — activity cross-correlation (abs, abs+smooth); direct overlays the FFT as a check
    crosscorr = plot(offs, corr_abs; label = "abs (fft) @$(round(peak_in_window(offs, corr_abs), digits=1)) s",
                     title = "cross-correlation", xlabel = "offset (s)",
                     xlims = (SEED_S - WINDOW_S, SEED_S + WINDOW_S))
    plot!(crosscorr, offs, corr_sm; label = "abs+smooth (fft) @$(round(offset, digits=1)) s")
    plot!(crosscorr, doff, dcor;    label = "abs+smooth (direct)")
    vline!(crosscorr, [offset]; label = "")

    push!(plts, traces, spectrum, crosscorr)
end
display(plot(plts...; layout = grid(5, 3), size = (1800, 1400), titlefontsize = 8))
