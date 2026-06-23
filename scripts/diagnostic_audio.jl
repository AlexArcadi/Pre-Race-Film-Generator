# Audio-only alignment diagnostic. Run in VS Code (Julia: "Execute File in REPL").
# Sibling of `diagnostic.jl` — same 3-panel layout (trace overlay / FFT magnitude /
# cross-correlation) but a single row, audio↔RPM only. Shows the global xcorr curve
# over the full ±SWEEP_S sweep, with the picked peak, top-K candidates, and the
# session-seed offset marked so you can see whether the alignment LOCKED (one
# dominant peak that the seed agrees with) or is lap-aliased (comb of similar
# peaks, seed disagrees).
using Revise
includet(joinpath(@__DIR__, "..", "src", "PreRaceFilm.jl"))
using .PreRaceFilm
using Serialization, FFTW, Statistics, Plots
plotly()

# ── what to look at ──────────────────────────────────────────────────────────
RACE = "25SON1"
CAR  = 16
LAP  = 50          # context only — alignment is session-wide

# ── alignment knobs (match the production `align_audio_rpm` defaults) ────────
# SWEEP is derived from signal length below — N samples gives ~N/frame_hz s of
# headroom. Production hardcodes 1800 s and silently rails on long offsets like
# Sonoma's ~-2330 s; the diagnostic computes the true max from the inputs.
WINDOW_S      = 60.0     # ± view around the located lock for panel 1's title
ENERGY_PCTILE = 0.4      # mask audio frames quieter than this percentile
SMOOTH_WIN_S  = 10.0     # smoothing for the boolean "is-racing" seed
RACING_THRESH = 5000.0   # RPM threshold for "racing" boolean
K_CANDIDATES  = 12
MIN_SPACING_S = 30.0

# ── cache (audio decode + STFT is the slow part) ─────────────────────────────
CACHE_DIR = joinpath(@__DIR__, "..", ".jl_scratch")
isdir(CACHE_DIR) || mkpath(CACHE_DIR)
CACHE = joinpath(CACHE_DIR, "diag_audio_$(RACE)_car$(CAR).jls")

# ── compute or load ──────────────────────────────────────────────────────────
function compute_traces()
    cfg  = getConfig(RACE)
    sess = find_car_session(cfg, CAR)
    println("video: ", sess.video)
    println("arrow: ", sess.arrow)

    time, rpm = PreRaceFilm.load_channels(sess.arrow, :Time, :EngineRotVel)
    t0_tel = time[1]

    audio = PreRaceFilm.extract_audio_mono(sess.video; sr = 4000)
    a = PreRaceFilm.audio_rpm_trace(audio, 4000;
                                    window_size = 2048, hop = 200,
                                    freq_band_hz = (300.0, 800.0), cylinders = 8)
    e_thresh  = quantile(a.energy, ENERGY_PCTILE)
    audio_rpm = Float64[a.energy[i] >= e_thresh ? a.rpm[i] : 0.0
                        for i in eachindex(a.rpm)]
    fps_eff   = a.frame_hz
    tel_rpm   = PreRaceFilm.rpm_proxy_signal(time, rpm; env_sr = round(Int, fps_eff))

    return (; audio_rpm, tel_rpm, fps_eff, t0_tel,
              active_frames = count(>(e_thresh), a.energy),
              total_frames  = length(a.energy))
end

d = if isfile(CACHE)
    println("loading cache: $CACHE")
    deserialize(CACHE)
else
    println("computing traces (first run — ~30–60 s)…")
    out = compute_traces()
    serialize(CACHE, out)
    println("cached to: $CACHE")
    out
end

audio_rpm = d.audio_rpm
tel_rpm   = d.tel_rpm
fps_eff   = d.fps_eff
t0_tel    = d.t0_tel
println("audio active frames: $(d.active_frames) / $(d.total_frames)  (energy mask)")

# ── cross-correlation (production recipe, inlined) ───────────────────────────
# max_lag = N-1 → use all the headroom _fft_xcorr_curve clamps to (n÷2-1, where
# n = nextpow(2, 2N)). No arbitrary seconds-cap. Edge correlations involve few
# overlapping samples but global-energy normalization keeps them honest.
N_xc    = min(length(tel_rpm), length(audio_rpm))
max_lag = N_xc - 1
SWEEP_S = max_lag / fps_eff   # for printing only
println("xcorr sweep: ±$(round(SWEEP_S, digits=1)) s  (max_lag = $max_lag samples, N = $N_xc)")
lags, vals = PreRaceFilm._fft_xcorr_curve(tel_rpm, audio_rpm, max_lag)

min_spacing = round(Int, MIN_SPACING_S * fps_eff)
top_k = PreRaceFilm._top_k_from_curve(lags, vals;
                                      k = K_CANDIDATES, min_spacing = min_spacing)
best_k, best_v = top_k[1]

# session seed: boolean "is-racing" xcorr — unique global shape, lap-alias free
smooth_win   = round(Int, SMOOTH_WIN_S * fps_eff)
tel_racing   = Float64[v > RACING_THRESH ? 1.0 : 0.0
                       for v in PreRaceFilm._moving_average(tel_rpm,   smooth_win)]
audio_racing = Float64[v > RACING_THRESH ? 1.0 : 0.0
                       for v in PreRaceFilm._moving_average(audio_rpm, smooth_win)]
seed_k, seed_conf = PreRaceFilm.fft_xcorr_lag(tel_racing, audio_racing, max_lag)

locked_offset_s = t0_tel - best_k / fps_eff
seed_offset_s   = t0_tel - seed_k / fps_eff
seed_ok         = seed_conf > 0.1
seed_agrees     = abs(best_k - seed_k) / fps_eff <= 60.0   # 60 s tolerance

println()
println("─ alignment summary ─")
println("  best peak    : $(round(locked_offset_s, digits=2)) s  (xcorr=$(round(best_v, digits=3)))")
println("  session seed : $(round(seed_offset_s,   digits=2)) s  (conf=$(round(seed_conf, digits=3))) — $(seed_ok ? "USABLE" : "WEAK")")
println("  seed agrees  : $(seed_agrees ? "YES" : "NO")  (Δ = $(round((best_k-seed_k)/fps_eff, digits=1)) s)")
println("  top-$(K_CANDIDATES) peaks (lag s → offset s, xcorr):")
for (lag, val) in top_k
    println("    $(rpad(round(lag/fps_eff, digits=2), 10)) → $(rpad(round(t0_tel - lag/fps_eff, digits=2), 10))   $(round(val, digits=3))")
end

# ── plotting ─────────────────────────────────────────────────────────────────
znorm(x) = (μ = mean(x); σ = std(x); σ == 0 ? x .- μ : (x .- μ) ./ σ)
function shifted(x, k)
    out = fill(NaN, length(x))
    @inbounds for i in eachindex(x)
        j = i + k
        1 <= j <= length(x) && (out[i] = x[j])
    end
    return out
end
thin(len; target = 8000) = 1:max(1, cld(len, target)):len

n = min(length(audio_rpm), length(tel_rpm))
seconds = (0:n-1) ./ fps_eff
tel_at_lock = shifted(znorm(tel_rpm[1:n]), best_k)
r = thin(n; target = 8000)

# panel 1 — traces at lock (audio on its own time, telem shifted onto it)
p_traces = plot(seconds[r], znorm(audio_rpm[1:n])[r];
                label = "audio RPM", xlabel = "s",
                title = "audio↔telem @ $(round(locked_offset_s, digits=1)) s   (race $RACE  car $CAR  lap $LAP)")
plot!(p_traces, seconds[r], tel_at_lock[r]; label = "telem RPM (shifted)")

# panel 2 — FFT magnitude (consistency: same low-freq structure?)
fft_v = abs.(rfft(audio_rpm[1:n]))
fft_t = abs.(rfft(tel_rpm[1:n]))
freqs = (0:n÷2) .* (fps_eff / n)
keep  = freqs .<= 1.0
p_fft = plot(freqs[keep], fft_v[keep]; label = "audio", yscale = :log10,
             title = "FFT magnitude", xlabel = "Hz")
plot!(p_fft, freqs[keep], fft_t[keep]; label = "telem")

# panel 3 — GLOBAL cross-correlation. The lock question lives here:
# one dominant peak vs comb of similar peaks. Top-K marked in gray;
# picked peak in red; session-seed offset in orange.
offs = lags ./ fps_eff
p_xcorr = plot(offs, vals; label = "xcorr", xlabel = "lag (s)",
               title = "global xcorr — peak $(round(best_k/fps_eff, digits=1)) s, seed $(round(seed_k/fps_eff, digits=1)) s  ($(seed_agrees ? "AGREE" : "DISAGREE"))")
for (lag, val) in top_k
    vline!(p_xcorr, [lag/fps_eff]; color = :gray, alpha = 0.25, label = "")
end
vline!(p_xcorr, [best_k/fps_eff]; color = :red,    label = "picked")
vline!(p_xcorr, [seed_k/fps_eff]; color = :orange, label = "seed")

display(plot(p_traces, p_fft, p_xcorr; layout = (1, 3),
             size = (1800, 500), titlefontsize = 9))
