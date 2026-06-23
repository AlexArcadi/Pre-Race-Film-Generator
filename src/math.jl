# Shared numeric primitives — used by the audio aligner, the visual aligner, and
# the diagnostics. Kept in one place (included first) so there's a single
# definition rather than per-file copies.

"""
    _resample(t, x, fs) -> (t0, values)

Resample `(t, x)` onto a uniform `fs`-Hz grid by linear interpolation, returning
the grid start `t0` and the values covering `[t[1], t[end]]`. `t` and the grid
are both monotonic, so a single forward sweep with a sample pointer suffices — no
per-query search. The grid is uniform, so query times (`t0 + (i-1)/fs`) are
implicit and never materialized.
"""
function _resample(t, x, fs::Float64)
    t0 = first(t); t1 = last(t); n = length(t)
    ngrid = floor(Int, (t1 - t0) * fs) + 1
    out = Vector{Float64}(undef, ngrid)
    inv_fs = 1.0 / fs                                # hoist the divide out of the loop
    k = 1                                            # bracket: t[k] ≤ q ≤ t[k+1]
    @inbounds for i in 1:ngrid
        q = muladd(i - 1, inv_fs, t0)                # t0 + (i-1)/fs, as an FMA
        while k < n - 1 && t[k + 1] < q
            k += 1
        end
        tk = t[k]; tk1 = t[k + 1]
        w = tk1 == tk ? 0.0 : (q - tk) / (tk1 - tk)
        out[i] = muladd(x[k + 1] - x[k], w, x[k])    # lerp as an FMA
    end
    return t0, out
end

"""
    _parabolic_peak(l, c, r) -> shift

Sub-sample peak offset (in samples, range ±0.5) from a 3-point parabolic fit to a
discrete peak `c` and its neighbours `l`, `r`. Used to de-quantize correlation and
spectral peaks in both aligners.
"""
function _parabolic_peak(l::Real, c::Real, r::Real)
    d = l - 2c + r
    return abs(d) < eps() ? 0.0 : clamp(0.5 * (l - r) / d, -1.0, 1.0)
end

"""
    _moving_average(x, n; f=identity) -> Vector{Float64}

Centred moving average over a width-`n` window, one pass via a prefix sum. `f` is
applied to each element while the prefix is built (fused), so passing `f=abs`
gives a rectified/activity envelope with no intermediate allocation.
"""
function _moving_average(x::Vector{Float64}, n::Int; f = identity)
    m = length(x)
    out = Vector{Float64}(undef, m)
    m == 0 && return out
    cs = Vector{Float64}(undef, m + 1); cs[1] = 0.0
    @inbounds for i in 1:m
        cs[i + 1] = cs[i] + f(x[i])
    end
    half = n ÷ 2
    @inbounds for i in 1:m
        lo = max(1, i - half); hi = min(m, i + half)
        out[i] = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)
    end
    return out
end
