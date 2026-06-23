
#region ALTERNATE
# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATE (parked — kept for later, NOT on the default path): per-pixel motion
# mask + per-frame edge/glare rejection. More robust in principle (rejects glare,
# fits the cockpit's true shape) but heavier. Reachable via video_forward_track_masked.
# ─────────────────────────────────────────────────────────────────────────────

# ── motion mask + per-frame edge/glare rejection ────────────────────────────
# The speed proxy must fire only on REAL streaming scene — not the static cockpit
# (A-pillars/hood/dash) and not glare/reflections. Two complementary stages:
#   (a) motion mask  — a first pass over the whole session keeps only pixels whose
#       intensity actually VARIES over time, rejecting the static cockpit by its
#       true (irregular) shape — better than any rectangular crop.
#   (b) per-frame edge diff — the flow runs on the Sobel GRADIENT of each frame,
#       so smooth glare/reflections (≈0 gradient every frame) contribute ≈0, while
#       a white bitmask drops sun/specular pixels (bright + low-saturation RGB).

# Sobel gradient magnitude of luma `Y` (w×h, raster order) into `E`; borders 0.
function _sobel_mag!(E::Vector{Float64}, Y::Vector{Float64}, w::Int, h::Int)
    fill!(E, 0.0)
    @inbounds for r in 2:h-1
        base = (r - 1) * w
        for c in 2:w-1
            i = base + c
            gx = (Y[i-w+1] + 2Y[i+1] + Y[i+w+1]) - (Y[i-w-1] + 2Y[i-1] + Y[i+w-1])
            gy = (Y[i+w-1] + 2Y[i+w] + Y[i+w+1]) - (Y[i-w-1] + 2Y[i-w] + Y[i-w+1])
            E[i] = sqrt(gx * gx + gy * gy)
        end
    end
    return E
end

"""
    _build_forward_mask(video_path; mw, mh, sample_fps, thresh, backend) -> Vector{Bool}

First pass: sample frames across the whole video and keep pixels whose temporal
std-dev exceeds `thresh`×max — the dynamic scene, with the static cockpit
(A-pillars/hood/dash) rejected by its actual shape.
"""
function _build_forward_mask(video_path::AbstractString; mw::Int = 320, mh::Int = 180,
                             sample_fps::Real = 1/6, thresh::Real = 0.40,
                             backend::FfmpegBackend = detect_backend())
    fsz = mw * mh
    bytes = with_backend(backend) do exe
        read(`$exe -hide_banner -loglevel error $(backend.hwaccel_args) -i $video_path -vf fps=$(sample_fps),scale=$(mw):$(mh),format=gray -f rawvideo pipe:1`)
    end
    n = length(bytes) ÷ fsz
    n < 2 && error("_build_forward_mask: only $n sample frames")
    acc = zeros(Float64, fsz); acc2 = zeros(Float64, fsz)
    @inbounds for k in 0:n-1
        off = k * fsz
        @simd for i in 1:fsz
            v = Float64(bytes[off + i]); acc[i] += v; acc2[i] += v * v
        end
    end
    sd = sqrt.(max.(acc2 ./ n .- (acc ./ n) .^ 2, 0.0))
    mx = maximum(sd)
    raw = Bool[sd[i] >= thresh * mx for i in 1:fsz]
    return _fill_window(raw, mw, mh)
end

# Fill the window interior: flood the static "exterior" inward from the image
# border through rejected (cold) pixels — that traces the cockpit frame. Anything
# the frame ENCLOSES (the view through the glass, holes and interior glare blobs
# included) becomes the ROI; the per-frame edge/white filter sorts glare from
# scene inside it. Captures the whole windshield opening by its true loop shape.
function _fill_window(raw::AbstractVector{Bool}, w::Int, h::Int)
    ext = falses(w * h)
    st = Int[]
    idx(r, c) = (r - 1) * w + c
    seed(i) = (@inbounds !raw[i] && !ext[i]) && (@inbounds ext[i] = true; push!(st, i))
    for c in 1:w; seed(idx(1, c)); seed(idx(h, c)); end
    for r in 1:h; seed(idx(r, 1)); seed(idx(r, w)); end
    @inbounds while !isempty(st)
        i = pop!(st); r = (i - 1) ÷ w + 1; c = (i - 1) % w + 1
        r > 1 && seed(idx(r-1, c)); r < h && seed(idx(r+1, c))
        c > 1 && seed(idx(r, c-1)); c < w && seed(idx(r, c+1))
    end
    return Bool[!ext[i] for i in 1:w*h]
end

"""
    video_forward_track_masked(video_path; start_s, dur_s, fps, mw, mh, mask, backend)
        -> (t, forward)

(Parked alternate.) Glare-robust speed proxy. STREAMS full frames (memory stays flat over long full-
fps windows), converts each to a Sobel-gradient image, and sums the inter-frame
change of that gradient over the motion `mask`, dropping near-white glare pixels.
Smooth glare/reflections → ≈0 gradient → ≈0 contribution; sharp streaming
road/walls dominate. `forward[k]` ∝ scene speed.
"""
function video_forward_track_masked(video_path::AbstractString;
                             start_s::Real = 0.0, dur_s::Real = Inf, fps::Real = 30.0,
                             mw::Int = 320, mh::Int = 180,
                             mask::Union{Nothing,AbstractVector{Bool}} = nothing,
                             backend::FfmpegBackend = detect_backend())
    fsz = mw * mh; fsz3 = 3 * fsz
    msk = mask === nothing ? trues(fsz) : mask
    length(msk) == fsz || error("mask length $(length(msk)) != $(fsz)")
    targs = isfinite(dur_s) ? ["-t", string(dur_s)] : String[]
    vf = "scale=$(mw):$(mh),format=rgb24"

    # Preallocate from the known frame count (dur_s*fps); index-fill, no per-frame
    # push!. resize! once at the end (and amortized-double only if we overrun).
    cap = isfinite(dur_s) ? ceil(Int, Float64(dur_s) * fps) + 8 : 4096
    t = Vector{Float64}(undef, cap); fwd = Vector{Float64}(undef, cap)
    buf = Vector{UInt8}(undef, fsz3)
    Y = Vector{Float64}(undef, fsz); white = Vector{Bool}(undef, fsz)
    Ecur = Vector{Float64}(undef, fsz); Eprev = Vector{Float64}(undef, fsz)
    j = 0
    with_backend(backend) do exe
        io = open(`$exe -hide_banner -loglevel error $(backend.hwaccel_args) -ss $start_s $targs -i $video_path -vf $vf -r $fps -f rawvideo pipe:1`, "r")
        k = 0
        while true
            try
                read!(io, buf)                  # exactly one frame, or EOFError at end
            catch e
                e isa EOFError && break
                rethrow()
            end
            @inbounds for i in 1:fsz
                r = Float64(buf[3i-2]); g = Float64(buf[3i-1]); b = Float64(buf[3i])
                Y[i] = (r + g + b) / 3
                mxc = max(r, g, b); mnc = min(r, g, b)
                white[i] = mxc > 220.0 && (mxc - mnc) < 0.18 * mxc + 1.0   # bright + low-sat = glare
            end
            _sobel_mag!(Ecur, Y, mw, mh)
            if k > 0
                s = 0.0; cnt = 0
                @inbounds for i in 1:fsz
                    if msk[i] && !white[i]
                        s += abs(Ecur[i] - Eprev[i]); cnt += 1
                    end
                end
                j += 1
                if j > length(fwd)              # overrun guard (amortized, not per-frame)
                    resize!(fwd, 2length(fwd)); resize!(t, 2length(t))
                end
                @inbounds fwd[j] = cnt > 0 ? s / cnt : 0.0
                @inbounds t[j]   = start_s + (k - 0.5) / fps
            end
            Ecur, Eprev = Eprev, Ecur
            k += 1
        end
        close(io)
    end
    resize!(fwd, j); resize!(t, j)
    j < 2 && error("video_forward_track: got $j frames at start=$start_s")
    return (t = t, forward = fwd)
end



"""
    align_forward_speed(video_path, arrow_path; start_s=300, dur_s=900, fps=30,
                        corr_fs=5, smooth_s=3, crop, side, n_angle, n_radius, backend)
        -> AlignEstimate

Offset from the forward-flow (Fourier-Mellin zoom) ↔ GPS-speed axis. The video
zoom rate is captured at full `fps`, then cross-correlated against telemetry speed
at the lower `corr_fs` (the speed envelope is slow). `start_s` should be past any
pre-green footage so the template lands inside the green-based telemetry; `dur_s`
must be shorter than the telemetry span. `detail` carries the candidate-peak comb
and the margin of the winning peak over the runner-up.
"""
function align_forward_speed(video_path::AbstractString, arrow_path::AbstractString;
                             start_s::Real = 300.0, dur_s::Real = 900.0, fps::Real = 30.0,
                             corr_fs::Real = 5.0, smooth_s::Real = 3.0,
                             crop = FORWARD_CROP_DEFAULT, side::Int = 128,
                             n_angle::Int = 128, n_radius::Int = 64,
                             backend::FfmpegBackend = detect_backend())
    track = video_forward_track(video_path; start_s = start_s, dur_s = dur_s, fps = fps,
                                crop = crop, side = side, n_angle = n_angle, n_radius = n_radius,
                                backend = backend)
    tel = load_telemetry(arrow_path)
    ref_time = Float64.(collect(tel.time)); ref_speed = Float64.(collect(tel.speed))
    keep = isfinite.(ref_time) .& isfinite.(ref_speed)
    ref_time = ref_time[keep]; ref_speed = ref_speed[keep]
    offsets, corr = _fwd_xcorr_search(track.t, track.forward, ref_time, ref_speed;
                                      fs = corr_fs, smooth_s = smooth_s)
    best = argmax(corr)
    peaks = _vs_top_k(offsets, corr, corr_fs; min_spacing_s = 20.0)
    margin = length(peaks) > 1 ? corr[best] - peaks[2][2] : corr[best]
    return AlignEstimate(offsets[best], corr[best], :forward_fmellin,
        (margin = margin, window = (start_s, dur_s), n_frames = length(track.forward),
         candidate_peaks = [(offset_s = d, conf = c) for (d, c) in peaks]))
end
