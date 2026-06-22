# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░                  MD Audio VAE Tiled Core – v3.0.1                   ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine / A&E Concepts
# ╠═ AUTHOR: Alexander Allan (MDMAchine)
# ╠═ LICENSE: GNU General Public License v3.0 (GPLv3)
# ║
# ║  Core math engine for MD Audio VAE Tiled Decoder. Features:
# ║    • Adaptive RMS leveling with Entropy-Driven gain clamping
# ║    • Shannon entropy monitoring (FFT power spectrum)
# ║    • Spectral crossfade integration (Cosine/Hann/Linear)
# ║    • Dual-Pass Staggered Decode (Phase-offset seam cancellation)
# ║    • Context-Aware Seam Repair (CASR)
# ║    • Adaptive Tile Sizing (Entropy-Driven dynamic chunking)
# ║
# ║  CHANGELOG v3.0.1 — GPU Performance Audit & Fixes:
# ║    [BIQUAD] _biquad_filter: scipy lfilter → torchaudio fallback → Python loop.
# ║             Was 14M Python iterations on 5-min track. Now vectorized CPU.
# ║    [HPF] _apply_highpass: scipy sosfilt (vectorized) replaces Python loop.
# ║    [HPC] _spectral_crossfade: removed B×C Python loops. Single batched
# ║           [BC,L] STFT. Frame weights vectorized via torch.arange.
# ║    [LSP] _generate_bridge_latents: SLERP vectorized via linspace t_vals.
# ║           Was 2×bridge_half Python iterations, now single tensor op.
# ║    [SCE] _enforce_spectral_continuity: single batched [BC,L] STFT replaces
# ║           nested B×C loops. Repair vectorized over all channels. no_grad.
# ║    [DENOISE] _apply_spectral_denoiser: batched [BC,L] STFT, vectorized
# ║           noise floor estimation via topk+gather, conv1d freq smoothing.
# ║           no_grad throughout. Was 2 nested loops + per-sample STFT.
# ║
# ║  CHANGELOG v3.0.0 — Next-Gen Decode Pipeline (5 Gamechangers):
# ║
# ║  [PLGR] Psychoacoustic Loudness-Aware Gain Riding. ITU-R BS.1770-4
# ║    K-weighted momentary loudness measurement replaces RMS gain ride.
# ║    Targets constant perceived loudness rather than constant energy.
# ║    Bass-heavy tiles no longer pump against vocal tiles.
# ║
# ║  [PCT] Phase-Coherent Triple-Pass Decode. Three decode passes at
# ║    offsets 0, 1/3, 2/3. Cross-correlation phase alignment before
# ║    trapezoidal merge eliminates whole-track phase smear.
# ║
# ║  [HPC] Harmonic Preservation Crossfade. STFT-domain OLA blend:
# ║    magnitudes blended via window, phase selected from higher-magnitude
# ║    source. Eliminates tonal swallowing at tile boundaries.
# ║
# ║  [LSP] Latent-Domain Seam Predictor. SLERP bridge latents generated
# ║    at each tile boundary before decode. Proactive seam prevention
# ║    rather than reactive repair.
# ║
# ║  [SCE] Spectral Continuity Enforcement. Post-decode harmonic
# ║    trajectory repair at tile seams. Detects magnitude discontinuities,
# ║    interpolates to restore continuity. Phase-aware reconstruction.
# ║
# ║  All 5 features independently togglable via module constants and
# ║  wrapper node UI params.
# ║
# ║  CHANGELOG v2.1.4 — Post-decode DSP chain:
# ║
# ║  [HPF] _apply_highpass() — 2nd order Butterworth IIR biquad. Removes
# ║    sub-sonic rumble below POST_DECODE_HIGHPASS_HZ. Default 20Hz.
# ║  [SOFTCLIP] _apply_soft_clip() — tanh saturation ceiling at
# ║    POST_DECODE_SOFT_CLIP_DB. Default -3.0dBFS.
# ║  [STEREO] _apply_stereo_width() — M/S width control via
# ║    POST_DECODE_STEREO_WIDTH. 1.0=unity, 0=mono, 2=wide.
# ║  All three wired as module-level constants, set by wrapper before call.
# ║
# ║  CHANGELOG v2.1.3 — Post-decode Wiener spectral denoiser:
# ║
# ║  [SPECTRAL_DENOISE] _apply_spectral_denoiser() added. STFT-based Wiener
# ║    filter estimates noise floor from quietest floor_pct of frames, computes
# ║    per-bin Wiener gain, smooths gain mask in frequency to prevent musical
# ║    noise, applies with strength blend and min_gain floor. Runs after hum
# ║    notch in execute_tiled_decode_logic().
# ║    Params (all tunable as constants):
# ║      SPECTRAL_DENOISE_ENABLED=True, STRENGTH=0.85, FLOOR_PCT=0.05
# ║      SMOOTH_HZ=80, MIN_GAIN=0.15
# ║    The 0.85 strength and 0.15 min_gain deliberately leave some noise floor
# ║    for naturalness — full suppression creates musical noise artifacts.
# ║
# ║  CHANGELOG v2.1.2 — Absolute RMS ceiling to kill long-run accumulation:
# ║
# ║  [RMS CEIL] Added CONST_RMS_ABSOLUTE_CEIL = 0.35 applied every tile after
# ║    the existing relative gain ride. Root cause: Adaptive gain ride chases
# ║    prev_rms (relative correction only) — if tile 1 is hot, all subsequent
# ║    tiles level to tile 1's hotness. Over 10+ tiles this compounds. The
# ║    absolute ceiling provides a fixed anchor independent of prior tile state.
# ║    Applied post-ride, downward-only, before prev_rms update. The LUFS
# ║    normalizer downstream handles final loudness; 0.35 provides headroom.
# ║
# ║  CHANGELOG v2.1.0 — Root cause fixes for all reported failure modes:
# ║
# ║  [PLACEMENT] Float anchor is the correct and complete solution for out_start.
# ║    out_start = int(round(start * upscale_factor)) for EVERY tile.
# ║    Removed the write-cursor accumulator introduced in v2.0.x — it
# ║    propagated VAE rounding slop forward causing cumulative drift.
# ║    With anchored ctx_skip, each tile writes exactly its latent region;
# ║    max position error is 0.5 samples/seam (completely inaudible).
# ║
# ║  [CTX_SKIP] Anchored to actual decoded tensor length (retained from v2.0.0).
# ║    ctx_skip = actual_decoded_len - expected_write_samples
# ║    Eliminates float-estimated skip drift that caused mid-track stutter.
# ║
# ║  [DUAL PASS] Pass B schedule now built directly from latents at its real
# ║    start position (latent_offset=tile_size//2). No zero-padding, no
# ║    shift arithmetic, no garbage entropy values at Pass B tile 0.
# ║    Both passes use float anchor → correct independent placement.
# ║
# ║  [SLERP BRIDGE → CASR] Oobleck is causal — 8-frame cold micro-decode
# ║    outputs silence/garbage for frames 0-6, audible click on frame 7.
# ║    Replaced with Context-Aware Seam Repair: decodes a full-context
# ║    window centred on each seam (same warm-up as main loop), extracts
# ║    only the seam-adjacent patch, applies a cosine bell crossfade that
# ║    fades in AND out — no hard cuts on either side.
# ║
# ║  [ADAPTIVE+DUAL] build_adaptive_tile_schedule takes latent_offset param.
# ║    Pass A: offset=0. Pass B: offset=tile_size//2. Both scan real content.
# ║    No coordinate shifting that could produce start<0 tiles.
# ║
# ╚════════════════════════════════════════════════════════════════════════════

import torch
import math
import logging

log = logging.getLogger("MD_Nodes.AudioTiledCore")

# =================================================================================
# == Global Caches & Constants
# =================================================================================
_WINDOW_CACHE    = {}
_TRAPEZOID_CACHE = {}

CONST_EPSILON              = 1e-8
CONST_GAIN_CLAMP_BASE      = 0.05    # ±5% base RMS gain ride
CONST_GAIN_CLAMP_MAX_RANGE = 0.20    # ±20% max on high-entropy seams
CONST_RMS_ABSOLUTE_CEIL    = 0.35    # Hard RMS ceiling per tile — prevents accumulation

# Post-decode DSP chain
POST_DECODE_HIGHPASS_HZ    = 20.0
POST_DECODE_SOFT_CLIP_DB   = -3.0    # Gold standard: -3.0. Overridden by node YAML.
POST_DECODE_STEREO_WIDTH   = 0.8

# Post-decode spectral denoiser (Wiener filter)
SPECTRAL_DENOISE_ENABLED   = True    # Global enable. Can be overridden per-call.
SPECTRAL_DENOISE_STRENGTH  = 0.85    # 0.0=off, 1.0=full Wiener, 0.85=default (leaves some noise floor for naturalness)
SPECTRAL_DENOISE_FLOOR_PCT = 0.05    # Bottom 5% of frames used to estimate noise floor
SPECTRAL_DENOISE_SMOOTH_HZ = 80.0    # Freq-domain smoothing bandwidth for gain mask (Hz)
SPECTRAL_DENOISE_MIN_GAIN  = 0.15    # Minimum gain (never fully silence a bin) — prevents musical noise
                                     # across long tracks. Applied after relative gain ride.
                                     # 0.35 = headroom for LUFS normalizer downstream.
CONST_LATENT_PADDING       = 4

# ── PLGR: Psychoacoustic Loudness-Aware Gain Riding ──────────────────
PLGR_ENABLED               = True
PLGR_TARGET_LUFS           = -18.0   # Target momentary loudness (LUFS)
# K-weighting biquad coefficients for 48kHz (ITU-R BS.1770-4)
_KWEIGHT_SHELF_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KWEIGHT_SHELF_A = [1.0, -1.69065929318241, 0.73248077421585]
_KWEIGHT_HP_B    = [1.0, -2.0, 1.0]
_KWEIGHT_HP_A    = [1.0, -1.99004745483398, 0.99007225036621]

# ── PCT: Phase-Coherent Triple-Pass Decode ────────────────────────────
PCT_TRIPLE_PASS            = False   # Enable triple-pass (overrides dual_pass)
PCT_PHASE_ALIGN            = True    # Cross-correlation phase alignment
PCT_PHASE_MAX_LAG_MS       = 2.0     # Max lag search window (ms)

# ── HPC: Harmonic Preservation Crossfade ──────────────────────────────
HPC_ENABLED                = True
HPC_MIN_OVERLAP_SAMPLES    = 1024    # Below this → fall back to time-domain
HPC_STFT_SIZE              = 1024
HPC_HOP                    = 256

# ── LSP: Latent-Domain Seam Predictor ─────────────────────────────────
LSP_ENABLED                = True
LSP_BRIDGE_FRAMES          = 16      # Half-width of bridge region (latent frames)

# ── LSS: Latent Spectral Suppressor ───────────────────────────────────
LSS_ENABLED                = True
LSS_STRENGTH               = 0.25   # Gold standard: 0.25. Overridden by node YAML.
LSS_VARIANCE_THRESHOLD     = 0.12
LSS_DC_REMOVE              = False  # PERMANENT OFF — metallic distortion

# ── SCE: Spectral Continuity Enforcement ──────────────────────────────────
SCE_ENABLED                = True
SCE_REPAIR_FRAMES          = 4       # STFT frames on each side of seam
SCE_MAX_FREQ_HZ            = 8000    # Only repair harmonics below this
SCE_THRESHOLD_DB           = 6.0     # Min amplitude discontinuity to trigger (dB)
SCE_STFT_SIZE              = 2048
SCE_HOP                    = 512

# =================================================================================
# == HUM SUPPRESSION: Multi-Band Notch + Bass Shelf
# =================================================================================
# Validated against ACE-Step XL Turbo VAE decoder hum artifact series.
# Root cause: VAE decoder resonance artifact manifesting as a harmonic series.
# Measured via Ableton EQ Eight on actual rendered output (2026-05-16).
#
# Harmonic series (Observer + EQ Eight confirmed):
#   76Hz   — sub fundamental (BPM-dependent: 75Hz@88BPM, 77Hz@172BPM)
#   211Hz  — harmonic 3
#   656Hz  — PRIMARY HUM (was incorrectly set to 426Hz)
#   1090Hz — harmonic 5
#
# Bass shelf: VAE decoder over-generates sub energy. Low shelf -2.0dB @120Hz
# compensates without killing kick punch.
#
# All bands can be individually disabled. Full chain disabled via HUM_NOTCH_ENABLED=False.
# =============================================================================

HUM_NOTCH_ENABLED  = True

# Individual band enable flags — defaults. All overridden by node YAML at runtime.
# V5 validated: cuts only. Boosts overcook presence — disabled by default.
HUM_NOTCH_BASS_SHELF_ENABLED = True
HUM_NOTCH_74_ENABLED         = True
HUM_NOTCH_94_ENABLED         = True
HUM_NOTCH_656_ENABLED        = True
HUM_NOTCH_353_ENABLED        = False  # boost — disabled
HUM_NOTCH_1090_ENABLED       = False  # boost — disabled
HUM_NOTCH_2060_ENABLED       = False  # boost — disabled
HUM_NOTCH_3520_ENABLED       = False  # boost — disabled

# Bass shelf — sub excess correction
HUM_BASS_SHELF_HZ    = 120.0   # Hz — shelf turnover
HUM_BASS_SHELF_DB    = -2.0    # dB — gentle, preserves kick
HUM_BASS_SHELF_SLOPE = 0.7     # shelf slope

# Notch/boost bands — all measured via Ableton EQ Eight 2026-05-16
# Cuts suppress VAE artifact. Boosts restore content the artifact was masking.
HUM_NOTCH_74_HZ    = 74.4     # Hz
HUM_NOTCH_74_DB    = -1.43    # dB
HUM_NOTCH_74_Q     = 6.27

HUM_NOTCH_94_HZ    = 94.0     # Hz
HUM_NOTCH_94_DB    = -1.86    # dB
HUM_NOTCH_94_Q     = 7.08

HUM_NOTCH_656_HZ   = 654.0    # Hz — PRIMARY HUM (corrected from 426Hz)
HUM_NOTCH_656_DB   = -15.0    # dB — measured in Ableton EQ Eight
HUM_NOTCH_656_Q    = 6.00     # Q=6 surgical — matches Ableton measurement

HUM_NOTCH_353_HZ   = 353.0    # Hz — presence restore (artifact dip)
HUM_NOTCH_353_DB   = +2.57    # dB — boost
HUM_NOTCH_353_Q    = 1.86

HUM_NOTCH_1090_HZ  = 1090.0   # Hz — air restore
HUM_NOTCH_1090_DB  = +4.71    # dB — boost (artifact was suppressing this)
HUM_NOTCH_1090_Q   = 2.47

HUM_NOTCH_2060_HZ  = 2060.0   # Hz
HUM_NOTCH_2060_DB  = +1.71    # dB
HUM_NOTCH_2060_Q   = 0.71

HUM_NOTCH_3520_HZ  = 3520.0   # Hz — air restore
HUM_NOTCH_3520_DB  = +3.43    # dB
HUM_NOTCH_3520_Q   = 1.72


def _peaking_biquad_coeffs(f0, gain_db, Q, sr):
    """Compute normalized peaking EQ biquad coefficients."""
    import math
    A      = 10.0 ** (gain_db / 40.0)
    w0     = 2.0 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha  = sin_w0 / (2.0 * Q)
    b0 = (1.0 + alpha * A) / (1.0 + alpha / A)
    b1 = (-2.0 * cos_w0)   / (1.0 + alpha / A)
    b2 = (1.0 - alpha * A) / (1.0 + alpha / A)
    a1 = (-2.0 * cos_w0)   / (1.0 + alpha / A)
    a2 = (1.0 - alpha / A) / (1.0 + alpha / A)
    return [b0, b1, b2], [1.0, a1, a2]


def _low_shelf_biquad_coeffs(f0, gain_db, slope, sr):
    """Compute normalized low shelf biquad coefficients."""
    import math
    A      = 10.0 ** (gain_db / 40.0)
    w0     = 2.0 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha  = sin_w0 / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    b0 =       A * ((A+1) - (A-1)*cos_w0 + 2*math.sqrt(A)*alpha)
    b1 =  2.0* A * ((A-1) - (A+1)*cos_w0)
    b2 =       A * ((A+1) - (A-1)*cos_w0 - 2*math.sqrt(A)*alpha)
    a0 =            (A+1) + (A-1)*cos_w0 + 2*math.sqrt(A)*alpha
    a1 = -2.0 *    ((A-1) + (A+1)*cos_w0)
    a2 =            (A+1) + (A-1)*cos_w0 - 2*math.sqrt(A)*alpha
    return [b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0]


def _apply_hum_notch(audio, sample_rate):
    """
    Multi-band hum suppression: low shelf (sub excess) + 4 peaking notches
    targeting the VAE decoder harmonic artifact series.
    Measured via Ableton EQ Eight on ACE-Step XL renders (2026-05-16).
    Operates on [B, C, L] tensor. Returns filtered tensor.
    """
    if not HUM_NOTCH_ENABLED:
        return audio

    import torch, numpy as np
    try:
        from scipy.signal import lfilter
    except ImportError:
        print("[AudioTiled] ⚠️ scipy not available, hum notch skipped. pip install scipy.")
        return audio

    sr = float(sample_rate)

    # Build filter chain — order: shelf first, then notches low→high
    filters = []
    if HUM_NOTCH_BASS_SHELF_ENABLED:
        filters.append(_low_shelf_biquad_coeffs(HUM_BASS_SHELF_HZ, HUM_BASS_SHELF_DB, HUM_BASS_SHELF_SLOPE, sr))
    if HUM_NOTCH_74_ENABLED:
        filters.append(_peaking_biquad_coeffs(HUM_NOTCH_74_HZ, HUM_NOTCH_74_DB, HUM_NOTCH_74_Q, sr))
    if HUM_NOTCH_94_ENABLED:
        filters.append(_peaking_biquad_coeffs(HUM_NOTCH_94_HZ, HUM_NOTCH_94_DB, HUM_NOTCH_94_Q, sr))
    if HUM_NOTCH_353_ENABLED:
        filters.append(_peaking_biquad_coeffs(HUM_NOTCH_353_HZ, HUM_NOTCH_353_DB, HUM_NOTCH_353_Q, sr))
    if HUM_NOTCH_656_ENABLED:
        filters.append(_peaking_biquad_coeffs(HUM_NOTCH_656_HZ, HUM_NOTCH_656_DB, HUM_NOTCH_656_Q, sr))
    if HUM_NOTCH_1090_ENABLED:
        filters.append(_peaking_biquad_coeffs(HUM_NOTCH_1090_HZ, HUM_NOTCH_1090_DB, HUM_NOTCH_1090_Q, sr))
    if HUM_NOTCH_2060_ENABLED:
        filters.append(_peaking_biquad_coeffs(HUM_NOTCH_2060_HZ, HUM_NOTCH_2060_DB, HUM_NOTCH_2060_Q, sr))
    if HUM_NOTCH_3520_ENABLED:
        filters.append(_peaking_biquad_coeffs(HUM_NOTCH_3520_HZ, HUM_NOTCH_3520_DB, HUM_NOTCH_3520_Q, sr))

    if not filters:
        return audio

    device = audio.device
    dtype  = audio.dtype
    x_np = audio.float().cpu().numpy()  # [B, C, L]

    for b_coeffs, a_coeffs in filters:
        x_np = lfilter(b_coeffs, a_coeffs, x_np, axis=-1)

    return torch.from_numpy(x_np.astype(np.float32)).to(device=device, dtype=dtype)


CONST_MIN_OLA_SAMPLES      = 8
CONST_VAE_CONTEXT_FRAMES   = 128     # Oobleck causal warm-up prefix (discarded)

CONST_ATS_TILE_MIN         = 128
CONST_ATS_TILE_MAX         = 1024
CONST_ATS_ENTROPY_SCAN_WIN = 16

CONST_DUAL_PASS_EDGE_TAPER = 0.25    # Trapezoidal ramp fraction for dual-pass merge

# Context-Aware Seam Repair
CONST_CASR_BLEND_FRAMES    = 32      # Latent frames decoded around each seam
CONST_CASR_PATCH_FRAMES    = 8       # Latent frames patched on each side of seam


# =================================================================================
# == Shannon Entropy Monitoring
# =================================================================================


def _apply_spectral_denoiser(audio, sample_rate,
                              strength=SPECTRAL_DENOISE_STRENGTH,
                              floor_pct=SPECTRAL_DENOISE_FLOOR_PCT,
                              smooth_hz=SPECTRAL_DENOISE_SMOOTH_HZ,
                              min_gain=SPECTRAL_DENOISE_MIN_GAIN):
    """
    Post-decode spectral denoiser using Wiener gain estimation.

    Algorithm:
      1. STFT the full decoded audio into frames.
      2. Estimate the noise power spectrum from the quietest floor_pct of frames
         (these are the frames most likely to be pure noise floor — silence,
         decay tails, brief gaps). Quietest = lowest RMS per frame.
      3. Compute Wiener gain per frequency bin:
             G(f) = P_signal(f) / (P_signal(f) + P_noise(f))
         where P_signal = max(P_observed - P_noise, 0).
      4. Smooth the gain mask in frequency to avoid musical noise artifacts
         (Wiener filters applied bin-by-bin create "birdies" — spectral
         modulation that sounds like its own artifact).
      5. Apply: output(f) = G(f)^strength × input(f).
         strength < 1.0 blends toward the original, preserving naturalness.
      6. ISTFT back to audio.

    The min_gain floor ensures no frequency bin is ever fully silenced —
    completely zeroing bins creates musical noise worse than the original hiss.

    Args:
        audio:       [B, C, L] float tensor, decoded audio.
        sample_rate: int, sample rate (Hz).
        strength:    float 0–1. 0=bypass, 1=full Wiener reduction.
        floor_pct:   float 0–1. Fraction of quietest frames used as noise ref.
        smooth_hz:   float. Frequency smoothing bandwidth for gain mask.
        min_gain:    float. Minimum per-bin gain (prevents full silencing).

    Returns:
        Denoised audio [B, C, L], same dtype as input.
    """
    if strength < 1e-4:
        return audio

    orig_dtype = audio.dtype
    device     = audio.device
    x          = audio.float()   # work in fp32

    B, C, L = x.shape

    # STFT parameters — 2048-point FFT with 75% overlap gives good frequency
    # resolution for audio noise estimation without excessive computation.
    n_fft   = 2048
    hop     = n_fft // 4   # 75% overlap
    win     = torch.hann_window(n_fft, device=device)

    with torch.no_grad():
        # Process all B×C in one batched STFT — GPU parallelism
        BC = B * C
        x_flat = x.reshape(BC, L)   # [BC, L]

        S = torch.stft(x_flat, n_fft=n_fft, hop_length=hop,
                       window=win, return_complex=True,
                       onesided=True, normalized=False,
                       pad_mode='reflect')   # [BC, F, T]

        mag_sq = S.abs().pow(2)   # [BC, F, T]
        F_dim, T_dim = mag_sq.shape[1], mag_sq.shape[2]

        # Noise floor estimation — vectorized over BC
        frame_rms  = mag_sq.mean(dim=1).sqrt()   # [BC, T]
        n_floor    = max(1, int(T_dim * floor_pct))
        _, idx     = torch.topk(frame_rms, n_floor, dim=1, largest=False)  # [BC, n_floor]
        # Gather noise frames
        idx_exp    = idx.unsqueeze(1).expand(-1, F_dim, -1)   # [BC, F, n_floor]
        noise_frames = torch.gather(mag_sq, 2, idx_exp)        # [BC, F, n_floor]
        noise_est  = noise_frames.mean(dim=2, keepdim=True)    # [BC, F, 1]

        # Wiener gain — fully vectorized
        signal_est = (mag_sq - noise_est).clamp(min=0.0)
        gain       = signal_est / (signal_est + noise_est + 1e-12)   # [BC, F, T]

        # Frequency smoothing via conv1d along F axis
        smooth_bins = max(1, int(smooth_hz / (sample_rate / n_fft)))
        if smooth_bins > 1:
            kern = torch.ones(1, 1, smooth_bins, device=device) / smooth_bins
            # gain: [BC, F, T] → conv along F: treat T as batch, F as signal
            g_t = gain.permute(0, 2, 1).reshape(BC * T_dim, 1, F_dim)
            g_t = torch.nn.functional.conv1d(g_t, kern, padding=smooth_bins // 2)
            gain = g_t.reshape(BC, T_dim, F_dim).permute(0, 2, 1)[..., :F_dim, :]

        gain = gain.clamp(min=min_gain, max=1.0)
        gain = gain * strength + (1.0 - strength)

        S_denoised = S * gain
        out_flat = torch.istft(S_denoised, n_fft=n_fft, hop_length=hop,
                               window=win, length=L, onesided=True, normalized=False)

    return out_flat.reshape(B, C, L).to(orig_dtype)



def _apply_highpass(audio, sample_rate, cutoff_hz):
    """
    Butterworth HPF — removes sub-sonic rumble.
    Uses scipy sosfilt (vectorized CPU) — orders of magnitude faster than Python loop.
    """
    if cutoff_hz < 1.0:
        return audio
    device, dtype = audio.device, audio.dtype
    try:
        from scipy.signal import butter, sosfilt
        import numpy as np
        sos = butter(2, cutoff_hz / (sample_rate * 0.5), btype='high', output='sos')
        x_np = audio.float().cpu().numpy()
        y_np = sosfilt(sos, x_np, axis=-1).astype(np.float32)
        return torch.from_numpy(y_np).to(device=device, dtype=dtype)
    except ImportError:
        # scipy unavailable — use _biquad_filter fallback path
        sr, f0 = float(sample_rate), float(cutoff_hz)
        w0 = 2.0 * math.pi * f0 / sr
        alpha = math.sin(w0) / (2.0 * 0.7071)
        b0 = (1.0 + math.cos(w0)) / 2.0
        b1 = -(1.0 + math.cos(w0))
        b2 = (1.0 + math.cos(w0)) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * math.cos(w0)
        a2 = 1.0 - alpha
        return _biquad_filter(audio.float(), [b0/a0, b1/a0, b2/a0],
                               [1.0, a1/a0, a2/a0]).to(dtype)


def _apply_soft_clip(audio, ceiling_db):
    """
    Soft clipper — tanh-based saturation with dBFS ceiling.
    ceiling_db=0.0 → bypass (treated as pass-through above 0dB — use -0.001 min).
    Provides gentle saturation rather than hard clipping at the ceiling.
    """
    if ceiling_db >= 0.0:
        return audio
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    x = audio.float()
    # Scale so ceiling_lin maps to tanh(1)=0.7616... then rescale back
    # tanh gives gentle saturation that approaches ceiling asymptotically
    scale = ceiling_lin / math.tanh(1.0)
    out = scale * torch.tanh(x / scale)
    return out.to(audio.dtype)


def _apply_stereo_width(audio, width):
    """
    Mid/Side stereo width control. Only applies to stereo (C=2).
    width=1.0 → unity (unchanged). 0.0 → mono. 2.0 → doubled side.
    """
    if abs(width - 1.0) < 1e-4 or audio.shape[1] != 2:
        return audio
    x   = audio.float()
    mid  = (x[:, 0:1, :] + x[:, 1:2, :]) * 0.5
    side = (x[:, 0:1, :] - x[:, 1:2, :]) * 0.5
    side = side * width
    L = mid + side
    R = mid - side
    out = torch.cat([L, R], dim=1)
    return out.to(audio.dtype)



# =================================================================================
# == PLGR: Psychoacoustic Loudness-Aware Gain Riding
# =================================================================================

def _biquad_filter(x, b, a):
    """
    Apply biquad IIR filter. x: [..., L], b/a: 3 coefficients.
    GPU-accelerated via torchaudio.functional.lfilter when available,
    falls back to scipy lfilter (CPU-vectorized, still fast), 
    last resort Python loop.
    """
    import numpy as np
    device, dtype = x.device, x.dtype
    x_cpu = x.float().cpu()
    b_np = np.array(b, dtype=np.float64)
    a_np = np.array(a, dtype=np.float64)
    try:
        from scipy.signal import lfilter as sp_lfilter
        y_np = sp_lfilter(b_np, a_np, x_cpu.numpy(), axis=-1)
        return torch.from_numpy(y_np.astype(np.float32)).to(device=device, dtype=dtype)
    except ImportError:
        pass
    # torchaudio fallback
    try:
        import torchaudio.functional as TAF
        b_t = torch.tensor(b, dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.float32)
        return TAF.lfilter(x.float(), a_t, b_t, clamp=False).to(dtype)
    except Exception:
        pass
    # Last resort: Python loop (slow but correct)
    y = torch.zeros_like(x)
    x1 = x2 = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
    y1 = y2 = torch.zeros_like(x1)
    for n in range(x.shape[-1]):
        xn = x[..., n]
        yn = b[0]*xn + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2
        y[..., n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


def _measure_loudness_lufs(audio_chunk, sample_rate):
    """
    ITU-R BS.1770-4 momentary loudness measurement.
    Returns LUFS value (float). Operates on [B, C, L] tensor.
    """
    x = audio_chunk.float().mean(dim=0)  # mix to mono-ish for measurement [C, L]
    # Stage 1: K-weighting
    x = _biquad_filter(x, _KWEIGHT_SHELF_B, _KWEIGHT_SHELF_A)
    x = _biquad_filter(x, _KWEIGHT_HP_B, _KWEIGHT_HP_A)
    # Stage 2: Mean square
    mean_sq = torch.mean(x ** 2).item()
    if mean_sq < 1e-12:
        return -70.0  # silence floor
    # Stage 3: LUFS
    return -0.691 + 10.0 * math.log10(mean_sq + CONST_EPSILON)


def _apply_plgr_gain(decoded_chunk, sample_rate, target_lufs, prev_lufs=None):
    """
    PLGR gain application. Returns (gained_chunk, current_lufs).
    First tile: measure and store, no gain change.
    Subsequent tiles: ride toward prev tile loudness (not absolute target).

    Downward-only riding — never amplifies. Prevents quiet tile → loud tile
    surge that occurs when targeting absolute LUFS on sparse/breakdown sections.
    The LUFS normalizer downstream handles absolute loudness; PLGR only
    corrects inter-tile consistency.
    """
    current_lufs = _measure_loudness_lufs(decoded_chunk, sample_rate)
    if prev_lufs is None:
        return decoded_chunk, current_lufs, 0.0

    if current_lufs < -60.0:  # near-silence, skip
        return decoded_chunk, current_lufs, 0.0

    # Chase prev tile loudness (not absolute target) — prevents overcorrection
    # on genuine dynamic variation (breakdowns, drops, etc.)
    gain_db = prev_lufs - current_lufs
    # Downward-only: only attenuate, never amplify
    # Clamp to -3dB max correction per tile — subtle leveling only
    gain_db = max(-1.5, min(0.0, gain_db))
    if abs(gain_db) < 0.3:  # sub-0.3dB correction not worth applying
        return decoded_chunk, current_lufs, 0.0
    gain_lin = 10.0 ** (gain_db / 20.0)
    return decoded_chunk * gain_lin, current_lufs, gain_db


# =================================================================================
# == PCT: Phase-Coherent Triple-Pass Decode
# =================================================================================

def _align_phase_xcorr(ref_chunk, target_chunk, max_lag_samples):
    """
    Cross-correlation phase alignment. Returns aligned target_chunk.
    Sub-sample precision via parabolic interpolation around xcorr peak.
    """
    ref = ref_chunk.float().flatten()
    tgt = target_chunk.float().flatten()
    L = min(len(ref), len(tgt))
    ref, tgt = ref[:L], tgt[:L]

    # Normalized cross-correlation via FFT
    n_fft = 1 << (2 * L - 1).bit_length()
    R = torch.fft.rfft(ref, n=n_fft)
    T = torch.fft.rfft(tgt, n=n_fft)
    xcorr = torch.fft.irfft(R * T.conj(), n=n_fft)

    # Search within ±max_lag
    center = 0
    search = xcorr[:max_lag_samples].clone()
    search_neg = xcorr[-max_lag_samples:].clone()
    combined = torch.cat([search_neg, search])
    peak_idx = combined.argmax().item()
    lag = peak_idx - max_lag_samples

    if abs(lag) < 1:
        return target_chunk  # no significant lag

    # Apply integer lag shift
    if lag > 0:
        shifted = torch.zeros_like(target_chunk)
        shifted[..., lag:] = target_chunk[..., :-lag]
    else:
        shifted = torch.zeros_like(target_chunk)
        shifted[..., :lag] = target_chunk[..., -lag:]

    return shifted.to(target_chunk.dtype)


def _execute_triple_pass(
    vae, latents, tile_size, overlap, upscale_factor, final_samples,
    output_channels, device, blending_mode, rms_leveling, context_prefix, B,
    adaptive_tiling, base_tile_size, base_overlap, sample_rate
):
    """
    Phase-Coherent Triple-Pass Decode.
    Three passes at offsets 0, 1/3, 2/3 of tile_size.
    Phase-aligned before trapezoidal merge.
    """
    W = latents.shape[-1]
    offset_b = tile_size // 3
    offset_c = 2 * tile_size // 3

    def _fixed_sched(start_lat, ts, ov):
        sched, cur = [], start_lat
        while cur < W:
            end = min(W, cur + ts)
            sched.append((cur, end, ov))
            hop = ts - ov
            if hop <= 0: hop = max(1, ts // 2)
            cur += hop
        return sched

    if adaptive_tiling:
        sched_a = build_adaptive_tile_schedule(latents, base_tile_size, base_overlap, device, 0)
        sched_b = build_adaptive_tile_schedule(latents, base_tile_size, base_overlap, device, offset_b)
        sched_c = build_adaptive_tile_schedule(latents, base_tile_size, base_overlap, device, offset_c)
    else:
        sched_a = _fixed_sched(0,        tile_size, overlap)
        sched_b = _fixed_sched(offset_b, tile_size, overlap)
        sched_c = _fixed_sched(offset_c, tile_size, overlap)

    audio_a, bounds_a = _execute_single_pass(
        vae, latents, sched_a, upscale_factor, final_samples,
        output_channels, device, blending_mode, rms_leveling, context_prefix, B)
    audio_b, bounds_b = _execute_single_pass(
        vae, latents, sched_b, upscale_factor, final_samples,
        output_channels, device, blending_mode, rms_leveling, context_prefix, B)
    audio_c, bounds_c = _execute_single_pass(
        vae, latents, sched_c, upscale_factor, final_samples,
        output_channels, device, blending_mode, rms_leveling, context_prefix, B)

    # Phase alignment
    if PCT_PHASE_ALIGN:
        max_lag = max(1, int(PCT_PHASE_MAX_LAG_MS * sample_rate / 1000.0))
        # Align B and C to A at each seam boundary
        chunk_size = max(2048, int(0.05 * sample_rate))  # 50ms chunks
        for bound in bounds_a:
            s = max(0, bound - chunk_size // 2)
            e = min(final_samples, bound + chunk_size // 2)
            if e - s < 256:
                continue
            ref = audio_a[..., s:e]
            audio_b[..., s:e] = _align_phase_xcorr(ref, audio_b[..., s:e], max_lag)
            audio_c[..., s:e] = _align_phase_xcorr(ref, audio_c[..., s:e], max_lag)

    # 3-pass trapezoidal merge
    weight_a = torch.zeros(final_samples, device=device)
    weight_b = torch.zeros(final_samples, device=device)
    weight_c = torch.zeros(final_samples, device=device)

    def _fill_w(buf, bounds, sched):
        for i, out_s in enumerate(bounds):
            s, e, _ = sched[i] if i < len(sched) else sched[-1]
            tile_len = int(round((e - s) * upscale_factor))
            out_e = min(final_samples, out_s + tile_len)
            L = out_e - out_s
            if L <= 0: continue
            trap = _get_trapezoidal_weight(L, CONST_DUAL_PASS_EDGE_TAPER, device)
            buf[out_s:out_e] = torch.max(buf[out_s:out_e], trap)

    _fill_w(weight_a, bounds_a, sched_a)
    _fill_w(weight_b, bounds_b, sched_b)
    _fill_w(weight_c, bounds_c, sched_c)

    total = weight_a + weight_b + weight_c + CONST_EPSILON
    wa = (weight_a / total).view(1, 1, -1)
    wb = (weight_b / total).view(1, 1, -1)
    wc = (weight_c / total).view(1, 1, -1)

    merged = audio_a * wa + audio_b * wb + audio_c * wc
    return merged, bounds_a, bounds_b


# =================================================================================
# == HPC: Harmonic Preservation Crossfade
# =================================================================================

def _spectral_crossfade(prev_tail, new_head, f_in_window, device):
    """
    STFT-domain crossfade with magnitude blend + phase selection.
    prev_tail, new_head: [B, C, L] same length.
    f_in_window: [L] fade-in ramp 0→1.
    Returns blended audio [B, C, L].
    """
    n_fft = HPC_STFT_SIZE
    hop   = HPC_HOP
    win   = torch.hann_window(n_fft, device=device)
    L     = prev_tail.shape[-1]
    B, C  = prev_tail.shape[:2]

    # Process all B×C simultaneously — STFT supports [*, L] batched input
    BC = B * C
    prev_flat = prev_tail.float().reshape(BC, L)   # [BC, L]
    new_flat  = new_head.float().reshape(BC, L)

    S_prev = torch.stft(prev_flat, n_fft=n_fft, hop_length=hop,
                        window=win, return_complex=True, onesided=True,
                        normalized=False, pad_mode='reflect')   # [BC, F, T]
    S_new  = torch.stft(new_flat, n_fft=n_fft, hop_length=hop,
                        window=win, return_complex=True, onesided=True,
                        normalized=False, pad_mode='reflect')

    mag_prev, phase_prev = S_prev.abs(), S_prev.angle()
    mag_new,  phase_new  = S_new.abs(),  S_new.angle()

    # Vectorized frame weight — no loop
    n_frames = S_prev.shape[-1]
    frame_centers = torch.arange(n_frames, device=device) * hop + n_fft // 2
    frame_centers = frame_centers.clamp(max=L - 1)
    frame_w = f_in_window[frame_centers]   # [T]
    w_in  = frame_w.view(1, 1, -1)        # broadcast to [BC, F, T]
    w_out = 1.0 - w_in

    mag_blend    = mag_prev * w_out + mag_new * w_in
    phase_select = torch.where(mag_prev > mag_new, phase_prev, phase_new)

    S_blend = mag_blend * torch.exp(1j * phase_select)
    out_spectral = torch.istft(S_blend, n_fft=n_fft, hop_length=hop,
                               window=win, length=L, onesided=True, normalized=False)

    # Blend spectral output with time-domain fallback — 70/30 reduces mid-high
    # presence from STFT phase selection while keeping the magnitude blend benefit.
    # Pure spectral was too present in 5-7kHz range.
    w_in_td  = w_in.squeeze(0)   # [1, T] → [T] for time interp
    # Reconstruct time-domain blend for comparison
    f_in_td  = f_in_window.view(1, 1, -1)
    f_out_td = 1.0 - f_in_td
    prev_flat2 = prev_tail.float().reshape(BC, L)
    new_flat2  = new_head.float().reshape(BC, L)
    out_timedomain = prev_flat2 * f_out_td.view(1, -1) + new_flat2 * f_in_td.view(1, -1)

    # Frequency-dependent spectral weight — full blend below 1kHz, fade to
    # time-domain above 1kHz. Kills vocal chorus without sacrificing bass seam quality.
    # Applied in STFT domain before ISTFT so it's spectrally precise.
    F_bins = out_spectral.shape[1] if out_spectral.dim() == 3 else (HPC_STFT_SIZE // 2 + 1)
    crossover_bin = max(1, int(1000.0 / (48000.0 / HPC_STFT_SIZE)))  # ~43 at 48kHz n_fft=1024
    freq_w = torch.ones(F_bins, device=prev_tail.device)
    if crossover_bin < F_bins:
        freq_w[crossover_bin:] = torch.linspace(1.0, 0.30, F_bins - crossover_bin,
                                                 device=prev_tail.device)
    # Re-blend in STFT: use freq_w as per-bin spectral weight
    # out_spectral came from pure spectral blend, out_timedomain from time-domain
    # Reconstruct freq-weighted output via STFT of time-domain + freq_w masking
    S_td = torch.stft(out_timedomain.float(), n_fft=HPC_STFT_SIZE, hop_length=HPC_HOP,
                      window=win, return_complex=True, onesided=True,
                      normalized=False, pad_mode='reflect')
    S_spec = torch.stft(out_spectral.float(), n_fft=HPC_STFT_SIZE, hop_length=HPC_HOP,
                        window=win, return_complex=True, onesided=True,
                        normalized=False, pad_mode='reflect')
    fw = freq_w.view(1, -1, 1)  # [1, F, 1] broadcast over [BC, F, T]
    S_blend_final = S_spec * fw + S_td * (1.0 - fw)
    out_flat = torch.istft(S_blend_final, n_fft=HPC_STFT_SIZE, hop_length=HPC_HOP,
                           window=win, length=L, onesided=True, normalized=False)

    return out_flat.reshape(B, C, L).to(prev_tail.dtype)


# =================================================================================
# == LSP: Latent-Domain Seam Predictor
# =================================================================================

def _slerp_latent(a, b, t):
    """Spherical linear interpolation on latent tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_norm = a_flat / (a_flat.norm() + CONST_EPSILON)
    b_norm = b_flat / (b_flat.norm() + CONST_EPSILON)
    dot = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return (a * (1 - t) + b * t).to(a.dtype)
    so = torch.sin(omega)
    result = (torch.sin((1 - t) * omega) / so) * a.float() +              (torch.sin(t * omega) / so) * b.float()
    return result.to(a.dtype)


def _generate_bridge_latents(latents, tile_schedule, bridge_half, device):
    """
    Generate SLERP bridge latents at each tile seam boundary.
    Returns dict: {seam_frame: bridge_latent [B, C, 2*bridge_half]}.
    """
    W = latents.shape[-1]
    bridges = {}

    for i in range(1, len(tile_schedule)):
        seam_frame = tile_schedule[i - 1][1]  # end of previous tile
        if seam_frame <= bridge_half or seam_frame >= W - bridge_half:
            continue

        bridge = []
        left  = latents[..., seam_frame - 1: seam_frame].clone().to(device)
        right = latents[..., seam_frame: seam_frame + 1].clone().to(device)

        # Vectorized SLERP — all t values at once
        n_frames = 2 * bridge_half
        t_vals = torch.linspace(0.0, 1.0, n_frames, device=device)  # [n_frames]
        a_f = left.float().flatten().to(device)   # [C*1] — force onto target device
        b_f = right.float().flatten().to(device)
        an = a_f / (a_f.norm() + CONST_EPSILON)
        bn = b_f / (b_f.norm() + CONST_EPSILON)
        dot = torch.clamp((an * bn).sum(), -1.0, 1.0)
        omega = torch.acos(dot)
        if omega.abs() < 1e-6:
            # Nearly parallel — linear interpolation
            bridge_flat = a_f.unsqueeze(0) * (1 - t_vals).unsqueeze(1) +                           b_f.unsqueeze(0) * t_vals.unsqueeze(1)
        else:
            so = torch.sin(omega)
            wa = (torch.sin((1 - t_vals) * omega) / so).unsqueeze(1)  # [T, 1]
            wb = (torch.sin(t_vals * omega) / so).unsqueeze(1)
            bridge_flat = wa * a_f.unsqueeze(0) + wb * b_f.unsqueeze(0)  # [T, C]
        # Reshape to [B, C, n_frames]
        orig_shape = left.shape  # [B, C, 1]
        bridge_tensor = bridge_flat.reshape(n_frames, *orig_shape[:-1]).permute(
            *range(1, len(orig_shape)), 0).contiguous()
        bridges[seam_frame] = bridge_tensor.to(left.dtype).to(device)

    return bridges


# =================================================================================
# == SCE: Spectral Continuity Enforcement
# =================================================================================

def _enforce_spectral_continuity(audio, boundaries, sample_rate):
    """
    Harmonic trajectory repair at tile seam boundaries.
    GPU-vectorized: single STFT over full [BC, L] batch, vectorized repair per seam.
    """
    if not boundaries or len(boundaries) < 2:
        return audio

    n_fft = SCE_STFT_SIZE
    hop   = SCE_HOP
    device = audio.device
    win   = torch.hann_window(n_fft, device=device)
    repair_frames = SCE_REPAIR_FRAMES
    max_freq_bin  = int(SCE_MAX_FREQ_HZ / (sample_rate / n_fft))
    threshold_lin = 10.0 ** (SCE_THRESHOLD_DB / 20.0)

    B, C, L = audio.shape

    with torch.no_grad():
        BC = B * C
        sig_flat = audio.float().reshape(BC, L)   # [BC, L]

        # Single STFT over all channels — GPU batched
        S = torch.stft(sig_flat, n_fft=n_fft, hop_length=hop,
                       window=win, return_complex=True, onesided=True,
                       normalized=False, pad_mode='reflect')   # [BC, F, T]
        F, T = S.shape[1], S.shape[2]
        mag   = S.abs()      # [BC, F, T]
        phase = S.angle()    # [BC, F, T]
        modified = False

        for seam_pos in boundaries:
            seam_frame = seam_pos // hop
            if seam_frame < repair_frames or seam_frame >= T - repair_frames:
                continue

            pre_frame  = max(0, seam_frame - repair_frames)
            post_frame = min(T - 1, seam_frame + repair_frames)

            # Vectorized discontinuity detection across all BC channels
            pre_mag  = mag[:, :max_freq_bin, seam_frame - 1]   # [BC, Fmax]
            post_mag = mag[:, :max_freq_bin, seam_frame]
            ratio    = (post_mag + CONST_EPSILON) / (pre_mag + CONST_EPSILON)
            disc_mask = (ratio > threshold_lin) | (ratio < 1.0 / threshold_lin)  # [BC, Fmax]

            if not disc_mask.any():
                continue

            # Vectorized interpolation over repair window
            repair_range = torch.arange(-repair_frames, repair_frames + 1,
                                        device=device, dtype=torch.float32)
            n_repair = len(repair_range)
            t_vals = ((repair_range + repair_frames) / (2 * repair_frames)).view(1, 1, n_repair)

            mag_pre  = mag[:, :max_freq_bin, pre_frame].unsqueeze(-1)    # [BC, Fmax, 1]
            mag_post = mag[:, :max_freq_bin, post_frame].unsqueeze(-1)
            mag_interp = mag_pre * (1 - t_vals) + mag_post * t_vals      # [BC, Fmax, n_repair]

            ph_pre  = phase[:, :max_freq_bin, pre_frame].unsqueeze(-1)
            ph_post = phase[:, :max_freq_bin, post_frame].unsqueeze(-1)
            ph_diff = ph_post - ph_pre
            ph_diff = ph_diff - 2 * math.pi * torch.round(ph_diff / (2 * math.pi))
            ph_interp = ph_pre + t_vals * ph_diff                         # [BC, Fmax, n_repair]

            # Apply to discontinuous bins only
            disc_exp = disc_mask.unsqueeze(-1).expand_as(mag_interp)     # [BC, Fmax, n_repair]
            for ri, t_idx in enumerate(range(seam_frame - repair_frames,
                                             seam_frame + repair_frames + 1)):
                if 0 <= t_idx < T:
                    mag_col   = mag[:, :max_freq_bin, t_idx]
                    phase_col = phase[:, :max_freq_bin, t_idx]
                    mag[:, :max_freq_bin, t_idx]   = torch.where(disc_mask,
                                                          mag_interp[:, :, ri], mag_col)
                    phase[:, :max_freq_bin, t_idx] = torch.where(disc_mask,
                                                          ph_interp[:, :, ri], phase_col)
                    modified = True

        if modified:
            S_repaired = mag * torch.exp(1j * phase)
            out_flat = torch.istft(S_repaired, n_fft=n_fft, hop_length=hop,
                                   window=win, length=L, onesided=True, normalized=False)
            return out_flat.reshape(B, C, L).to(audio.dtype)

    return audio



# =================================================================================
# == LSS: Latent Spectral Suppressor
# =================================================================================

def _apply_latent_suppressor(latents, strength=None, var_threshold=None, dc_remove=None):
    """
    Latent Spectral Suppressor (LSS).

    Suppresses low-variance bias channels in the latent tensor before VAE decode.
    The ringing/hum artifact in ACE-Step latents manifests as channels with:
      1. Persistent mean offset (DC bias) — encoder bias baked into specific channels
      2. Low variance relative to signal channels — carry noise floor not content

    Per-channel operation on [B, C, T] — acts on the 64 latent channels, not audio freqs.
    Applied once to the full latent before tiling — every tile benefits consistently
    without the artifact compounding across tile boundaries.
    """
    _strength      = strength      if strength      is not None else LSS_STRENGTH
    _var_threshold = var_threshold if var_threshold is not None else LSS_VARIANCE_THRESHOLD
    _dc_remove     = dc_remove     if dc_remove     is not None else LSS_DC_REMOVE

    with torch.no_grad():
        orig_dtype = latents.dtype
        x = latents.float()  # [B, C, T]

        # DC removal — subtract per-channel temporal mean
        if _dc_remove:
            ch_mean = x.mean(dim=-1, keepdim=True)  # [B, C, 1]
            x = x - ch_mean

        # Per-channel variance across time, averaged over batch → [C]
        ch_var  = x.var(dim=-1).mean(dim=0)
        var_min = ch_var.min()
        var_max = ch_var.max()
        var_norm = (ch_var - var_min) / (var_max - var_min + CONST_EPSILON)  # [C] in [0,1]

        # Smooth suppression gain: low-variance → suppress, high-variance → keep
        below = var_norm < _var_threshold
        smooth = (1.0 - _strength) + _strength * (var_norm / (_var_threshold + CONST_EPSILON))
        smooth = smooth.clamp(1.0 - _strength, 1.0)
        gain   = torch.where(below, smooth, torch.ones_like(var_norm))  # [C]

        n_suppressed = int(below.sum().item())
        gain_min = float(gain.min().item())
        # Store stats as tensor attributes for reporting
        result = x * gain.view(1, -1, 1)
        result._lss_n_suppressed = n_suppressed
        result._lss_gain_min = gain_min
        result._lss_var_range = (float(var_norm.min().item()), float(var_norm.max().item()))
        return result.to(orig_dtype)


def calculate_spectral_entropy_logic(tensor):
    """
    Shannon entropy monitor. Entropy on flattened absolute values.
    Returns scalar float (bits). Higher = more complex latent content.
    """
    with torch.no_grad():
        p     = torch.abs(tensor).flatten()
        p_sum = p.sum() + CONST_EPSILON
        p     = p / p_sum
        return (-torch.sum(p * torch.log2(p + CONST_EPSILON))).item()


# =================================================================================
# == DSP Utilities — Cached Window Factories
# =================================================================================

def _get_cached_window(length, mode, device):
    """Fade-in ramp [0→1], `length` samples. Cached by (length, mode, device)."""
    key = (length, mode, str(device))
    if key not in _WINDOW_CACHE:
        if mode == "Hann":
            win = torch.hann_window(length * 2, periodic=False, device=device)[:length]
        elif mode == "Cosine":
            x   = torch.linspace(0, math.pi / 2, length, device=device)
            win = torch.sin(x)
        else:
            win = torch.linspace(0, 1, length, device=device)
        _WINDOW_CACHE[key] = win
    return _WINDOW_CACHE[key]


def _get_trapezoidal_weight(length, edge_taper_frac, device):
    """Flat 1.0 center, ramps to 0.5 at both edges. For dual-pass merge."""
    key = (length, edge_taper_frac, str(device))
    if key not in _TRAPEZOID_CACHE:
        taper = max(CONST_MIN_OLA_SAMPLES, int(length * edge_taper_frac))
        w     = torch.ones(length, device=device)
        ramp  = torch.linspace(0.5, 1.0, taper, device=device)
        w[:taper]  = ramp
        w[-taper:] = ramp.flip(0)
        _TRAPEZOID_CACHE[key] = w
    return _TRAPEZOID_CACHE[key]


# =================================================================================
# == Format Detection
# =================================================================================

def detect_audio_format_logic(audio_tensor):
    """Detects [B,C,L] vs [B,L,C]. Handles 2-D and 4-D edge cases."""
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(0)
    if audio_tensor.ndim == 4:
        if audio_tensor.shape[2] == 1:
            return 'BCL', audio_tensor.squeeze(2)
        return 'BCL', audio_tensor
    if audio_tensor.ndim != 3:
        return 'BCL', audio_tensor
    B, d1, d2 = audio_tensor.shape
    if d1 <= 16 and d2 > 16:
        return 'BCL', audio_tensor
    elif d1 > 16 and d2 <= 16:
        return 'BLC', audio_tensor
    return 'BCL', audio_tensor


# =================================================================================
# == Core Decode Primitive
# =================================================================================

def _decode_tile(vae, latents, ctx_start, tile_end, expected_write_samples, device):
    """
    Decodes latents[ctx_start:tile_end] with Oobleck context warm-up.

    ctx_skip is ANCHORED to actual decoded length:
        ctx_skip = actual_decoded_len - expected_write_samples
    This absorbs VAE rounding slop into the discarded context region,
    ensuring the returned tensor is exactly expected_write_samples long
    (or as close as possible if the VAE under-produces).

    Returns: decoded [B, C, expected_write_samples] in BCL on device.
    """
    chunk = latents[..., ctx_start : tile_end].to(device)
    with torch.no_grad():
        decoded = vae.decode(chunk)
    fmt, decoded = detect_audio_format_logic(decoded)
    if fmt == 'BLC':
        decoded = decoded.transpose(1, 2).contiguous()
    decoded = decoded.to(device)

    actual_len = decoded.shape[-1]
    ctx_skip   = max(0, actual_len - expected_write_samples)
    if ctx_skip > 0 and actual_len > ctx_skip:
        decoded = decoded[..., ctx_skip:]

    return decoded


# =================================================================================
# == OLA Write Primitive
# =================================================================================

def _ola_write(output_audio, decoded_chunk, out_start, overlap_audio,
               blending_mode, device, final_samples):
    """
    Writes decoded_chunk into output_audio at out_start with OLA crossfade.
    overlap_audio is clamped to half of decoded_chunk length before use.
    """
    valid_len     = min(decoded_chunk.shape[-1], final_samples - out_start)
    if valid_len <= 0:
        return
    decoded_chunk = decoded_chunk[..., :valid_len]
    overlap_audio = min(overlap_audio, valid_len // 2)

    if out_start > 0 and overlap_audio > 0:
        blend_len    = min(overlap_audio, valid_len)
        prev_tail    = output_audio[..., out_start : out_start + blend_len]
        new_head     = decoded_chunk[..., :blend_len]
        actual_blend = min(prev_tail.shape[-1], new_head.shape[-1])

        if actual_blend > 0:
            f_in = _get_cached_window(actual_blend, blending_mode, device).view(1, 1, -1)
            # HPC: spectral crossfade for tonal preservation
            if HPC_ENABLED and actual_blend >= HPC_MIN_OVERLAP_SAMPLES:
                f_in_1d = f_in.view(-1)
                blended = _spectral_crossfade(
                    prev_tail[..., :actual_blend],
                    new_head[..., :actual_blend],
                    f_in_1d, device)
                output_audio[..., out_start : out_start + actual_blend] = blended
            else:
                f_out = (torch.sqrt((1.0 - f_in ** 2).clamp(min=0.0))
                         if blending_mode == "Cosine" else (1.0 - f_in))
                output_audio[..., out_start : out_start + actual_blend] = \
                    prev_tail[..., :actual_blend] * f_out + new_head[..., :actual_blend] * f_in

        if valid_len > actual_blend:
            output_audio[..., out_start + actual_blend : out_start + valid_len] = \
                decoded_chunk[..., actual_blend:]
    else:
        output_audio[..., out_start : out_start + valid_len] = decoded_chunk


# =================================================================================
# == Adaptive Tile Schedule Builder
# =================================================================================

def build_adaptive_tile_schedule(latents, base_tile_size, base_overlap,
                                 device, latent_offset=0):
    """
    Entropy-Driven Adaptive Tile Sizing.

    Scans latents starting at latent_offset, builds per-frame entropy map,
    maps entropy → tile size (inverted: high entropy → smaller tile).

    latent_offset allows Pass B to scan its real content region directly —
    no zero-padding, no coordinate shifting, no start<0 tiles.

    Returns list of (start, end, overlap) in absolute latent frame indices.
    """
    W        = latents.shape[-1]
    scan_win = CONST_ATS_ENTROPY_SCAN_WIN

    entropy_values, positions = [], []
    for i in range(latent_offset, W, scan_win):
        chunk = latents[..., i : i + scan_win].to(device)
        entropy_values.append(calculate_spectral_entropy_logic(chunk))
        positions.append(i)

    if not entropy_values:
        return [(latent_offset, W, base_overlap)]

    e_min, e_max  = min(entropy_values), max(entropy_values)
    e_range       = (e_max - e_min) + CONST_EPSILON
    overlap_ratio = base_overlap / max(base_tile_size, 1)

    frame_tile = {}
    for pos, e in zip(positions, entropy_values):
        norm_e = (e - e_min) / e_range
        t      = int(CONST_ATS_TILE_MAX - norm_e * (CONST_ATS_TILE_MAX - CONST_ATS_TILE_MIN))
        t      = max(CONST_ATS_TILE_MIN, (t // 64) * 64)
        ov     = max(8, int(t * overlap_ratio))
        for f in range(pos, min(pos + scan_win, W)):
            frame_tile[f] = (t, ov)

    schedule, cursor = [], latent_offset
    while cursor < W:
        tile_sz, tile_ov = frame_tile.get(cursor, (base_tile_size, base_overlap))
        end = min(W, cursor + tile_sz)
        schedule.append((cursor, end, tile_ov))
        hop = tile_sz - tile_ov
        if hop <= 0: hop = max(1, tile_sz // 2)
        cursor += hop

    return schedule


# =================================================================================
# == Single-Pass OLA Decode Engine
# =================================================================================

def _execute_single_pass(
    vae, latents, tile_schedule, upscale_factor, final_samples,
    output_channels, device, blending_mode, rms_leveling, context_prefix, B
):
    """
    OLA decode over tile_schedule.

    out_start = int(round(start * upscale_factor)) for every tile.
    This is correct and complete:
      - Max error 0.5 samples/seam from int(round()) — completely inaudible
      - Each tile anchors to its own latent position, no error accumulation
      - Works for Pass B (start > 0) without any special-casing
    The write-cursor accumulator (v2.0.x) is removed — it propagated
    VAE rounding slop forward causing drift that grew with track length.
    """
    output_audio = torch.zeros((B, output_channels, final_samples), device=device)
    boundaries   = []
    prev_rms     = None
    prev_entropy = None
    prev_lufs    = None
    _plgr_gains  = []

    for (start, end, overlap) in tile_schedule:
        ctx_start              = max(0, start - context_prefix)
        overlap_audio          = int(round(overlap * upscale_factor))
        expected_write_samples = int(round((end - start) * upscale_factor))

        decoded_chunk = _decode_tile(
            vae, latents, ctx_start, end, expected_write_samples, device
        )

        overlap_audio = min(overlap_audio, decoded_chunk.shape[-1] // 2)

        # Gain riding: PLGR (psychoacoustic loudness) or legacy RMS
        if PLGR_ENABLED:
            decoded_chunk, _lufs, _plgr_db = _apply_plgr_gain(
                decoded_chunk, 48000, PLGR_TARGET_LUFS, prev_lufs)
            _plgr_gains.append(_plgr_db)
            prev_lufs = _lufs
        elif rms_leveling and prev_rms is not None:
            current_rms = torch.sqrt(torch.mean(decoded_chunk ** 2) + CONST_EPSILON)
            current_ent = calculate_spectral_entropy_logic(decoded_chunk)
            ent_delta   = abs(current_ent - prev_entropy) if prev_entropy is not None else 0.0
            clamp_range = CONST_GAIN_CLAMP_BASE + min(
                ent_delta * 0.10, CONST_GAIN_CLAMP_MAX_RANGE - CONST_GAIN_CLAMP_BASE
            )
            gain = torch.clamp(
                prev_rms / (current_rms + CONST_EPSILON),
                1.0 - clamp_range, 1.0 + clamp_range
            )
            decoded_chunk = decoded_chunk * gain
            prev_entropy  = current_ent

        # Absolute RMS ceiling — prevents energy accumulation across long tracks.
        # Applied every tile regardless of rms_leveling flag.
        # The relative gain ride above corrects seam discontinuities; this corrects
        # the slow upward drift that the relative ride can't catch (it chases prev_rms).
        _tile_rms = torch.sqrt(torch.mean(decoded_chunk ** 2) + CONST_EPSILON)
        if _tile_rms > CONST_RMS_ABSOLUTE_CEIL:
            decoded_chunk = decoded_chunk * (CONST_RMS_ABSOLUTE_CEIL / _tile_rms)

        prev_rms = torch.sqrt(torch.mean(decoded_chunk ** 2) + CONST_EPSILON)
        if prev_entropy is None:
            prev_entropy = calculate_spectral_entropy_logic(decoded_chunk)

        out_start = int(round(start * upscale_factor))
        boundaries.append(out_start)
        _ola_write(output_audio, decoded_chunk, out_start, overlap_audio,
                   blending_mode, device, final_samples)

    return output_audio, boundaries


# =================================================================================
# == Dual-Pass Staggered Decode
# =================================================================================

def _execute_dual_pass(
    vae, latents, tile_size, overlap, upscale_factor, final_samples,
    output_channels, device, blending_mode, rms_leveling, context_prefix, B,
    adaptive_tiling, base_tile_size, base_overlap
):
    """
    Phase-offset dual-pass OLA with trapezoidal weight merge.

    Pass A: tiles from latent frame 0.
    Pass B: tiles from latent frame (tile_size // 2).

    Both passes use float anchor placement — correct independent positioning
    in the same audio buffer without any coordinate shifting.

    For adaptive tiling: each pass's entropy scan starts at its own offset,
    scanning real latent content (not zero-padded garbage).
    """
    W      = latents.shape[-1]
    offset = tile_size // 2

    if adaptive_tiling:
        sched_a = build_adaptive_tile_schedule(
            latents, base_tile_size, base_overlap, device, latent_offset=0
        )
        sched_b = build_adaptive_tile_schedule(
            latents, base_tile_size, base_overlap, device, latent_offset=offset
        )
    else:
        def _fixed_sched(start_lat, ts, ov):
            sched, cur = [], start_lat
            while cur < W:
                end = min(W, cur + ts)
                sched.append((cur, end, ov))
                hop = ts - ov
                if hop <= 0: hop = max(1, ts // 2)
                cur += hop
            return sched
        sched_a = _fixed_sched(0,      tile_size, overlap)
        sched_b = _fixed_sched(offset, tile_size, overlap)

    audio_a, bounds_a = _execute_single_pass(
        vae, latents, sched_a, upscale_factor, final_samples,
        output_channels, device, blending_mode, rms_leveling, context_prefix, B
    )
    audio_b, bounds_b = _execute_single_pass(
        vae, latents, sched_b, upscale_factor, final_samples,
        output_channels, device, blending_mode, rms_leveling, context_prefix, B
    )

    # Build trapezoidal weight maps from actual boundary positions
    weight_a = torch.zeros(final_samples, device=device)
    weight_b = torch.zeros(final_samples, device=device)

    def _fill_weights(weight_buf, boundaries, schedule):
        for i, out_s in enumerate(boundaries):
            s, e, _ = schedule[i] if i < len(schedule) else schedule[-1]
            tile_len = int(round((e - s) * upscale_factor))
            out_e    = min(final_samples, out_s + tile_len)
            L        = out_e - out_s
            if L <= 0:
                continue
            trap = _get_trapezoidal_weight(L, CONST_DUAL_PASS_EDGE_TAPER, device)
            weight_buf[out_s : out_e] = torch.max(weight_buf[out_s : out_e], trap)

    _fill_weights(weight_a, bounds_a, sched_a)
    _fill_weights(weight_b, bounds_b, sched_b)

    total_w  = weight_a + weight_b + CONST_EPSILON
    weight_a = (weight_a / total_w).view(1, 1, -1)
    weight_b = (weight_b / total_w).view(1, 1, -1)

    return audio_a * weight_a + audio_b * weight_b, bounds_a, bounds_b


# =================================================================================
# == Context-Aware Seam Repair  (CASR — replaces broken SLERP bridge)
# =================================================================================

def _apply_seam_repair(
    output_audio, latents, tile_schedule, boundaries,
    upscale_factor, vae, device, final_samples, blending_mode, context_prefix
):
    """
    Context-Aware Seam Repair (CASR).

    Why SLERP bridge failed with Oobleck:
      - 8-frame cold micro-decode: frames 0-6 output near-silence (causal
        warm-up not satisfied), frame 7 output is a click artifact.
      - Bridge wrote at float-computed seam position, not actual boundary.
      - Asymmetric fade_in-only: existing audio faded out but bridge only
        faded in at start — hard cut at bridge end into next tile.

    CASR strategy:
      For each seam between tile N and tile N+1:
        1. Decode a CONST_CASR_BLEND_FRAMES window centred on the seam,
           with full context_prefix warm-up (same strategy as main loop).
           This gives the VAE enough context to produce valid audio at the
           seam centre.
        2. Extract CONST_CASR_PATCH_FRAMES on each side of the seam centre.
        3. Apply a symmetric cosine bell (peak=1.0 at centre, 0.0 at both
           edges) — fades IN from existing audio, peaks at seam, fades BACK
           OUT to existing audio. No hard cuts anywhere.
        4. The patch is centred on the actual seam audio position (from
           boundary list + tile length, not float-computed from schedule).
    """
    if len(tile_schedule) < 2:
        return output_audio

    half_window = CONST_CASR_BLEND_FRAMES // 2
    half_patch  = int(round(CONST_CASR_PATCH_FRAMES * upscale_factor))

    for i in range(1, len(tile_schedule)):
        # Seam position: end of tile N in audio samples
        # Use actual boundary + tile write length (not float from schedule end)
        tile_n_start, tile_n_end, _ = tile_schedule[i - 1]
        tile_n_write_len = int(round((tile_n_end - tile_n_start) * upscale_factor))
        seam_audio       = min(boundaries[i - 1] + tile_n_write_len, final_samples)

        seam_lat  = tile_n_end
        ctx_start = max(0, seam_lat - half_window - context_prefix)
        win_end   = min(latents.shape[-1], seam_lat + half_window)
        # Tile start for this decode is the beginning of the blend window
        tile_start_for_decode = seam_lat - half_window
        if tile_start_for_decode < 0:
            tile_start_for_decode = 0
        expected_write = int(round((win_end - tile_start_for_decode) * upscale_factor))

        try:
            patch_decoded = _decode_tile(
                vae, latents, ctx_start, win_end, expected_write, device
            )
        except Exception:
            continue

        patch_len    = patch_decoded.shape[-1]
        patch_centre = patch_len // 2
        patch_half   = min(half_patch, patch_centre, patch_len - patch_centre)
        if patch_half < 4:
            continue

        patch_region = patch_decoded[..., patch_centre - patch_half : patch_centre + patch_half]
        pr_len       = patch_region.shape[-1]

        write_start = max(0, seam_audio - patch_half)
        write_end   = min(final_samples, seam_audio + patch_half)
        actual_len  = min(write_end - write_start, pr_len)
        if actual_len < 4:
            continue

        # Symmetric cosine bell: 0 at both edges, 1 at centre
        bell_half = actual_len // 2
        if bell_half < 2:
            continue
        fade_up   = _get_cached_window(bell_half, "Cosine", device)
        fade_down = fade_up.flip(0)
        bell      = torch.cat([fade_up, fade_down], dim=0)[:actual_len].view(1, 1, -1)

        existing  = output_audio[..., write_start : write_start + actual_len]
        patch_seg = patch_region[..., :actual_len]
        output_audio[..., write_start : write_start + actual_len] = \
            existing * (1.0 - bell) + patch_seg * bell

    return output_audio


# =================================================================================
# == Master Tiled Decode Engine  (Public API)
# =================================================================================

def execute_tiled_decode_logic(
    vae,
    latents,
    tile_size,
    overlap,
    device,
    blending_mode   = "Cosine",
    rms_leveling    = True,
    context_prefix  = CONST_VAE_CONTEXT_FRAMES,
    dual_pass       = False,
    slerp_bridge    = False,   # routes to CASR; param name kept for wrapper compat
    adaptive_tiling = False,
    sample_rate     = 48000,   # ACE-Step default
):
    """
    Master OLA Synthesis Engine — v3.0.1.
    See module header for full changelog.

    Returns: (output_audio [B, C, L], boundaries [list[int]], audio_format str)
    """
    if hasattr(vae, 'first_stage_model'):
        vae.first_stage_model.to(device)

    B = latents.shape[0]
    W = latents.shape[-1]

    # Calibration pilot
    test_len = min(tile_size, W)
    with torch.no_grad():
        dec_test = vae.decode(latents[..., :test_len].to(device))
    fmt_test, dec_test = detect_audio_format_logic(dec_test)
    if fmt_test == 'BLC':
        dec_test = dec_test.transpose(1, 2).contiguous().to(device)

    raw_upscale = float(dec_test.shape[-1]) / float(test_len)
    # Snap to nearest integer: ACE-Step Oobleck upscale is always an exact integer
    # (44100 / 5 = 8820 samples/frame). Float measurement from the pilot tile
    # introduces sub-sample error that compounds — tile 9 can be 24+ samples off,
    # audible as snare timing drift on long tracks. Snap only if within 0.5% of
    # an integer (non-ACE-Step VAEs with non-integer ratios keep the raw float).
    snapped        = round(raw_upscale)
    upscale_factor = float(snapped) if abs(snapped - raw_upscale) / (raw_upscale + 1e-8) < 0.005 \
                     else raw_upscale
    final_samples   = int(round(W * upscale_factor))
    output_channels = dec_test.shape[1]
    del dec_test

    # Primary tile schedule
    if adaptive_tiling:
        tile_schedule = build_adaptive_tile_schedule(
            latents, tile_size, overlap, device, latent_offset=0
        )
    else:
        tile_schedule, cursor = [], 0
        while cursor < W:
            end = min(W, cursor + tile_size)
            tile_schedule.append((cursor, end, overlap))
            hop = tile_size - overlap
            if hop <= 0: hop = max(1, tile_size // 2)
            cursor += hop

    # Ensure latents are on the target device before any LSP/dispatch work
    latents = latents.to(device)

    # LSS: Latent Spectral Suppressor — suppress bias channels before decode
    # Removes persistent low-variance encoder bias channels that the VAE decodes
    # as ringing/hum. Applied once to full latent before tiling so all tiles
    # benefit equally without compounding across tile boundaries.
    if LSS_ENABLED:
        latents = _apply_latent_suppressor(latents)

    # LSP: Generate bridge latents at seam boundaries
    _lsp_bridges = {}
    if LSP_ENABLED:
        _lsp_schedule = tile_schedule if not adaptive_tiling else build_adaptive_tile_schedule(
            latents, tile_size, overlap, device, latent_offset=0)
        _lsp_bridges = _generate_bridge_latents(
            latents, _lsp_schedule, LSP_BRIDGE_FRAMES, device)
        # Pre-decode latent injection DISABLED — Oobleck causal decoder is
        # sensitive to latent direction changes even after magnitude clamping.
        # Bridge latents reserved for future post-decode CASR integration.
        pass

    # Dispatch — returns bounds_a and bounds_b separately for visualization
    bounds_a = []
    bounds_b = []
    if PCT_TRIPLE_PASS:
        output_audio, bounds_a, bounds_b = _execute_triple_pass(
            vae, latents, tile_size, overlap, upscale_factor, final_samples,
            output_channels, device, blending_mode, rms_leveling, context_prefix, B,
            adaptive_tiling, tile_size, overlap, sample_rate
        )
        boundaries = bounds_a
    elif dual_pass:
        output_audio, bounds_a, bounds_b = _execute_dual_pass(
            vae, latents, tile_size, overlap, upscale_factor, final_samples,
            output_channels, device, blending_mode, rms_leveling, context_prefix, B,
            adaptive_tiling, tile_size, overlap
        )
        boundaries = bounds_a   # CASR uses Pass A seam positions
    else:
        output_audio, bounds_a = _execute_single_pass(
            vae, latents, tile_schedule, upscale_factor, final_samples,
            output_channels, device, blending_mode, rms_leveling, context_prefix, B
        )
        boundaries = bounds_a

    # Context-Aware Seam Repair
    if slerp_bridge:
        output_audio = _apply_seam_repair(
            output_audio, latents, tile_schedule, boundaries,
            upscale_factor, vae, device, final_samples, blending_mode, context_prefix
        )

    # HUM SUPPRESSION: Multi-band notch + bass shelf post-decode
    # Targets full VAE decoder harmonic series: 76, 211, 656, 1090Hz + sub shelf.
    # Corrected from single 426Hz notch (2026-05-16). Measured via Ableton EQ Eight.
    output_audio = _apply_hum_notch(output_audio, sample_rate)

    # POST-DECODE SPECTRAL DENOISER (Wiener filter) — runs BEFORE SCE
    # Critical order: denoise the VAE noise floor artifact BEFORE SCE sees it.
    # If denoiser runs after SCE, SCE detects the malformed pad/hum artifact as
    # a "harmonic trajectory" and repairs it into a coherent ring. Denoising first
    # removes the artifact at source so SCE only repairs genuine seam discontinuities.
    if SPECTRAL_DENOISE_ENABLED:
        output_audio = _apply_spectral_denoiser(output_audio, sample_rate)

    # SCE: Spectral Continuity Enforcement (harmonic repair at seams)
    # Now operates on denoised audio — repairs genuine seam discontinuities only,
    # not VAE noise floor artifacts.
    if SCE_ENABLED and boundaries:
        all_bounds = sorted(set(bounds_a + bounds_b))
        output_audio = _enforce_spectral_continuity(output_audio, all_bounds, sample_rate)

    # POST-DECODE DSP CHAIN
    # High-pass filter — removes sub-sonic rumble
    if POST_DECODE_HIGHPASS_HZ > 0.5:
        output_audio = _apply_highpass(output_audio, sample_rate, POST_DECODE_HIGHPASS_HZ)
    # Stereo width — M/S processing
    if abs(POST_DECODE_STEREO_WIDTH - 1.0) > 1e-4:
        output_audio = _apply_stereo_width(output_audio, POST_DECODE_STEREO_WIDTH)
    # Soft clipper — before LUFS normalizer
    if POST_DECODE_SOFT_CLIP_DB < -0.001:
        output_audio = _apply_soft_clip(output_audio, POST_DECODE_SOFT_CLIP_DB)

    # ── FEATURE ANALYTICS REPORT ─────────────────────────────────────────────
    _feat_lines = []
    _feat_lines.append("\n" + "="*62)
    _feat_lines.append("🔬 [VAE Tiled] FEATURE ANALYTICS")
    _feat_lines.append("="*62)

    # LSS
    if LSS_ENABLED and hasattr(latents, '_lss_n_suppressed'):
        ns  = latents._lss_n_suppressed
        gm  = latents._lss_gain_min
        vr  = latents._lss_var_range
        _feat_lines.append(f"  🧹 LSS: {ns} bias channels suppressed | "
                           f"min_gain={gm:.3f} | var_range=[{vr[0]:.3f},{vr[1]:.3f}] | "
                           f"dc_remove={LSS_DC_REMOVE} strength={LSS_STRENGTH:.2f}")
    elif LSS_ENABLED:
        _feat_lines.append(f"  🧹 LSS: ACTIVE (strength={LSS_STRENGTH:.2f} "
                           f"threshold={LSS_VARIANCE_THRESHOLD:.2f} dc={LSS_DC_REMOVE})")
    else:
        _feat_lines.append("  🧹 LSS: OFF")

    # PLGR
    if PLGR_ENABLED:
        _feat_lines.append(f"  📊 PLGR: ACTIVE (target={PLGR_TARGET_LUFS:.1f} LUFS | downward-only -1.5dB max)")
    else:
        _feat_lines.append("  📊 PLGR: OFF (using legacy RMS leveling)")

    # HPC
    if HPC_ENABLED:
        _feat_lines.append(f"  🎼 HPC: ACTIVE (freq-dependent blend: 100%@<1kHz→30%@Nyquist | "
                           f"min_overlap={HPC_MIN_OVERLAP_SAMPLES} | "
                           f"n_fft={HPC_STFT_SIZE})")
    else:
        _feat_lines.append("  🎼 HPC: OFF")

    # LSP
    if LSP_ENABLED:
        _feat_lines.append(f"  🌉 LSP: bridge generation ACTIVE | "
                           f"injection DISABLED (Oobleck causal sensitivity) | "
                           f"frames={LSP_BRIDGE_FRAMES}")
    else:
        _feat_lines.append("  🌉 LSP: OFF")

    # SCE
    if SCE_ENABLED:
        n_seams = len(sorted(set(bounds_a + bounds_b))) if bounds_a or bounds_b else 0
        _feat_lines.append(f"  🔧 SCE: ACTIVE | {n_seams} seams analyzed | "
                           f"threshold={SCE_THRESHOLD_DB:.1f}dB | "
                           f"max_freq={SCE_MAX_FREQ_HZ:.0f}Hz")
    else:
        _feat_lines.append("  🔧 SCE: OFF")

    # Denoiser
    if SPECTRAL_DENOISE_ENABLED:
        _feat_lines.append(f"  🔇 Wiener: ACTIVE (strength={SPECTRAL_DENOISE_STRENGTH:.2f} "
                           f"floor={SPECTRAL_DENOISE_FLOOR_PCT:.2f} "
                           f"min_gain={SPECTRAL_DENOISE_MIN_GAIN:.2f})")
    else:
        _feat_lines.append("  🔇 Wiener: OFF")

    # PCT / dual pass
    if PCT_TRIPLE_PASS:
        _feat_lines.append(f"  🔀 PCT: triple-pass | phase_align={PCT_PHASE_ALIGN}")
    elif bounds_b:
        _feat_lines.append(f"  🔀 Dual-pass: A={len(bounds_a)} seams | B={len(bounds_b)} seams")
    else:
        _feat_lines.append(f"  🔀 Single-pass: {len(bounds_a)} seams")

    # DSP chain
    dsp_parts = []
    if POST_DECODE_HIGHPASS_HZ > 0.5:
        dsp_parts.append(f"HPF={POST_DECODE_HIGHPASS_HZ:.0f}Hz")
    if abs(POST_DECODE_STEREO_WIDTH - 1.0) > 1e-4:
        dsp_parts.append(f"Width={POST_DECODE_STEREO_WIDTH:.1f}")
    if POST_DECODE_SOFT_CLIP_DB < -0.001:
        dsp_parts.append(f"SoftClip={POST_DECODE_SOFT_CLIP_DB:.1f}dBFS")
    _feat_lines.append(f"  🎛️  DSP: HumNotch=76/211/656/1090Hz+shelf | " + (" | ".join(dsp_parts) if dsp_parts else "passthrough"))

    _feat_lines.append("="*62 + "\n")
    print("\n".join(_feat_lines))

    # Return bounds_a and bounds_b separately — wrapper colours them differently in plot
    return output_audio, bounds_a, bounds_b, fmt_test


# =================================================================================
# == Tiled Encode (unchanged from v1.5.1)
# =================================================================================

def execute_tiled_encode_logic(vae, audio, tile_size_seconds, overlap_seconds):
    """Industrial-grade tiled encoder. Matches decode OLA for manifold stability."""
    waveform = audio['waveform']
    sr       = audio['sample_rate']
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    B, C, L      = waveform.shape
    tile_samples = int(tile_size_seconds * sr)
    overlap_samp = int(overlap_seconds   * sr)
    device       = waveform.device

    if hasattr(vae, 'first_stage_model'):
        vae.first_stage_model.to(device)

    pilot_size = min(tile_samples, L)
    with torch.no_grad():
        pilot_latent = vae.encode(waveform[..., :pilot_size])
    ratio          = float(pilot_size) / float(pilot_latent.shape[-1])
    final_lat_len  = int(L / ratio)
    lat_channels   = pilot_latent.shape[1]
    overlap_latent = int(overlap_samp / ratio)
    output_latent  = torch.zeros((B, lat_channels, final_lat_len), device=device)
    current_sample = 0

    while current_sample < L:
        start = max(0, current_sample - overlap_samp)
        end   = min(L, current_sample + tile_samples)
        if end == L and (end - start) < overlap_samp and L > tile_samples:
            start = max(0, L - tile_samples - overlap_samp)
        chunk = waveform[..., start : end].to(device)
        with torch.no_grad():
            chunk_latent = vae.encode(chunk)
        out_start    = int(start / ratio)
        valid_len    = min(chunk_latent.shape[-1], final_lat_len - out_start)
        chunk_latent = chunk_latent[..., :valid_len]
        if start > 0:
            blend_len    = min(overlap_latent, valid_len)
            prev_tail    = output_latent[..., out_start : out_start + blend_len]
            new_head     = chunk_latent[..., :blend_len]
            actual_blend = min(prev_tail.shape[-1], new_head.shape[-1])
            if actual_blend > 0:
                f_in  = torch.linspace(0, 1, actual_blend, device=device).view(1, 1, -1)
                output_latent[..., out_start : out_start + actual_blend] = \
                    prev_tail[..., :actual_blend] * (1.0 - f_in) + new_head[..., :actual_blend] * f_in
            if valid_len > actual_blend:
                output_latent[..., out_start + actual_blend : out_start + valid_len] = \
                    chunk_latent[..., actual_blend:]
        else:
            output_latent[..., out_start : out_start + valid_len] = chunk_latent
        current_sample += (tile_samples - overlap_samp)

    if output_latent.shape[-1] < CONST_LATENT_PADDING:
        pad = CONST_LATENT_PADDING - output_latent.shape[-1]
        output_latent = torch.cat(
            [output_latent, torch.zeros((B, lat_channels, pad), device=device)], dim=-1
        )
    return output_latent


# =================================================================================
# == Health Report
# =================================================================================

def get_core_health_report():
    return {
        "version":        "3.0.1",
        "backend":        "Pure PyTorch/CUDA",
        "cache_size":     len(_WINDOW_CACHE) + len(_TRAPEZOID_CACHE),
        "precision":      str(torch.get_default_dtype()),
        "entropy_monitor": "Active",
        "dual_pass":      "Available",
        "seam_repair":    "CASR (Context-Aware)",
        "adaptive_tiling":"Available",
        "license":        "GPLv3"
    }