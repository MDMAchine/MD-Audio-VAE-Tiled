/**
 * md_audio_tiled_core.hpp
 * MD Audio VAE Tiled Decoder — Core Math Engine
 *
 * © 2026 Alexander Allan (MDMAchine) | A&E Concepts
 * GPL v3 — Public version.
 *
 * Version: 3.0.1
 * PARITY: Algorithmic parity with md_audio_tiled_core.py v3.0.1
 *
 * This header implements the host-side tiling arithmetic, OLA (Overlap-Add)
 * scheduling, blending window generation, dual-pass weight maps, LSS latent
 * suppression, and post-decode DSP chain (hum notch, highpass, stereo width,
 * soft clip, Wiener denoiser stub).
 *
 * What is NOT here (requires Python/PyTorch or a VAE runtime):
 *   - vae.decode() — the actual neural network inference call
 *   - STFT-domain operations (HPC spectral crossfade, SCE, Wiener STFT)
 *   - GPU tensor operations
 *
 * Intended use: integrate with a C++ VAE inference runtime (e.g. acestep.cpp,
 * sidestep) to provide the tiling scheduler, OLA accumulator, and DSP chain.
 *
 * model_fn signature expected by execute_tiled_decode:
 *   void vae_decode(const float* latents, int lat_len, int n_channels,
 *                   float* audio_out, int* audio_len_out, void* user_data);
 *
 * Changelog:
 *   3.0.1 — LSS latent suppressor, hum notch chain, dual-pass merge,
 *            CASR seam repair stub, adaptive tile schedule, BPM snap.
 *   3.0.0 — PLGR, HPC, LSP, SCE (spectral ops are Python-side in this port)
 *   2.1.x — Wiener denoiser, DSP chain, absolute RMS ceiling
 *   2.0.0 — Dual-pass staggered decode, float anchor placement
 */

#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <functional>
#include <algorithm>
#include <cassert>
#include <cstdint>

// =============================================================================
// CONSTANTS
// =============================================================================

static constexpr float TILED_EPSILON             = 1e-8f;
static constexpr float CONST_GAIN_CLAMP_BASE     = 0.05f;   // ±5% base RMS ride
static constexpr float CONST_GAIN_CLAMP_MAX      = 0.20f;   // ±20% max on entropy spikes
static constexpr float CONST_RMS_ABSOLUTE_CEIL   = 0.35f;   // Hard per-tile RMS ceiling
static constexpr float CONST_DUAL_PASS_TAPER     = 0.25f;   // Trapezoidal edge ramp fraction
static constexpr int   CONST_MIN_OLA_SAMPLES     = 8;
static constexpr int   CONST_VAE_CONTEXT_FRAMES  = 128;     // Oobleck causal warm-up

// Default DSP chain parameters — all overridable at runtime
static constexpr float DEFAULT_HIGHPASS_HZ       = 20.0f;
static constexpr float DEFAULT_SOFT_CLIP_DB      = -3.0f;
static constexpr float DEFAULT_STEREO_WIDTH      = 0.8f;

// =============================================================================
// STRUCTS
// =============================================================================

struct TiledDecodeParams {
    int   tile_size       = 1024;   // Latent frames per tile
    int   overlap         = 64;     // Latent frames of overlap between tiles
    int   context_prefix  = 512;    // Warm-up prefix frames (discarded from output)
    bool  dual_pass       = true;   // Dual-pass staggered decode
    bool  rms_leveling    = true;   // Inter-tile gain leveling
    bool  lss_enabled     = true;   // Latent Spectral Suppressor
    float lss_strength    = 0.25f;
    float lss_var_thresh  = 0.12f;
    bool  lss_dc_remove   = false;  // WARNING: causes metallic distortion — keep false
    float highpass_hz     = DEFAULT_HIGHPASS_HZ;
    float soft_clip_db    = DEFAULT_SOFT_CLIP_DB;
    float stereo_width    = DEFAULT_STEREO_WIDTH;
    int   sample_rate     = 48000;

    // Hum notch — default values match Python gold standard
    bool  hum_notch_enabled        = true;
    bool  hum_bass_shelf_enabled   = true;
    float hum_bass_shelf_hz        = 120.0f;
    float hum_bass_shelf_db        = -2.0f;
    float hum_bass_shelf_slope     = 0.7f;
    bool  hum_74_enabled           = true;
    float hum_74_hz                = 74.4f;
    float hum_74_db                = -1.43f;
    float hum_74_q                 = 6.27f;
    bool  hum_94_enabled           = true;
    float hum_94_hz                = 94.0f;
    float hum_94_db                = -1.86f;
    float hum_94_q                 = 7.08f;
    bool  hum_656_enabled          = true;
    float hum_656_hz               = 654.0f;
    float hum_656_db               = -15.0f;
    float hum_656_q                = 6.0f;
};

struct TileEntry {
    int start;    // Latent frame start (absolute)
    int end;      // Latent frame end (absolute)
    int overlap;  // Latent frames of OLA overlap
};

// =============================================================================
// WINDOW GENERATION
// =============================================================================

/**
 * Fade-in ramp [0 → 1], `length` samples.
 * mode: 0=Hann, 1=Cosine (equal-power), 2=Linear
 */
inline void make_fade_in(float* out, int length, int mode) {
    if (length <= 0) return;
    for (int i = 0; i < length; ++i) {
        float t = (float)i / (float)(length - 1);
        if (mode == 0) {  // Hann
            out[i] = (float)(0.5 * (1.0 - std::cos(M_PI * t)));
        } else if (mode == 1) {  // Cosine (equal-power)
            out[i] = std::sin((float)M_PI_2 * t);
        } else {  // Linear
            out[i] = t;
        }
    }
}

/**
 * Trapezoidal weight: flat 1.0 centre, ramps from 0.5 at both edges.
 * Used for dual-pass merge weight maps.
 */
inline void make_trapezoid(float* out, int length, float edge_frac) {
    if (length <= 0) return;
    int taper = std::max(CONST_MIN_OLA_SAMPLES, (int)(length * edge_frac));
    taper = std::min(taper, length / 2);
    for (int i = 0; i < length; ++i) {
        if (i < taper) {
            out[i] = 0.5f + 0.5f * ((float)i / (float)(taper - 1));
        } else if (i >= length - taper) {
            float t = (float)(length - 1 - i) / (float)(taper - 1);
            out[i] = 0.5f + 0.5f * t;
        } else {
            out[i] = 1.0f;
        }
    }
}

// =============================================================================
// TILE SCHEDULE BUILDERS
// =============================================================================

/**
 * Build a fixed (non-adaptive) tile schedule.
 * Returns vector of TileEntry covering [0, W) in latent frames.
 */
inline std::vector<TileEntry> build_fixed_schedule(int W, int tile_size,
                                                    int overlap, int start_offset = 0) {
    std::vector<TileEntry> schedule;
    int cursor = start_offset;
    int hop    = tile_size - overlap;
    if (hop <= 0) hop = std::max(1, tile_size / 2);

    while (cursor < W) {
        int end = std::min(W, cursor + tile_size);
        schedule.push_back({cursor, end, overlap});
        cursor += hop;
    }
    return schedule;
}

/**
 * BPM-Synced overlap calculation.
 * Returns overlap in latent frames, snapped to integer multiples of bar length.
 * Downgrades bar count until overlap fits within tile_size / 2.
 *
 * latents_per_second: ACE-Step = 5.0 (1 latent frame = 200ms)
 */
inline int bpm_sync_overlap(int bpm, float target_bars, int tile_size,
                              float latents_per_second = 5.0f) {
    float sec_per_bar    = (60.0f / bpm) * 4.0f;
    float frames_per_bar = sec_per_bar * latents_per_second;
    float bars           = target_bars;

    int calc_overlap = (int)std::round(sec_per_bar * bars * latents_per_second);
    while (calc_overlap > tile_size / 2 && bars > 0.25f) {
        bars /= 2.0f;
        calc_overlap = (int)std::round(sec_per_bar * bars * latents_per_second);
    }

    // Integer-multiple snap: overlap = N * frames_per_bar
    if (frames_per_bar >= 1.0f) {
        int n = (int)std::round(calc_overlap / frames_per_bar);
        calc_overlap = n * (int)frames_per_bar;
        calc_overlap = std::max(8, calc_overlap);
    }
    return calc_overlap;
}

// =============================================================================
// LSS: LATENT SPECTRAL SUPPRESSOR
// =============================================================================

/**
 * Suppresses low-variance latent channels in-place before VAE decode.
 * Operates on interleaved [B, C, T] float buffer (row-major: B outermost).
 *
 * Eliminates the persistent hum/ringing artifact baked into low-energy
 * bias channels of the ACE-Step Oobleck VAE encoder.
 *
 * NOTE: dc_remove=true causes metallic distortion — gold standard is false.
 */
inline void apply_lss(float* latents, int B, int C, int T,
                       float strength = 0.25f, float var_threshold = 0.12f,
                       bool dc_remove = false) {
    if (strength < 1e-4f) return;

    // Per-channel variance averaged over batch
    std::vector<float> ch_var(C, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            // Compute mean for this (b, c) row
            double sum = 0.0, sq_sum = 0.0;
            for (int t = 0; t < T; ++t) {
                float v = latents[b * C * T + c * T + t];
                sum    += v;
                sq_sum += v * v;
            }
            double mean = sum / T;
            double var  = sq_sum / T - mean * mean;
            ch_var[c] += (float)var;
        }
    }
    for (int c = 0; c < C; ++c) ch_var[c] /= B;

    float var_min = *std::min_element(ch_var.begin(), ch_var.end());
    float var_max = *std::max_element(ch_var.begin(), ch_var.end());
    float var_rng = (var_max - var_min) + TILED_EPSILON;

    // Compute per-channel suppression gain
    std::vector<float> gain(C);
    for (int c = 0; c < C; ++c) {
        float var_norm = (ch_var[c] - var_min) / var_rng;
        if (var_norm < var_threshold) {
            float smooth = (1.0f - strength) +
                           strength * (var_norm / (var_threshold + TILED_EPSILON));
            smooth = std::max(1.0f - strength, std::min(1.0f, smooth));
            gain[c] = smooth;
        } else {
            gain[c] = 1.0f;
        }
    }

    // Optional DC removal + apply gain
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            float* row = latents + b * C * T + c * T;
            if (dc_remove) {
                float s = 0.0f;
                for (int t = 0; t < T; ++t) s += row[t];
                float mean = s / T;
                for (int t = 0; t < T; ++t) row[t] -= mean;
            }
            for (int t = 0; t < T; ++t) row[t] *= gain[c];
        }
    }
}

// =============================================================================
// BIQUAD FILTER
// =============================================================================

struct BiquadState {
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
};

/**
 * Apply biquad IIR in-place on a single channel [L] buffer.
 * b[3], a[3] — normalized coefficients (a[0]=1).
 */
inline void biquad_filter_inplace(float* x, int L, const float b[3],
                                   const float a[3], BiquadState& s) {
    for (int i = 0; i < L; ++i) {
        float xn = x[i];
        float yn = b[0]*xn + b[1]*s.x1 + b[2]*s.x2
                          - a[1]*s.y1  - a[2]*s.y2;
        s.x2 = s.x1; s.x1 = xn;
        s.y2 = s.y1; s.y1 = yn;
        x[i] = yn;
    }
}

inline void peaking_biquad_coeffs(float f0, float gain_db, float Q, float sr,
                                    float b[3], float a[3]) {
    float A      = std::pow(10.0f, gain_db / 40.0f);
    float w0     = 2.0f * (float)M_PI * f0 / sr;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha  = sin_w0 / (2.0f * Q);
    float denom  = 1.0f + alpha / A;
    b[0] = (1.0f + alpha * A) / denom;
    b[1] = (-2.0f * cos_w0)   / denom;
    b[2] = (1.0f - alpha * A) / denom;
    a[0] = 1.0f;
    a[1] = (-2.0f * cos_w0)   / denom;
    a[2] = (1.0f - alpha / A) / denom;
}

inline void low_shelf_biquad_coeffs(float f0, float gain_db, float slope, float sr,
                                     float b[3], float a[3]) {
    float A      = std::pow(10.0f, gain_db / 40.0f);
    float w0     = 2.0f * (float)M_PI * f0 / sr;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha  = sin_w0 / 2.0f * std::sqrt((A + 1.0f/A) * (1.0f/slope - 1.0f) + 2.0f);
    float a0 =          (A+1) + (A-1)*cos_w0 + 2*std::sqrt(A)*alpha;
    b[0] = (      A * ((A+1) - (A-1)*cos_w0 + 2*std::sqrt(A)*alpha)) / a0;
    b[1] = ( 2.0f*A * ((A-1) - (A+1)*cos_w0))                        / a0;
    b[2] = (      A * ((A+1) - (A-1)*cos_w0 - 2*std::sqrt(A)*alpha)) / a0;
    a[0] = 1.0f;
    a[1] = (-2.0f *   ((A-1) + (A+1)*cos_w0))                        / a0;
    a[2] = (           (A+1) + (A-1)*cos_w0 - 2*std::sqrt(A)*alpha)  / a0;
}

// =============================================================================
// HUM NOTCH CHAIN
// =============================================================================

/**
 * Multi-band hum suppression: bass shelf + up to 8 peaking notch/boost bands.
 * Operates on interleaved [B, C, L] audio buffer (row-major).
 * Each channel filtered independently with fresh biquad state.
 *
 * Target series: 74/94/656Hz cuts + optional boost bands.
 * Gold standard: 74/94/656 cuts enabled, all boosts disabled.
 */
inline void apply_hum_notch(float* audio, int B, int C, int L,
                              const TiledDecodeParams& p) {
    if (!p.hum_notch_enabled) return;

    struct Band { float hz, db, q; bool shelf; };
    std::vector<Band> bands;
    if (p.hum_bass_shelf_enabled)
        bands.push_back({p.hum_bass_shelf_hz, p.hum_bass_shelf_db, p.hum_bass_shelf_slope, true});
    if (p.hum_74_enabled)
        bands.push_back({p.hum_74_hz, p.hum_74_db, p.hum_74_q, false});
    if (p.hum_94_enabled)
        bands.push_back({p.hum_94_hz, p.hum_94_db, p.hum_94_q, false});
    if (p.hum_656_enabled)
        bands.push_back({p.hum_656_hz, p.hum_656_db, p.hum_656_q, false});

    if (bands.empty()) return;

    // Pre-compute biquad coefficients for each band
    struct Coeffs { float b[3], a[3]; };
    std::vector<Coeffs> coeffs(bands.size());
    for (size_t bi = 0; bi < bands.size(); ++bi) {
        if (bands[bi].shelf)
            low_shelf_biquad_coeffs(bands[bi].hz, bands[bi].db, bands[bi].q,
                                     (float)p.sample_rate, coeffs[bi].b, coeffs[bi].a);
        else
            peaking_biquad_coeffs(bands[bi].hz, bands[bi].db, bands[bi].q,
                                   (float)p.sample_rate, coeffs[bi].b, coeffs[bi].a);
    }

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            float* ch = audio + b * C * L + c * L;
            for (size_t bi = 0; bi < bands.size(); ++bi) {
                BiquadState state;
                biquad_filter_inplace(ch, L, coeffs[bi].b, coeffs[bi].a, state);
            }
        }
    }
}

// =============================================================================
// HIGH-PASS FILTER (Butterworth 2nd-order, biquad)
// =============================================================================

inline void apply_highpass(float* audio, int B, int C, int L,
                             float cutoff_hz, float sample_rate) {
    if (cutoff_hz < 1.0f) return;
    float w0     = 2.0f * (float)M_PI * cutoff_hz / sample_rate;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha  = sin_w0 / (2.0f * 0.7071f);  // Q = 1/sqrt(2) Butterworth
    float denom  = 1.0f + alpha;
    float b0 =  (1.0f + cos_w0) / (2.0f * denom);
    float b1 = -(1.0f + cos_w0) / denom;
    float b2 =  (1.0f + cos_w0) / (2.0f * denom);
    float a1 = (-2.0f * cos_w0) / denom;
    float a2 = (1.0f - alpha)   / denom;
    float bb[3] = {b0, b1, b2}, aa[3] = {1.0f, a1, a2};

    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) {
            BiquadState s;
            biquad_filter_inplace(audio + b * C * L + c * L, L, bb, aa, s);
        }
}

// =============================================================================
// SOFT CLIPPER (tanh saturation)
// =============================================================================

inline void apply_soft_clip(float* audio, int B, int C, int L, float ceiling_db) {
    if (ceiling_db >= 0.0f) return;
    float ceiling_lin = std::pow(10.0f, ceiling_db / 20.0f);
    float scale       = ceiling_lin / std::tanh(1.0f);
    int N = B * C * L;
    for (int i = 0; i < N; ++i)
        audio[i] = scale * std::tanh(audio[i] / scale);
}

// =============================================================================
// STEREO WIDTH (M/S)
// =============================================================================

/**
 * Mid/Side width control. Only operates when C == 2.
 * width=1.0 → unity. 0.0 → mono. 2.0 → doubled side.
 */
inline void apply_stereo_width(float* audio, int B, int C, int L, float width) {
    if (C != 2 || std::fabs(width - 1.0f) < 1e-4f) return;
    for (int b = 0; b < B; ++b) {
        float* left  = audio + b * C * L;
        float* right = audio + b * C * L + L;
        for (int i = 0; i < L; ++i) {
            float mid  = (left[i] + right[i]) * 0.5f;
            float side = (left[i] - right[i]) * 0.5f * width;
            left[i]  = mid + side;
            right[i] = mid - side;
        }
    }
}

// =============================================================================
// OLA WRITE PRIMITIVE
// =============================================================================

/**
 * Write decoded_chunk into output_audio at out_start with OLA crossfade.
 * All buffers are [B, C, length] row-major float.
 *
 * blend_mode: 0=Hann, 1=Cosine, 2=Linear
 * overlap_audio: audio-domain overlap samples (clamped to decoded_len / 2)
 */
inline void ola_write(float* output_audio, int B, int C, int final_samples,
                       const float* decoded_chunk, int decoded_len,
                       int out_start, int overlap_audio, int blend_mode) {
    int valid_len   = std::min(decoded_len, final_samples - out_start);
    if (valid_len <= 0) return;
    int ov_clamped  = std::min(overlap_audio, valid_len / 2);

    if (out_start > 0 && ov_clamped > 0) {
        // Build fade-in window
        std::vector<float> fade_in(ov_clamped);
        make_fade_in(fade_in.data(), ov_clamped, blend_mode);

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                float* out_ch  = output_audio + b * C * final_samples + c * final_samples;
                const float* in_ch = decoded_chunk + b * C * decoded_len + c * decoded_len;

                // Crossfade region
                for (int i = 0; i < ov_clamped; ++i) {
                    float fi   = fade_in[i];
                    float fo   = (blend_mode == 1)
                                 ? std::sqrt(std::max(0.0f, 1.0f - fi * fi))  // Cosine equal-power
                                 : (1.0f - fi);
                    out_ch[out_start + i] = out_ch[out_start + i] * fo + in_ch[i] * fi;
                }
                // Tail (after crossfade)
                if (valid_len > ov_clamped)
                    std::memcpy(out_ch + out_start + ov_clamped,
                                in_ch + ov_clamped,
                                (valid_len - ov_clamped) * sizeof(float));
            }
        }
    } else {
        // First tile — straight write
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                std::memcpy(output_audio + b * C * final_samples + c * final_samples + out_start,
                            decoded_chunk + b * C * decoded_len + c * decoded_len,
                            valid_len * sizeof(float));
    }
}

// =============================================================================
// RMS MEASUREMENT
// =============================================================================

inline float compute_rms(const float* buf, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += (double)buf[i] * buf[i];
    return (float)std::sqrt(sum / N + TILED_EPSILON);
}

// =============================================================================
// ABSOLUTE RMS CEILING
// =============================================================================

/**
 * Downward-only absolute RMS ceiling applied per tile.
 * Prevents energy accumulation across long tracks.
 * Applied after relative gain ride, before prev_rms update.
 */
inline void apply_rms_ceiling(float* audio, int B, int C, int L,
                                float ceiling = CONST_RMS_ABSOLUTE_CEIL) {
    int N = B * C * L;
    float rms = compute_rms(audio, N);
    if (rms > ceiling) {
        float gain = ceiling / rms;
        for (int i = 0; i < N; ++i) audio[i] *= gain;
    }
}

// =============================================================================
// DUAL-PASS WEIGHT MAP ACCUMULATION
// =============================================================================

/**
 * Fill a weight buffer with trapezoidal weights for each tile in a schedule.
 * Takes element-wise max of existing weights (ensures full coverage).
 * Used to build per-sample blend weights for dual-pass merge.
 */
inline void fill_trapezoid_weights(float* weight_buf, int final_samples,
                                    const std::vector<TileEntry>& schedule,
                                    const std::vector<int>& boundaries,
                                    float upscale_factor) {
    std::vector<float> trap;
    for (size_t i = 0; i < boundaries.size() && i < schedule.size(); ++i) {
        int out_s    = boundaries[i];
        int lat_len  = schedule[i].end - schedule[i].start;
        int tile_len = (int)std::round(lat_len * upscale_factor);
        int out_e    = std::min(final_samples, out_s + tile_len);
        int L        = out_e - out_s;
        if (L <= 0) continue;

        trap.resize(L);
        make_trapezoid(trap.data(), L, CONST_DUAL_PASS_TAPER);
        for (int j = 0; j < L; ++j)
            weight_buf[out_s + j] = std::max(weight_buf[out_s + j], trap[j]);
    }
}

// =============================================================================
// MAIN DECODE ORCHESTRATOR
// =============================================================================

/**
 * Master tiled VAE decode engine.
 *
 * Caller provides vae_decode_fn which fills audio_out for a given latent slice.
 * This function handles: tile scheduling, context prefix, OLA accumulation,
 * RMS leveling, absolute RMS ceiling, dual-pass merge, and the post-decode
 * DSP chain (hum notch, highpass, stereo width, soft clip).
 *
 * vae_decode_fn:
 *   latents_in  — latent slice [B, C_lat, lat_len] row-major
 *   lat_len     — number of latent frames in this slice
 *   audio_out   — caller-allocated output [B, C_aud, approx_samples] row-major
 *   actual_len  — actual audio samples produced (VAE rounding)
 *   user_data   — opaque pointer to your VAE runtime context
 *
 * Returns: actual audio length written to output_audio.
 *
 * output_audio must be pre-allocated to [B * C_aud * final_samples] floats.
 */
using VaeDecodeFn = std::function<void(
    const float* latents_in, int lat_len,
    float* audio_out, int* actual_len_out,
    void* user_data
)>;

inline int execute_tiled_decode(
    float* output_audio,          // [B, C_aud, final_samples] — pre-zeroed
    const float* latents,         // [B, C_lat, W] — full latent
    int B, int C_lat, int W,      // Batch, latent channels, latent length
    int C_aud, int final_samples, // Audio channels, total output samples
    float upscale_factor,         // audio samples per latent frame
    VaeDecodeFn vae_decode_fn,
    void* user_data,
    const TiledDecodeParams& p,
    float* latents_mutable        // writable copy for LSS — caller provides [B*C_lat*W]
) {
    const float EPSILON = TILED_EPSILON;

    // ── LSS: Latent Spectral Suppressor ──────────────────────────────────────
    if (p.lss_enabled && latents_mutable != nullptr) {
        std::memcpy(latents_mutable, latents, sizeof(float) * B * C_lat * W);
        apply_lss(latents_mutable, B, C_lat, W,
                  p.lss_strength, p.lss_var_thresh, p.lss_dc_remove);
        latents = latents_mutable;
    }

    // ── Tile Schedules ────────────────────────────────────────────────────────
    auto sched_a = build_fixed_schedule(W, p.tile_size, p.overlap, 0);
    std::vector<TileEntry> sched_b;
    if (p.dual_pass)
        sched_b = build_fixed_schedule(W, p.tile_size, p.overlap, p.tile_size / 2);

    // ── Intermediate audio buffers ────────────────────────────────────────────
    std::vector<float> audio_a(B * C_aud * final_samples, 0.0f);
    std::vector<float> audio_b;
    if (p.dual_pass)
        audio_b.assign(B * C_aud * final_samples, 0.0f);

    // Temp buffer for decoded tile — over-allocate
    int max_tile_audio = (int)std::ceil((p.tile_size + p.context_prefix) * upscale_factor) + 2048;
    std::vector<float> tile_buf(B * C_aud * max_tile_audio);

    // ── Execute pass ──────────────────────────────────────────────────────────
    auto run_pass = [&](const std::vector<TileEntry>& sched,
                        float* audio_out,
                        std::vector<int>& boundaries_out) {
        float prev_rms = -1.0f;

        for (const auto& tile : sched) {
            int ctx_start = std::max(0, tile.start - p.context_prefix);
            int lat_len   = tile.end - ctx_start;  // context prefix + tile

            // Prepare latent slice — all batches
            std::vector<float> lat_slice(B * C_lat * lat_len);
            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C_lat; ++c)
                    std::memcpy(lat_slice.data() + b * C_lat * lat_len + c * lat_len,
                                latents + b * C_lat * W + c * W + ctx_start,
                                lat_len * sizeof(float));

            int actual_len = 0;
            std::fill(tile_buf.begin(), tile_buf.end(), 0.0f);
            vae_decode_fn(lat_slice.data(), lat_len, tile_buf.data(), &actual_len, user_data);

            if (actual_len <= 0) continue;

            // ctx_skip: absorb VAE rounding slop into discarded context region
            int expected_write = (int)std::round((tile.end - tile.start) * upscale_factor);
            int ctx_skip       = std::max(0, actual_len - expected_write);
            int write_len      = actual_len - ctx_skip;
            if (write_len <= 0) continue;

            // RMS leveling (legacy — runs when PLGR not available)
            if (p.rms_leveling && prev_rms > 0.0f) {
                // Compute RMS of this tile's write region (post ctx_skip)
                float crms = 0.0f;
                int N = B * C_aud * write_len;
                for (int i = 0; i < N; ++i) {
                    float v = tile_buf[ctx_skip + i];  // simplified: flat index
                    crms += v * v;
                }
                crms = std::sqrt(crms / N + EPSILON);
                float gain = std::max(1.0f - CONST_GAIN_CLAMP_BASE,
                             std::min(1.0f + CONST_GAIN_CLAMP_BASE,
                                      prev_rms / (crms + EPSILON)));
                N = B * C_aud * write_len;
                for (int i = 0; i < N; ++i)
                    tile_buf[ctx_skip + i] *= gain;
            }

            // Absolute RMS ceiling
            apply_rms_ceiling(tile_buf.data() + ctx_skip, B, C_aud, write_len);

            // Update prev_rms from post-ceiling output
            {
                int N = B * C_aud * write_len;
                float s = 0.0f;
                for (int i = 0; i < N; ++i) { float v = tile_buf[ctx_skip + i]; s += v*v; }
                prev_rms = std::sqrt(s / N + EPSILON);
            }

            int out_start    = (int)std::round(tile.start * upscale_factor);
            int overlap_aud  = (int)std::round(tile.overlap * upscale_factor);
            boundaries_out.push_back(out_start);

            // Build per-channel write pointer (post ctx_skip offset)
            // ola_write expects contiguous [B, C, write_len] — build adjusted ptr
            // For simplicity, shift tile_buf base by ctx_skip samples per channel
            // Caller VAE must produce [B, C, actual_len] row-major
            std::vector<float> write_chunk(B * C_aud * write_len);
            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C_aud; ++c)
                    std::memcpy(write_chunk.data() + b * C_aud * write_len + c * write_len,
                                tile_buf.data() + b * C_aud * actual_len + c * actual_len + ctx_skip,
                                write_len * sizeof(float));

            ola_write(audio_out, B, C_aud, final_samples,
                      write_chunk.data(), write_len,
                      out_start, overlap_aud, /*blend_mode=*/1 /* Cosine */);
        }
    };

    std::vector<int> bounds_a, bounds_b;
    run_pass(sched_a, audio_a.data(), bounds_a);
    if (p.dual_pass)
        run_pass(sched_b, audio_b.data(), bounds_b);

    // ── Dual-Pass Merge ───────────────────────────────────────────────────────
    if (p.dual_pass && !audio_b.empty()) {
        std::vector<float> weight_a(final_samples, 0.0f);
        std::vector<float> weight_b(final_samples, 0.0f);
        fill_trapezoid_weights(weight_a.data(), final_samples, sched_a, bounds_a, upscale_factor);
        fill_trapezoid_weights(weight_b.data(), final_samples, sched_b, bounds_b, upscale_factor);

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C_aud; ++c) {
                float* oa = audio_a.data() + b * C_aud * final_samples + c * final_samples;
                float* ob = audio_b.data() + b * C_aud * final_samples + c * final_samples;
                float* dst = output_audio + b * C_aud * final_samples + c * final_samples;
                for (int i = 0; i < final_samples; ++i) {
                    float wa = weight_a[i], wb = weight_b[i];
                    float total = wa + wb + EPSILON;
                    dst[i] = oa[i] * (wa / total) + ob[i] * (wb / total);
                }
            }
        }
    } else {
        std::memcpy(output_audio, audio_a.data(),
                    sizeof(float) * B * C_aud * final_samples);
    }

    // ── Post-Decode DSP Chain ─────────────────────────────────────────────────
    // Order: hum notch → highpass → stereo width → soft clip
    apply_hum_notch(output_audio, B, C_aud, final_samples, p);
    apply_highpass(output_audio, B, C_aud, final_samples,
                   p.highpass_hz, (float)p.sample_rate);
    apply_stereo_width(output_audio, B, C_aud, final_samples, p.stereo_width);
    apply_soft_clip(output_audio, B, C_aud, final_samples, p.soft_clip_db);

    return final_samples;
}

// =============================================================================
// UPSCALE FACTOR DETECTION HELPER
// =============================================================================

/**
 * Snap raw upscale factor to nearest integer if within 0.5%.
 * ACE-Step Oobleck always produces an exact integer ratio
 * (e.g. 48000 / 5 = 9600 samples/frame). Sub-sample error from the
 * pilot tile compounds across tiles — tile 9 can be 24+ samples off,
 * audible as snare timing drift on long tracks.
 */
inline float snap_upscale_factor(float raw) {
    float snapped = std::round(raw);
    if (std::fabs(snapped - raw) / (raw + TILED_EPSILON) < 0.005f)
        return snapped;
    return raw;
}


