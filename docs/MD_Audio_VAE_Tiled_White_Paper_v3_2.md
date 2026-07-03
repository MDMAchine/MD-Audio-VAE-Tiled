# MD Audio VAE Tiled Decoder
## Technical White Paper for the Oobleck Causal VAE Chunked-Decode Artifact Problem

**Alexander Allan (MDMAchine)**  
A&E Concepts, New Bedford MA  

**Version:** v3.2, Pre-print  
**Date:** July 2026  
**arXiv target:** cs.SD (Sound), cross-list eess.AS  
**Supersedes:** v3.1.0, v3.0.0, v2.1.1, v2.0.x

*IP Notice: This paper describes MD Audio VAE Tiled, released under GPL v3, and the
VAE Regrind / Layer Regrind training program used to bake artifact suppression into
model weights. All mathematics and implementation details in this paper are clean-room
and intended for public distribution. © 2026 Alexander Allan / A&E Concepts. All rights
reserved.*

---

## Abstract

Modern diffusion-based audio generation models based on ACE-Step produce audio by decoding a sequence of latent frames through a causal VAE (Variational Autoencoder) called Oobleck. The Oobleck decoder is architecturally causal: it cannot process arbitrarily long inputs in a single pass without exceeding GPU memory limits around 30 seconds, and its recurrent structure means it requires a warm-up context before producing valid output. Naive chunked decoding without overlap and warm-up produces audible artifacts at every chunk boundary.

MD Audio VAE Tiled solves this problem with a multi-layer approach: Overlap-Add (OLA) tiling with BPM-synced boundaries, dual-pass staggered decoding for seam phase cancellation, harmonic preservation crossfade at blend zones, context-aware seam repair, and a post-decode spectral cleanup chain specifically calibrated to the Oobleck VAE artifact profile. The system supports audio of unlimited duration without audible seam artifacts and includes a regrinded model stack (VAE Regrind V7, Layer Regrind V7) that addresses the artifact at weight level rather than post-hoc filtering alone.

This document describes the technical design of each subsystem, the hum artifact characterization that drove design decisions, and the regrind training program for artifact suppression at the model level.

---

## 1. Problem Statement

### 1.1 Oobleck VAE Constraints

The ACE-Step audio pipeline uses the Oobleck VAE, an autoencoder with an upscale factor of 1920 audio samples per latent frame (at 48kHz). A 5-minute track at 48kHz produces approximately 1500 latent frames. The Oobleck decoder is architecturally causal and recurrent, which introduces two constraints:

**Memory constraint:** Decoding 1500 latent frames in a single forward pass requires memory proportional to the full sequence. On consumer GPUs, the practical limit is roughly 150-200 frames before OOM (Out of Memory) errors occur.

**Cold-start constraint:** The Oobleck decoder requires a warm-up period. On a cold start (no prior context), the first 50-100 frames of decoded output are corrupted, the decoder's recurrent state has not stabilized. Naive chunked decoding without a context prefix produces a distinctive click/silence at the start of every chunk.

### 1.2 Native Decoder Artifacts

Beyond chunking artifacts, the Oobleck VAE decoder produces a static harmonic artifact at specific frequencies that is present in all generations regardless of track length or content. This artifact has two components with separate origins (detailed in §4):

- **VAE decoder bias:** Reconstruction bias baked into the Oobleck decoder weights. Static hum present in all generations. Manifests as a harmonic series centered on 654Hz (primary), 211Hz, 1090Hz, and sub fundamentals at 74Hz and 94Hz.
- **DiT body resonance:** Builds over 60+ seconds of generation. Not a VAE artifact, originates in the DiT transformer layers. Hypothesized to arise from ODE manifold shearing in the early denoising steps of the diffusion process (see §4 for full discussion — this is an inferred mechanism, not a confirmed one).

---

## 2. Tiling Architecture

### 2.1 Fixed Tile Schedule

The base tiling mode builds a fixed schedule of `(start, end, overlap)` tuples covering the full latent sequence. Each tile overlaps the previous by `overlap` latent frames. The BPM-sync engine calculates overlap to align with bar boundaries:

```
sec_per_bar = (60 / bpm) * 4
overlap_frames = round(sec_per_bar * target_bars * 5.0)  # 5.0 = ACE-Step latents/sec
```

If the calculated overlap exceeds `tile_size / 2`, the bar count is halved iteratively until it fits. Tile size and overlap are further snapped to integer multiples of frames-per-bar, ensuring seam positions align with the musical grid.

### 2.2 Float Anchor Placement

A critical implementation detail: every tile's write position in the output buffer is computed as:

```python
out_start = int(round(start * upscale_factor))
```

Each tile independently anchors to its own latent start position. This produces a maximum positional error of 0.5 samples per seam, completely inaudible, and critically, does not accumulate across tiles.

The alternative, a write-cursor accumulator (`write_cursor += tile_samples - overlap_samples`), was used in v2.0.x and was removed in v2.1.0. The accumulator propagates VAE rounding slop forward across tiles. By tile 9 on a long track, the write cursor can be 20-30 samples off from the correct position, which manifests as snare timing drift and subtle pitch-shifting on pitched elements.

### 2.3 Context Prefix and ctx_skip

Each tile is decoded with a context prefix of `context_prefix` latent frames prepended. This frames the tile decode as if it were part of a longer sequence, allowing the Oobleck recurrent state to stabilize before the region of interest.

After decoding, the output is trimmed:

```python
ctx_skip = actual_decoded_len - expected_write_samples
decoded = decoded[..., ctx_skip:]
```

The skip is anchored to the actual decoded length, not an estimated value. This absorbs any integer rounding in the VAE's upscaling into the discarded context region rather than into the output samples.

### 2.4 Adaptive Entropy-Driven Tiling

Optional mode. A Shannon entropy scan passes over the latent in `CONST_ATS_ENTROPY_SCAN_WIN`-frame windows, computing entropy on the absolute value distribution. High-entropy regions (dense content, transients) get smaller tiles and more overlap for higher reconstruction accuracy. Low-entropy regions (sparse content, silences) get larger tiles for GPU efficiency.

Entropy is measured on the latent distribution directly: `H = -sum(p * log2(p))` where `p` is the normalized absolute value distribution of the latent chunk.

The adaptive scheduler respects the `latent_offset` parameter, allowing Pass B to scan starting at `tile_size // 2` rather than 0, so its entropy map reflects actual content at its starting position.

---

## 3. Seam Repair Subsystems

### 3.1 Dual-Pass Staggered Decode

Standard single-pass tiling places seams at a fixed grid. Dual-pass staggers a second decode pass by `tile_size // 2` latent frames, so Pass B's seams land in the center of Pass A's clean (non-seam) regions, and vice versa.

Both passes run through `_execute_single_pass` independently, producing two full-length audio buffers. The merge uses trapezoidal weight maps:

```
weight[i] = trapezoidal ramp, flat 1.0 at center, ramps to 0.5 at tile edges
merged = audio_a * (weight_a / total) + audio_b * (weight_b / total)
```

Pass A seam positions become weight-valley positions, regions where Pass B's weight is maximal. The result is that neither pass dominates at its own seam positions, and both seams cancel against each other's clean centers.

### 3.2 Phase-Coherent Triple-Pass (PCT)

Three decode passes at offsets 0, `tile_size//3`, and `2*tile_size//3`. Before the trapezoidal merge, cross-correlation phase alignment is applied: for each Pass A seam boundary, a 50ms analysis window is extracted and used as the reference. Passes B and C are shifted via integer lag to maximize cross-correlation with Pass A at each seam.

Phase alignment uses sub-sample precision via parabolic interpolation around the xcorr peak. Maximum search lag is configurable (default 2ms).

PCT is disabled by default (`PCT_TRIPLE_PASS = False`) due to the 3x decode time cost.

### 3.3 Context-Aware Seam Repair (CASR)

CASR replaced the old SLERP bridge micro-decode in v2.1.0. The original approach decoded 8-frame micro-windows at each seam boundary and spliced the result. This failed with Oobleck because the decoder requires at least 50-100 frame warm-up, an 8-frame cold decode produces silence or garbage on frames 0-6 and a click on frame 7 when the state partially stabilizes.

CASR instead decodes a full-context window centered on each seam boundary, the same window size used in the main decode loop. It extracts only the seam-adjacent patch (frames within `CONST_CASR_PATCH_FRAMES` of center), then applies a symmetric cosine bell:

```
bell = [fade_up | fade_down]  # 0 at both edges, 1 at center
patched = existing * (1 - bell) + patch * bell
```

The cosine bell fades in AND out, so the patch blends smoothly with both the pre-seam and post-seam audio without introducing hard cuts at either edge.

### 3.4 Harmonic Preservation Crossfade (HPC)

Standard time-domain OLA blending (fade one chunk out while fading the next in) causes tonal swallowing at tile boundaries. When two pitched signals with the same fundamental but different phases are linearly crossfaded, the magnitudes partially cancel, producing an audible dip in tonal content at the blend zone.

HPC addresses this in the STFT domain. At each blend zone, both the outgoing tail and the incoming head are STFT-analyzed:

```
mag_blend = mag_prev * f_out + mag_new * f_in   # blend magnitudes via fade window
phase_select = where(mag_prev > mag_new, phase_prev, phase_new)  # take phase from louder source
S_blend = mag_blend * exp(j * phase_select)
```

Phase is taken from the louder source rather than blended, which avoids the magnitude cancellation that occurs when blending phases that are not aligned.

A frequency-dependent weight further blends the spectral output with a time-domain fallback: full spectral blend below 1kHz (where magnitude cancellation is most perceptible), tapering to 30% spectral at Nyquist. This prevents excessive presence in the 5-7kHz range that full spectral blending introduces.

### 3.5 Spectral Continuity Enforcement (SCE)

SCE operates on the fully assembled (post-OLA, post-CASR) audio. For each tile boundary position, it performs a batched STFT of the full audio and examines the magnitude spectrum at STFT frames adjacent to each seam position.

For frequency bins below `SCE_MAX_FREQ_HZ` (8kHz), if the magnitude ratio between the pre-seam and post-seam frame exceeds `SCE_THRESHOLD_DB` (6dB), the bin is flagged as discontinuous. Flagged bins are interpolated across a repair window of `SCE_REPAIR_FRAMES` STFT frames on each side of the seam, using linear interpolation of both magnitude and phase (with phase unwrapping).

SCE must run on Wiener-denoised audio. If run on raw VAE output, it detects the noise floor artifact as a "harmonic trajectory" and interpolates it into a coherent ring artifact. Denoising first removes the artifact at source so SCE only processes genuine seam discontinuities.

---

## 4. Hum Artifact Characterization

### 4.1 Measurement Methodology

The Oobleck hum artifact was characterized via direct measurement in Ableton Live EQ Eight on ACE-Step XL renders with the regrinded base model. The EQ Eight parametric bands were manually tuned to null each component of the artifact, giving frequency, depth, and Q values.

### 4.2 Harmonic Series

The artifact manifests as a harmonic series:

| Frequency | Measured Cut | Artifact Role |
|---|---|---|
| 74.4 Hz | -1.43 dB, Q=6.27 | Sub fundamental, BPM-dependent |
| 94.0 Hz | -1.86 dB, Q=7.08 | Upper sub, BPM-dependent |
| 211 Hz | ~-7 dB | 3rd harmonic of fundamental |
| 654–669 Hz | -12 to -15 dB | Primary hum, most perceptible |
| 1090 Hz | ~-7 dB | 5th harmonic |

The primary hum at ~654Hz exhibits a BPM-dependent frequency shift of approximately 2Hz per doubling of BPM. At 88 BPM the fundamental sits at ~75Hz; at 172 BPM it shifts to ~77Hz. This tempo-locking behavior implicates the decoder's temporal processing in the artifact generation mechanism.

An additional observation: air reduction of -17.02% to -17.03% was measured consistently across both ACE-Step 2.6B and XL models under observer analysis. This figure is identical across model sizes, confirming it is a VAE decoder characteristic rather than a model-size effect.

### 4.3 Dual Origin

The hum has two mechanistically distinct components:

**VAE decoder bias:** Static reconstruction bias baked into the Oobleck decoder weights during its original training. Present in all generations at consistent amplitude, independent of generation length. Addressable at the decoder weight level via VAE regrind.

**DiT body resonance:** Accumulates over the denoising trajectory, becoming perceptible after approximately 60 seconds of generation. Hypothesized to arise from ODE manifold shearing in the early denoising steps where the model traverses frequency-space manifolds that create resonant modes in the FFN activations of decoder layers. Addressable via LoRA regrind of the DiT FFN layers.

Both components must be targeted independently for full suppression. The VAE-side component is addressed by VAE Regrind + LSS + hum notch. The DiT-side component is addressed by Layer Regrind.

---

## 5. Latent Spectral Suppressor (LSS)

### 5.1 Mechanism

The Oobleck encoder produces a 64-channel latent representation. The channels are not equally informative, some carry genuine content variation, others carry persistent low-variance bias with mean offsets (DC components) that the decoder reconstructs as static spectral artifacts.

LSS operates on the latent before decode, computing per-channel variance across time:

```python
ch_var = latents.var(dim=-1).mean(dim=0)  # [C] variance per channel, averaged over batch
var_norm = (ch_var - var_min) / (var_max - var_min)  # normalize to [0, 1]
```

Channels with normalized variance below `var_threshold` receive a suppression gain:

```python
smooth = (1 - strength) + strength * (var_norm / var_threshold)
gain[c] = smooth if var_norm[c] < var_threshold else 1.0
```

At `strength=0.25` and `var_threshold=0.12`, low-variance channels are suppressed to at minimum 75% of their original amplitude, with linear recovery toward full amplitude as variance approaches the threshold. High-variance channels are untouched.

### 5.2 DC Removal, Permanently Disabled

The LSS includes a DC removal option (`lss_dc_remove`) that subtracts the per-channel temporal mean before suppression. This was found to cause metallic distortion, the DC offset removal disrupts temporal dependencies in the latent that the decoder relies on for frequency stability. `lss_dc_remove` is permanently disabled. The gain-based suppression alone is sufficient and produces no audible distortion at the gold standard settings.

---

## 6. Post-Decode DSP Chain

### 6.1 Pipeline Order

The post-decode DSP chain runs in this fixed order after OLA synthesis is complete:

```
CASR → Hum Notch (biquad chain) → Wiener Denoiser → SCE → HPF → Stereo Width → Soft Clip
```

The critical constraint is Wiener before SCE. SCE analyzes STFT magnitude trajectories to detect seam discontinuities. If the Wiener denoiser has not run first, the VAE noise floor artifact presents as structured spectral energy that SCE interprets as a harmonic trajectory and "repairs" into a coherent ring. Denoising first ensures SCE sees only genuine seam artifacts.

PLGR (loudness gain riding) runs per-tile inside the decode loop, before OLA write, not as a post-process stage.

### 6.2 Wiener Spectral Denoiser

The denoiser estimates the noise floor from the quietest `floor_pct` (5%) of STFT frames:

```
noise_est[f] = mean(mag_sq[f, quietest_frames])
gain[f, t] = signal_est[f, t] / (signal_est[f, t] + noise_est[f])
  where signal_est = (mag_sq - noise_est).clamp(min=0)
```

The gain mask is smoothed along the frequency axis via `conv1d` with a kernel width corresponding to `smooth_hz=80Hz` in frequency resolution. This prevents per-bin gain variation that creates "musical noise", the buzzing artifact that results from unstable per-bin modulation.

`min_gain=0.15` ensures no frequency bin is ever fully silenced. Full suppression of low-gain bins creates musical noise in sparse audio regions. The 0.15 floor leaves a trace of the noise floor in the output, which is perceptually preferable to the buzzing alternative.

`strength=0.85` means the gain mask is applied at 85% and 15% passthrough is maintained. The formula is `gain = gain * strength + (1 - strength)`, so even a fully-suppressed bin passes at 15%.

### 6.3 Biquad Filter Implementation

All biquad filters (hum notch, HPF) use the normalized direct form II transposed structure. The coefficient computation follows the Audio EQ Cookbook formulas. The filter dispatch chain tries scipy first (vectorized CPU, orders of magnitude faster than Python loops), falls back to torchaudio, falls back to Python loop:

```python
try:
    from scipy.signal import lfilter
    # vectorized, fast
except ImportError:
    try:
        import torchaudio.functional as TAF
        # GPU-accelerated path
    except:
        # Python loop, correct but slow
```

Install scipy for production use.

### 6.4 PLGR, Psychoacoustic Loudness-Aware Gain Riding

Standard RMS gain riding corrects inter-tile level differences by chasing the previous tile's RMS energy. This works for spectrally consistent content but fails when bass-heavy tiles alternate with sparse tiles, a kick-heavy tile reads as loud relative to a breakdown tile and over-amplifies it, causing pumping.

PLGR uses ITU-R BS.1770-4 K-weighted momentary loudness (LUFS) instead of RMS. The K-weighting filters (pre-filter high-shelf + HPF stage) approximate the equal-loudness contours of human hearing, de-emphasizing bass energy relative to mid-high energy. This makes the loudness measurement perceptually accurate, a bass-heavy tile no longer reads as disproportionately loud.

PLGR is downward-only: it attenuates tiles that are louder than the previous tile, but never amplifies quiet tiles. Maximum correction is -1.5dB per tile. This prevents quiet breakdown sections from being boosted, preserving genuine dynamics while correcting tile-to-tile level inconsistency.

---

## 7. VAE Regrind Program

### 7.1 V7, Deployed

VAE Regrind V7 targets the Oobleck decoder weights directly. The regrind fine-tunes the decoder on a reconstruction loss that includes a hum-penalty component `L_hum` targeting the measured artifact frequencies.

**Results:**
- Validation loss: 0.0506 (baseline 0.0593), -14.7% improvement
- L_hum at convergence: 0.0000
- VRAM: 2.44GB (fits on most consumer cards alongside inference)
- Converged at epoch 15 (mean-based variance stopping criterion, threshold < 0.0001, minimum 15 epochs)

The convergence-detection logic in V7 was corrected from V6. V6 used a last-value-based variance check with threshold 0.001, which triggered prematurely at epoch 10. V7 uses mean-based variance with threshold 0.0001 and a minimum of 15 epochs.

**Public release:** `acestep_1.5_vae_Regrind_V7.safetensors` on HuggingFace (`mdmachine/ACEStep-XL-Regrind-V1`).

### 7.2 V8, Pre-Registered

V8 targets stronger hum suppression through architectural improvements to the training setup:

**Adversarial hum injection:** Synthetic artifact frequencies injected at -25dB during training (30% probability). Forces the decoder to actively learn hum suppression rather than passively avoid hum-generating activations.

**Frequency-sweep validation set:** 75 pure tones at 60-800Hz in 10Hz steps, 30 seconds each. This provides an honest benchmark that cannot hide behind mix reconstruction quality, a model that suppresses hum only in complex content will fail on pure tones.

**90-second training clips:** V7 used 8-second clips, which are too short to observe the accumulation behavior of the DiT body resonance. 90-second clips expose the temporal build-up within the training window.

**DJ set dataset (Omni_Manifold_DJSet_V1):** 3076 hours, average 14-minute tracks, bass-forward electronic music. Both provides long sequences and ensures the bass-heavy content that excites the artifact harmonics is well-represented.

**Tightened convergence criterion:** threshold 0.00005 (was 0.0001), minimum epoch 20 (was 15).

Target GPU: RTX 5060 Ti (GPU1), 2.4GB footprint, simultaneous with Layer Regrind V8 on GPU0.

---

## 8. Layer Regrind Program

### 8.1 V7, Deployed and Baked

Layer Regrind V7 targets the DiT-side resonance accumulation by fine-tuning the FFN (feed-forward network) layers of the ACE-Step XL transformer with a LoRA adapter.

**Training:** Layers 24-31 FFN-only LoRA. Ran to a fixed 80-epoch ceiling (min_loss=0.4492), the convergence criterion never triggered due to an oscillation bug in the internal stability metric used for early stopping (values alternated between two degenerate states, neither of which registered as stable).

**Baked base:** The LoRA adapter was baked into xl-turbo-official via the formula:
```
W_fused = W_base + (alpha/r) * (lora_up @ lora_down) * strength
```

A key-mapping fix was required: xl-turbo-comfyui uses bare `decoder.layers.N.mlp.X.weight` keys without the `diffusion_model.` prefix, while LoRA keys carry the prefix. The bake script strips the prefix on match.

Output: `acestep_xl_Regrind_V1.safetensors` (~8.6GB full base model).

**Perceptual result:** Resonance reduced but not eliminated. V7 is a baseline, not a ceiling. V8 required for stronger suppression.

### 8.2 V8, Pre-Registered

**Extended layer window:** Layers 20-31 (12 layers, ~2.9% of parameters vs V7's 8 layers at 1.9%).

**L_res temporal accumulation loss:** New loss term penalizing positive spectral energy slope at artifact bands over time. V7 only penalized static hum amplitude; V8 penalizes the temporal growth of resonance that produces the 60-second accumulation effect.

**Stability-metric fix:** The degenerate zone includes both a near-zero value and values approaching the metric's upper bound, which oscillate back to near-zero on the next step. V8 patches the stopping criterion to treat both degenerate states as non-converged.

Target GPU: RTX 4070 Ti Super (GPU0), simultaneous with VAE Regrind V8.

---

## 9. Observer Measurement Findings

### 9.1 Model Size Independence

Air reduction of -17.02% to -17.03% was measured identically across ACE-Step 2.6B and XL models. This confirms the artifact is a VAE decoder characteristic, not dependent on DiT model size. The VAE decoder weights are the same across model variants.

---

## 10. Deployment Architecture

### 10.1 ComfyUI Target

The Python implementation consists of two files: the core math engine (`md_audio_tiled_core.py`) and the ComfyUI node wrapper (`MD_Audio_VAE_Tiled.py`). The wrapper loads the core via a Universal Binary Loader that tries a compiled binary (`md_audio_tiled_core_bin`) first, falls back to source. All runtime parameters are passed via a YAML string input, with a complete set of defaults in `_VAE_YAML_DEFAULTS`. The wrapper outputs the active parameter set as a `yaml_applied` string for reproducibility.

The API client path (`AudioTiledApiClient`) dispatches decoding to a remote server. Serialization uses `torch.save` to `BytesIO` with base64+zlib encoding. The `pickle` module is not used, this was a security fix in v2.1.1 that eliminated a remote code execution surface on API response deserialization.

### 10.2 HOT-Step-CPP Target

The Lua implementation (`md_audio_tiled_core.lua`) provides the host-side tiling arithmetic as a pure-Lua library with no external dependencies. The plugin adapter (`md_audio_tiled_V2.lua`) wraps the core to conform to the HOT-Step postprocess plugin contract:

```lua
function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
    local audio = core.execute_tiled_decode(vae_decode_fn, latents, ...)
    return audio, final_samples
end
```

The `vae_decode_fn` callback is provided by the engine and handles the actual neural network inference call in C++.

The C++ header (`md_audio_tiled_core.hpp`) provides the same functionality as the Lua core for direct C++ integration, using `std::function` for the VAE callback.

**STFT-domain features (HPC, SCE, Wiener, CASR) are Python-only.** The Lua and C++ targets implement: tiling, OLA, LSS, dual-pass merge, RMS leveling, hum notch, HPF, stereo width, soft clip. The Python target adds all five STFT-domain subsystems.

### 10.3 Algorithmic Parity

All three targets implement the same algorithms for the non-STFT subsystems. Key parity points verified across all targets:

- Float anchor placement: identical floating-point math
- ctx_skip: anchored to `actual_len - expected_write_samples`
- LSS: identical variance computation and suppression gain formula
- Biquad coefficient computation: identical for peaking EQ and low shelf
- Trapezoidal weight maps: identical edge taper fraction (0.25) and ramp shape
- RMS absolute ceiling: 0.35, downward-only, every tile
- `lss_dc_remove = false`: enforced in all targets with warning comments

---

## 11. Known Limitations and Pre-Registered Fixes

| Issue | Status |
|---|---|
| DiT body resonance 60s+ | PARTIAL FIX, V8 layer regrind with L_res loss and 90s training clips pending |
| VAE decoder static hum | PARTIAL FIX, V8 VAE regrind with adversarial injection pending |
| Stability-metric streak oscillation | PRE-REGISTERED, V8 patches the stopping criterion to handle both degenerate oscillation states |
| STFT ops in Lua/C++ | BY DESIGN, no native STFT library in target environment |
| LSP bridge injection | DISABLED, Oobleck causal sensitivity to latent direction changes |

---

## Appendix A, Hum Notch Biquad Parameters (Deployed)

```
Bass shelf:   120Hz, -2.0dB, slope=0.7
74Hz cut:     74.4Hz, -1.43dB, Q=6.27
94Hz cut:     94.0Hz, -1.86dB, Q=7.08
654Hz cut:    654.0Hz, -15.0dB, Q=6.00  ← PRIMARY HUM
```

Boost bands (disabled by default, preserved for experimentation):
```
353Hz boost:  353.0Hz, +2.57dB, Q=1.86
1090Hz boost: 1090.0Hz, +4.71dB, Q=2.47
2060Hz boost: 2060.0Hz, +1.71dB, Q=0.71
3520Hz boost: 3520.0Hz, +3.43dB, Q=1.72
```

---

## Appendix B, Core Constants

```python
CONST_EPSILON              = 1e-8
CONST_GAIN_CLAMP_BASE      = 0.05    # ±5% base RMS ride range
CONST_GAIN_CLAMP_MAX_RANGE = 0.20    # ±20% max on high-entropy seams
CONST_RMS_ABSOLUTE_CEIL    = 0.35    # Hard per-tile RMS ceiling
CONST_VAE_CONTEXT_FRAMES   = 128     # Oobleck causal warm-up prefix
CONST_CASR_BLEND_FRAMES    = 32      # Latent frames decoded around each seam
CONST_CASR_PATCH_FRAMES    = 8       # Latent frames patched on each side
CONST_DUAL_PASS_EDGE_TAPER = 0.25    # Trapezoidal ramp fraction
PLGR_TARGET_LUFS           = -18.0   # Target momentary loudness
SPECTRAL_DENOISE_STRENGTH  = 0.85
SPECTRAL_DENOISE_MIN_GAIN  = 0.15
LSS_STRENGTH               = 0.25    # Gold standard
LSS_VARIANCE_THRESHOLD     = 0.12    # Gold standard
```

## References

1. Evans, Z., Carr, C. J., Taylor, J., Hawley, S. H., & Pons, J. (2024). Stable Audio Open. *arXiv:2407.14358*.

2. ACE-Step: A Step Towards Music Generation Foundation Models. ByteDance Research, 2025.

3. International Telecommunication Union. (2015). Algorithms to measure audio programme loudness and true-peak audio level. *Recommendation ITU-R BS.1770-4*.

4. Allen, J. B. (1977). Short term spectral analysis, synthesis, and modification by discrete Fourier transform. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 25(3), 235-238.

5. Scalart, P., & Filho, J. V. (1996). Speech enhancement based on a priori signal to noise estimation. *IEEE ICASSP 1996*.

6. Allan, A. (MDMAchine). (2026). STORM: Stabilized Taylor Oscillation with Runge-Kutta Memory, An Adaptive Stiffness-Switching ODE Sampler for Flow-Matching Diffusion Models. *A&E Concepts preprint*.

---

*© 2026 Alexander Allan (MDMAchine) · A&E Concepts*  
*GPL v3 · All Rights Reserved*  
*Version 3.2, July 2026*
