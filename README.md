![MD Audio VAE Tiled](assets/header_animated.svg)

# MD Audio VAE Tiled Decoder
### Enterprise OLA Synthesis Engine for Infinite-Duration Audio
**Seamless tiled VAE decoding for ACE-Step / Oobleck VAE, BPM-synced, seam-free, OOM-safe**

*Alexander Allan (MDMAchine) · A&E Concepts · GPL v3*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Version](https://img.shields.io/badge/Version-3.0.1-orange.svg)]()

---

## What Is MD Audio VAE Tiled?

MD Audio VAE Tiled is a full-stack tiled VAE decoder for infinite-duration audio generation with ACE-Step / Oobleck VAE. It solves a specific hardware problem: the Oobleck VAE is causal and recurrent, it cannot decode long audio in a single forward pass without exhausting GPU memory, and it requires a causal warm-up context before producing valid output. Naive chunked decoding without overlap and warm-up produces a loud click artifact at every chunk boundary.

MD Audio VAE Tiled solves this with multi-layer seam elimination: BPM-synced Overlap-Add synthesis, dual-pass staggered decoding, STFT-domain harmonic preservation crossfade, and context-aware seam repair, plus a post-decode spectral cleanup chain specifically calibrated to the Oobleck VAE artifact profile. The result is seamless audio of any duration with no audible tile boundaries, full spectral continuity across seams, and a fully characterized hum suppression stack.

**Validated on:** ACE-Step XL Turbo 4B, ACE-Step 2.6B, Oobleck VAE (all variants)
**Deployment:** ComfyUI node · C++ header-only · Lua plugin (HOT-Step-CPP)
**License:** GPL v3 · Commercial dual-license: contact A&E Concepts

---

## Architecture

```
Latent [B, C_lat, W]
  +-> LSS (suppress low-variance bias channels, once on full latent)
        +-> LSP (generate SLERP bridge latents at seam positions)
              +-> Decode dispatch:
                    Per-tile: context prefix warm-up -> vae.decode
                              -> PLGR K-weighted gain ride (per tile)
                              -> Absolute RMS ceiling (0.35)
                              -> OLA write with HPC spectral crossfade
                    Dual-pass: Pass A (offset 0) + Pass B (offset tile/2)
                               -> Trapezoidal weight merge
                    Triple-pass: Pass A/B/C + xcorr phase align -> merge
                         +-> CASR (full-context seam patch, cosine bell blend)
                               +-> Hum notch (bass shelf + 74/94/654Hz)
                                     +-> Wiener spectral denoiser
                                           +-> SCE (harmonic trajectory repair)
                                                 +-> HPF (20Hz Butterworth)
                                                       +-> Stereo width (M/S)
                                                             +-> Soft clip (tanh)
                                                                   +-> Output
```

### Dual-Pass Staggered Decode

Two full decode passes at offset phase: Pass A tiles from latent frame 0, Pass B tiles from latent frame `tile_size // 2`. Pass A seams land in the center of Pass B's clean regions, and vice versa. Both passes are merged with trapezoidal weight maps, each pass's weight ramps to zero at its own seam positions, where the other pass is at maximum weight. Tile start positions use float anchor placement (`out_start = int(round(start * upscale_factor))`), not a write-cursor accumulator, which propagates VAE rounding slop and causes timing drift on long tracks.

### Context-Aware Seam Repair (CASR)

Replaced the old SLERP bridge micro-decode in v2.1.0. Oobleck is causal, an 8-frame cold decode outputs silence or garbage for frames 0-6, then a click on frame 7 as the recurrent state partially stabilizes. CASR instead decodes a full-context window centred on each seam (same warm-up as the main decode loop), extracts only the seam-adjacent patch, and applies a symmetric cosine bell that fades in AND out, no hard cuts on either edge.

### Harmonic Preservation Crossfade (HPC)

Standard time-domain OLA causes tonal swallowing at tile boundaries, two pitched signals at different phases partially cancel during the linear crossfade, producing a magnitude dip in tonal content. HPC addresses this in the STFT domain: magnitudes are blended via the crossfade window, but phase is taken from whichever source has higher magnitude. This prevents cancellation while preserving the correct phase trajectory. A frequency-dependent weight blends toward time-domain at high frequencies to prevent presence artifacts in the 5-7kHz range.

### Latent Spectral Suppressor (LSS)

The Oobleck encoder produces 64 latent channels of unequal variance. Low-variance channels carry persistent bias offsets rather than content, the decoder reconstructs these as static spectral artifacts. LSS computes per-channel variance across time, identifies channels below `var_threshold`, and applies a proportional gain suppression before any tiling occurs. Applied once to the full latent so all tiles benefit equally. `lss_dc_remove` is permanently disabled, subtracting per-channel mean disrupts temporal dependencies the decoder relies on for frequency stability.

### PLGR (Psychoacoustic Loudness-Aware Gain Riding)

Standard RMS gain riding fails on spectrally imbalanced material, a bass-heavy tile reads as disproportionately loud relative to a sparse tile, causing pumping. PLGR measures ITU-R BS.1770-4 K-weighted momentary LUFS instead of RMS, which de-emphasizes bass energy in proportion to human equal-loudness curves. Riding is downward-only at max -1.5dB/tile, quiet breakdown sections are never boosted, preserving genuine dynamics while correcting seam-level loudness inconsistency. Runs per-tile inside the decode loop before OLA write.

### Wiener Spectral Denoiser + SCE

The Wiener denoiser estimates per-bin noise floor from the quietest 5% of STFT frames and applies a gain mask with 80Hz frequency smoothing and a 0.15 minimum gain floor (prevents musical noise in sparse regions). It runs before SCE, this ordering is non-negotiable. SCE detects magnitude discontinuities at seam boundaries and interpolates harmonic trajectories to restore continuity. If SCE runs on un-denoised audio, it interprets the VAE noise floor artifact as a harmonic trajectory and interpolates it into a coherent ring artifact.

---

## Validated Parameters

```yaml
# Gold Standard (2026-05-14)
overlap_mode: "BPM-Synced (Max 8 Bars)"
tile_size: 1024
context_prefix: 512
blending_mode: "Hann"
dual_pass: true
spectral_denoise: true
denoise_strength: 1.00
highpass_hz: 20
soft_clip_db: -3.0
stereo_width: 0.8
plgr_enabled: true
hpc_enabled: true
lsp_enabled: true
lss_enabled: true
lss_strength: 0.25
lss_variance_threshold: 0.12
lss_dc_remove: false          # NEVER true, metallic distortion
sce_enabled: true
sce_threshold_db: 6.0
```

### Hum Artifact Series (Measured via Ableton EQ Eight)

The Oobleck VAE decoder produces a characteristic harmonic artifact present in all ACE-Step generations. The series has two origins requiring separate treatment: VAE decoder reconstruction bias (static, addressed by VAE Regrind + LSS + hum notch) and DiT body resonance (accumulates over 60s+, addressed by Layer Regrind).

| Frequency | Depth | Character |
|---|---|---|
| 74.4 Hz | -1.43 dB | Sub fundamental, BPM-dependent |
| 94.0 Hz | -1.86 dB | Upper sub resonance, BPM-dependent |
| 211 Hz | ~-7 dB | 3rd harmonic |
| 654-669 Hz | -12 to -15 dB | **Primary hum**, BPM shift ~2Hz/octave |
| 1090 Hz | ~-7 dB | 5th harmonic |

---

## Deployment

### ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MDMAchine/MD-Audio-VAE-Tiled
pip install -r MD-Audio-VAE-Tiled/requirements.txt
```

The node appears as **MD: Audio VAE Decode (Tiled) v3** under `MD_Nodes/Audio`.

**Minimal workflow:**
1. Load base model: `acestep_xl_Regrind_V1.safetensors` (DiT loader)
2. Load VAE: `acestep_1.5_vae_Regrind_V7.safetensors` (VAE loader)
3. Run sampler, connect latents to MD Audio VAE Tiled
4. Set `bpm` to your track's BPM
5. Connect `audio` output to audio save/preview node

All decode parameters are configured via the `yaml_settings_str` input. Connect the node's `yaml_template` output to a text preview to see the full parameter reference.

### Recommended Model Stack

For best results, use all three components from [ACEStep-XL-Regrind-V1](https://huggingface.co/mdmachine/ACEStep-XL-Regrind-V1):

| Component | File | Where to load |
|---|---|---|
| Base model | `acestep_xl_Regrind_V1.safetensors` | DiT loader |
| VAE decoder | `acestep_1.5_vae_Regrind_V7.safetensors` | VAE loader |
| LoRA (optional) | `acestep_xl_LayerRegrind_V7.safetensors` | LoRA loader, 0.25 to 0.50 |

LoRA on raw xl-turbo: 0.50 to 1.00. No trigger words. License: CC BY-NC-SA 4.0.

### C++ (acestep.cpp / HOT-Step-CPP)

`core/md_audio_tiled_core.hpp` is a self-contained header-only implementation. Drop into your project and include:

```cpp
#include "md_audio_tiled_core.hpp"

md_tiled::Config cfg;
cfg.tile_size = 1024;
cfg.overlap = 64;
cfg.lss_strength = 0.25f;
cfg.dual_pass = true;
// ... configure as needed

md_tiled::run(latents, W, vae_decode_fn, userdata, cfg);
```

### Lua ([HOT-Step-CPP](https://github.com/scragnog/HOT-Step-CPP))

Copy `core/md_audio_tiled_core.lua` and your choice of plugin adapter to `engine/lua/plugins/`:
- `hotstep/md_audio_tiled.lua`: V1, minimal UI (6 params)
- `hotstep/md_audio_tiled_V2.lua`: V2, extended UI (15+ params)

```bash
cd engine && cmake --build build --config Release
```

The plugin appears as **MD Audio Tiled Decoder V2** in the postprocess plugin list.

> **Parity note:** Lua and C++ targets implement tiling, OLA, LSS, dual-pass merge, hum notch, HPF, stereo width, and soft clip. STFT-domain operations (HPC, SCE, Wiener, CASR) require PyTorch and are Python-only.

---

## Node Outputs

| Output | Type | Description |
|---|---|---|
| `audio` | AUDIO | Decoded waveform |
| `debug_plot` | IMAGE | Waveform + tile boundary visualization (Pass A red, Pass B amber) |
| `analytics_plot` | IMAGE | RMS envelope + active feature summary panel |
| `console_log` | STRING | Per-tile decode log |
| `yaml_applied` | STRING | Active parameters as YAML, paste back to reproduce |
| `yaml_template` | STRING | Full template with all parameters and comments |

---

## Performance Notes

Decode time is dominated by `vae.decode()` calls. Dual-pass = 2x decode, triple-pass = 3x. The v3.0.1 GPU audit vectorized all previously looped DSP:
- Wiener: batched `[BC, L]` STFT, vectorized noise floor via topk+gather
- SCE: single batched STFT, vectorized seam repair across all channels
- HPC: removed `B×C` Python loops, frame weights vectorized via `torch.arange`
- Biquad filters: scipy, torchaudio, Python loop fallback chain

Install scipy. The Python loop fallback is orders of magnitude slower on long tracks.

---

## Files

| File | Description |
|---|---|
| `README.md` | This file |
| `LICENSE` | GPL v3 |
| `__init__.py` | ComfyUI node registration |
| `pyproject.toml` | Version, deps, metadata |
| `requirements.txt` | pip fallback |
| `.gitignore` | Ignore rules |
| `assets/header_animated.svg` | Animated SVG header |
| `core/__init__.py` | Package init |
| `core/md_audio_tiled_core.py` | Core math engine (pure PyTorch/CUDA) |
| `core/md_audio_tiled_core.lua` | Lua port (tiling + DSP, no STFT ops) |
| `core/md_audio_tiled_core.hpp` | C++ header-only (same scope as Lua) |
| `core/md_audio_tiled_core_original.lua` | Legacy reference |
| `vae/__init__.py` | Package init |
| `vae/MD_Audio_VAE_Tiled.py` | ComfyUI node wrapper, YAML config, BPM engine |
| `hotstep/md_audio_tiled.lua` | V1 plugin adapter (minimal UI) |
| `hotstep/md_audio_tiled_V2.lua` | V2 plugin adapter (extended params UI) |
| `docs/MD_Audio_VAE_Tiled_White_Paper_v3_2.md` | Technical white paper |

---

## Acknowledgments

- **[scragnog / Captain HOT-Step](https://github.com/scragnog/HOT-Step-CPP)**, HOT-Step-CPP maintainer. HOT-Step integration support and early testing.
- **ACE-Step community**, seam artifact reporting and perceptual validation across hardware configs.

---

## License

**GPL v3**, free for open-source use. See [`LICENSE`](LICENSE).

Commercial closed-source integration requires a dual-license commercial exemption. Contact: A&E Concepts on GitHub.

---

## White Paper

Full technical treatment of the tiling architecture, seam repair subsystems, hum artifact characterization, and regrind program:

[`docs/MD_Audio_VAE_Tiled_White_Paper_v3_2.md`](docs/MD_Audio_VAE_Tiled_White_Paper_v3_2.md)

---

*© 2026 Alexander Allan (MDMAchine) · A&E Concepts*
