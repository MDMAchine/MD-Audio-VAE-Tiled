# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░                 MD Audio VAE Tiled Wrapper – v3.0.1                 ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine / A&E Concepts
# ╠═ AUTHOR: Alexander Allan (MDMAchine)
# ╠═ LICENSE: GNU General Public License v3.0 (GPLv3)
# ╠═ DESCRIPTION:
# ║  Enterprise-grade Audio VAE Tiled Decoder for infinite-duration audio.
# ║  Overlap-Add (OLA) synthesis with multi-pass seam repair and spectral crossfade.
# ║
# ╠═ CHANGELOG v2.1.1:
# ║  [SECURITY] API client: Replaced pickle.loads on server response with
# ║             JSON + base64-encoded torch tensor (torch.save/load via BytesIO).
# ║             Eliminates RCE attack surface on API response deserialization.
# ║             Outbound payload likewise migrated from pickle.dumps to the same
# ║             torch.save + JSON envelope — `pickle` import removed entirely.
# ║             Matches fix already applied in noise_decay_core_V02.py v1.7.0.
# ╠═ CHANGELOG v2.0.0:
# ║  + Dual-Pass Staggered Decode toggle (dual_pass BOOLEAN)
# ║    Two phase-offset decode passes merged via trapezoidal weighting.
# ║    Pass A seams land in Pass B's clean center → seam artifacts cancel.
# ║  + SLERP Latent Bridge toggle (slerp_bridge BOOLEAN)
# ║    Spherical linear interpolation across tile boundaries in latent space.
# ║    Decodes manifold-smooth bridge audio, surgically splices seam region.
# ║  + Adaptive Tile Sizing toggle (adaptive_tiling BOOLEAN)
# ║    Entropy-driven dynamic tile lengths. High-entropy regions → smaller tiles
# ║    + more overlap. Low-entropy regions → larger tiles for GPU efficiency.
# ║  + Adaptive RMS Gain Clamping: clamp range widens on entropy-delta spikes.
# ║  + Enhanced debug plot: dual-pass B boundaries shown in distinct color.
# ║  + core health report surfaced to console at debug_level >= 1.
# ║  + BPM snap now enforces integer-multiple alignment post-calculation.
# ║  + OOM auto-reduction loop retained; tile_schedule rebuilt on each retry.
# ╠═ CORE FEATURES:
# ║  ✅ Dual-Pass Staggered Decode (seam phase-cancellation)
# ║  ✅ SLERP Latent Bridge (manifold seam repair)
# ║  ✅ Adaptive Tile Sizing (entropy-driven chunking)
# ║  ✅ BPM-Synced Overlap (1/2/4/8/16/32 Bars, integer-multiple snapped)
# ║  ✅ Adaptive AEG RMS Leveling (±5%→±20% dynamic clamp)
# ║  ✅ Hann/Cosine/Linear Blending Curves
# ║  ✅ Performance Profiling & Analytics
# ║  ✅ Triple-Branch Execution Safety (Local Binary → Source → API)
# ╚════════════════════════════════════════════════════════════════════════════

import os
import sys
import torch
import logging
import math
import time
import io
import json
import zlib
import base64
import requests
import numpy as np
from PIL import Image
import comfy.model_management
import folder_paths
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# =================================================================================
# == Third-Party Dependency Checks
# =================================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print(f"[AudioTiled] ✅ Matplotlib {matplotlib.__version__} OK (Agg).")
except Exception as _mpl_err:
    MATPLOTLIB_AVAILABLE = False
    print(f"[AudioTiled] ❌ Matplotlib FAILED: {type(_mpl_err).__name__}: {_mpl_err}")

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1 UPS)
# =================================================================================

def find_core_paths():
    """
    Intelligently locates the core directory regardless of nesting depth.
    Search Strategy: Direct relatives + Repository root crawl (up to 4 levels).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates  = [
        os.path.join(current_dir, "core"),
        os.path.join(current_dir, "..", "core"),
        os.path.join(current_dir, "..", "..", "core"),
    ]
    pointer = current_dir
    for _ in range(4):
        if os.path.basename(pointer) == "ComfyUI_MD_Nodes":
            candidates.append(os.path.join(pointer, "core"))
            break
        parent = os.path.dirname(pointer)
        if parent == pointer:
            break
        pointer = parent
    return list(dict.fromkeys([os.path.abspath(c) for c in candidates if os.path.exists(c)]))

CORE_LOCATIONS = find_core_paths()
for loc in CORE_LOCATIONS:
    if loc not in sys.path:
        sys.path.insert(0, loc)

CQ_CORE_LOADED = False
CQ_CORE_TYPE   = "NONE"
CQ_CORE_ERROR  = None

try:
    import md_audio_tiled_core_bin as core
    CQ_CORE_LOADED = True
    CQ_CORE_TYPE   = "BINARY"
except ImportError as e1:
    try:
        import md_audio_tiled_core as core
        CQ_CORE_LOADED = True
        CQ_CORE_TYPE   = "SOURCE"
    except ImportError as e2:
        CQ_CORE_ERROR = f"Binary: {e1} | Source: {e2}"

# =================================================================================
# == Global Constants
# =================================================================================

CONST_JS_MAX_SAFE_INTEGER   = 9007199254740991
CONST_DEFAULT_SAMPLE_RATE   = 44100
CONST_VAE_HOP_SIZE          = 2048
CONST_ACE_LATENTS_PER_SECOND = 5.0           # ACE-Step Oobleck: 1 frame = 200ms
CONST_LATENTS_PER_SECOND    = CONST_ACE_LATENTS_PER_SECOND
CONST_ACE_VAE_CONTEXT_FRAMES = 128           # Oobleck cold-start warm-up (discarded)

CONST_DEFAULT_TILE_SIZE     = 512
CONST_DEFAULT_OVERLAP       = 64
CONST_MIN_TILE_SIZE         = 64
CONST_MAX_TILE_SIZE         = 8192
CONST_PLOT_DPI              = 120
CONST_MAX_PLOT_SAMPLES      = 150000

# MD Visual Identity
CONST_BACKGROUND_COLOR      = '#1e1e1e'
CONST_WAVEFORM_COLOR        = '#87CEEB'      # Sky Blue
CONST_PEAK_COLOR            = 'orangered'
CONST_BOUNDARY_COLOR_A      = '#FF4444'      # Pass A seam — bright red
CONST_BOUNDARY_COLOR_B      = '#FFB347'      # Pass B seam — amber (dual-pass only)
CONST_BOUNDARY_COLOR_SINGLE = '#FF4444'      # Single-pass seam

# =================================================================================
# == PerformanceProfiler  (MD_Nodes Standard v1.5.3)
# =================================================================================

class PerformanceProfiler:
    def __init__(self, enabled=True):
        self.enabled     = enabled
        self.timings     = {}
        self.start_times = {}

    def start(self, name):
        if not self.enabled: return
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if not self.enabled: return
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times.pop(name)
            self.timings.setdefault(name, []).append(elapsed)

    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(v) for v in self.timings.values())

    def print_report(self):
        if not self.enabled or not self.timings: return
        print("\n" + "=" * 60)
        print("📊 [AudioTiled] ANALYTICS REPORT")
        print("=" * 60)
        print("⏱️  PERFORMANCE:")
        print(f"    • Total Wall Time: {self.get_total_time():.4f}s")
        for name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            tag = f"({len(times)}x)" if len(times) > 1 else ""
            print(f"    • {name:<28}: {avg:.4f}s {tag}")
        print("=" * 60 + "\n")

# =================================================================================
# == Enterprise API Client  (MD Standard v2.1 — unchanged)
# =================================================================================

class AudioTiledApiClient:
    @staticmethod
    def execute_remote(latents, tile_size, overlap, blending_mode, rms_leveling,
                       dual_pass, slerp_bridge, adaptive_tiling,
                       context_prefix, api_url, api_key):
        """Dispatches VAE decoding to remote enterprise server."""
        try:
            # ── Serialize latent tensor via torch.save (no pickle on user data) ──
            tensor_buf = io.BytesIO()
            torch.save(latents.cpu(), tensor_buf)
            tensor_buf.seek(0)
            encoded_tensor = base64.b64encode(
                zlib.compress(tensor_buf.read())
            ).decode('utf-8')

            payload = {
                "tensor": encoded_tensor,
                "params": {
                    "tile_size":       tile_size,
                    "overlap":         overlap,
                    "blending_mode":   blending_mode,
                    "rms_leveling":    rms_leveling,
                    "dual_pass":       dual_pass,
                    "slerp_bridge":    slerp_bridge,
                    "adaptive_tiling": adaptive_tiling,
                    "context_prefix":  context_prefix,
                },
            }

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=300
            )

            if response.status_code != 200:
                logging.error(f"[AudioTiled] API Error {response.status_code}: {response.text}")
                return None, []

            result_json = response.json()
            if result_json.get("status") != "success":
                logging.error(f"[AudioTiled] Remote failure: {result_json.get('message')}")
                return None, []

            # ── Deserialize response via torch.load — no pickle.loads ──────────
            result_tensor_bytes = zlib.decompress(
                base64.b64decode(result_json["audio_tensor"])
            )
            audio_tensor = torch.load(
                io.BytesIO(result_tensor_bytes), map_location="cpu",
                weights_only=True
            )
            boundaries = result_json.get("boundaries", [])
            return audio_tensor, boundaries

        except Exception as e:
            logging.error(f"[AudioTiled] API Exception: {e}")
            return None, []

# =================================================================================
# == Main Node Class
# =================================================================================

# ─── VAE YAML Template + Parser (inserted before class MD_Audio_VAE_Tiled) ───

YAML_TEMPLATE_VAE_TILED = (
    "# ======================================================================\n"
    "# MD Audio VAE Tiled - COMPLETE Parameter Reference v3.2.0\n"
    "# ALL parameters listed. Paste into yaml_settings_str.\n"
    "# Edit values directly — no uncommenting needed.\n"
    "# ======================================================================\n"
    "\n"
    "# -- GOLD STANDARD (2026-05-16) --\n"
    "# overlap_mode: BPM-Synced (Max 4 Bars) | blending_mode: Hann\n"
    "# tile_size: 1024 | context_prefix: 512 | dual_pass: true | triple_pass: false\n"
    "# lss_strength: 0.25 | lss_dc_remove: false | soft_clip_db: -3.0\n"
    "# plgr_target_lufs: -18.0 | hum_notch_enabled: true (cuts only)\n"
    "\n"
    "# -- TILING & OVERLAP --\n"
    "overlap_mode: \"BPM-Synced (Max 4 Bars)\"\n"
    "tile_size: 1024\n"
    "overlap: 64\n"
    "context_prefix: 512\n"
    "blending_mode: \"Hann\"\n"
    "use_tiling: true\n"
    "adaptive_tiling: false\n"
    "\n"
    "# -- DECODE STRATEGY --\n"
    "dual_pass: true\n"
    "triple_pass: false\n"
    "phase_align: true\n"
    "slerp_bridge: true\n"
    "\n"
    "# -- LATENT SUPPRESSOR (LSS) --\n"
    "lss_enabled: true\n"
    "lss_strength: 0.25\n"
    "lss_variance_threshold: 0.12\n"
    "lss_dc_remove: false\n"
    "\n"
    "# -- SPECTRAL DENOISER (Wiener) --\n"
    "spectral_denoise: true\n"
    "denoise_strength: 1.00\n"
    "denoise_floor_pct: 0.05\n"
    "denoise_min_gain: 0.15\n"
    "\n"
    "# -- HARMONIC PRESERVATION CROSSFADE (HPC) --\n"
    "hpc_enabled: true\n"
    "hpc_min_overlap_samples: 1024\n"
    "\n"
    "# -- SPECTRAL CONTINUITY (SCE) --\n"
    "sce_enabled: true\n"
    "sce_threshold_db: 6.0\n"
    "sce_repair_frames: 4\n"
    "sce_max_freq_hz: 8000\n"
    "\n"
    "# -- LATENT SEAM PREDICTOR (LSP) --\n"
    "lsp_enabled: true\n"
    "lsp_bridge_frames: 16\n"
    "\n"
    "# -- LOUDNESS GAIN RIDING (PLGR) --\n"
    "plgr_enabled: true\n"
    "plgr_target_lufs: -18.0\n"
    "rms_leveling: true\n"
    "\n"
    "# -- POST-DECODE DSP --\n"
    "highpass_hz: 20\n"
    "soft_clip_db: -3.0\n"
    "stereo_width: 0.8\n"
    "\n"
    "# -- HUM NOTCH (multi-band biquad chain) --\n"
    "# V5 validated: cuts only. Boosts overcook presence.\n"
    "# Air and presence recover naturally once hum is removed.\n"
    "hum_notch_enabled: true\n"
    "# Bass shelf\n"
    "hum_notch_bass_shelf_enabled: true\n"
    "hum_bass_shelf_hz: 120.0\n"
    "hum_bass_shelf_db: -2.0\n"
    "hum_bass_shelf_slope: 0.7\n"
    "# 74Hz sub fundamental\n"
    "hum_notch_74_enabled: true\n"
    "hum_notch_74_hz: 74.4\n"
    "hum_notch_74_db: -1.43\n"
    "hum_notch_74_q: 6.27\n"
    "# 94Hz sub harmonic\n"
    "hum_notch_94_enabled: true\n"
    "hum_notch_94_hz: 94.0\n"
    "hum_notch_94_db: -1.86\n"
    "hum_notch_94_q: 7.08\n"
    "# 654Hz PRIMARY HUM\n"
    "hum_notch_656_enabled: true\n"
    "hum_notch_656_hz: 654.0\n"
    "hum_notch_656_db: -15.0\n"
    "hum_notch_656_q: 6.00\n"
    "# 353Hz presence (BOOST — disabled by default)\n"
    "hum_notch_353_enabled: false\n"
    "hum_notch_353_hz: 353.0\n"
    "hum_notch_353_db: 2.57\n"
    "hum_notch_353_q: 1.86\n"
    "# 1090Hz air restore (BOOST — disabled by default)\n"
    "hum_notch_1090_enabled: false\n"
    "hum_notch_1090_hz: 1090.0\n"
    "hum_notch_1090_db: 4.71\n"
    "hum_notch_1090_q: 2.47\n"
    "# 2060Hz presence (BOOST — disabled by default)\n"
    "hum_notch_2060_enabled: false\n"
    "hum_notch_2060_hz: 2060.0\n"
    "hum_notch_2060_db: 1.71\n"
    "hum_notch_2060_q: 0.71\n"
    "# 3520Hz air (BOOST — disabled by default)\n"
    "hum_notch_3520_enabled: false\n"
    "hum_notch_3520_hz: 3520.0\n"
    "hum_notch_3520_db: 3.43\n"
    "hum_notch_3520_q: 1.72\n"
    "\n"
    "# -- EXECUTION --\n"
    "execution_mode: \"Local\"\n"
)


def _parse_vae_yaml(yaml_str, defaults):
    """Parse YAML override string, merge with defaults. Returns merged dict."""
    result = dict(defaults)
    if not yaml_str or not yaml_str.strip():
        return result
    try:
        if _YAML_AVAILABLE:
            parsed = _yaml.safe_load(yaml_str) or {}
        else:
            parsed = {}
            for line in yaml_str.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    k, _, v = line.partition(':')
                    k = k.strip()
                    v = v.strip().split('#')[0].strip().strip('"').strip("'")
                    parsed[k] = v
        for k, v in parsed.items():
            if k in defaults:
                orig = defaults[k]
                try:
                    if isinstance(orig, bool):
                        result[k] = v if isinstance(v, bool) else str(v).lower() in ('true','1','yes')
                    elif isinstance(orig, int):
                        result[k] = int(float(str(v)))
                    elif isinstance(orig, float):
                        result[k] = float(str(v))
                    else:
                        result[k] = str(v)
                except Exception:
                    pass
    except Exception as e:
        print(f"[AudioTiled] YAML parse error: {e}")
    return result


def _build_yaml_applied(params):
    """Build YAML string of all active parameter values."""
    sections = [
        ("TILING & OVERLAP",  ["overlap_mode","tile_size","overlap","context_prefix","blending_mode","use_tiling","adaptive_tiling"]),
        ("DECODE STRATEGY",   ["dual_pass","triple_pass","phase_align","slerp_bridge"]),
        ("LSS",               ["lss_enabled","lss_strength","lss_variance_threshold","lss_dc_remove"]),
        ("DENOISER",          ["spectral_denoise","denoise_strength","denoise_floor_pct","denoise_min_gain"]),
        ("HPC",               ["hpc_enabled","hpc_min_overlap_samples"]),
        ("SCE",               ["sce_enabled", "sce_threshold_db", "sce_repair_frames", "sce_max_freq_hz"]),
        ("LSP",               ["lsp_enabled","lsp_bridge_frames"]),
        ("PLGR",              ["plgr_enabled","plgr_target_lufs","rms_leveling"]),
        ("DSP",               ["highpass_hz", "soft_clip_db", "stereo_width"]),
        ("EXECUTION",         ["execution_mode"]),
        ("HUM NOTCH",         [
            "hum_notch_enabled",
            "hum_notch_bass_shelf_enabled", "hum_bass_shelf_hz", "hum_bass_shelf_db", "hum_bass_shelf_slope",
            "hum_notch_74_enabled",  "hum_notch_74_hz",  "hum_notch_74_db",  "hum_notch_74_q",
            "hum_notch_94_enabled",  "hum_notch_94_hz",  "hum_notch_94_db",  "hum_notch_94_q",
            "hum_notch_656_enabled", "hum_notch_656_hz", "hum_notch_656_db", "hum_notch_656_q",
            "hum_notch_353_enabled", "hum_notch_353_hz", "hum_notch_353_db", "hum_notch_353_q",
            "hum_notch_1090_enabled","hum_notch_1090_hz","hum_notch_1090_db","hum_notch_1090_q",
            "hum_notch_2060_enabled","hum_notch_2060_hz","hum_notch_2060_db","hum_notch_2060_q",
            "hum_notch_3520_enabled","hum_notch_3520_hz","hum_notch_3520_db","hum_notch_3520_q",
        ]),
    ]
    lines = [
        "# ======================================================================",
        "# MD Audio VAE Tiled - Active Parameters (yaml_applied)",
        "# Paste into yaml_settings_str to reproduce this exact run.",
        "# ======================================================================","",
    ]
    for section, keys in sections:
        lines.append(f"# -- {section} --")
        for k in keys:
            if k in params:
                v = params[k]
                if isinstance(v, bool):
                    lines.append(f"{k}: {str(v).lower()}")
                elif isinstance(v, str):
                    lines.append(f'{k}: "{v}"')
                else:
                    lines.append(f"{k}: {v}")
        lines.append("")
    return "\n".join(lines)


_VAE_YAML_DEFAULTS = {
    # Tiling
    "overlap_mode": "BPM-Synced (Max 4 Bars)",
    "tile_size": 1024, "overlap": 64, "context_prefix": 512,
    "blending_mode": "Hann", "use_tiling": True, "adaptive_tiling": False,
    # Decode strategy
    "dual_pass": True, "triple_pass": False, "phase_align": True, "slerp_bridge": True,
    # LSS
    "lss_enabled": True, "lss_strength": 0.25, "lss_variance_threshold": 0.12, "lss_dc_remove": False,
    # Denoiser
    "spectral_denoise": True, "denoise_strength": 1.00, "denoise_floor_pct": 0.05, "denoise_min_gain": 0.15,
    # HPC
    "hpc_enabled": True, "hpc_min_overlap_samples": 1024,
    # SCE
    "sce_enabled": True, "sce_threshold_db": 6.0, "sce_repair_frames": 4, "sce_max_freq_hz": 8000,
    # LSP
    "lsp_enabled": True, "lsp_bridge_frames": 16,
    # PLGR
    "plgr_enabled": True, "plgr_target_lufs": -18.0, "rms_leveling": True,
    # Post-decode DSP
    "highpass_hz": 20.0, "soft_clip_db": -3.0, "stereo_width": 0.8,
    # Execution
    "execution_mode": "Local",
    # Hum notch — enable flags
    "hum_notch_enabled": True,
    "hum_notch_bass_shelf_enabled": True,
    "hum_notch_74_enabled": True,
    "hum_notch_94_enabled": True,
    "hum_notch_656_enabled": True,
    "hum_notch_353_enabled": False,   # boost — disabled (overcooks presence)
    "hum_notch_1090_enabled": False,  # boost — disabled
    "hum_notch_2060_enabled": False,  # boost — disabled
    "hum_notch_3520_enabled": False,  # boost — disabled
    # Hum notch — bass shelf
    "hum_bass_shelf_hz": 120.0, "hum_bass_shelf_db": -2.0, "hum_bass_shelf_slope": 0.7,
    # Hum notch — cut bands (measured Ableton EQ Eight 2026-05-16)
    "hum_notch_74_hz": 74.4,   "hum_notch_74_db": -1.43,  "hum_notch_74_q": 6.27,
    "hum_notch_94_hz": 94.0,   "hum_notch_94_db": -1.86,  "hum_notch_94_q": 7.08,
    "hum_notch_656_hz": 654.0, "hum_notch_656_db": -15.0, "hum_notch_656_q": 6.00,
    # Hum notch — boost bands (disabled by default, kept for experimentation)
    "hum_notch_353_hz": 353.0,  "hum_notch_353_db": 2.57,  "hum_notch_353_q": 1.86,
    "hum_notch_1090_hz": 1090.0,"hum_notch_1090_db": 4.71, "hum_notch_1090_q": 2.47,
    "hum_notch_2060_hz": 2060.0,"hum_notch_2060_db": 1.71, "hum_notch_2060_q": 0.71,
    "hum_notch_3520_hz": 3520.0,"hum_notch_3520_db": 3.43, "hum_notch_3520_q": 1.72,
}

class MD_Audio_VAE_Tiled:
    """
    MD: Audio VAE Decode (Tiled) v3.0.1
    Enterprise OLA synthesis engine for infinite-duration audio generation.

    Key features (v3.0.x):
        • PLGR            -- ITU-R BS.1770-4 psychoacoustic loudness gain riding.
        • PCT             -- Phase-Coherent Triple-Pass decode.
        • HPC             -- Harmonic Preservation Crossfade (STFT-domain OLA).
        • SCE             -- Spectral Continuity Enforcement at seam boundaries.
        • LSP             -- Latent-Domain Seam Predictor (SLERP bridge).
        • CASR            -- Context-Aware Seam Repair (replaced SLERP micro-decode).
        • dual_pass       -- Phase-offset staggered decode + trapezoidal merge.
        • adaptive_tiling -- Entropy-driven dynamic tile sizing.
    """
    def __init__(self):
        self.logger = logging.getLogger("MD_Nodes.Audio.VAETiled")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "Audio VAE model (AudioOobleckVAE or ACE standard)."}),
                "samples": ("LATENT", {"tooltip": "Audio latents from sampler."}),
                "sample_rate": ("INT", {"default": 48000, "min": 22050, "max": 96000, "step": 50,
                    "tooltip": "Output sample rate. ACE-Step default: 48000."}),
                "bpm": ("INT", {"default": 120, "min": 1, "max": 999,
                    "tooltip": "Track BPM. Used for BPM-synced overlap calculation."}),
                "yaml_settings_str": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "YAML CONFIGURATION\n"
                        "Paste yaml_template output here and uncomment/edit what you need.\n"
                        "All decode parameters live here. Base UI only has vae/samples/bpm/sample_rate/debug."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "1 - Info",
                    "tooltip": "Console verbosity. 2=full per-tile trace + feature analytics report."
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("audio", "debug_plot", "analytics_plot", "console_log", "yaml_applied", "yaml_template")
    FUNCTION      = "decode"
    CATEGORY      = "MD_Nodes/Audio"

    # ─────────────────────────────────────────────────────────────────────────
    def decode(self, vae, samples, sample_rate, bpm,
               yaml_settings_str="", debug_mode="1 - Info"):

        # ══════════════════════════════════════════════════════════════════════
        # 1. YAML PARSE — merge overrides with defaults
        # ══════════════════════════════════════════════════════════════════════
        p = _parse_vae_yaml(yaml_settings_str, _VAE_YAML_DEFAULTS)

        # Unpack all params from merged dict
        overlap_mode         = p["overlap_mode"]
        tile_size            = int(p["tile_size"])
        overlap              = int(p["overlap"])
        context_prefix       = int(p["context_prefix"])
        blending_mode        = p["blending_mode"]
        use_tiling           = bool(p["use_tiling"])
        adaptive_tiling      = bool(p["adaptive_tiling"])
        dual_pass            = bool(p["dual_pass"])
        triple_pass          = bool(p["triple_pass"])
        phase_align          = bool(p["phase_align"])
        slerp_bridge         = bool(p["slerp_bridge"])
        lss_enabled          = bool(p["lss_enabled"])
        lss_strength         = float(p["lss_strength"])
        lss_variance_threshold = float(p["lss_variance_threshold"])
        lss_dc_remove        = bool(p["lss_dc_remove"])
        spectral_denoise     = bool(p["spectral_denoise"])
        denoise_strength     = float(p["denoise_strength"])
        denoise_floor_pct    = float(p["denoise_floor_pct"])
        denoise_min_gain     = float(p["denoise_min_gain"])
        hpc_enabled          = bool(p["hpc_enabled"])
        hpc_min_overlap_samples = int(p["hpc_min_overlap_samples"])
        sce_enabled          = bool(p["sce_enabled"])
        sce_threshold_db     = float(p["sce_threshold_db"])
        sce_repair_frames    = int(p["sce_repair_frames"])
        sce_max_freq_hz      = float(p["sce_max_freq_hz"])
        lsp_enabled          = bool(p["lsp_enabled"])
        lsp_bridge_frames    = int(p["lsp_bridge_frames"])
        plgr_enabled         = bool(p["plgr_enabled"])
        plgr_target_lufs     = float(p["plgr_target_lufs"])
        rms_leveling         = bool(p["rms_leveling"])
        highpass_hz          = float(p["highpass_hz"])
        soft_clip_db         = float(p["soft_clip_db"])
        stereo_width         = float(p["stereo_width"])
        execution_mode       = p["execution_mode"]

        # Hum notch params
        hum_notch_enabled         = bool(p["hum_notch_enabled"])
        hum_notch_bass_shelf_en   = bool(p["hum_notch_bass_shelf_enabled"])
        hum_bass_shelf_hz         = float(p["hum_bass_shelf_hz"])
        hum_bass_shelf_db         = float(p["hum_bass_shelf_db"])
        hum_bass_shelf_slope      = float(p["hum_bass_shelf_slope"])
        hum_notch_74_en           = bool(p["hum_notch_74_enabled"])
        hum_notch_74_hz           = float(p["hum_notch_74_hz"])
        hum_notch_74_db           = float(p["hum_notch_74_db"])
        hum_notch_74_q            = float(p["hum_notch_74_q"])
        hum_notch_94_en           = bool(p["hum_notch_94_enabled"])
        hum_notch_94_hz           = float(p["hum_notch_94_hz"])
        hum_notch_94_db           = float(p["hum_notch_94_db"])
        hum_notch_94_q            = float(p["hum_notch_94_q"])
        hum_notch_656_en          = bool(p["hum_notch_656_enabled"])
        hum_notch_656_hz          = float(p["hum_notch_656_hz"])
        hum_notch_656_db          = float(p["hum_notch_656_db"])
        hum_notch_656_q           = float(p["hum_notch_656_q"])
        hum_notch_353_en          = bool(p["hum_notch_353_enabled"])
        hum_notch_353_hz          = float(p["hum_notch_353_hz"])
        hum_notch_353_db          = float(p["hum_notch_353_db"])
        hum_notch_353_q           = float(p["hum_notch_353_q"])
        hum_notch_1090_en         = bool(p["hum_notch_1090_enabled"])
        hum_notch_1090_hz         = float(p["hum_notch_1090_hz"])
        hum_notch_1090_db         = float(p["hum_notch_1090_db"])
        hum_notch_1090_q          = float(p["hum_notch_1090_q"])
        hum_notch_2060_en         = bool(p["hum_notch_2060_enabled"])
        hum_notch_2060_hz         = float(p["hum_notch_2060_hz"])
        hum_notch_2060_db         = float(p["hum_notch_2060_db"])
        hum_notch_2060_q          = float(p["hum_notch_2060_q"])
        hum_notch_3520_en         = bool(p["hum_notch_3520_enabled"])
        hum_notch_3520_hz         = float(p["hum_notch_3520_hz"])
        hum_notch_3520_db         = float(p["hum_notch_3520_db"])
        hum_notch_3520_q          = float(p["hum_notch_3520_q"])

        yaml_applied_str = _build_yaml_applied(p)

        # ══════════════════════════════════════════════════════════════════════
        # 2. PRE-EXECUTION CHECKS
        # ══════════════════════════════════════════════════════════════════════
        if not CQ_CORE_LOADED and execution_mode == "Local":
            logging.error(f"[AudioTiled] CRITICAL: Core math missing. {CQ_CORE_ERROR}")
            empty = self._empty_output()
            empty_img = self._empty_image()
            return (empty, empty_img, empty_img, "", yaml_applied_str, YAML_TEMPLATE_VAE_TILED)

        try:
            debug_level = int(debug_mode.split(" ")[0])
        except Exception:
            debug_level = 0

        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_decode")

        latents = samples["samples"]

        # Surface core health at info level
        if debug_level >= 1 and CQ_CORE_LOADED:
            report = core.get_core_health_report()
            print(f"🧠 [AudioTiled] Core v{report['version']} [{report['backend']}] "
                  f"| Cache: {report['cache_size']} windows | Type: {CQ_CORE_TYPE}")

        # ══════════════════════════════════════════════════════════════════════
        # 3. BPM-SYNC ENGINE — Integer-Multiple Snapped
        # ══════════════════════════════════════════════════════════════════════
        actual_overlap = overlap
        target_bars    = 1.0

        if overlap_mode != "Manual":
            bar_map = {
                "32 Bars": 32.0, "16 Bars": 16.0, "8 Bars": 8.0,
                "4 Bars":   4.0, "2 Bars":   2.0, "1 Bar":  1.0,
            }
            for label, bars in bar_map.items():
                if label in overlap_mode:
                    target_bars = bars
                    break

            sec_per_bar  = (60.0 / bpm) * 4.0
            calc_overlap = int(round(sec_per_bar * target_bars * CONST_LATENTS_PER_SECOND))

            # Downgrade until overlap fits inside half a tile
            while calc_overlap > (tile_size // 2) and target_bars > 0.25:
                if debug_level >= 1:
                    print(f"⚠️ [AudioTiled] Overlap {calc_overlap} > tile_size/2. "
                          f"Downgrading from {target_bars} bars...")
                target_bars  /= 2.0
                calc_overlap  = int(round(sec_per_bar * target_bars * CONST_LATENTS_PER_SECOND))

            # ── Integer-multiple snap: ensure overlap is exact multiple of bar ──
            frames_per_bar = sec_per_bar * CONST_LATENTS_PER_SECOND
            if frames_per_bar >= 1.0:
                calc_overlap = int(round(calc_overlap / frames_per_bar)) * int(frames_per_bar)
                calc_overlap = max(8, calc_overlap)

            # Also snap tile_size to integer-multiple of bar length for seam alignment
            if frames_per_bar >= 1.0:
                tile_size = max(
                    CONST_MIN_TILE_SIZE,
                    int(round(tile_size / frames_per_bar)) * int(frames_per_bar)
                )

            actual_overlap = calc_overlap

            if debug_level >= 1:
                print(f"🎵 [AudioTiled] BPM Sync: {bpm} BPM | "
                      f"Overlap: {actual_overlap} frames ({target_bars} bars) | "
                      f"Tile: {tile_size} frames")

        # ══════════════════════════════════════════════════════════════════════
        # 3. EXECUTION DISPATCH
        # ══════════════════════════════════════════════════════════════════════
        decoded_audio = None
        boundaries    = []
        bounds_b      = []

        # Branch 1: API
        if execution_mode == "API":
            api_url = os.environ.get("MD_AUDIO_API_URL", "http://localhost:8000/v1/decode")
            api_key = os.environ.get("MD_AUDIO_API_KEY", "")
            profiler.start("api_execution")
            decoded_audio, boundaries = AudioTiledApiClient.execute_remote(
                latents, tile_size, actual_overlap, blending_mode, rms_leveling,
                dual_pass, slerp_bridge, adaptive_tiling, context_prefix, api_url, api_key
            )
            profiler.stop("api_execution")

        # Branch 2: Local Core
        if decoded_audio is None and execution_mode != "API Only":
            profiler.start("local_execution")
            decoded_audio, boundaries, bounds_b = self._execute_local(
                vae, latents, tile_size, actual_overlap, blending_mode,
                rms_leveling, dual_pass, slerp_bridge, adaptive_tiling,
                context_prefix, use_tiling, debug_level, profiler, sample_rate,
                spectral_denoise, denoise_strength, denoise_floor_pct, denoise_min_gain,
                highpass_hz, soft_clip_db, stereo_width,
                plgr_enabled, plgr_target_lufs, triple_pass, phase_align,
                hpc_enabled, lsp_enabled, lsp_bridge_frames,
                lss_enabled, lss_strength, lss_variance_threshold, lss_dc_remove,
                sce_enabled, sce_threshold_db, sce_repair_frames, sce_max_freq_hz,
                hpc_min_overlap_samples=hpc_min_overlap_samples,
                hum_notch_enabled=hum_notch_enabled,
                hum_notch_bass_shelf_en=hum_notch_bass_shelf_en,
                hum_bass_shelf_hz=hum_bass_shelf_hz,
                hum_bass_shelf_db=hum_bass_shelf_db,
                hum_bass_shelf_slope=hum_bass_shelf_slope,
                hum_notch_74_en=hum_notch_74_en, hum_notch_74_hz=hum_notch_74_hz,
                hum_notch_74_db=hum_notch_74_db, hum_notch_74_q=hum_notch_74_q,
                hum_notch_94_en=hum_notch_94_en, hum_notch_94_hz=hum_notch_94_hz,
                hum_notch_94_db=hum_notch_94_db, hum_notch_94_q=hum_notch_94_q,
                hum_notch_656_en=hum_notch_656_en, hum_notch_656_hz=hum_notch_656_hz,
                hum_notch_656_db=hum_notch_656_db, hum_notch_656_q=hum_notch_656_q,
                hum_notch_353_en=hum_notch_353_en, hum_notch_353_hz=hum_notch_353_hz,
                hum_notch_353_db=hum_notch_353_db, hum_notch_353_q=hum_notch_353_q,
                hum_notch_1090_en=hum_notch_1090_en, hum_notch_1090_hz=hum_notch_1090_hz,
                hum_notch_1090_db=hum_notch_1090_db, hum_notch_1090_q=hum_notch_1090_q,
                hum_notch_2060_en=hum_notch_2060_en, hum_notch_2060_hz=hum_notch_2060_hz,
                hum_notch_2060_db=hum_notch_2060_db, hum_notch_2060_q=hum_notch_2060_q,
                hum_notch_3520_en=hum_notch_3520_en, hum_notch_3520_hz=hum_notch_3520_hz,
                hum_notch_3520_db=hum_notch_3520_db, hum_notch_3520_q=hum_notch_3520_q,
            )
            profiler.stop("local_execution")

        profiler.stop("total_decode")

        if decoded_audio is None:
            logging.error("[AudioTiled] Critical Failure: No output from any branch.")
            empty_img = self._empty_image()
            return (self._empty_output(), empty_img, empty_img, "Critical failure: no output.", yaml_applied_str, YAML_TEMPLATE_VAE_TILED)

        # Capture console log
        import io as _io, sys as _sys
        console_log = ""

        if debug_level >= 1:
            profiler.print_report()
            seam_info = (f"📉 [AudioTiled] Output shape: {list(decoded_audio.shape)} | "
                         f"Seams (A): {len(boundaries)} | Seams (B): {len(bounds_b)} | "
                         f"Total unique: {len(sorted(set(boundaries + bounds_b)))}")
            print(seam_info)
            console_log += seam_info + "\n"

        # ══════════════════════════════════════════════════════════════════════
        # 4. VISUALIZATION
        # ══════════════════════════════════════════════════════════════════════
        debug_img     = self._empty_image()
        analytics_img = self._empty_image()
        print(f"[AudioTiled] 🎨 Visualization check: MATPLOTLIB_AVAILABLE={MATPLOTLIB_AVAILABLE}")
        if MATPLOTLIB_AVAILABLE:
            profiler.start("visualization")
            debug_img = self._generate_debug_plot(
                decoded_audio, boundaries, bounds_b, dual_pass, adaptive_tiling, sample_rate
            )
            analytics_img = self._generate_analytics_plot(
                decoded_audio, boundaries, bounds_b, p, sample_rate
            )
            profiler.stop("visualization")

        return (
            {"waveform": decoded_audio.cpu(), "sample_rate": sample_rate},
            debug_img,
            analytics_img,
            console_log,
            yaml_applied_str,
            YAML_TEMPLATE_VAE_TILED
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _execute_local(self, vae, latents, tile_size, overlap, blending_mode,
                       rms_leveling, dual_pass, slerp_bridge, adaptive_tiling,
                       context_prefix, use_tiling, debug_level, profiler, sample_rate=48000,
                       spectral_denoise=True, denoise_strength=0.85,
                       denoise_floor_pct=0.05, denoise_min_gain=0.15,
                       highpass_hz=20.0, soft_clip_db=-3.0, stereo_width=0.8,
                       plgr_enabled=True, plgr_target_lufs=-18.0,
                       triple_pass=False, phase_align=True,
                       hpc_enabled=True, lsp_enabled=True, lsp_bridge_frames=16,
                       lss_enabled=True, lss_strength=0.25,
                       lss_variance_threshold=0.12, lss_dc_remove=False,
                       sce_enabled=True, sce_threshold_db=6.0,
                       sce_repair_frames=4, sce_max_freq_hz=8000.0,
                       hpc_min_overlap_samples=1024,
                       hum_notch_enabled=True,
                       hum_notch_bass_shelf_en=True, hum_bass_shelf_hz=120.0,
                       hum_bass_shelf_db=-2.0, hum_bass_shelf_slope=0.7,
                       hum_notch_74_en=True, hum_notch_74_hz=74.4,
                       hum_notch_74_db=-1.43, hum_notch_74_q=6.27,
                       hum_notch_94_en=True, hum_notch_94_hz=94.0,
                       hum_notch_94_db=-1.86, hum_notch_94_q=7.08,
                       hum_notch_656_en=True, hum_notch_656_hz=654.0,
                       hum_notch_656_db=-15.0, hum_notch_656_q=6.0,
                       hum_notch_353_en=False, hum_notch_353_hz=353.0,
                       hum_notch_353_db=2.57, hum_notch_353_q=1.86,
                       hum_notch_1090_en=False, hum_notch_1090_hz=1090.0,
                       hum_notch_1090_db=4.71, hum_notch_1090_q=2.47,
                       hum_notch_2060_en=False, hum_notch_2060_hz=2060.0,
                       hum_notch_2060_db=1.71, hum_notch_2060_q=0.71,
                       hum_notch_3520_en=False, hum_notch_3520_hz=3520.0,
                       hum_notch_3520_db=3.43, hum_notch_3520_q=1.72):
        """
        Local execution with OOM-resilience retry loop.
        Returns (decoded_audio, boundaries_a, boundaries_b).
        boundaries_b is non-empty only when dual_pass=True.
        """
        device         = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        decoded_audio = None
        boundaries    = []
        bounds_b      = []

        # Global (non-tiled) path
        if not use_tiling:
            profiler.start("global_decode")
            try:
                vae.first_stage_model.to(device)
                decoded = vae.decode(latents.to(device))
                fmt, decoded = core.detect_audio_format_logic(decoded)
                if fmt == 'BLC':
                    decoded = decoded.transpose(1, 2).contiguous().to(device)
                vae.first_stage_model.to(offload_device)
                profiler.stop("global_decode")
                return decoded, [], []
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("⚠️ [AudioTiled] Global OOM — falling back to tiled mode.")
                    use_tiling = True
                    comfy.model_management.soft_empty_cache()
                else:
                    raise e
            profiler.stop("global_decode")

        # Tiled path with auto tile-size reduction on OOM
        profiler.start("tiled_decode")
        current_tile = tile_size

        while current_tile >= CONST_MIN_TILE_SIZE:
            try:
                safe_overlap = min(overlap, current_tile // 2)

                if debug_level >= 2:
                    flags = (f"dual={'ON' if dual_pass else 'off'} | "
                             f"slerp={'ON' if slerp_bridge else 'off'} | "
                             f"adaptive={'ON' if adaptive_tiling else 'off'}")
                    print(f"🧩 [AudioTiled] tile={current_tile} overlap={safe_overlap} | {flags}")

                # Apply spectral denoiser settings to core constants before decode
                core.SPECTRAL_DENOISE_ENABLED   = spectral_denoise
                core.SPECTRAL_DENOISE_STRENGTH  = denoise_strength
                core.SPECTRAL_DENOISE_FLOOR_PCT = denoise_floor_pct
                core.SPECTRAL_DENOISE_MIN_GAIN  = denoise_min_gain
                core.POST_DECODE_HIGHPASS_HZ    = highpass_hz
                core.POST_DECODE_SOFT_CLIP_DB   = soft_clip_db
                core.POST_DECODE_STEREO_WIDTH   = stereo_width
                core.PLGR_ENABLED               = plgr_enabled
                core.PLGR_TARGET_LUFS           = plgr_target_lufs
                core.PCT_TRIPLE_PASS            = triple_pass
                core.PCT_PHASE_ALIGN            = phase_align
                core.HPC_ENABLED                = hpc_enabled
                core.HPC_MIN_OVERLAP_SAMPLES    = hpc_min_overlap_samples
                core.LSP_ENABLED                = lsp_enabled
                core.LSP_BRIDGE_FRAMES          = lsp_bridge_frames
                core.LSS_ENABLED               = lss_enabled
                core.LSS_STRENGTH              = lss_strength
                core.LSS_VARIANCE_THRESHOLD    = lss_variance_threshold
                core.LSS_DC_REMOVE             = lss_dc_remove
                core.SCE_ENABLED                = sce_enabled
                core.SCE_THRESHOLD_DB           = sce_threshold_db
                core.SCE_REPAIR_FRAMES          = sce_repair_frames
                core.SCE_MAX_FREQ_HZ            = sce_max_freq_hz
                # Hum notch
                core.HUM_NOTCH_ENABLED              = hum_notch_enabled
                core.HUM_NOTCH_BASS_SHELF_ENABLED   = hum_notch_bass_shelf_en
                core.HUM_BASS_SHELF_HZ              = hum_bass_shelf_hz
                core.HUM_BASS_SHELF_DB              = hum_bass_shelf_db
                core.HUM_BASS_SHELF_SLOPE           = hum_bass_shelf_slope
                core.HUM_NOTCH_74_ENABLED           = hum_notch_74_en
                core.HUM_NOTCH_74_HZ                = hum_notch_74_hz
                core.HUM_NOTCH_74_DB                = hum_notch_74_db
                core.HUM_NOTCH_74_Q                 = hum_notch_74_q
                core.HUM_NOTCH_94_ENABLED           = hum_notch_94_en
                core.HUM_NOTCH_94_HZ                = hum_notch_94_hz
                core.HUM_NOTCH_94_DB                = hum_notch_94_db
                core.HUM_NOTCH_94_Q                 = hum_notch_94_q
                core.HUM_NOTCH_656_ENABLED          = hum_notch_656_en
                core.HUM_NOTCH_656_HZ               = hum_notch_656_hz
                core.HUM_NOTCH_656_DB               = hum_notch_656_db
                core.HUM_NOTCH_656_Q                = hum_notch_656_q
                core.HUM_NOTCH_353_ENABLED          = hum_notch_353_en
                core.HUM_NOTCH_353_HZ               = hum_notch_353_hz
                core.HUM_NOTCH_353_DB               = hum_notch_353_db
                core.HUM_NOTCH_353_Q                = hum_notch_353_q
                core.HUM_NOTCH_1090_ENABLED         = hum_notch_1090_en
                core.HUM_NOTCH_1090_HZ              = hum_notch_1090_hz
                core.HUM_NOTCH_1090_DB              = hum_notch_1090_db
                core.HUM_NOTCH_1090_Q               = hum_notch_1090_q
                core.HUM_NOTCH_2060_ENABLED         = hum_notch_2060_en
                core.HUM_NOTCH_2060_HZ              = hum_notch_2060_hz
                core.HUM_NOTCH_2060_DB              = hum_notch_2060_db
                core.HUM_NOTCH_2060_Q               = hum_notch_2060_q
                core.HUM_NOTCH_3520_ENABLED         = hum_notch_3520_en
                core.HUM_NOTCH_3520_HZ              = hum_notch_3520_hz
                core.HUM_NOTCH_3520_DB              = hum_notch_3520_db
                core.HUM_NOTCH_3520_Q               = hum_notch_3520_q

                result = core.execute_tiled_decode_logic(
                    vae, latents, current_tile, safe_overlap, device,
                    blending_mode, rms_leveling,
                    context_prefix=context_prefix,
                    dual_pass=dual_pass,
                    slerp_bridge=slerp_bridge,
                    adaptive_tiling=adaptive_tiling,
                    sample_rate=sample_rate
                )
                # Unpack: (audio, bounds_a, bounds_b, audio_format)
                decoded_audio = result[0]
                boundaries    = result[1]   # Pass A seam positions
                bounds_b      = result[2]   # Pass B seam positions (empty if single-pass)

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ [AudioTiled] OOM at tile={current_tile}. Decimating...")
                    current_tile //= 2
                    comfy.model_management.soft_empty_cache()
                else:
                    raise e

        try:
            vae.first_stage_model.to(offload_device)
        except Exception:
            pass
        torch.cuda.empty_cache()
        profiler.stop("tiled_decode")

        return decoded_audio, boundaries, bounds_b

    # ─────────────────────────────────────────────────────────────────────────
    def _generate_debug_plot(self, audio, boundaries, bounds_b, dual_pass, adaptive_tiling, sample_rate=48000):
        """
        Enterprise waveform + seam visualization.
        Dual-pass: Pass A seams in red, Pass B seams in amber, overlap zones shaded.
        Single-pass: Seams in red with energy envelope overlay.
        """
        try:
            import matplotlib
            import traceback as _tb
            # Force Agg inside the call — other nodes may have switched the backend
            # at import time. matplotlib.use() after init requires force=True on 3.x.
            try:
                matplotlib.use('Agg', force=True)
                import matplotlib.pyplot as plt
            except Exception:
                pass  # Already Agg or switched safely

            plt.style.use('dark_background')
            fig, (ax_wave, ax_energy) = plt.subplots(
                2, 1, figsize=(14, 4), dpi=CONST_PLOT_DPI,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
            )
            fig.patch.set_facecolor(CONST_BACKGROUND_COLOR)

            # ── Waveform panel ────────────────────────────────────────────────
            ax_wave.set_facecolor('#0d0d0d')

            data      = audio[0, 0].cpu().numpy()
            ds_factor = max(1, len(data) // CONST_MAX_PLOT_SAMPLES)
            plot_data = data[::ds_factor]
            x_axis    = np.arange(len(plot_data)) * ds_factor
            total_samples = len(data)

            # Stereo shadow (channel 1 if available) — subtle depth
            if audio.shape[1] > 1:
                data_r      = audio[0, 1].cpu().numpy()
                plot_data_r = data_r[::ds_factor]
                ax_wave.fill_between(x_axis, plot_data_r, 0,
                                     color='#1a4a6e', alpha=0.25, linewidth=0)

            # Main waveform fill + line
            ax_wave.fill_between(x_axis, plot_data, 0,
                                  color=CONST_WAVEFORM_COLOR, alpha=0.15, linewidth=0)
            ax_wave.plot(x_axis, plot_data, color=CONST_WAVEFORM_COLOR,
                         alpha=0.92, linewidth=0.6, label="Audio")

            # Overlap zone shading between adjacent seams
            all_seams_a = sorted([b for b in (boundaries or []) if b > 0])
            if dual_pass and bounds_b:
                # shade overlap zones: from each Pass B seam to the next Pass A seam
                seams_b = sorted(bounds_b)
                for sb in seams_b:
                    # find the next Pass A seam after this Pass B seam
                    next_a = next((sa for sa in all_seams_a if sa > sb), None)
                    if next_a is not None:
                        ax_wave.axvspan(sb, next_a, color='#FFB347', alpha=0.06, linewidth=0)

            # Pass A seam lines — sharp red
            if boundaries:
                for i, b in enumerate(boundaries):
                    ax_wave.axvline(x=b, color='#FF3333', alpha=0.8,
                                    linestyle='--', linewidth=0.9,
                                    label="Pass A" if i == 0 else "")

            # Pass B seam lines — amber, offset dotted
            if dual_pass and bounds_b:
                for i, b in enumerate(bounds_b):
                    ax_wave.axvline(x=b, color='#FFB347', alpha=0.65,
                                    linestyle=':', linewidth=1.1,
                                    label="Pass B" if i == 0 else "")

            ax_wave.set_xlim(0, total_samples)
            ax_wave.set_ylim(-1.05, 1.05)
            ax_wave.set_ylabel("Amplitude", color='#666666', fontsize=7)
            ax_wave.tick_params(axis='both', colors='#555555', labelsize=6,
                                 bottom=False, labelbottom=False)
            ax_wave.yaxis.set_tick_params(length=3)
            ax_wave.grid(True, color='#1a1a1a', linestyle='-', alpha=0.8, linewidth=0.4)
            for spine in ax_wave.spines.values():
                spine.set_edgecolor('#2a2a2a')

            # Legend
            legend = ax_wave.legend(
                loc='upper right', fontsize=6.5, framealpha=0.6,
                facecolor='#111111', edgecolor='#333333'
            )
            for text in legend.get_texts():
                text.set_color('#aaaaaa')

            # ── Build title ───────────────────────────────────────────────────
            n_a      = len(boundaries or [])
            n_b      = len(bounds_b or [])
            dur_s    = total_samples / sample_rate
            dur_str  = f"{int(dur_s//60)}:{int(dur_s%60):02d}"
            flags    = []
            if dual_pass:        flags.append(f"Dual-Pass  A:{n_a} B:{n_b}")
            else:                flags.append(f"{n_a} Seams")
            if adaptive_tiling:  flags.append("Adaptive")
            flag_str = "  ·  ".join(flags)

            ax_wave.set_title(
                f"MD Audio VAE Tiled  v3.0.1   ·   {flag_str}   ·   {dur_str}  @{sample_rate}Hz",
                color='#cccccc', fontsize=8.5, pad=5,
                fontdict={'fontweight': 'bold'}
            )

            # ── Energy envelope panel ─────────────────────────────────────────
            ax_energy.set_facecolor('#080808')

            # RMS energy in chunks
            chunk = max(1, len(data) // 512)
            rms_vals, rms_x = [], []
            for j in range(0, len(data) - chunk, chunk):
                seg = data[j : j + chunk]
                rms = float(np.sqrt(np.mean(seg ** 2)))
                rms_vals.append(rms)
                rms_x.append(j + chunk // 2)

            rms_vals = np.array(rms_vals)
            rms_x    = np.array(rms_x)

            ax_energy.fill_between(rms_x, rms_vals, 0,
                                    color='#2ecc71', alpha=0.5, linewidth=0)
            ax_energy.plot(rms_x, rms_vals, color='#2ecc71',
                           alpha=0.85, linewidth=0.7)

            # Mark seams on energy panel too
            if boundaries:
                for b in boundaries:
                    ax_energy.axvline(x=b, color='#FF3333', alpha=0.5,
                                      linestyle='--', linewidth=0.7)
            if dual_pass and bounds_b:
                for b in bounds_b:
                    ax_energy.axvline(x=b, color='#FFB347', alpha=0.4,
                                      linestyle=':', linewidth=0.8)

            ax_energy.set_xlim(0, total_samples)
            ax_energy.set_ylim(0, max(rms_vals.max() * 1.15, 0.01) if len(rms_vals) else 0.1)
            ax_energy.set_ylabel("RMS", color='#555555', fontsize=6)
            ax_energy.set_xlabel("Sample", color='#555555', fontsize=6.5)
            ax_energy.tick_params(colors='#444444', labelsize=5.5)
            ax_energy.grid(True, color='#141414', linestyle='-', alpha=0.8, linewidth=0.3)
            for spine in ax_energy.spines.values():
                spine.set_edgecolor('#1e1e1e')

            # ── Render ────────────────────────────────────────────────────────
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight',
                        dpi=CONST_PLOT_DPI, facecolor=CONST_BACKGROUND_COLOR)
            plt.close(fig)
            buf.seek(0)

            img_array = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)

        except Exception as e:
            import traceback
            print(f"[AudioTiled] ❌ Plot generation failed: {e}")
            print(f"[AudioTiled] Full traceback:\n{traceback.format_exc()}")
            return self._empty_image()

    # ─────────────────────────────────────────────────────────────────────────
    def _generate_analytics_plot(self, audio, boundaries, bounds_b, params, sample_rate=48000):
        """
        Feature analytics plot — second matplotlib figure.
        Shows: RMS envelope with seam markers + active feature summary panel.
        """
        try:
            import matplotlib
            try:
                matplotlib.use('Agg', force=True)
                import matplotlib.pyplot as plt
            except Exception:
                pass

            plt.style.use('dark_background')
            fig, (ax_rms, ax_feat) = plt.subplots(
                1, 2, figsize=(14, 4), dpi=CONST_PLOT_DPI,
                gridspec_kw={'width_ratios': [2, 1]}
            )
            fig.patch.set_facecolor(CONST_BACKGROUND_COLOR)
            fig.suptitle("MD Audio VAE Tiled  v3.0.1   ·   Analytics",
                         color='#cccccc', fontsize=9, y=0.98)

            # ── Left: RMS envelope with seam colors by feature ──────────────
            ax_rms.set_facecolor('#0d0d0d')
            data      = audio[0, 0].cpu().numpy()
            n_samples = len(data)
            chunk     = max(1, n_samples // 500)
            rms_vals  = []
            for i in range(0, n_samples, chunk):
                seg = data[i:i+chunk]
                rms_vals.append(float((seg**2).mean()**0.5))
            import numpy as np
            rms_x = np.arange(len(rms_vals)) * chunk
            ax_rms.fill_between(rms_x, rms_vals, alpha=0.7, color='#22cc66')
            ax_rms.plot(rms_x, rms_vals, color='#33ff88', linewidth=0.8, alpha=0.9)

            # Seam markers — colored by pass
            for b in (boundaries or []):
                ax_rms.axvline(x=b, color='#FF3333', alpha=0.7, linewidth=0.8, linestyle='--')
            for b in (bounds_b or []):
                ax_rms.axvline(x=b, color='#FFB347', alpha=0.5, linewidth=0.8, linestyle=':')

            ax_rms.set_xlim(0, n_samples)
            ax_rms.set_ylim(0, max(rms_vals) * 1.15 if rms_vals else 1.0)
            ax_rms.set_title("RMS Envelope + Seam Map", color='#aaaaaa', fontsize=8)
            ax_rms.tick_params(colors='#666666', labelsize=6)
            ax_rms.set_xlabel("Sample", color='#666666', fontsize=7)
            ax_rms.set_ylabel("RMS", color='#666666', fontsize=7)

            # ── Right: Feature status panel ──────────────────────────────────
            ax_feat.set_facecolor('#0d0d0d')
            ax_feat.axis('off')

            def feat_line(name, enabled, detail=""):
                icon = "✅" if enabled else "⬜"
                return f"{icon} {name}" + (f"\n    {detail}" if detail else "")

            feature_lines = [
                feat_line("LSS", params.get("lss_enabled", True),
                          f"str={params.get('lss_strength',0.25):.2f} thr={params.get('lss_variance_threshold',0.12):.2f} dc={params.get('lss_dc_remove',False)}"),
                feat_line("PLGR", params.get("plgr_enabled", True),
                          f"target={params.get('plgr_target_lufs',-18.0):.1f} LUFS | downward-only ±1.5dB"),
                feat_line("HPC", params.get("hpc_enabled", True),
                          f"70/30 spectral blend | min_overlap={params.get('hpc_min_overlap_samples',1024)}"),
                feat_line("SCE", params.get("sce_enabled", True),
                          f"threshold={params.get('sce_threshold_db',6.0):.1f}dB"),
                feat_line("LSP", params.get("lsp_enabled", True),
                          "bridge gen ACTIVE | injection DISABLED"),
                feat_line("Wiener", params.get("spectral_denoise", True),
                          f"strength={params.get('denoise_strength',1.0):.2f} floor={params.get('denoise_floor_pct',0.05):.2f}"),
                feat_line("Dual-Pass" if not params.get("triple_pass") else "Triple-Pass",
                          params.get("dual_pass", True) or params.get("triple_pass", False),
                          f"A={len(boundaries or [])} B={len(bounds_b or [])} seams"),
                feat_line("DSP Chain", True,
                          f"HPF={params.get('highpass_hz',20):.0f}Hz | W={params.get('stereo_width',0.8):.1f} | Clip={params.get('soft_clip_db',-3.0):.1f}dB"),
            ]
            text = "\n\n".join(feature_lines)
            ax_feat.text(0.05, 0.95, text, transform=ax_feat.transAxes,
                         color='#cccccc', fontsize=7, va='top', ha='left',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111', alpha=0.8))
            ax_feat.set_title("Active Features", color='#aaaaaa', fontsize=8)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight',
                        facecolor=CONST_BACKGROUND_COLOR, dpi=CONST_PLOT_DPI)
            plt.close(fig)
            buf.seek(0)
            img_array = np.array(Image.open(buf).convert('RGB')).astype(np.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
        except Exception as e:
            print(f"[AudioTiled] Analytics plot failed: {e}")
            return self._empty_image()

    def _empty_output(self):
        return {"waveform": torch.zeros((1, 2, CONST_DEFAULT_SAMPLE_RATE)),
                "sample_rate": CONST_DEFAULT_SAMPLE_RATE}

    def _empty_image(self):
        return torch.zeros((1, 64, 64, 3))


# =================================================================================
# == Embedded Unit Tests
# =================================================================================

if __name__ == "__main__":
    print("\n🧪 MD_Audio_VAE_Tiled v3.0.1 — Self-Tests\n")
    passed = 0
    failed = 0

    try:
        assert CONST_JS_MAX_SAFE_INTEGER == 9007199254740991
        print("✅ JS_MAX_SAFE_INTEGER constant: OK")
        passed += 1
    except AssertionError:
        print("❌ JS_MAX_SAFE_INTEGER constant: FAIL")
        failed += 1

    try:
        paths = find_core_paths()
        print(f"✅ Core path discovery: {len(paths)} candidate(s) found")
        passed += 1
    except Exception as e:
        print(f"❌ Core path discovery: {e}")
        failed += 1

    try:
        p = PerformanceProfiler(enabled=True)
        p.start("test"); time.sleep(0.001); p.stop("test")
        assert "test" in p.timings
        print("✅ PerformanceProfiler: OK")
        passed += 1
    except Exception as e:
        print(f"❌ PerformanceProfiler: {e}")
        failed += 1

    try:
        # BPM snap math sanity check: 120BPM, 4 bars = 10 frames @5Hz
        bpm = 120; bars = 4.0; lps = 5.0
        expected = int(round((60.0 / bpm) * 4.0 * bars * lps))
        assert expected == 10
        print(f"✅ BPM snap math (120BPM, 4 bars → {expected} frames): OK")
        passed += 1
    except AssertionError:
        print("❌ BPM snap math: FAIL")
        failed += 1

    print(f"\n{'='*40}")
    print(f"Summary: {passed} passed | {failed} failed\n")

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS      = {"MD_Audio_VAE_Tiled": MD_Audio_VAE_Tiled}
NODE_DISPLAY_NAME_MAPPINGS = {"MD_Audio_VAE_Tiled": "MD: Audio VAE Decode (Tiled) v3"}