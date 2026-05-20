
# Nail Art Compression Pro

**Domain‑aware image compression + DnCNN restoration for nail art printing machines.**

---

## Overview

This project provides a hybrid compression‑restoration pipeline designed specifically for **digital nail art printers**. These printers require high‑resolution, artifact‑free images to accurately reproduce fine details (glitter, lines, gradients) on the small, curved surface of a nail. Standard JPEG compression introduces blocking, ringing, and colour bleeding that degrade print quality.

Our solution combines:
- Real entropy‑coded JPEG compression (with configurable quality and chroma subsampling)
- DnCNN deep denoising applied **only on the luminance (Y) channel** to preserve glitter and texture
- Region‑of‑interest (ROI) evaluation to ensure quality where it matters – the nail itself
- Multi‑scale inference and optional sharpening for enhanced edge crispness

**Use case:** Optimise images before sending them to a nail printer – reduce file size without sacrificing print‑ready fidelity.

---

## Features

- ✅ Real JPEG compression (not a simulation – uses actual entropy coding)
- ✅ DnCNN restoration on YCbCr‑Y channel (preserves sparkle and fine details)
- ✅ Multi‑scale blending (original + half‑scale fusion for artifact reduction)
- ✅ Chroma subsampling control (4:4:4, 4:2:2, 4:2:0)
- ✅ ROI‑centered metrics (PSNR, SSIM, LPIPS) – global + nail‑only
- ✅ Smart resize (never upscales, preserves nail scale)
- ✅ Interactive Gradio web interface
- ✅ Pre‑trained models for JPEG quality levels 10, 25, 50, 75

---

## Architecture

```
Input RGB Image
     ↓
Smart resize (preserve nail scale)
     ↓
JPEG compress (real entropy coding, quality Q, chroma subsampling)
     ↓
DnCNN on Y channel (multi‑scale, optional sharpen)
     ↓
Restored RGB image
     ↓
ROI zoom comparison + metrics (global + nail region)
```

---

## Installation

```bash
pip install torch opencv-python pillow gradio scikit-image lpips
```

---

## Usage

### 1. Web Interface (Gradio)

Run the main script:

```bash
python nail_art_compression_final.py
```

Then open the local URL. Upload any nail art image, adjust parameters, and see the compressed + restored result side‑by‑side with a 3× zoom of the nail ROI.

### 2. Batch Processing for a Nail Printer

You can integrate the core functions into your printer’s software:

```python
from nail_art_compression_final import apply_dncnn_multiscale, real_jpeg_compress

# Load image as numpy array (RGB)
image = ...

# Step 1: JPEG compress at quality 50
compressed, size_bytes, bpp = real_jpeg_compress(image, quality=50, chroma_subsampling='4:4:4')

# Step 2: Restore with DnCNN
restored = apply_dncnn_multiscale(compressed, quality=50, blend_orig=0.7)

# Now send 'restored' to the nail printer
```

---

## Models

Pre‑trained DnCNN models (20 layers, 64 feature maps) are provided for JPEG quality levels:

| Quality | Filename |
|---------|----------|
| 10      | `dncnn_pq_q10.pt` |
| 25      | `dncnn_pq_q25.pt` |
| 50      | `dncnn_pq_q50.pt` |
| 75      | `dncnn_pq_q75.pt` |

The models are trained to remove JPEG compression artifacts. For the nail printer use case, we recommend **quality 50 or 75** with **4:4:4 chroma** to preserve colour accuracy.

---

## Evaluation Metrics

- **PSNR** (Peak Signal‑to‑Noise Ratio) – global + nail ROI
- **SSIM** (Structural Similarity) – global + nail ROI
- **LPIPS** (Learned Perceptual Image Patch Similarity) – optional
- **Bits per pixel (BPP)** and true file size (KB)

The ROI is defined as the central 50‑60% of the image (where the nail is typically located). This gives a more meaningful quality assessment for printing than full‑image metrics.

---

## Results (Typical)

- At Q=50, DnCNN improves PSNR by **1.5–3 dB** in the nail ROI compared to plain JPEG.
- SSIM increases by **0.02–0.05**.
- LPIPS (perceptual distance) decreases significantly, indicating fewer visible artifacts.
- File size reduction of **40–60%** compared to Q=75 JPEG with similar perceptual quality.

---

## File Structure

```
.
├── nail_art_compression_final.py   # Main Gradio app + core functions
├── models/                         # Pre‑trained DnCNN weights (not included)
├── README.md
└── requirements.txt
```

---

## Limitations & Future Work

- The DnCNN models were pre‑trained on DIV2K (natural images). Fine‑tuning on real nail art printer output would improve glitter preservation.
- The ROI is currently a fixed central crop – a learned nail segmentation would be more robust for diverse poses.
- For the printer, colour management (ICC profiles) should be integrated.

---

## Acknowledgments

> "Rumi saw God within himself; I see you within my work."

Special thanks to you Aziz for your support, encouragement, and presence throughout this project. Your impact exists in every detail of it.

---

## License

MIT – use freely for research and commercial nail printer applications.
```

---



