# Nail Art Compression Pro

## Abstract

We present Nail Art Compression Pro, a domain-aware image compression and restoration pipeline designed for high-frequency cosmetic imagery such as nail art photographs. The system combines real JPEG entropy-coded compression with a deep convolutional denoising network (DnCNN) operating in the luminance domain (YCbCr-Y) to mitigate compression artifacts while preserving perceptually critical fine-grained structures. We further introduce a region-of-interest (ROI) evaluation protocol focused on nail-centric image regions to better reflect perceptual quality in application-specific settings. The framework integrates chroma subsampling control, multi-scale inference, and perceptual metrics (PSNR, SSIM, LPIPS) within an interactive Gradio-based interface.

---

## 1. Introduction

Standard image compression algorithms (e.g., JPEG) are optimized for general-purpose imagery and often fail to preserve high-frequency details in structured cosmetic imagery such as nail art. This leads to perceptible artifacts including blocking, ringing, and color banding, which are particularly detrimental in e-commerce and portfolio contexts.

We address this limitation by introducing a hybrid compression-restoration pipeline that combines:

* Standard JPEG entropy-coded compression for true bitrate reduction
* A learned denoising model (DnCNN) applied to luminance components
* ROI-based perceptual evaluation tailored to nail-centric content

---

## 2. Method

### 2.1 Overview

Given an input RGB image (I), the pipeline performs:

1. Color space conversion: RGB → YCbCr
2. Optional spatial resizing with scale preservation heuristics
3. JPEG compression with configurable quality factor Q and chroma subsampling
4. DnCNN-based restoration on luminance channel Y
5. Multi-scale fusion of restored outputs
6. Reconstruction: YCbCr → RGB

---

### 2.2 JPEG Compression Module

We employ standard entropy-coded JPEG compression with explicit control over:

* Quality factor Q ∈ {25, 50, 75, 90}
* Chroma subsampling ratios: 4:2:0, 4:2:2, 4:4:4

Unlike tensor-based approximations, bitrate measurements are derived from actual encoded JPEG bitstreams.

---

### 2.3 DnCNN Restoration

We adopt a 20-layer DnCNN architecture operating exclusively on the luminance channel. The model is trained to learn a residual mapping:

[
R(Y) = Y_{clean} - Y_{compressed}
]

The restored output is computed as:

[
\hat{Y} = Y_{compressed} + R(Y_{compressed})
]

Multi-scale inference is performed by applying the model at both original and downsampled resolutions, followed by weighted fusion.

---

### 2.4 ROI-Based Evaluation

To better reflect application-specific perceptual quality, we define a central region-of-interest (ROI) covering the primary nail region. Metrics are computed both globally and within ROI crops:

* Global image metrics
* ROI-centered metrics (central 50% crop)

This mitigates bias introduced by background regions in full-image averaging.

---

## 3. Experimental Setup

### 3.1 Metrics

We evaluate performance using:

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)
* LPIPS (Learned Perceptual Image Patch Similarity)
* Bits-per-pixel (BPP)
* True JPEG file size (entropy-coded output)

---

### 3.2 Implementation Details

* Framework: PyTorch 2.0
* Interface: Gradio 4.0
* Image processing: PIL, OpenCV, NumPy
* Perceptual metrics: scikit-image, LPIPS

---

## 4. System Architecture

Input RGB Image
→ Color Conversion (YCbCr)
→ JPEG Compression (entropy-coded)
→ DnCNN (Y channel)
→ Multi-scale fusion
→ Reconstruction (RGB output)

Optional modules:

* Chroma subsampling control
* Unsharp masking post-processing
* ROI metric extraction

---

## 5. Results

The system demonstrates improved perceptual fidelity in ROI regions compared to baseline JPEG compression, particularly in high-frequency textures such as glitter, fine lines, and reflective surfaces.

Typical trends observed:

* Increased SSIM and PSNR in ROI regions relative to global metrics
* Significant reduction in blocking artifacts at moderate compression levels (Q50–Q75)
* Improved perceptual similarity as measured by LPIPS

---

## 6. Discussion

The proposed pipeline highlights the importance of domain-aware compression strategies. While general-purpose codecs optimize for average perceptual quality, application-specific evaluation (e.g., nail-centric ROI analysis) reveals localized improvements that are not captured by global metrics alone.

Limitations include dependency on accurate implicit ROI assumptions and potential generalization gaps outside cosmetic imagery.

---

## 7. Conclusion

We introduce a hybrid compression-restoration framework tailored for nail art imagery that integrates classical JPEG compression with deep learning-based artifact removal. The system emphasizes perceptual quality preservation in regions of interest and provides a reproducible pipeline for application-specific image compression research.

---

## Acknowledgments

> "Rumi saw God within himself; I see you within my work."

Special thanks to you Aziz for your support, encouragement, and presence throughout this project. Your impact exists in every detail of it.

## References

* Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising", 2017.
* Wallace, G. K., "The JPEG Still Picture Compression Standard", 1992.

---

## Appendix A: Implementation Stack

* PyTorch
* Gradio
* OpenCV
* PIL
* NumPy
* LPIPS
* scikit-image

---

## Appendix B: Notes on Reproducibility

The system uses deterministic JPEG encoding settings where applicable. Variations in entropy coding may introduce minor bitrate fluctuations depending on image content.

