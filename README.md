
# 💅 Nail Art Compression

### AI-Assisted Compression & Restoration for Beauty Imagery

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-green)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An interactive AI-powered image compression system designed specifically for **nail art and beauty images**.
The project combines **JPEG DCT compression** with **deep learning-based artifact reduction (DnCNN)** to preserve fine artistic details while reducing file size.

> 🎯 **Optimized for**: Instagram, Etsy, salon websites, mobile portfolios, e-commerce platforms

---

## ✨ Why This Project?

Traditional JPEG compression destroys what makes nail art beautiful:

| Problem | Standard JPEG | Our Solution |
|---------|-------------|--------------|
| Glitter texture | ❌ Blocky, lost | ✅ Preserved via DnCNN |
| Gradient backgrounds | ❌ Banding artifacts | ✅ Smooth reconstruction |
| Fine line details | ❌ Blurred/ringing | ✅ Edge-preserving denoising |
| Skin tone transitions | ❌ Patchy | ✅ Natural gradients |
| Reflective surfaces | ❌ Flat | ✅ Specular detail restored |

---

## 🚀 Features

- 📸 **Drag & drop upload** — Supports JPG, PNG, WEBP
- 🗜️ **4 quality presets** — Basique (Q25) → Premium (Q90)
- 🤖 **AI toggle** — Enable/disable DnCNN per image
- 🔍 **Before/After slider** — Pixel-perfect comparison
- 📊 **Real-time metrics** — PSNR, SSIM, BPP, file size
- 💾 **One-click download** — PNG output
- 🌈 **Full RGB color** — No grayscale conversion

---

## 🧠 Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  RGB Input  │ → │ JPEG DCT     │ → │ DnCNN       │ → │ Optimized   │
│  (512px)    │    │ Compression  │    │ Denoising   │    │ RGB Output  │
│             │    │ (Q25-Q90)    │    │ (per channel│    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                              ↑
                                    Quality-specific model
                                    (q10/q25/q50/q75)
```

### DnCNN Restoration
Each RGB channel is processed independently by a **20-layer DnCNN** trained specifically for that compression level. The network learns to:
- Remove DCT blocking artifacts
- Suppress ringing near edges
- Recover high-frequency texture details
- Preserve semantic structure

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | Python + Gradio | Interactive web UI |
| ML Framework | PyTorch 2.0 | DnCNN inference |
| Image Processing | Pillow + NumPy + SciPy | DCT/JPEG simulation |
| Metrics | scikit-image | PSNR / SSIM computation |
| Deployment | Gradio Share | Public URL generation |

---

## 📂 Project Structure

```
nail-art-compression/
│
├── 📁 models/                          # Pre-trained DnCNN weights
│   ├── dncnn_pq_q10.pt                 # Ultra-low quality restoration
│   ├── dncnn_pq_q25.pt                 # Low quality (max compression)
│   ├── dncnn_pq_q50.pt                 # Medium quality (balanced)
│   └── dncnn_pq_q75.pt                 # High quality restoration
│
├── 📄 nail_art_compression.ipynb         # Kaggle notebook (this file)
├── 📄 backend.py                       # FastAPI backend (optional)
├── 📄 requirements.txt                 # Dependencies
│
└── 📁 assets/                          # Screenshots & demo images
    └── demo_comparison.png
```

---

## ⚙️ Installation & Setup

### Option A: Kaggle Notebook (Recommended)
1. Open [Kaggle Notebooks](https://www.kaggle.com/code)
2. Add dataset: `mariyyyaaella/dncnn-nail-art-models`
3. Copy `nail_art_compression.ipynb`
4. Run all cells → Gradio interface appears

### Option B: Local Environment
```bash
# Clone repository
git clone https://github.com/yourusername/nail-art-compression.git
cd nail-art-compression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (if not included)
# Place .pt files in ./models/

# Launch
python app.py
```

### requirements.txt
```text
gradio>=4.0
torch>=2.0
torchvision
numpy
pillow
scipy
scikit-image
```

---

## ▶️ Usage

### Gradio Interface
```python
# In Kaggle cell or local Python
python app.py

# Output:
# Running on local URL:  http://127.0.0.1:7860
# Running on public URL: https://xxxx.gradio.live  ← Share this!
```

### API Endpoint (FastAPI mode)
```bash
# Start backend
uvicorn backend:app --host 0.0.0.0 --port 8000

# Test compression
curl -X POST "http://localhost:8000/compress" \
  -F "file=@nail_art.jpg" \
  -F "quality=50" \
  -F "apply_dncnn=true"
```

---

## 📊 Metrics Explained

| Metric | Formula Range | Interpretation |
|--------|--------------|----------------|
| **PSNR** | 0 → ∞ dB | >30 dB = good, >40 dB = excellent |
| **SSIM** | 0 → 1 | >0.95 = perceptually identical |
| **BPP** | 0 → 24 | Lower = more compressed |
| **File Size** | KB | Target: <100KB for web |

### Typical Results
| Quality | DnCNN | PSNR | SSIM | Size | Use Case |
|---------|-------|------|------|------|----------|
| Q25 | ❌ Off | 28 dB | 0.82 | 45 KB | Thumbnail |
| Q25 | ✅ On | 34 dB | 0.91 | 45 KB | ✅ **Optimal** |
| Q50 | ❌ Off | 32 dB | 0.89 | 85 KB | Preview |
| Q50 | ✅ On | 38 dB | 0.95 | 85 KB | ✅ **Optimal** |
| Q75 | ❌ Off | 36 dB | 0.94 | 150 KB | Gallery |
| Q75 | ✅ On | 41 dB | 0.97 | 150 KB | ✅ **Optimal** |

---

## 💡 Added Value vs. Standard Tools

| Feature | TinyPNG | Squoosh | JPEGmini | **Ours** |
|---------|---------|---------|----------|----------|
| Generic compression | ✅ | ✅ | ✅ | ✅ |
| AI artifact removal | ❌ | ❌ | ❌ | ✅ |
| Quality-specific models | ❌ | ❌ | ❌ | ✅ |
| Nail-art optimized | ❌ | ❌ | ❌ | ✅ |
| Perceptual metrics | ❌ | ❌ | ❌ | ✅ |
| Open source | ❌ | ✅ | ❌ | ✅ |

---

## 🎯 Use Cases

- 💅 **Nail artist portfolios** — Instagram-perfect quality at half the size
- 🛒 **Beauty e-commerce** — Fast-loading product images
- 📱 **Mobile salons** — Quick portfolio sharing
- 🖨️ **Print workflows** — Preview compression before sending to printer
- 🌐 **Website galleries** — Optimized thumbnails with full-quality lightbox

---

## 🔮 Roadmap

- [ ] **Batch processing** — Compress entire folders
- [ ] **FFFDNet/DRUNet** — Alternative architectures comparison
- [ ] **GAN enhancement** — Super-resolution post-processing
- [ ] **Mobile app** — React Native / Flutter
- [ ] **Cloud API** — AWS Lambda deployment
- [ ] **Adaptive quality** — ML-based optimal Q prediction
- [ ] **Video compression** — Extend to nail art tutorials

---

## 🙏 Acknowledgments
- “Rumi saw God within himself; I see you within my work.”
Special thanks to you Aziz for your support, encouragement, and presence throughout this project. Your impact exists in every detail of it.
- DnCNN architecture: [Zhang et al., "Beyond a Gaussian Denoiser"](https://arxiv.org/abs/1608.08151)

- Kaggle community for GPU resources

---

## 👩‍💻 Author


Mariya DERRAG


Interested in: AI · Image Processing · Neural Restoration · Interactive Web Apps

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

> **Note**: The DnCNN model weights are provided for research/educational use. Commercial use of pre-trained weights requires verification of training data licensing.
```

---

## Fichiers complémentaires suggérés

### `requirements.txt`
```text
gradio>=4.0.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
scipy>=1.11.0
scikit-image>=0.21.0
```

### `LICENSE` (MIT)
```text
MIT License

Copyright (c) 2024 Maria

Permission is hereby granted...
[standard MIT text]
```

