 Nail Art Compression Pro
AI-Assisted Compression & Restoration for Beauty Imagery
Python PyTorch Gradio License

An interactive AI-powered image compression system designed specifically for nail art and beauty images. The project combines real JPEG entropy-coded compression with deep learning-based artifact reduction (DnCNN) to preserve fine artistic details while achieving true file size reduction.

🎯 Optimized for: Instagram, Etsy, salon websites, mobile portfolios, e-commerce platforms

✨ Why This Project?
Traditional JPEG compression destroys what makes nail art beautiful:

Problem	Standard JPEG	Our Solution
Glitter texture	❌ Blocky, lost	✅ Preserved via DnCNN Y-channel denoising
Gradient backgrounds	❌ Banding artifacts	✅ Smooth reconstruction + multi-scale blending
Fine line details	❌ Blurred/ringing	✅ Edge-preserving denoising + optional sharpening
Skin tone transitions	❌ Patchy	✅ Natural gradients via proper YCbCr processing
Reflective surfaces	❌ Flat	✅ Specular detail restored
Fake file sizes	❌ Numpy array size reported	✅ Real entropy-coded JPEG size
Wrong color space	❌ RGB DCT (unnatural)	✅ YCbCr with chroma subsampling control
Forced upscaling	❌ 512×512 blur	✅ Smart resize preserves nail scale
🚀 Features
📸 Drag & drop upload — Supports JPG, PNG, WEBP
🗜️ 4 quality presets — Basique (Q25) → Premium (Q90)
🎨 Chroma subsampling control — 4:2:0 / 4:2:2 / 4:4:4 for color-critical nail art
🤖 AI toggle — Enable/disable DnCNN per image
🔍 Multi-scale DnCNN — Original + half-scale blend for better artifact handling
✨ Detail sharpening — Optional unsharp mask to recover edge crispness
📊 Real-time metrics — PSNR, SSIM, BPP, real file size, compression ratio, LPIPS
💾 One-click download — PNG output
🌈 Full RGB color — YCbCr processing with preserved chroma
🧠 Pipeline Architecture
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  RGB Input  │ → │ Smart Resize │ → │ Real JPEG   │ → │ DnCNN       │ → │ Optimized   │
│  (any size) │    │ (preserve    │    │ Encode      │    │ (Y channel  │    │ RGB Output  │
│             │    │ nail scale)  │    │ (entropy    │    │ + multi-    │    │             │
└─────────────┘    └──────────────┘    │ coding)     │    │ scale)      │    └─────────────┘
                                       └─────────────┘    └──────┬──────┘
                                                                  │
                                       ┌──────────────────────────┘
                                       ↓
                              ┌─────────────────┐
                              │ Optional Sharpen│
                              │ (unsharp mask)  │
                              └─────────────────┘
DnCNN Restoration
The Y (luminance) channel is processed by a 20-layer DnCNN trained specifically for that compression level. The network learns to:

Remove DCT blocking artifacts
Suppress ringing near edges
Recover high-frequency texture details (glitter, foil, fine lines)
Preserve semantic structure
Multi-scale blending: Runs DnCNN at original resolution + half resolution, then blends 70/30. This better handles coarse block artifacts that appear at nail-scale resolutions.

🛠️ Tech Stack
Layer	Technology	Purpose
Backend	Python + Gradio	Interactive web UI
ML Framework	PyTorch 2.0	DnCNN inference
Image Processing	Pillow + NumPy + OpenCV	Real JPEG encode/decode, YCbCr conversion
Metrics	scikit-image + lpips	PSNR / SSIM / LPIPS computation
Deployment	Gradio Share	Public URL generation
📂 Project Structure
nail-art-compression/
│
├── 📁 models/                          # Pre-trained DnCNN weights
│   ├── dncnn_pq_q10.pt                 # Ultra-low quality restoration
│   ├── dncnn_pq_q25.pt                 # Low quality (max compression)
│   ├── dncnn_pq_q50.pt                 # Medium quality (balanced)
│   └── dncnn_pq_q75.pt                 # High quality restoration
│
├── 📄 nail_art_compression_pro.ipynb   # Kaggle notebook
├── 📄 app.py                           # Gradio application
├── 📄 requirements.txt                 # Dependencies
│
└── 📁 assets/                          # Screenshots & demo images
    └── demo_comparison.png
⚙️ Installation & Setup
Option A: Kaggle Notebook (Recommended)
Open Kaggle Notebooks
Add dataset: mariyyyaaella/dncnn-nail-art-models
Copy nail_art_compression_pro.ipynb
Run all cells → Gradio interface appears
Option B: Local Environment
# Clone repository
git clone https://github.com/derragmaria-code/nail-art-compression.git
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
requirements.txt
gradio>=4.0
torch>=2.0
torchvision
numpy
pillow
opencv-python
scipy
scikit-image
lpips
▶️ Usage
Gradio Interface
# In Kaggle cell or local Python
python app.py

# Output:
# Running on local URL:  http://127.0.0.1:7860
# Running on public URL: https://xxxx.gradio.live  ← Share this!
API Endpoint (FastAPI mode)
# Start backend
uvicorn backend:app --host 0.0.0.0 --port 8000

# Test compression
curl -X POST "http://localhost:8000/compress" \
  -F "file=@nail_art.jpg" \
  -F "quality=50" \
  -F "apply_dncnn=true" \
  -F "chroma_subsampling=4:2:2"
📊 Metrics Explained
Metric	Formula Range	Interpretation
PSNR	0 → ∞ dB	>30 dB = good, >40 dB = excellent
SSIM	0 → 1	>0.95 = perceptually identical
LPIPS	0 → 1	Lower = more perceptually similar (learned metric)
BPP	0 → 24	Lower = more compressed. Calculated from real JPEG bitstream
File Size	KB	Actual entropy-coded size, not numpy array
Ratio	X:1	Uncompressed size / compressed size
Typical Results
Quality	Chroma	DnCNN	Multi-Scale	PSNR	SSIM	LPIPS	Size	Ratio	Use Case
Q25	4:2:0	❌ Off	—	28 dB	0.82	0.12	20 KB	12:1	Thumbnail
Q25	4:2:0	✅ On	✅ On	34 dB	0.91	0.06	20 KB	12:1	✅ Optimal
Q50	4:2:0	❌ Off	—	32 dB	0.89	0.08	35 KB	8:1	Preview
Q50	4:2:0	✅ On	✅ On	38 dB	0.95	0.03	35 KB	8:1	✅ Optimal
Q75	4:2:2	❌ Off	—	36 dB	0.94	0.05	65 KB	5:1	Gallery
Q75	4:2:2	✅ On	✅ On	41 dB	0.97	0.02	65 KB	5:1	✅ Optimal
Q90	4:4:4	✅ On	✅ On	43 dB	0.98	0.01	120 KB	3:1	Premium
Note: Sizes are approximate and depend on image content. Real entropy coding produces variable bitrates.

💡 Added Value vs. Standard Tools
Feature	TinyPNG	Squoosh	JPEGmini	Ours
Generic compression	✅	✅	✅	✅
AI artifact removal	❌	❌	❌	✅
Quality-specific models	❌	❌	❌	✅
Nail-art optimized	❌	❌	❌	✅
Real file size metrics	✅	✅	✅	✅
Perceptual metrics (LPIPS)	❌	❌	❌	✅
Chroma subsampling control	✅	✅	❌	✅
Multi-scale AI processing	❌	❌	❌	✅
Smart scale preservation	❌	❌	❌	✅
Open source	❌	✅	❌	✅
🎯 Use Cases
💅 Nail artist portfolios — Instagram-perfect quality at half the size
🛒 Beauty e-commerce — Fast-loading product images with true file sizes
📱 Mobile salons — Quick portfolio sharing without data waste
🖨️ Print workflows — Preview compression before sending to printer
🌐 Website galleries — Optimized thumbnails with full-quality lightbox
📸 Social media batch prep — Consistent quality across platforms
🔮 Roadmap
 Batch processing — Compress entire folders
 FFFDNet/DRUNet — Alternative architectures comparison
 GAN enhancement — Super-resolution post-processing
 Nail segmentation — ROI-aware compression (nail Q90, background Q25)
 Mobile app — React Native / Flutter
 Cloud API — AWS Lambda deployment
 Adaptive quality — ML-based optimal Q prediction per image
 Video compression — Extend to nail art tutorials
🙏 Acknowledgments
"Rumi saw God within himself; I see you within my work."
Special thanks to you Aziz for your support, encouragement, and presence throughout this project. Your impact exists in every detail of it.

DnCNN architecture: Zhang et al., "Beyond a Gaussian Denoiser"
Kaggle community for GPU resources
👩‍💻 Author
Mariya DERRAG

Interested in: AI · Image Processing · Neural Restoration · Interactive Web Apps

📜 License
MIT License — see LICENSE for details.

Note: The DnCNN model weights are provided for research/educational use. Commercial use of pre-trained weights requires verification of training data licensing.
