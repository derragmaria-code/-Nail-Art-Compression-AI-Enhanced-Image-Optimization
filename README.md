# 💅 Nail Art Compression

### AI-Assisted Compression & Restoration for Beauty Imagery

An interactive AI-powered image compression system designed specifically for **nail art and beauty images**.
The project combines **JPEG compression** with **deep learning-based artifact reduction (DnCNN)** to preserve fine artistic details while reducing file size.

---

# ✨ Overview

Traditional image compression often damages:

* fine nail patterns
* glitter textures
* decorative micro-details
* smooth gradients
* reflective surfaces

This project focuses on preserving the **visual aesthetics** of nail art imagery while optimizing images for:

* web uploads
* social media
* mobile devices
* online portfolios
* printing workflows

---

# 🚀 Features

* 📸 Upload custom nail art images
* 🗜️ Adjustable JPEG compression quality
* 🤖 AI-based post-processing using DnCNN
* 🔍 Before/After comparison viewer
* 📊 Quality metrics:

  * PSNR
  * SSIM
  * BPP
  * File size
* 💾 Download compressed images
* 🌐 Interactive web interface
* ⚡ FastAPI backend + React frontend

---

# 🧠 Pipeline

```text
Input Image
     ↓
JPEG Compression
     ↓
AI Restoration (DnCNN)
     ↓
Optimized Output Image
```

The neural restoration stage helps reduce:

* JPEG blocking artifacts
* ringing effects
* excessive smoothing
* texture degradation

---

# 🛠️ Technologies Used

## Backend

* FastAPI
* PyTorch
* OpenCV
* NumPy
* Pillow
* Uvicorn

## Frontend

* React
* HTML/CSS
* JavaScript
* Babel

## Deployment

* ngrok
* Google Colab
* Gradio

---

# 📂 Project Structure

```text
project/
│
├── backend.py
├── models/
│   ├── dncnn_q25.pt
│   ├── dncnn_q50.pt
│   ├── dncnn_q75.pt
│   └── dncnn_q90.pt
│
├── frontend/
│   └── index.html
│
├── utils/
│   ├── metrics.py
│   └── preprocessing.py
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone repository

```bash
git clone https://github.com/yourusername/nail-art-compression.git
cd nail-art-compression
```

---

## 2. Install dependencies

```bash
pip install fastapi uvicorn torch torchvision opencv-python numpy pillow pyngrok
```

---

# ▶️ Running the Backend

Start FastAPI server:

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

---

# 🌐 Expose Public URL (ngrok)

```python
from pyngrok import ngrok

public_url = ngrok.connect(8000)
print(public_url)
```

Use the generated URL inside the frontend:

```javascript
const backendUrl = "YOUR_NGROK_URL";
```

---

# 📊 Metrics

The system evaluates image quality using:

| Metric    | Description                 |
| --------- | --------------------------- |
| PSNR      | Peak Signal-to-Noise Ratio  |
| SSIM      | Structural Similarity Index |
| BPP       | Bits Per Pixel              |
| File Size | Final compressed image size |

---

# 💡 Added Value

Unlike generic image compressors, this project is designed specifically for:

```text
detail-sensitive cosmetic and nail-art imagery
```

The system prioritizes:

* artistic detail preservation
* perceptual image quality
* texture clarity
* aesthetic appearance

rather than focusing only on compression ratio.

---

# 🎯 Use Cases

* Nail artist portfolios
* Instagram optimization
* Beauty ecommerce platforms
* Mobile-friendly uploads
* Online galleries
* Print preview workflows

---

# 🔮 Future Improvements

* Batch image compression
* Mobile application
* GAN-based restoration
* Adaptive quality prediction
* Cloud deployment
* Real-time processing

---

# 👩‍💻 Author

Mariya DERRAG

Interested in:

* AI
* image processing
* neural restoration
* compression systems
* interactive web applications

---

# 📜 License

MIT License


