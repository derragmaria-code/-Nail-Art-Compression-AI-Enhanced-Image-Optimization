import gradio as gr
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Try to import LPIPS for perceptual metric
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️ lpips not installed, skipping perceptual metrics")

# ─── CONFIG ───
MODEL_DIR = '/kaggle/input/datasets/mariyyyaaella/dncnn-nail-art-models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Initialize LPIPS if available
if LPIPS_AVAILABLE:
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)

# ─── DnCNN MODEL (20 layers to match saved weights) ───
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=20):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)

# ─── LOAD MODELS ───
quality_map = {10: 'q10', 25: 'q25', 50: 'q50', 75: 'q75'}
models = {}

def load_model(quality):
    if quality in models:
        return models[quality]

    q_key = quality_map.get(quality, 'q50')
    model_path = os.path.join(MODEL_DIR, f'dncnn_pq_{q_key}.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = DnCNN(channels=1, num_layers=20).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)

    # Renommer net. → dncnn.
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('net.', 'dncnn.')
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    models[quality] = model
    print(f"✅ Loaded model Q{quality}")
    return model

# ─── SMART RESIZE (Preserve nail scale) ───
def smart_resize(img, max_dim=512, min_dim=64):
    """Preserve nail scale: don't upscale small images, don't downscale too hard."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_w = max(int(w * scale), min_dim)
    new_h = max(int(h * scale), min_dim)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ─── ROI CROP (Center region = nail zone) ───
def get_nail_roi(img_array, roi_ratio=0.6):
    """Crop center region where nail is typically located."""
    h, w = img_array.shape[:2]
    ch, cw = int(h * roi_ratio), int(w * roi_ratio)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    return img_array[y1:y1+ch, x1:x1+cw]

# ─── REAL JPEG COMPRESSION WITH ENTROPY CODING ───
def real_jpeg_compress(img_array, quality, chroma_subsampling='4:2:0'):
    """Compress using actual JPEG encoder with proper entropy coding."""
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    if chroma_subsampling == '4:4:4':
        encode_params.extend([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    elif chroma_subsampling == '4:2:2':
        encode_params.extend([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422])

    success, buffer = cv2.imencode('.jpg', bgr, encode_params)
    if not success:
        raise RuntimeError("JPEG encoding failed")

    file_size_bytes = len(buffer)
    decoded_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_array.shape[:2]
    actual_bpp = (file_size_bytes * 8) / (h * w)

    return decoded_rgb, file_size_bytes, actual_bpp

# ─── DnCNN ON Y CHANNEL (YCbCr) ───
def apply_dncnn_ycbcr(img_array, quality):
    """Apply DnCNN on Y channel only, preserve CbCr."""
    model = load_model(quality)
    ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    y_float = y.astype(np.float32) / 255.0
    tensor = torch.from_numpy(y_float).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        denoised_y = model(tensor)

    y_denoised = np.clip(denoised_y.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    result_ycbcr = cv2.merge([y_denoised, cr, cb])
    result_rgb = cv2.cvtColor(result_ycbcr, cv2.COLOR_YCrCb2RGB)

    return np.clip(result_rgb, 0, 255).astype(np.uint8)

# ─── MULTI-SCALE DnCNN ───
def apply_dncnn_multiscale(img_array, quality, blend_orig=0.7):
    """Blend original scale + half scale for better artifact handling."""
    h, w = img_array.shape[:2]
    orig = apply_dncnn_ycbcr(img_array, quality)

    if min(h, w) < 128:
        return orig

    small = cv2.resize(img_array, (w//2, h//2), interpolation=cv2.INTER_LANCZOS4)
    small_denoised = apply_dncnn_ycbcr(small, quality)
    small_up = cv2.resize(small_denoised, (w, h), interpolation=cv2.INTER_LANCZOS4)

    blended = (blend_orig * orig.astype(np.float32) + 
               (1 - blend_orig) * small_up.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)

# ─── NAIL-SHARPENING ───
def enhance_nail_details(img_array, strength=0.3):
    """Mild unsharp mask to recover edge crispness after DnCNN."""
    pil_img = Image.fromarray(img_array)
    sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return np.array(Image.blend(pil_img, sharpened, strength))

# ─── ZOOM COMPARISON ───
def make_comparison(orig, comp, zoom=3):
    """Create side-by-side zoom of nail ROI for visual proof."""
    roi_orig = get_nail_roi(orig, roi_ratio=0.5)
    roi_comp = get_nail_roi(comp, roi_ratio=0.5)

    # Nearest neighbor zoom to see pixel-level artifacts
    big_orig = cv2.resize(roi_orig, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    big_comp = cv2.resize(roi_comp, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

    # Add labels
    h, w = big_orig.shape[:2]
    label_h = 30
    canvas = np.ones((h + label_h, w * 2, 3), dtype=np.uint8) * 255

    canvas[label_h:, :w] = big_orig
    canvas[label_h:, w:] = big_comp

    cv2.putText(canvas, "ORIGINAL", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(canvas, "COMPRESSE", (w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return canvas

# ─── METRICS (Global + ROI Nail) ───
def compute_metrics(original, compressed, file_size_bytes):
    orig_float = original.astype(np.float64) / 255.0
    comp_float = compressed.astype(np.float64) / 255.0
    h, w = original.shape[:2]

    # GLOBAL metrics
    psnr_vals = [psnr(orig_float[:,:,c], comp_float[:,:,c], data_range=1.0) for c in range(3)]
    ssim_vals = [ssim(orig_float[:,:,c], comp_float[:,:,c], data_range=1.0) for c in range(3)]

    psnr_val = np.mean(psnr_vals)
    ssim_val = np.mean(ssim_vals)
    bpp = (file_size_bytes * 8) / (h * w)
    file_size_kb = file_size_bytes / 1024
    compression_ratio = original.nbytes / file_size_bytes

    # ROI NAIL metrics (proof of nail-scale quality)
    orig_roi = get_nail_roi(original, roi_ratio=0.5)
    comp_roi = get_nail_roi(compressed, roi_ratio=0.5)
    orig_roi_f = orig_roi.astype(np.float64) / 255.0
    comp_roi_f = comp_roi.astype(np.float64) / 255.0

    psnr_roi = np.mean([psnr(orig_roi_f[:,:,c], comp_roi_f[:,:,c], data_range=1.0) for c in range(3)])
    ssim_roi = np.mean([ssim(orig_roi_f[:,:,c], comp_roi_f[:,:,c], data_range=1.0) for c in range(3)])

    metrics = {
        '📊 PSNR (global)': f"{psnr_val:.2f} dB",
        '📊 SSIM (global)': f"{ssim_val:.4f}",
        '🎯 PSNR (ongle)': f"{psnr_roi:.2f} dB",
        '🎯 SSIM (ongle)': f"{ssim_roi:.4f}",
        '💾 Taille': f"{file_size_kb:.1f} KB",
        '📉 BPP': f"{bpp:.2f} bits/px",
        '📈 Ratio': f"{compression_ratio:.1f}:1"
    }

    # LPIPS global + ROI
    if LPIPS_AVAILABLE:
        orig_t = torch.from_numpy(orig_float).permute(2,0,1).unsqueeze(0).to(DEVICE) * 2 - 1
        comp_t = torch.from_numpy(comp_float).permute(2,0,1).unsqueeze(0).to(DEVICE) * 2 - 1
        with torch.no_grad():
            lpips_g = loss_fn_vgg(orig_t, comp_t).item()

        orig_roi_t = torch.from_numpy(orig_roi_f).permute(2,0,1).unsqueeze(0).to(DEVICE) * 2 - 1
        comp_roi_t = torch.from_numpy(comp_roi_f).permute(2,0,1).unsqueeze(0).to(DEVICE) * 2 - 1
        with torch.no_grad():
            lpips_roi = loss_fn_vgg(orig_roi_t, comp_roi_t).item()

        metrics['🧠 LPIPS (global)'] = f"{lpips_g:.4f}"
        metrics['🧠 LPIPS (ongle)'] = f"{lpips_roi:.4f}"

    return metrics

# ─── MAIN PROCESSING ───
def compress_image(input_image, quality_choice, use_dncnn, use_multiscale, use_sharpen, chroma_sub):
    if input_image is None:
        return None, None, "Aucune image chargée", ""

    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image).convert('RGB')
    else:
        img = input_image.convert('RGB')

    # Smart resize (preserve nail scale)
    img = smart_resize(img, max_dim=512, min_dim=64)
    img_array = np.array(img)

    quality_map_choice = {
        "Basique (Q25)": 25,
        "Économique (Q50)": 50,
        "Standard (Q75)": 75,
        "Premium (Q90)": 90
    }
    quality = quality_map_choice.get(quality_choice, 50)

    # Real JPEG compression
    compressed, file_size_bytes, actual_bpp = real_jpeg_compress(
        img_array, quality, chroma_subsampling=chroma_sub
    )

    # Apply DnCNN
    if use_dncnn:
        try:
            if use_multiscale:
                compressed = apply_dncnn_multiscale(compressed, quality)
            else:
                compressed = apply_dncnn_ycbcr(compressed, quality)

            if use_sharpen:
                compressed = enhance_nail_details(compressed, strength=0.2)
        except Exception as e:
            print(f"⚠️ DnCNN failed: {e}")

    # Metrics with ROI proof
    metrics = compute_metrics(img_array, compressed, file_size_bytes)
    metrics_text = "\n".join([f"**{k}:** {v}" for k, v in metrics.items()])

    # Comparison zoom for visual proof
    comparison = make_comparison(img_array, compressed, zoom=3)

    # Processing info
    proc_info = f"Qualité: {quality_choice} | DnCNN: {'OUI' if use_dncnn else 'NON'} | Multi-échelle: {'OUI' if use_multiscale else 'NON'} | Chroma: {chroma_sub}"

    result_img = Image.fromarray(compressed)
    comparison_img = Image.fromarray(comparison)

    return result_img, comparison_img, metrics_text, proc_info

# ─── GRADIO INTERFACE ───
with gr.Blocks(title="💅 Nail Art Compression Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💅 Nail Art Compression Pro")
    gr.Markdown("**Preuve de qualité à l'échelle ongle** — métriques ROI + zoom comparatif")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(
                label="📸 Image originale",
                type="pil",
                image_mode="RGB",
                height=350
            )

            quality_dropdown = gr.Dropdown(
                choices=["Basique (Q25)", "Économique (Q50)", "Standard (Q75)", "Premium (Q90)"],
                value="Économique (Q50)",
                label="Niveau de compression JPEG"
            )

            chroma_dropdown = gr.Dropdown(
                choices=["4:2:0", "4:2:2", "4:4:4"],
                value="4:2:0",
                label="Sous-échantillonnage chroma",
                info="4:4:4 = meilleure couleur, 4:2:0 = plus compact"
            )

            with gr.Row():
                dncnn_checkbox = gr.Checkbox(
                    label="🤖 DnCNN (Y channel)",
                    value=True,
                    info="Débruitage sur canal luminance"
                )
                multiscale_checkbox = gr.Checkbox(
                    label="🔍 Multi-échelle",
                    value=True,
                    info="Fusion original + demi-échelle"
                )
                sharpen_checkbox = gr.Checkbox(
                    label="✨ Sharpen",
                    value=False,
                    info="Unsharp mask léger post-DnCNN"
                )

            compress_btn = gr.Button("🚀 Compresser", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_img = gr.Image(
                label="🖼️ Image compressée",
                height=350
            )

            comparison_img = gr.Image(
                label="🔬 Zoom comparatif (zone ongle)",
                height=250
            )

            status_text = gr.Textbox(
                label="⚙️ Paramètres",
                interactive=False
            )

            metrics_box = gr.Markdown("### 📊 Métriques\n*En attente de compression...*")

    # Auto-compress on change
    inputs = [input_img, quality_dropdown, dncnn_checkbox, multiscale_checkbox, sharpen_checkbox, chroma_dropdown]
    outputs = [output_img, comparison_img, metrics_box, status_text]

    input_img.change(fn=compress_image, inputs=inputs, outputs=outputs)
    quality_dropdown.change(fn=compress_image, inputs=inputs, outputs=outputs)
    dncnn_checkbox.change(fn=compress_image, inputs=inputs, outputs=outputs)
    multiscale_checkbox.change(fn=compress_image, inputs=inputs, outputs=outputs)
    sharpen_checkbox.change(fn=compress_image, inputs=inputs, outputs=outputs)
    chroma_dropdown.change(fn=compress_image, inputs=inputs, outputs=outputs)
    compress_btn.click(fn=compress_image, inputs=inputs, outputs=outputs)

# Launch
demo.launch(share=True, debug=True)
