import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from unet import UNET

# ----------------------------
# Konfigurasi & Load Model
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_road_extraction.pth"
IMAGE_SIZE = (512, 512)
GSD = 0.5  # Ground Sampling Distance (meter/pixel)

# Load model
model = UNET(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----------------------------
# Fungsi Estimasi Jarak Jalan
# ----------------------------
def calculate_road_length(mask_tensor):
    mask_np = mask_tensor.squeeze().cpu().numpy()
    binary_mask = (mask_np > 0).astype(np.uint8)
    skeleton = skeletonize(binary_mask)
    pixel_count = np.count_nonzero(skeleton)
    length_meters = pixel_count * GSD
    return length_meters, skeleton

# ----------------------------
# Fungsi Gradio
# ----------------------------
def predict(image: Image.Image):
    # Preprocessing input
    image_np = np.array(image.convert("RGB"))
    image_resized = cv2.resize(image_np, IMAGE_SIZE)
    image_norm = image_resized / 255.0
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.no_grad():
        output = model(image_tensor)
        output_binary = (output > 0.5).float()

    # Estimasi panjang jalan
    length_m, skeleton = calculate_road_length(output_binary)

    # Konversi ke gambar untuk ditampilkan
    pred_mask = output_binary.squeeze().cpu().numpy() * 255
    skeleton_img = skeleton.astype(np.uint8) * 255

    # Gabungkan hasil ke 3 kolom
    input_disp = cv2.resize(image_np, IMAGE_SIZE)
    mask_disp = cv2.cvtColor(pred_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    skel_disp = cv2.cvtColor(skeleton_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    concat = np.concatenate([input_disp, mask_disp, skel_disp], axis=1)

    # Konversi ke PIL
    result_img = Image.fromarray(concat)
    result_text = f"Estimasi Panjang Jalan: {length_m:.2f} meter"
    return result_img, result_text

# ----------------------------
# Gradio UI
# ----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Unggah Citra Jalan"),
    outputs=[
        gr.Image(type="pil", label="Citra | Prediksi Masker | Skeleton"),
        gr.Textbox(label="Estimasi Panjang Jalan")
    ],
    title="Segmentasi Jalan dengan UNET",
    description="Aplikasi ini memprediksi peta jalan dari citra satelit dan mengestimasi total panjang jalan yang terdeteksi."
)

if __name__ == "__main__":
    demo.launch()
