#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Load model segmentasi
model = YOLO("best.pt")  # ganti ke best.pt hasil training kamu

# Mapping label (urut sesuai data.yaml)
label_map = {
    0: "coral",
    1: "pipeline",
    2: "shipwreck"
}

# Warna RGB untuk mask per kelas
color_map = {
    0: (0, 255, 0),     # hijau
    1: (255, 0, 0),     # merah
    2: (0, 0, 255)      # biru
}

def predict_segmentation_with_legend(image):
    results = model.predict(image, conf=0.5, iou=0.5)
    result = results[0]

    image_np = np.array(image).copy()
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Draw masks
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):
            class_id = class_ids[i]
            label = label_map.get(class_id, str(class_id))
            color = color_map.get(class_id, (255, 255, 0))
            score = scores[i]
            box = boxes[i].astype(int)

            # Blend mask onto image
            color_mask = np.zeros_like(image_np)
            for c in range(3):
                color_mask[:, :, c] = mask * color[c]
            image_np = np.where(color_mask > 0, image_np * 0.5 + color_mask * 0.5, image_np)

            # Draw box
            draw.rectangle(box.tolist(), outline=color, width=3)
            text = f"{label}: {score:.2f}"
            draw.text((box[0], box[1] - 10), text, fill="white", font=font)

    # Convert back to image
    final_image = Image.fromarray(image_np.astype(np.uint8))

    # Add legend (kecil di bawah)
    legend = Image.new("RGB", (400, 50), (255, 255, 255))
    draw_legend = ImageDraw.Draw(legend)
    x = 10
    for cid, label in label_map.items():
        draw_legend.rectangle([x, 10, x+20, 30], fill=color_map[cid])
        draw_legend.text((x + 25, 10), label, fill="black", font=font)
        x += 120

    # Gabungkan hasil dan legend vertikal
    combined = Image.new("RGB", (final_image.width, final_image.height + 50))
    combined.paste(final_image, (0, 0))
    combined.paste(legend, (10, final_image.height))

    return combined

# Interface Gradio
iface = gr.Interface(
    fn=predict_segmentation_with_legend,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="YOLOv11 Segmentasi + Mask + Legend",
    description="Upload gambar sonar, hasil segmentasi ditampilkan dengan mask, kotak, skor dan legenda warna."
)

if __name__ == "__main__":
    iface.launch()

