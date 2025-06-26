import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# --- Paths ---
data_dir = "augmented_phone_view"
output_features = "clip_features.npy"
output_labels = "clip_labels.npy"

# --- Load CLIP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Label mapping ---
class_names = sorted(os.listdir(data_dir))
label_map = {cls: idx for idx, cls in enumerate(class_names)}

# --- Storage ---
features = []
labels = []

# --- Iterate through folders ---
for cls in tqdm(class_names, desc="🔍 Extracting Features"):
    class_path = os.path.join(data_dir, cls)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_features = clip_model.get_image_features(**inputs)
            features.append(img_features.squeeze().numpy())
            labels.append(label_map[cls])
        except Exception as e:
            print(f"⚠️ Skipped {img_file}: {e}")

# --- Save arrays ---
features = np.array(features)
labels = np.array(labels)

np.save(output_features, features)
np.save(output_labels, labels)

print(f"\n✅ Saved {len(features)} features to {output_features}")
print(f"✅ Saved labels to {output_labels}")
