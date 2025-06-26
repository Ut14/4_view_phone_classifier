import os
import torch
import numpy as np
from PIL import Image
import joblib
from transformers import CLIPProcessor, CLIPModel


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


svm_model = joblib.load("svm_phone_view_model.joblib")


label_map = {0: "back", 1: "bottom", 2: "front", 3: "top"}


def extract_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().numpy()


def predict_view(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    embedding = extract_clip_embedding(image_path)
    probs = svm_model.predict_proba([embedding])[0]
    pred_index = np.argmax(probs)
    prediction = label_map[pred_index]
    confidence = probs[pred_index] * 100

    print(f"\nImage: {image_path}")
    print(f"Predicted View: {prediction.upper()}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    # 👇 Replace with your image path
    test_image_path = "top.jpg"
    predict_view(test_image_path)
