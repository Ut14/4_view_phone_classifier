# app.py
import os
import torch
import numpy as np
from PIL import Image
import joblib
import gradio as gr
from transformers import CLIPProcessor, CLIPModel

# --- Load CLIP Model and Processor ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Load Trained SVM Model ---
svm_model = joblib.load("svm_phone_view_model.joblib")

# --- Label Mapping ---
label_map = {0: "back", 1: "bottom", 2: "front", 3: "top"}

# --- Function to Extract CLIP Embedding ---
def extract_clip_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().numpy()

# --- Gradio prediction function ---
def predict_image_view(image):
    embedding = extract_clip_embedding(image)
    probs = svm_model.predict_proba([embedding])[0]
    pred_index = np.argmax(probs)
    prediction = label_map[pred_index]
    confidence = probs[pred_index] * 100
    return f"View: {prediction.upper()} ({confidence:.2f}%)"

# --- Launch Gradio interface ---
demo = gr.Interface(
    fn=predict_image_view,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Phone View Classifier (4-class)",
    description="Upload an image of a phone and classify it as one of: Front, Back, Top, Bottom"
)

if __name__ == "__main__":
    demo.launch()
