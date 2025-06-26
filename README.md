# 📱 4-View Phone Classifier

A lightweight web application that classifies phone images into **Front**, **Back**, **Top**, or **Bottom** views using a combination of the **CLIP Vision Model** and an **SVM classifier**.

Hosted on [🤗 Hugging Face Spaces](https://huggingface.co/spaces/Ut14/Phone_view_classfier) using **Gradio**.

---

## 🚀 Demo

[![Gradio App](https://img.shields.io/badge/Live%20Demo-Huggingface-blue?logo=huggingface)](https://huggingface.co/spaces/Ut14/Phone_view_classfier)

---

## 📂 Project Structure

```
phone_view_gradio/
├── app.py                        # Gradio web app code
├── svm_phone_view_model.joblib  # Trained SVM model on CLIP features
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 🧠 Model Details

- **Feature Extractor**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Classifier**: Scikit-learn's SVM (`svm_phone_view_model.joblib`)
- **Classes**:
  - `Front`
  - `Back`
  - `Top`
  - `Bottom`

---

## 🖼️ How to Use

1. Upload a phone image showing one of the 4 views.
2. The app returns:
   - Predicted view (`FRONT`, `BACK`, `TOP`, or `BOTTOM`)
   - Confidence score (%)

---

## ⚙️ Installation (For Local Use)

```bash
git clone https://github.com/Ut14/Phone_view_gradio.git
cd Phone_view_gradio
pip install -r requirements.txt
python app.py
```

---

## 🧪 Model Training

The training involved:
- Augmenting image dataset with torchvision
- Extracting CLIP image embeddings
- Training an SVM classifier using Scikit-learn

Scripts used for training (available in the original GitHub repo):
- `augmentation.py`
- `clip_feature_extraction.py`
- `classifier.py`

---

## 🙋‍♂️ Author

**Utkarsh Tripathy**  
🔗 [LinkedIn](https://www.linkedin.com/in/utkarsh-tripathy/) | [GitHub](https://github.com/Ut14)  
📧 ut140203@gmail.com

---

## 📄 License

This project is open source under the MIT License.
