import os
from PIL import Image
import torchvision.transforms as T
import random


input_root = "phone_view"                
output_root = "augmented_phone_view"    
os.makedirs(output_root, exist_ok=True)


augment = T.Compose([
    T.Resize((224, 224)), 
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
])


AUG_PER_IMAGE = 10


for label in os.listdir(input_root):
    class_input_path = os.path.join(input_root, label)
    class_output_path = os.path.join(output_root, label)
    os.makedirs(class_output_path, exist_ok=True)

    for img_file in os.listdir(class_input_path):
        img_path = os.path.join(class_input_path, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            base_name = os.path.splitext(img_file)[0]

            img_resized = img.resize((224, 224))
            img_resized.save(os.path.join(class_output_path, f"{base_name}_orig.jpg"))

            for i in range(AUG_PER_IMAGE):
                aug_img = augment(img)
                aug_img.save(os.path.join(class_output_path, f"{base_name}_aug{i}.jpg"))

            print(f"✅ Processed: {img_file} in '{label}'")
        except Exception as e:
            print(f"⚠️ Error with {img_file}: {e}")
