# inference_with_preprocessing.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix

# -----------------------
# 1) CONFIGURATION
# -----------------------
MODEL_PATH  = r"E:\BHI Paper Stuff\Results\vgg16_fall_classifier_3000gen.pt"
FOLDER_PATH = r'E:\BHI Paper Stuff\Code and Misc Files\Real World Fall Data Processed'
BATCH_SIZE  = 8
IMAGE_SIZE  = (224, 224)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# 2) MODEL SETUP
# -----------------------
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------
# 3) TRANSFORM
# -----------------------
common_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------
# 4) PREPROCESSING HELPER
# -----------------------
def preprocess_image(path):
    # Load grayscale spectrogram
    img = Image.open(path).convert('L')
    w, h = img.size
    # ADL segments are recorded at double length → compress horizontally by factor 2
    if 'adl' in os.path.basename(path).lower():
        img = img.resize((w // 2, h), resample=Image.BILINEAR)
    # Convert back to 3-channel RGB for VGG
    img = img.convert('RGB')
    # Finally, resize to network input
    img = img.resize(IMAGE_SIZE, resample=Image.BILINEAR)
    return common_transform(img)

# -----------------------
# 5) INFERENCE & METRICS
# -----------------------
def evaluate_folder(folder_path):
    true_labels = []
    preds = []
    paths = []

    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
    # Batch processing
    for i in range(0, len(file_list), BATCH_SIZE):
        batch_files = file_list[i:i + BATCH_SIZE]
        imgs = []
        for fname in batch_files:
            path = os.path.join(folder_path, fname)
            imgs.append(preprocess_image(path))
            true_labels.append(1 if 'fall' in fname.lower() else 0)
            paths.append(fname)

        batch_tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            out = model(batch_tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
            preds.extend(batch_preds.tolist())

    # Metrics
    acc = np.mean(np.array(preds) == np.array(true_labels)) * 100
    print(f"Accuracy: {acc:.2f}%")
    cm = confusion_matrix(true_labels, preds)
    print("Confusion Matrix:\n", cm)

    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ADL','Fall'], yticklabels=['ADL','Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_folder(FOLDER_PATH)
