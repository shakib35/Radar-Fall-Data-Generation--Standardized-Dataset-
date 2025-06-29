# vgg_transfer_binary.py
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import vgg16
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# -----------------------
# 1) CONFIGURATION
# -----------------------
# Absolute path to your dataset root (must contain 'Fall' and 'ADL' subfolders)
DATA_DIR         = r"E:\BHI Paper Stuff\3000 Generated Samples"
# Absolute path to save the trained model
SAVE_MODEL_PATH  = r"E:\BHI Paper Stuff\Results\vgg16_fall_classifier_3000gen.pt"
# Metrics output file (will be created in current working directory or absolute path if you change it)
METRICS_FILE     = r"E:\BHI Paper Stuff\Results\vgg16_generated_metrics_3000gen.txt"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
IMAGE_SIZE = 224  # resize all images to 112x112
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# -----------------------
# 2) DATA TRANSFORMS
# -----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # replicate single-channel to 3
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------
# 3) TRAIN + EVALUATE FUNCTION
# -----------------------
def train_and_evaluate():
    # Load dataset
    full_dataset = ImageFolder(DATA_DIR, transform=transform)
    class_names = full_dataset.classes

    # Split train/validation
    total = len(full_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Initialize pretrained VGG16
    model = vgg16(pretrained=True)
    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace final classifier layer
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Clear metrics file
    open(METRICS_FILE, 'w').close()

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / train_size

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        # Collect metrics
        report = classification_report(all_labels, all_preds, target_names=class_names)
        cm = confusion_matrix(all_labels, all_preds)
        with open(METRICS_FILE, 'a') as f:
            f.write(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f}\n")
            if epoch == EPOCHS:
                f.write("\nClassification Report:\n")
                f.write(report + "\n")
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + "\n")
                f.write("="*40 + "\n")

        print(f"Epoch {epoch} complete. Loss: {epoch_loss:.4f}")

    # Save the final trained model
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Training complete. Model saved to {SAVE_MODEL_PATH}")
    print(f"Metrics saved to {METRICS_FILE}")

# -----------------------
# 4) MAIN
# -----------------------
if __name__ == '__main__':
    train_and_evaluate()
