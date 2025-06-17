# diffusion_train.py
import os
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -----------------------
# 1) CONFIGURATION
# -----------------------
DATA_DIR      = r"C:\Users\squddus\Documents\Radar-Fall-Data-Generation--Standardized-Dataset-\IEEE Radar Dataset\dataset\fall"
# Save model checkpoint to specified directory
SAVE_MODEL    = r"C:\Users\squddus\Documents\Radar-Fall-Data-Generation--Standardized-Dataset-\fall_ddpm_unet.pt"
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE    = 64
IMAGE_SIZE    = 112                # adjust to your spectrogram resolution
CHANNELS      = 1                  # grayscale
LR            = 1e-4
EPOCHS        = 250

# Diffusion hyperparameters
TIMESTEPS     = 1000
BETA_START    = 1e-4
BETA_END      = 0.02

# -----------------------
# 2) DATASET DEFINITION
# -----------------------
# Top-level scaling helper to avoid pickle issues
def scale_to_neg1_pos1(tensor):
    return tensor * 2 - 1

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, image_size):
        # load all PNGs in root_dir (only fall images)
        self.paths = glob(os.path.join(root_dir, '*.png'))
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),               # maps [0,255] to [0,1]
            transforms.Normalize((0.5,), (0.5,)), # maps [0,1] to [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img)

# -----------------------
# 3) UNet ARCHITECTURE
# -----------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, c_in, c_base=64):
        super().__init__()
        # Downsampling
        self.conv1 = conv_block(c_in,  c_base)
        self.conv2 = conv_block(c_base, c_base*2)
        self.conv3 = conv_block(c_base*2, c_base*4)
        self.conv4 = conv_block(c_base*4, c_base*8)
        self.pool  = nn.MaxPool2d(2)
        # Upsampling
        self.up3   = nn.ConvTranspose2d(c_base*8, c_base*4, 2, stride=2)
        self.conv_up3 = conv_block(c_base*8, c_base*4)
        self.up2   = nn.ConvTranspose2d(c_base*4, c_base*2, 2, stride=2)
        self.conv_up2 = conv_block(c_base*4, c_base*2)
        self.up1   = nn.ConvTranspose2d(c_base*2, c_base, 2, stride=2)
        self.conv_up1 = conv_block(c_base*2, c_base)
        # Final
        self.out = nn.Conv2d(c_base, c_in, 1)

    def forward(self, x, t_embed):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        # Decoder with skip connections
        u3 = self.up3(x4)
        x3 = torch.cat([u3, x3], dim=1)
        x3 = self.conv_up3(x3)
        u2 = self.up2(x3)
        x2 = torch.cat([u2, x2], dim=1)
        x2 = self.conv_up2(x2)
        u1 = self.up1(x2)
        x1 = torch.cat([u1, x1], dim=1)
        x1 = self.conv_up1(x1)
        return self.out(x1)

# -----------------------
# 4) DIFFUSION SCHEDULE
# -----------------------
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alpha_cum = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise=None):
    """Generate noisy sample at step t"""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = alpha_cum[t].sqrt().view(-1,1,1,1)
    sqrt_mb = (1 - alpha_cum[t]).sqrt().view(-1,1,1,1)
    return sqrt_ab * x0 + sqrt_mb * noise

# -----------------------
# 5) MODEL + OPTIMIZER
# -----------------------
model = UNet(c_in=CHANNELS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------
# 6) MAIN TRAINING FUNCTION
# -----------------------
def main():
    dataset = SpectrogramDataset(DATA_DIR, IMAGE_SIZE)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs in loop:
            imgs = imgs.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (imgs.size(0),), device=DEVICE).long()
            noise = torch.randn_like(imgs)
            x_noisy = q_sample(imgs, t, noise)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), SAVE_MODEL)
    print("Training complete. Model saved to", SAVE_MODEL)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
