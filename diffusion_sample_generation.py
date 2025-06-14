# diffusion_sample.py

import os
from glob import glob
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

# -----------------------
# 1) CONFIGURATION
# -----------------------
MODEL_DIR   = r"C:\Users\squddus\Documents\Radar-Fall-Data-Generation--Standardized-Dataset-"
MODEL_PATH  = os.path.join(MODEL_DIR, "fall_ddpm_unet.pt")
OUTPUT_DIR  = os.path.join(MODEL_DIR, "generated_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE  = 128      # must match training
CHANNELS    = 1
TIMESTEPS   = 1000

# -----------------------
# 2) UNet ARCHITECTURE
#    (should match your training script)
# -----------------------
import torch
from torchvision import transforms
from PIL import Image

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
        # Down
        self.conv1 = conv_block(c_in,  c_base)
        self.conv2 = conv_block(c_base, c_base*2)
        self.conv3 = conv_block(c_base*2, c_base*4)
        self.conv4 = conv_block(c_base*4, c_base*8)
        self.pool  = nn.MaxPool2d(2)
        # Up
        self.up3   = nn.ConvTranspose2d(c_base*8, c_base*4, 2, stride=2)
        self.conv_up3 = conv_block(c_base*8, c_base*4)
        self.up2   = nn.ConvTranspose2d(c_base*4, c_base*2, 2, stride=2)
        self.conv_up2 = conv_block(c_base*4, c_base*2)
        self.up1   = nn.ConvTranspose2d(c_base*2, c_base, 2, stride=2)
        self.conv_up1 = conv_block(c_base*2, c_base)
        # Out
        self.out = nn.Conv2d(c_base, c_in, 1)

    def forward(self, x, t):
        # encoding
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        # decoding
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
# 3) DIFFUSION SCHEDULE
# -----------------------
betas     = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alphas    = 1.0 - betas
alpha_cum = torch.cumprod(alphas, dim=0)

@torch.no_grad()
def p_sample(model, x_t, t):
    beta_t = betas[t]
    sqrt_alpha = torch.sqrt(alphas[t])
    sqrt_one_minus_ac = torch.sqrt(1 - alpha_cum[t])

    eps_pred = model(x_t, t.repeat(x_t.size(0)))

    mean = (1 / sqrt_alpha) * (x_t - (beta_t / sqrt_one_minus_ac) * eps_pred)

    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise
    else:
        return mean

@torch.no_grad()
def sample_loop(model, shape):
    x = torch.randn(shape, device=DEVICE)
    for t in tqdm(reversed(range(TIMESTEPS)), desc="Sampling"):
        x = p_sample(model, x, torch.tensor([t], device=DEVICE))
    return x

# -----------------------
# 4) MAIN GENERATION
# -----------------------
def main():
    model = UNet(c_in=CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    n_samples = 200
    samples = sample_loop(model, (n_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    # rescale [-1,1] => [0,1]
    samples = (samples + 1) / 2

    # save each sample individually
    for idx, img in enumerate(samples):
        out_path = os.path.join(OUTPUT_DIR, f"sample_{idx+1}.png")
        save_image(img, out_path)

    print(f"Saved {n_samples} generated images to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
