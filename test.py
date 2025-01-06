import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from attn_unet import UNet

# Define metrics
def absrel(pred, target):
    return torch.mean(torch.abs(pred - target) / target)

def mse(pred, target):
    return torch.mean((pred - target) ** 2)

def rmse(pred, target):
    return torch.sqrt(mse(pred, target))

# Paths
depth_path = "C:/Projects/attn-unet/data/depth_output/depth/test"
image_path = "C:/Projects/attn-unet/data/depth_output/image/test"
checkpoint_path = "C:/Projects/attn-unet/checkpoints/model.pth"

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / x.max())
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(3, 1).to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Compute metrics
absrel_list, mse_list, rmse_list = [], [], []
with torch.no_grad():
    for file_name in tqdm(os.listdir(image_path), desc="Evaluating Metrics"):
        if not file_name.endswith(".png"):
            continue
        
        # Load image and depth
        img = Image.open(os.path.join(image_path, file_name)).convert("RGB")
        depth = Image.open(os.path.join(depth_path, file_name)).convert("F")
        
        img = transform(img).unsqueeze(0).to(device)
        depth = target_transform(depth).unsqueeze(0).to(device)
        
        # Predict
        pred = model(img)
        pred = torch.clamp(pred, min=1e-5)
        
        # Calculate metrics
        absrel_list.append(absrel(pred, depth).item())
        mse_list.append(mse(pred, depth).item())
        rmse_list.append(rmse(pred, depth).item())

# Average metrics
avg_absrel = np.mean(absrel_list)
avg_mse = np.mean(mse_list)
avg_rmse = np.mean(rmse_list)

print(f"Average AbsRel: {avg_absrel}")
print(f"Average MSE: {avg_mse}")
print(f"Average RMSE: {avg_rmse}")
