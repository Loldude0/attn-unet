import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from attn_unet import UNet
from tqdm import tqdm
import sys
import csv
import wandb

wandb.init(
    project="attn-unet-depth-estimation",
    group="attn-unet-depth-estimation-loss_func_comp",
    config={
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs": 50,
        "optimizer": "Adam",
        "loss_function": "ScaleInvariantLoss",
        "scheduler": "ReduceLROnPlateau",
    },
)
config = wandb.config

wandb.run.name = f"{config.loss_function}_{config.optimizer}_{config.scheduler}"

patience = 5
delta = 0.0001
count = 0
best = float("inf")
checkpoints_dir =r"C:\Projects\attn-unet\checkpoints"

sys.path.append("nyuv2-python-toolkit")
from nyuv2 import NYUv2

class ScaleInvariantLoss(nn.Module):
    def __init__(self, lam=0.5):
        super(ScaleInvariantLoss, self).__init__()
        self.lam = lam

    def forward(self, y_pred, y_true):
        log_diff = torch.log(y_pred) - torch.log(y_true)
        return torch.mean(log_diff ** 2) - self.lam * (torch.mean(log_diff)) ** 2

transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

target_transform = transforms.Compose([transforms.ToTensor(),
transforms.Lambda(lambda x: x /x.max())])

train = NYUv2(
    root="C:/Projects/attn-unet/data/depth_output",
    split="train",
    target_type="depth",
    transform=transform,
    target_transform=target_transform,
)

test = NYUv2(
    root="C:/Projects/attn-unet/data/depth_output",
    split="test",
    target_type="depth",
    transform=transform,
    target_transform=target_transform,
)

train_loader = DataLoader(train, batch_size=4, shuffle=True)
test_loader = DataLoader(test, batch_size=4, shuffle=False)
model = UNet(3, 1).to("cuda")
crit = ScaleInvariantLoss()
opt = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3, factor=0.1)

wandb.watch(model)

epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
        for images, depth in tepoch:
            images = images.float().to("cuda")
            depth = depth.float().to("cuda")
            opt.zero_grad()
            outputs = model(images)
            outputs = torch.clamp(outputs, min=1e-5)
            loss = crit(outputs, depth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            opt.step()
            running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} loss: {epoch_loss}")
    wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with tqdm(test_loader, desc="Validation") as vepoch:
            for images, depth in vepoch:
                images = images.to("cuda")
                depth = depth.to("cuda")
                outputs = model(images)
                outputs = torch.clamp(outputs, min=1e-5)
                loss = crit(outputs, depth)
                running_loss += loss.item() * images.size(0)

    val_loss = running_loss / len(test_loader.dataset)
    print(f"Validation loss: {val_loss}")
    scheduler.step(val_loss)
    wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "learning_rate": opt.param_groups[0]["lr"]})

    if val_loss < best - delta:
        best = val_loss
        torch.save(model.state_dict(), f"{checkpoints_dir}/model.pth")
        count = 0
    else:
        count += 1
        if count >= patience:
            print("Early stopping")
            break

torch.save(model.state_dict(), "model.pth")
wandb.finish()