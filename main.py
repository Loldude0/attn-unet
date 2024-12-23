import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from attn_unet import UNet
from tqdm import tqdm
import sys

sys.path.append("nyuv2-python-toolkit")
from nyuv2 import NYUv2

transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.ToTensor()])

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

train_loader = DataLoader(train, batch_size=8, shuffle=True)
test_loader = DataLoader(test, batch_size=8, shuffle=False)
model = UNet(3, 1).to("cuda")
crit = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-4)

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
            loss = crit(outputs, depth)
            loss.backward()
            opt.step()
            running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} loss: {epoch_loss}")

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with tqdm(test_loader, desc="Validation") as vepoch:
            for images, depth in vepoch:
                images = images.to("cuda")
                depth = depth.to("cuda")
                outputs = model(images)
                loss = crit(outputs, depth)
                running_loss += loss.item() * images.size(0)

    val_loss = running_loss / len(test_loader.dataset)
    print(f"Validation loss: {val_loss}")

torch.save(model.state_dict(), "model.pth")