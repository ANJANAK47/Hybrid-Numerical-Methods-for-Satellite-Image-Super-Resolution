import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ✅ SRCNN Model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)  # 1st layer
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)  # 2nd layer
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)  # Output layer

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# ✅ EuroSAT Dataset Class
class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Load all image paths
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_file in os.listdir(category_path):
                    self.image_paths.append(os.path.join(category_path, img_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # Load image
        img = img.resize((64, 64))  # Resize all images to 64x64 for uniformity

        # Create Low-Resolution version (Bicubic downsampling)
        lr_img = img.resize((32, 32), Image.BICUBIC).resize((64, 64), Image.BICUBIC)

        if self.transform:
            img = self.transform(img)
            lr_img = self.transform(lr_img)

        return lr_img, img  # (Low-Res, High-Res)

# ✅ Training Settings
batch_size = 32
num_epochs = 10
learning_rate = 1e-4

# ✅ Define Image Transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# ✅ Load Train & Test Datasets
train_dataset = EuroSATDataset(root_dir="train_EuroSAT", transform=transform)
test_dataset = EuroSATDataset(root_dir="test_EuroSAT", transform=transform)

# ✅ Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ✅ Initialize Model, Loss Function, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        # Forward Pass
        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.6f}")

print("✅ Training Completed!")

# ✅ Function to Show Image Comparisons
def show_images(lr, sr, hr):
    lr = np.transpose(lr.cpu().detach().numpy(), (1, 2, 0))
    sr = np.transpose(sr.cpu().detach().numpy(), (1, 2, 0))
    hr = np.transpose(hr.cpu().detach().numpy(), (1, 2, 0))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr)
    axs[0].set_title("Low-Res (Bicubic)")
    axs[0].axis("off")

    axs[1].imshow(sr)
    axs[1].set_title("SRCNN Output")
    axs[1].axis("off")

    axs[2].imshow(hr)
    axs[2].set_title("Ground Truth")
    axs[2].axis("off")

    plt.show()

# ✅ Evaluate SRCNN on Test Images
model.eval()
with torch.no_grad():
    for lr_imgs, hr_imgs in test_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        sr_imgs = model(lr_imgs)

        # Show first test result
        show_images(lr_imgs[0], sr_imgs[0], hr_imgs[0])
        break

# ✅ Save Model
torch.save(model.state_dict(), "srcnn_eurosat.pth")
print("✅ Model Saved!")

# ✅ Load Model
model.load_state_dict(torch.load("srcnn_eurosat.pth"))
model.eval()
print("✅ Model Loaded Successfully!")
