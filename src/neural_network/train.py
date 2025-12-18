import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleCNN

# ======================
# Paths
# ======================
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_DIR = "models"
RESULTS_DIR = "results"
DOCS_DIR = "docs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ======================
# Hyperparameters
# ======================
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 30

# ======================
# Transforms
# ======================
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# ======================
# Datasets & Loaders
# ======================
train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_set = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_set.classes)
print("Classes:", train_set.classes)

# Save class names
with open(os.path.join(MODEL_DIR, "classes.txt"), "w") as f:
    for c in train_set.classes:
        f.write(c + "\n")

# ======================
# Model
# ======================
device = torch.device("cpu")
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======================
# Tracking
# ======================
history_file = open(os.path.join(RESULTS_DIR, "training_history.csv"), "w", newline="")
csv_writer = csv.writer(history_file)
csv_writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

train_losses = []
val_losses = []

# ======================
# Training Loop
# ======================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total

    # ---------- Validation ----------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Acc: {train_acc*100:.2f}% | "
        f"Val Acc: {val_acc*100:.2f}%"
    )

    csv_writer.writerow([epoch + 1, train_loss, val_loss, train_acc, val_acc])

# ======================
# Save Model
# ======================
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
history_file.close()

# ======================
# Plot Loss Curve
# ======================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig(os.path.join(DOCS_DIR, "loss_curve.png"))
plt.close()

print("Training complete. Model and loss curve saved.")
