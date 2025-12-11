import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

train_dir = "data/train"
val_dir = "data/val"
model_dir = "models"

os.makedirs(model_dir, exist_ok=True)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_set = datasets.ImageFolder(train_dir, transform=transform)
val_set = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
num_classes = len(train_set.classes)
print("Classes:", train_set.classes)

# save class names
with open(os.path.join(model_dir, "classes.txt"), "w") as f:
    for c in train_set.classes:
        f.write(c + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_acc = correct / total * 100

    # validation
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, pred = out.max(1)
            val_total += labels.size(0)
            val_correct += pred.eq(labels).sum().item()

    val_acc = val_correct / val_total * 100

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
    )

torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
print("Model saved in models/model.pth")
