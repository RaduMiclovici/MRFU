import torch
import json
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import SimpleCNN

# --------------------
# Paths
# --------------------
TEST_DIR = "data/test"
MODEL_PATH = "models/model.pth"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------
# Transform (same as validation)
# --------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# --------------------
# Dataset & Loader
# --------------------
test_set = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# --------------------
# Model
# --------------------
device = torch.device("cpu")
model = SimpleCNN(num_classes=len(test_set.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"Test Accuracy: {acc*100:.2f}%")
print(f"F1-score (macro): {f1:.2f}")

# --------------------
# Save metrics
# --------------------
metrics = {"test_accuracy": acc, "f1_macro": f1}

with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
