import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append("src/neural_network")
from model import SimpleCNN

MODEL_PATH = "models/model.pth"
CLASSES_PATH = "models/classes.txt"

# load classes
with open(CLASSES_PATH, "r") as f:
    CLASSES = [x.strip() for x in f.readlines()]

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def load_model():
    model = SimpleCNN(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(img_path):
    model = load_model()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        conf, idx = torch.softmax(out, dim=1).max(1)

    return CLASSES[idx.item()], conf.item()
