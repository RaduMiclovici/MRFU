import os
import random
from PIL import Image

# folders
RAW = "data/raw"
TRAIN = "data/train"
VAL = "data/val"

# image settings
SIZE = (224, 224)
VAL_RATIO = 0.2
EX = (".jpg")


def save_img(in_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.open(in_path).convert("RGB")
    img = img.resize(SIZE)
    img.save(out_path)

def main():
    print("raw =", os.path.abspath(RAW))
    classes = os.listdir(RAW)
    print("found classes:", classes)

    for cls in classes:
        p = os.path.join(RAW, cls)
        if not os.path.isdir(p):
            continue

        imgs = [f for f in os.listdir(p) if f.lower().endswith(EX)]
        print(f"{cls}: {len(imgs)} imgs")

        if len(imgs) == 0:
            print(" -> skip")
            continue

        random.shuffle(imgs)
        cut = int(len(imgs) * (1 - VAL_RATIO))
        tr = imgs[:cut]
        val = imgs[cut:]

        print(f"  train: {len(tr)}, val: {len(val)}")

        for x in tr:
            save_img(os.path.join(p, x), os.path.join(TRAIN, cls, x))
        for x in val:
            save_img(os.path.join(p, x), os.path.join(VAL, cls, x))

    print("done.")

if __name__ == "__main__":
    main()
