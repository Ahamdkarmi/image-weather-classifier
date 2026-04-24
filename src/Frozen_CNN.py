import os
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
CSV_PATH = "dataset.csv"
IMG_COL = "Image URL"
LABEL_COL = "Weather"
OUT_NPZ = "features_convnext_tiny_224.npz"
BATCH_SIZE = 32
def main():
    data = pd.read_csv(CSV_PATH)
    data[LABEL_COL] = data[LABEL_COL].astype(str)
    image_paths = data[IMG_COL].astype(str).str.strip().tolist()
    labels = data[LABEL_COL].astype(str).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    feats = []
    valid_paths = []
    valid_labels = []
    batch_tensors = []
    batch_meta = []
    def flush_batch():
        nonlocal batch_tensors, batch_meta, feats, valid_paths, valid_labels
        if not batch_tensors:
            return
        imgs = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = model(imgs)
        feats.append(out.detach().cpu().numpy())
        for pth, lab in batch_meta:
            valid_paths.append(pth)
            valid_labels.append(lab)
        batch_tensors = []
        batch_meta = []
    for i, (path, lab) in enumerate(zip(image_paths, labels)):
        if i % 25 == 0:
            print(f"Processing {i}/{len(image_paths)}")
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                t = transform(im)
            batch_tensors.append(t)
            batch_meta.append((path, lab))
            if len(batch_tensors) >= BATCH_SIZE:
                flush_batch()
        except Exception as e:
            print(f"Skipping unreadable file: {path} | error: {e}")
            continue
    flush_batch()
    if len(feats) == 0:
        raise RuntimeError("No valid images were loaded. Check 'Image URL' paths in dataset.csv")
    X = np.concatenate(feats, axis=0)  # (N, D)
    y = np.array(valid_labels, dtype=object)
    paths = np.array(valid_paths, dtype=object)
    print("Saved feature matrix shape:", X.shape)
    print("Saved labels shape:", y.shape)
    np.savez_compressed(
        OUT_NPZ,
        X=X,
        y=y,
        paths=paths,
        backbone="convnext_tiny",
        image_size=224
    )
    print(f"Done. Saved -> {OUT_NPZ}")
if __name__ == "__main__":
    main()
