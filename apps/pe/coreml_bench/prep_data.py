"""Preprocess Imagenette exactly like the repo's get_image_transform.

Writes val.npy (50 images per class), val_labels.npy and calib.npy (8 per class, train split).
"""
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from core.vision_encoder.transforms import get_image_transform

DATA = Path(sys.argv[1])
ROOT = DATA / "imagenette2-320"
if not ROOT.exists():
    with tarfile.open(DATA / "imagenette2-320.tgz") as t:
        t.extractall(DATA, filter="data")

CLASSES = sorted(p.name for p in (ROOT / "val").iterdir() if p.is_dir())
transform = get_image_transform(336)


def load(split, per_class):
    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        files = sorted((ROOT / split / cls).glob("*.JPEG"))
        step = max(1, len(files) // per_class)
        for f in files[::step][:per_class]:
            xs.append(transform(Image.open(f)).numpy().astype(np.float32))
            ys.append(label)
    return np.stack(xs), np.array(ys)


val, val_y = load("val", 50)
calib, _ = load("train", 8)
np.save(DATA / "val.npy", val)
np.save(DATA / "val_labels.npy", val_y)
np.save(DATA / "calib.npy", calib)
print("classes:", CLASSES)
print("val", val.shape, "calib", calib.shape, "range", val.min(), val.max())
