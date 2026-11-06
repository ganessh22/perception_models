"""Compare saved embeddings to the PyTorch FP32 CPU reference, and score zero-shot top-1 on Imagenette."""
import sys
from pathlib import Path

import numpy as np
import torch

import core.vision_encoder.pe as pe
from core.vision_encoder.transforms import get_text_tokenizer

HERE = Path(__file__).parent
CLASSES = ["tench", "English springer spaniel", "cassette player", "chainsaw", "church",
           "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]

cache = HERE / "embeds" / "_text.pt"
if cache.exists():
    text = torch.load(cache)
else:
    clip = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).eval()
    tok = get_text_tokenizer(clip.context_length)(["a photo of a " + c for c in CLASSES])
    with torch.no_grad():
        text = clip.encode_text(tok, normalize=True)
    torch.save(text, cache)
text = text.numpy()

labels = np.load(HERE / "data" / "val_labels.npy")
norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
ref = norm(np.load(HERE / "embeds" / "torch_fp32_cpu.npy"))
ref_pred = (ref @ text.T).argmax(1)

print(f"{'config':44s} {'n':>4s} {'cos mean':>9s} {'cos min':>9s} {'top1':>6s} {'agree':>6s}")
for path in sorted((HERE / "embeds").glob("*.npy")):
    e = norm(np.load(path))
    n = len(e)
    idx = np.linspace(0, len(ref) - 1, n).astype(int)  # same sampling as bench.py; ref must be the full set
    cos = (e * ref[idx]).sum(1)
    pred = (e @ text.T).argmax(1)
    print(f"{path.stem:44s} {n:4d} {cos.mean():9.6f} {cos.min():9.6f} "
          f"{(pred == labels[idx]).mean():6.3f} {(pred == ref_pred[idx]).mean():6.3f}")
