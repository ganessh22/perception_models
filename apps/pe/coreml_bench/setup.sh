#!/usr/bin/env bash
# Creates a local Python environment and downloads the weights and test images.
# Everything except the Hugging Face weights cache (~/.cache/huggingface) stays inside this folder.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # the repo root, for `core`

if [ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]; then
  echo "This benchmark needs an Apple Silicon Mac (found $(uname -s)-$(uname -m))." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV=uv
else
  echo "== Installing uv into ./.uv (your PATH and shell profile are not changed)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.uv" UV_NO_MODIFY_PATH=1 sh
  UV="$PWD/.uv/uv"
fi
export UV_PYTHON_INSTALL_DIR="$PWD/.uv/python"

echo "== Creating Python 3.11 environment"
"$UV" venv --python 3.11 --allow-existing .venv
"$UV" pip install --python .venv/bin/python -r requirements.lock

echo "== Downloading PE-Core-L14-336 weights"
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('facebook/PE-Core-L14-336', 'PE-Core-L14-336.pt'))"

mkdir -p data
if [ ! -f data/val.npy ] || [ ! -f data/calib.npy ]; then
  if [ ! -f data/imagenette2-320.tgz ]; then
    echo "== Downloading Imagenette (about 340 MB)"
    curl -fL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -o data/imagenette2-320.tgz.part
    mv data/imagenette2-320.tgz.part data/imagenette2-320.tgz
  fi
  echo "== Preprocessing images"
  .venv/bin/python prep_data.py "$PWD/data"
fi

echo
echo "Setup finished. Next, run:  bash run.sh"
