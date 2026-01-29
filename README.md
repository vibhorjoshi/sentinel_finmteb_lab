# sentinel_finmteb_lab

## GPU setup

This project will use CUDA automatically when available. To force CUDA on a machine
with an NVIDIA GPU, set `SENTINEL_DEVICE=cuda` before running any scripts. If CUDA
is unavailable, the runtime falls back to CPU and logs a warning.

```bash
export SENTINEL_DEVICE=cuda
python run_sentinel_100k.py
```

To explicitly run on CPU, set `SENTINEL_DEVICE=cpu`.

### Installing CUDA-enabled PyTorch

If you see `No matching distribution found for torch==2.1.2+cu121`, install a
CUDA 12.1 build that exists for your Python version (for example 2.2.x+cu121 or
newer). Example:

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
  -f https://download.pytorch.org/whl/cu121/torch_stable.html
```

Adjust the version if your Python environment requires a newer build from the
PyTorch CUDA 12.1 index.
