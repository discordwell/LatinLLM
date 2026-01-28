# AGENTS.md

## Notes: PyTorch MPS on macOS 26.x

### Problem
- `torch.backends.mps.is_built()` returns True but `mps_available()` returns False
- RuntimeError: "MPS backend is supported on MacOS 14.0+"
- Cause: PyTorch stable wheels mis-detect macOS 26.x

### Solution: Build PyTorch from Source

```bash
# 1. Clone PyTorch (shallow to save time)
git clone --depth 1 https://github.com/pytorch/pytorch.git ~/Projects/pytorch
cd ~/Projects/pytorch
git submodule update --init --recursive --depth 1

# 2. Install build deps in your venv
source /path/to/your/.venv/bin/activate
pip install cmake ninja

# 3. Build with MPS enabled, CUDA disabled
export TMPDIR=/tmp/pytorch_build_tmp  # Fix clang temp file permission errors
mkdir -p $TMPDIR
USE_MPS=1 USE_METAL=1 USE_CUDA=0 USE_CUDNN=0 BUILD_TEST=0 \
  MACOSX_DEPLOYMENT_TARGET=14.0 MAX_JOBS=8 \
  python setup.py develop
```

Build takes ~30-45 min on M4 Max. Verify with:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
torch.tensor([1.0]).to('mps')  # Should work
```

### References
- https://github.com/pytorch/pytorch/issues/167679
- https://developer.apple.com/metal/pytorch/

### CTC Loss on MPS
CTC loss (`aten::_ctc_loss`) is not implemented on MPS. Run training with:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_mms.py ...
```
This uses CPU fallback for CTC loss while keeping other ops on GPU.

