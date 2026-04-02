# MCMH-Net: A Multi-Scale CNN-Mamba2 Hybrid Network for Low-Dose CT Denoising

**Paper:** MCMH-Net: A Multi-Scale CNN-Mamba2 Hybrid Network for Low-Dose CT Denoising  
**Authors:** Ye Li, Wei Zhang  
**Institution:** University of Shanghai for Science and Technology  

## Code Release Status

| Component | Status |
|-----------|--------|
| Model architecture | ✅ Available |
| Training code | 🔒 Coming upon acceptance |
| Validation & evaluation code | 🔒 Coming upon acceptance |

## Model Architecture

该 `model/` directory contains the implementation of:
- MCMH-Net encoder-decoder architecture
- Content-Aware Fusion Attention (CAFA) module
- Dynamic Cross-scale Serpentine Mamba2 (DCSMamba2) module
- Residual DCSMamba2 Block (RDB)

## Requirements
```
Python >= 3.8
PyTorch >= 1.12
mamba-ssm
```

## Citation
If you find this work useful, please cite:
(Citation information will be updated upon acceptance.)
