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
PyTorch >= 2.3.1
mamba-ssm >= 2.2.2
causal_conv1d >= 1.4.0
```

## Citation
If you find this work useful, please cite:
https://www.sciencedirect.com/science/article/abs/pii/S1568494626007799
