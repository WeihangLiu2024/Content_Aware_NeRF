# A-CAQ for radiance field

This is the reference code for paper "Content-Aware Radiance Fields: Aligning Model Complexity with Scene Intricacy Through Learned Bitwidth Quantization" submitted to ECCV 2024

Supported public datasets should be initially downloaded from the internet

- Synthetic-NeRF

- Mip-NeRF360 (COLMAP required)

- RTMV
  
  and put in folder `./data`

## Install

```bash
pip install -r requirements.txt

# install all extension modules
bash scripts/install_ext.sh
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

## Tested environments

Ubuntu 20.04 with torch 2.0.0 & CUDA 11.8 on a RTX 4090

## Usage

Examples see `train.sh`, `train_penalty.sh`, `train_PTQ_LSQ.sh`

To find the parameter definition, see  `./config/GetConfig.py`
