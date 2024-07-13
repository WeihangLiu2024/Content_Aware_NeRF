# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty

python main.py data/nerf_synthetic/chair     --workspace workspace/CA_final/nerf_synthetic/chair     -O -MDL --data_format nerf
