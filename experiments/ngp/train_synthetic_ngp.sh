# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x

exp_name='NGP'
data_format='nerf'
downscale=1

./train_ngp.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} & sleep 10s

./train_ngp.sh -d nerf_synthetic/drums -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} & sleep 10s

./train_ngp.sh -d nerf_synthetic/ficus -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} & sleep 10s

./train_ngp.sh -d nerf_synthetic/hotdog -l ${exp_name} -f ${data_format} -b 1.3 --downscale ${downscale} & sleep 500s

./train_ngp.sh -d nerf_synthetic/lego -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} & sleep 10s

./train_ngp.sh -d nerf_synthetic/materials -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} & sleep 10s

./train_ngp.sh -d nerf_synthetic/mic -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} & sleep 10s

./train_ngp.sh -d nerf_synthetic/ship -l ${exp_name} -f ${data_format} -b 1.3 --downscale ${downscale}
