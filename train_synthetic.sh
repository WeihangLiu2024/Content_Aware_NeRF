# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x

exp_name='CA_final'
data_format='nerf'
quantization='MDL'
alpha='True'
downscale=1
update_hash=100

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} &
sleep 10s

./train.sh -d nerf_synthetic/drums -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} &
sleep 10s

./train.sh -d nerf_synthetic/ficus -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} &
sleep 10s

./train.sh -d nerf_synthetic/hotdog -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.3 --downscale ${downscale} --update_hash ${update_hash} &
sleep 1000s

./train.sh -d nerf_synthetic/lego -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} &
sleep 20s

./train.sh -d nerf_synthetic/materials -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} &
sleep 20s

./train.sh -d nerf_synthetic/mic -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} &
sleep 20s

./train.sh -d nerf_synthetic/ship -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.3 --downscale ${downscale} --update_hash ${update_hash} &
