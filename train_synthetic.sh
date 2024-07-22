# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty

exp_name='CA_final'
data_format='nerf'
quantization='MDL'
alpha='True'

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
sleep 20s

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
sleep 20s

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
sleep 20s

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
sleep 20s

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
sleep 20s

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
sleep 20s

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
sleep 20s

./train.sh -d nerf_synthetic/chair -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha}
