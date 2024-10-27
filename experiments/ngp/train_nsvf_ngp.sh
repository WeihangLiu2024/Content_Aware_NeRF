# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x

exp_name='NGP'
data_format='nsvf'
downscale=1
log2_hashmap_size=19

./train_ngp.sh -d Synthetic_NSVF/Bike -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d Synthetic_NSVF/Lifestyle -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d Synthetic_NSVF/Palace -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d Synthetic_NSVF/Robot -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d Synthetic_NSVF/Spaceship -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d Synthetic_NSVF/Steamtrain -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d Synthetic_NSVF/Toad -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d Synthetic_NSVF/Wineholder -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s
