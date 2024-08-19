# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x
# tank: ["Barn", "Caterpillar", "Family", "Ignatius", "Truck"]

exp_name='NGP'
data_format='tank'
downscale=1
log2_hashmap_size=19

./train_ngp.sh -d TanksAndTemple/Barn -l ${exp_name} -f ${data_format} -b 3.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d TanksAndTemple/Caterpillar -l ${exp_name} -f ${data_format} -b 3.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d TanksAndTemple/Family -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d TanksAndTemple/Ignatius -l ${exp_name} -f ${data_format} -b 1.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s

./train_ngp.sh -d TanksAndTemple/Truck -l ${exp_name} -f ${data_format} -b 3.0 --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} & sleep 10s
