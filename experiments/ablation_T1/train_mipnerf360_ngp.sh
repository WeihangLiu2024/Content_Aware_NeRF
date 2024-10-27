# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x

exp_name='LSQ_5bit'
data_format='colmap'
bound=8.0
downscale=4
log2_hashmap_size=19
bit_width=5

./train_ngp.sh -d mipnerf360/bicycle/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 10s

./train_ngp.sh -d mipnerf360/bonsai/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 10s

./train_ngp.sh -d mipnerf360/counter/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 10s

./train_ngp.sh -d mipnerf360/flowers/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 800s

./train_ngp.sh -d mipnerf360/garden/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 10s

./train_ngp.sh -d mipnerf360/kitchen/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 10s

./train_ngp.sh -d mipnerf360/room/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 10s

./train_ngp.sh -d mipnerf360/stump/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} & sleep 800s

./train_ngp.sh -d mipnerf360/treehill/ -l ${exp_name} -f ${data_format} -b ${bound} --downscale ${downscale} --log2_hashmap_size ${log2_hashmap_size} --bit_width ${bit_width} &