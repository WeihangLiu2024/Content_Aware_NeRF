# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x

exp_name='CA_final_v0_2_MGL30dB'
data_format='colmap'
quantization='MGL'
alpha='True'
bound=8.0
downscale=4
update_hash=50
hash_interval=5
target=30

./train_MGL.sh -d mipnerf360/bicycle/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 30s

./train_MGL.sh -d mipnerf360/bonsai/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 30s

./train_MGL.sh -d mipnerf360/counter/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 30s

./train_MGL.sh -d mipnerf360/flowers/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 600s

./train_MGL.sh -d mipnerf360/garden/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 30s

./train_MGL.sh -d mipnerf360/kitchen/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 30s

./train_MGL.sh -d mipnerf360/room/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 30s

./train_MGL.sh -d mipnerf360/stump/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&
sleep 600s

./train_MGL.sh -d mipnerf360/treehill/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} --target ${target}&