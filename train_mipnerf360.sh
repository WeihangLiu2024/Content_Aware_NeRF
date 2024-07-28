# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x

exp_name='CA_final'
data_format='colmap'
quantization='MDL'
alpha='True'
bound=8.0

./train.sh -d mipnerf360/bicycle/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 10s

./train.sh -d mipnerf360/bonsai/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 10s

./train.sh -d mipnerf360/counter/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 10s

./train.sh -d mipnerf360/flowers/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 1000s

./train.sh -d mipnerf360/garden/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 60s

./train.sh -d mipnerf360/kitchen/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 60s

./train.sh -d mipnerf360/room/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 60s

./train.sh -d mipnerf360/stump/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &
sleep 1000s

./train.sh -d mipnerf360/treehill/ -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b ${bound} &