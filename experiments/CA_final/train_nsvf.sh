# The script of all content-aware NeRF experiments, including
# === Quantitative Results ===
# 1) overall
# === Ablation study ===
# 2) scalable hash
# 3) content-aware Quantization
# 4) uncertainty
#set -x

exp_name='CA_final_v0_2'
data_format='nsvf'
quantization='MDL'
alpha='True'
downscale=1
update_hash=50
hash_interval=5

./train.sh -d Synthetic_NSVF/Bike -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} &
sleep 30s

./train.sh -d Synthetic_NSVF/Lifestyle -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} &
sleep 30s

./train.sh -d Synthetic_NSVF/Palace -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} &
sleep 30s

./train.sh -d Synthetic_NSVF/Robot -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} &
sleep 600s

./train.sh -d Synthetic_NSVF/Spaceship -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} &
sleep 30s

./train.sh -d Synthetic_NSVF/Steamtrain -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval} &
sleep 30s

./train.sh -d Synthetic_NSVF/Toad -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash 100 --hash_interval 10 &
sleep 30s

./train.sh -d Synthetic_NSVF/Wineholder -l ${exp_name} -f ${data_format} -q ${quantization} -a ${alpha} -b 1.0 --downscale ${downscale} --update_hash ${update_hash} --hash_interval ${hash_interval}
