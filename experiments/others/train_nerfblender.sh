time=$(date "+%Y-%m-%d_%H:%M:%S")
log2_hashmap_size=12
exp_name="NGP"

python ../../main.py ../../data/nerf_blender/teamug --workspace ../../workspace/${exp_name}/nerf_blender/teamug/${time} -O --data_format nerf --bound 1.0 --downscale 1 --update_hash 0 --log2_hashmap_size ${log2_hashmap_size} --offset 0 0 -1 &
sleep 10s

python ../../main.py ../../data/nerf_blender/plant --workspace ../../workspace/${exp_name}/nerf_blender/plant/${time} -O --data_format nerf --bound 14.0 --downscale 1 --update_hash 0 --log2_hashmap_size ${log2_hashmap_size} --offset 0 0 -1 &
sleep 10s

python ../../main.py ../../data/nerf_blender/spaceship --workspace ../../workspace/${exp_name}/nerf_blender/spaceship/${time} -O --data_format nerf --bound 5.0 --downscale 1 --update_hash 0 --log2_hashmap_size ${log2_hashmap_size} --offset 0 0 -1 &
