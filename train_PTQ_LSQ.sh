# PTQ & LSQ+
# ============== FQR 7.6 =====================
# synthetic NeRF
#python main.py data/nerf_synthetic/chair     --workspace workspace/LSQp_6bit/nerf_synthetic/chair     -O -MGL --data_format nerf --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/nerf_synthetic/drums     --workspace workspace/LSQp_6bit/nerf_synthetic/drums     -O -MGL --data_format nerf --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/nerf_synthetic/lego      --workspace workspace/LSQp_6bit/nerf_synthetic/lego      -O -MGL --data_format nerf --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/nerf_synthetic/ship      --workspace workspace/LSQp_6bit/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/nerf_synthetic/ficus     --workspace workspace/LSQp_6bit/nerf_synthetic/ficus     -O -MGL --data_format nerf --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/LSQp_6bit/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/nerf_synthetic/materials --workspace workspace/LSQp_6bit/nerf_synthetic/materials -O -MGL --data_format nerf --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/nerf_synthetic/mic       --workspace workspace/LSQp_6bit/nerf_synthetic/mic       -O -MGL --data_format nerf --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32

## RTMV
#python main.py data/RTMV/V8                  --workspace workspace/LSQp_6bit/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32

# mipnerf360
#python main.py data/mipnerf360/bonsai/       --workspace workspace/LSQp_6bit/mipnerf360/bonsai        -O -MGL --data_format colmap --downscale 4 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/mipnerf360/room/         --workspace workspace/LSQp_6bit/mipnerf360/room          -O -MGL --data_format colmap --downscale 4 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/mipnerf360/bicycle/     --workspace workspace/LSQp_6bit/mipnerf360/bicycle         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
#python main.py data/mipnerf360/garden/      --workspace workspace/LSQp_6bit/mipnerf360/garden         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6 --target 32
python main.py data/mipnerf360/counter/       --workspace workspace/LSQp_6bit/mipnerf360/counter     -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6
python main.py data/mipnerf360/kitchen/       --workspace workspace/LSQp_6bit/mipnerf360/kitchen     -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6
python main.py data/mipnerf360/stump/       --workspace workspace/LSQp_6bit/mipnerf360/stump         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 6 6 30 6 6 6 6 6 6 6 6 6 6 6 6

# ============== FQR 6.6 =====================
# synthetic NeRF
#python main.py data/nerf_synthetic/chair     --workspace workspace/LSQp_5bit/nerf_synthetic/chair     -O -MGL --data_format nerf --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/nerf_synthetic/drums     --workspace workspace/LSQp_5bit/nerf_synthetic/drums     -O -MGL --data_format nerf --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/nerf_synthetic/lego      --workspace workspace/LSQp_5bit/nerf_synthetic/lego      -O -MGL --data_format nerf --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/nerf_synthetic/ship      --workspace workspace/LSQp_5bit/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/nerf_synthetic/ficus     --workspace workspace/LSQp_5bit/nerf_synthetic/ficus     -O -MGL --data_format nerf --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/LSQp_5bit/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5 --target 32
#python main.py data/nerf_synthetic/materials --workspace workspace/LSQp_5bit/nerf_synthetic/materials -O -MGL --data_format nerf --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/nerf_synthetic/mic       --workspace workspace/LSQp_5bit/nerf_synthetic/mic       -O -MGL --data_format nerf --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5

## RTMV
#python main.py data/RTMV/V8                  --workspace workspace/LSQp_5bit/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5

# mipnerf360
#python main.py data/mipnerf360/bonsai/       --workspace workspace/LSQp_5bit/mipnerf360/bonsai        -O -MGL --data_format colmap --downscale 4 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/mipnerf360/room/         --workspace workspace/LSQp_5bit/mipnerf360/room          -O -MGL --data_format colmap --downscale 4 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/mipnerf360/bicycle/     --workspace workspace/LSQp_5bit/mipnerf360/bicycle         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
#python main.py data/mipnerf360/garden/      --workspace workspace/LSQp_5bit/mipnerf360/garden         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5

python main.py data/mipnerf360/counter/       --workspace workspace/LSQp_5bit/mipnerf360/counter     -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
python main.py data/mipnerf360/kitchen/       --workspace workspace/LSQp_5bit/mipnerf360/kitchen     -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5
python main.py data/mipnerf360/stump/       --workspace workspace/LSQp_5bit/mipnerf360/stump         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 5 5 29 5 5 5 5 5 5 5 5 5 5 5 5

# ============== FQR 5.6 =====================
# synthetic NeRF
#python main.py data/nerf_synthetic/chair     --workspace workspace/LSQp_4bit/nerf_synthetic/chair     -O -MGL --data_format nerf --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/nerf_synthetic/drums     --workspace workspace/LSQp_4bit/nerf_synthetic/drums     -O -MGL --data_format nerf --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/nerf_synthetic/lego      --workspace workspace/LSQp_4bit/nerf_synthetic/lego      -O -MGL --data_format nerf --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/nerf_synthetic/ship      --workspace workspace/LSQp_4bit/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/nerf_synthetic/ficus     --workspace workspace/LSQp_4bit/nerf_synthetic/ficus     -O -MGL --data_format nerf --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/LSQp_4bit/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/nerf_synthetic/materials --workspace workspace/LSQp_4bit/nerf_synthetic/materials -O -MGL --data_format nerf --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/nerf_synthetic/mic       --workspace workspace/LSQp_4bit/nerf_synthetic/mic       -O -MGL --data_format nerf --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4

## RTMV
#python main.py data/RTMV/V8                  --workspace workspace/LSQp_4bit/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/RTMV/V8                  --workspace workspace/MGL_30dB/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5 --T1 --bit_set 5 4 19 4 4 7 4 3 7 5 6 4 5 3 5


# mipnerf360
#python main.py data/mipnerf360/bonsai/       --workspace workspace/LSQp_4bit/mipnerf360/bonsai        -O -MGL --data_format colmap --downscale 4 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/mipnerf360/room/         --workspace workspace/LSQp_4bit/mipnerf360/room          -O -MGL --data_format colmap --downscale 4 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/mipnerf360/bicycle/     --workspace workspace/LSQp_4bit/mipnerf360/bicycle         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
#python main.py data/mipnerf360/garden/      --workspace workspace/LSQp_4bit/mipnerf360/garden           -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4

python main.py data/mipnerf360/counter/       --workspace workspace/LSQp_4bit/mipnerf360/counter     -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
python main.py data/mipnerf360/kitchen/       --workspace workspace/LSQp_4bit/mipnerf360/kitchen     -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
python main.py data/mipnerf360/stump/       --workspace workspace/LSQp_4bit/mipnerf360/stump         -O -MDL --data_format colmap --downscale 4 --T1 --bit_set 4 4 28 4 4 4 4 4 4 4 4 4 4 4 4
