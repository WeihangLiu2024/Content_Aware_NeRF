# ===== colmap for mipnerf360 =====
#echo "Y" | python scripts/colmap2nerf.py --images ./data/mipnerf360/garden/images/ --run_colmap # if use images
#echo "Y" | python scripts/colmap2nerf.py --images ./data/mipnerf360/room/images/ --run_colmap # if use images
#echo "Y" | python scripts/colmap2nerf.py --images ./data/mipnerf360/counter/images/ --run_colmap # if use images
#echo "Y" | python scripts/colmap2nerf.py --images ./data/mipnerf360/kitchen/images/ --run_colmap # if use images
#echo "Y" | python scripts/colmap2nerf.py --images ./data/mipnerf360/stump/images/ --run_colmap # if use images

# =========================================================== fp ========================================================
python main.py data/nerf_synthetic/drums --workspace workspace/fp/nerf_synthetic/drums_log19 -O --data_format nerf
python main.py data/nerf_synthetic/drums --workspace workspace/fp/nerf_synthetic/drums_log15 -O --data_format nerf --log2_hashmap_size 15
python main.py data/nerf_synthetic/drums --workspace workspace/fp/nerf_synthetic/drums_log12 -O --data_format nerf --log2_hashmap_size 12

python main.py data/nerf_blender/teamug --workspace workspace/fp/nerf_blender/teamug_log19 -O --data_format nerf --offset 0 0 -1
python main.py data/nerf_blender/teamug --workspace workspace/fp/nerf_blender/teamug_log15 -O --data_format nerf --offset 0 0 -1 --log2_hashmap_size 15
python main.py data/nerf_blender/teamug --workspace workspace/fp/nerf_blender/teamug_log12 -O --data_format nerf --offset 0 0 -1 --log2_hashmap_size 12

# =========================================================== MGL =======================================================
## synthetic NeRF
#python main.py data/nerf_synthetic/chair     --workspace workspace/MGL_30dB/nerf_synthetic/chair     -O -MGL --data_format nerf
#python main.py data/nerf_synthetic/drums     --workspace workspace/MGL_30dB/nerf_synthetic/drums     -O -MGL --data_format nerf
#python main.py data/nerf_synthetic/lego      --workspace workspace/MGL_30dB/nerf_synthetic/lego      -O -MGL --data_format nerf
#python main.py data/nerf_synthetic/ship      --workspace workspace/MGL_30dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3
#python main.py data/nerf_synthetic/ficus     --workspace workspace/MGL_30dB/nerf_synthetic/ficus     -O -MGL --data_format nerf
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/MGL_30dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3
#python main.py data/nerf_synthetic/materials --workspace workspace/MGL_30dB/nerf_synthetic/materials -O -MGL --data_format nerf
#python main.py data/nerf_synthetic/mic       --workspace workspace/MGL_30dB/nerf_synthetic/mic       -O -MGL --data_format nerf
#
## RTMV
#python main.py data/RTMV/V8                  --workspace workspace/MGL_30dB/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5

## mipnerf360
#python main.py data/mipnerf360/bonsai/       --workspace workspace/MGL_30dB/mipnerf360/bonsai        -O -MGL --data_format colmap --downscale 4
#python main.py data/mipnerf360/room/         --workspace workspace/MGL_30dB/mipnerf360/room          -O -MGL --data_format colmap --downscale 4
#python main.py data/mipnerf360/counter/       --workspace workspace/MGL_30dB/mipnerf360/counter        -O -MGL --data_format colmap --downscale 4
#python main.py data/mipnerf360/kitchen/         --workspace workspace/MGL_30dB/mipnerf360/kitchen      -O -MGL --data_format colmap --downscale 4
#python main.py data/mipnerf360/stump/       --workspace workspace/MGL_30dB/mipnerf360/stump            -O -MGL --data_format colmap --downscale 4
#python main.py data/mipnerf360/garden/         --workspace workspace/MGL_30dB/mipnerf360/garden      -O -MGL --data_format colmap --downscale 4
#python main.py data/mipnerf360/bicycle/       --workspace workspace/MGL_30dB/mipnerf360/bicycle            -O -MGL --data_format colmap --downscale 4

## ===================================================== MGL2 target 32 ==================================================
## synthetic NeRF
#python main.py data/nerf_synthetic/chair     --workspace workspace/MGL_32dB/nerf_synthetic/chair     -O -MGL --data_format nerf --target 32
#python main.py data/nerf_synthetic/drums     --workspace workspace/MGL_32dB/nerf_synthetic/drums     -O -MGL --data_format nerf --target 32
#python main.py data/nerf_synthetic/lego      --workspace workspace/MGL_32dB/nerf_synthetic/lego      -O -MGL --data_format nerf --target 32
#python main.py data/nerf_synthetic/ship      --workspace workspace/MGL_32dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32
#python main.py data/nerf_synthetic/ficus     --workspace workspace/MGL_32dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --target 32
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/MGL_32dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32
#python main.py data/nerf_synthetic/materials --workspace workspace/MGL_32dB/nerf_synthetic/materials -O -MGL --data_format nerf --target 32
#python main.py data/nerf_synthetic/mic       --workspace workspace/MGL_32dB/nerf_synthetic/mic       -O -MGL --data_format nerf --target 32

## RTMV
#python main.py data/RTMV/V8                  --workspace workspace/MGL_32dB/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5 --target 32

## mipnerf360
#python main.py data/mipnerf360/bonsai/       --workspace workspace/MGL_32dB/mipnerf360/bonsai        -O -MGL --data_format colmap --downscale 4 --target 32
#python main.py data/mipnerf360/room/         --workspace workspace/MGL_32dB/mipnerf360/room          -O -MGL --data_format colmap --downscale 4 --target 32
#python main.py data/mipnerf360/counter/       --workspace workspace/MGL_32dB/mipnerf360/counter        -O -MGL --data_format colmap --downscale 4 --target 32
#python main.py data/mipnerf360/kitchen/         --workspace workspace/MGL_32dB/mipnerf360/kitchen          -O -MGL --data_format colmap --downscale 4 --target 32
#python main.py data/mipnerf360/stump/         --workspace workspace/MGL_32dB/mipnerf360/stump          -O -MGL --data_format colmap --downscale 4 --target 32
#python main.py data/mipnerf360/garden/         --workspace workspace/MGL_32dB/mipnerf360/garden          -O -MGL --data_format colmap --downscale 4 --target 32
#python main.py data/mipnerf360/bicycle/         --workspace workspace/MGL_32dB/mipnerf360/bicycle          -O -MGL --data_format colmap --downscale 4 --target 32

## ========================================================== MDL ========================================================
## synthetic NeRF
#python main.py data/nerf_synthetic/chair     --workspace workspace/MDL/nerf_synthetic/chair     -O -MDL --data_format nerf
#python main.py data/nerf_synthetic/drums     --workspace workspace/MDL/nerf_synthetic/drums     -O -MDL --data_format nerf
#python main.py data/nerf_synthetic/lego      --workspace workspace/MDL/nerf_synthetic/lego      -O -MDL --data_format nerf
#python main.py data/nerf_synthetic/ship      --workspace workspace/MDL/nerf_synthetic/ship      -O -MDL --data_format nerf --selfbound --bound 1.3
#python main.py data/nerf_synthetic/ficus     --workspace workspace/MDL/nerf_synthetic/ficus     -O -MDL --data_format nerf
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/MDL/nerf_synthetic/hotdog    -O -MDL --data_format nerf --selfbound --bound 1.3
#python main.py data/nerf_synthetic/materials --workspace workspace/MDL/nerf_synthetic/materials -O -MDL --data_format nerf
#python main.py data/nerf_synthetic/mic       --workspace workspace/MDL/nerf_synthetic/mic       -O -MDL --data_format nerf

## RTMV
#python main.py data/RTMV/V8                 --workspace workspace/MDL/RTMV/V8                   -O -MDL --data_format nerf --selfbound --bound 0.5

## mipnerf360
#python main.py data/mipnerf360/bonsai/      --workspace workspace/MDL/mipnerf360/bonsai         -O -MDL --data_format colmap --downscale 4
#python main.py data/mipnerf360/bicycle/     --workspace workspace/MDL/mipnerf360/bicycle        -O -MDL --data_format colmap --downscale 4
#python main.py data/mipnerf360/garden/      --workspace workspace/MDL/mipnerf360/garden         -O -MDL --data_format colmap --downscale 4
#python main.py data/mipnerf360/room/        --workspace workspace/MDL/mipnerf360/room           -O -MDL --data_format colmap --downscale 4
#python main.py data/mipnerf360/counter/     --workspace workspace/MDL/mipnerf360/counter        -O -MDL --data_format colmap --downscale 4
#python main.py data/mipnerf360/kitchen/     --workspace workspace/MDL/mipnerf360/kitchen         -O -MDL --data_format colmap --downscale 4
#python main.py data/mipnerf360/stump/       --workspace workspace/MDL/mipnerf360/stump           -O -MDL --data_format colmap --downscale 4

# =============================== [image gradient] v.s. [MSE degradation] ================================
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/blender/teamug/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/blender/plant/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/blender/spaceship/train
#
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/chair/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/drums/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/ficus/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/hotdog/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/lego/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/materials/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/mic/train
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/nerf_synthetic/ship/train
#
##python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/RTMV/V8
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/mipnerf360/bonsai/images
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/mipnerf360/room/images
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/mipnerf360/bicycle/images
#python image_gradient.py /home/weihangliu/Documents/code/proj/ngp_qat_v6/data/mipnerf360/garden/images

# ================================== ablation MDL ========================
# nerf_synthetic
#python main.py data/nerf_synthetic/chair     --workspace workspace/MDL_T2/nerf_synthetic/chair     -O -MDL --data_format nerf --T2
#python main.py data/nerf_synthetic/drums     --workspace workspace/MDL_T2/nerf_synthetic/drums     -O -MDL --data_format nerf --T2
#python main.py data/nerf_synthetic/lego      --workspace workspace/MDL_T2/nerf_synthetic/lego      -O -MDL --data_format nerf --T2
#python main.py data/nerf_synthetic/ship      --workspace workspace/MDL_T2/nerf_synthetic/ship      -O -MDL --data_format nerf --selfbound --bound 1.3 --T2
#python main.py data/nerf_synthetic/ficus     --workspace workspace/MDL_T2/nerf_synthetic/ficus     -O -MDL --data_format nerf --T2
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/MDL_T2/nerf_synthetic/hotdog    -O -MDL --data_format nerf --selfbound --bound 1.3 --T2
#python main.py data/nerf_synthetic/materials --workspace workspace/MDL_T2/nerf_synthetic/materials -O -MDL --data_format nerf --T2
#python main.py data/nerf_synthetic/mic       --workspace workspace/MDL_T2/nerf_synthetic/mic       -O -MDL --data_format nerf --T2

# RTMV
#python main.py data/RTMV/V8                 --workspace workspace/MDL_T2/RTMV/V8                  -O -MDL --data_format nerf --selfbound --bound 0.5 --T2

# mip-nerf 360
#python main.py data/mipnerf360/bonsai/      --workspace workspace/MDL_T2/mipnerf360/bonsai         -O -MDL --data_format colmap --downscale 4 --T2
#python main.py data/mipnerf360/bicycle/     --workspace workspace/MDL_T2/mipnerf360/bicycle        -O -MDL --data_format colmap --downscale 4 --T2
#python main.py data/mipnerf360/garden/      --workspace workspace/MDL_T2/mipnerf360/garden         -O -MDL --data_format colmap --downscale 4 --T2
#python main.py data/mipnerf360/room/        --workspace workspace/MDL_T2/mipnerf360/room           -O -MDL --data_format colmap --downscale 4 --T2


# ================================== ablation 30dB ========================
# nerf_synthetic
#python main.py data/nerf_synthetic/chair     --workspace workspace/MGL_T2_30dB/nerf_synthetic/chair     -O -MGL --data_format nerf --T2
#python main.py data/nerf_synthetic/drums     --workspace workspace/MGL_T2_30dB/nerf_synthetic/drums     -O -MGL --data_format nerf --T2
#python main.py data/nerf_synthetic/lego      --workspace workspace/MGL_T2_30dB/nerf_synthetic/lego      -O -MGL --data_format nerf --T2
#python main.py data/nerf_synthetic/ship      --workspace workspace/MGL_T2_30dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --T2
#python main.py data/nerf_synthetic/ficus     --workspace workspace/MGL_T2_30dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --T2
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/MGL_T2_30dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --T2
#python main.py data/nerf_synthetic/materials --workspace workspace/MGL_T2_30dB/nerf_synthetic/materials -O -MGL --data_format nerf --T2
#python main.py data/nerf_synthetic/mic       --workspace workspace/MGL_T2_30dB/nerf_synthetic/mic       -O -MGL --data_format nerf --T2

## RTMV
#python main.py data/RTMV/V8                 --workspace workspace/MGL_T2_30dB/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5 --T2

## mip-nerf 360
#python main.py data/mipnerf360/bonsai/      --workspace workspace/MGL_T2_30dB/mipnerf360/bonsai         -O -MGL --data_format colmap --downscale 4 --T2
#python main.py data/mipnerf360/bicycle/     --workspace workspace/MGL_T2_30dB/mipnerf360/bicycle        -O -MGL --data_format colmap --downscale 4 --T2
#python main.py data/mipnerf360/garden/      --workspace workspace/MGL_T2_30dB/mipnerf360/garden         -O -MGL --data_format colmap --downscale 4 --T2
#python main.py data/mipnerf360/room/        --workspace workspace/MGL_T2_30dB/mipnerf360/room           -O -MGL --data_format colmap --downscale 4 --T2
#
## ================================== ablation 32dB ========================
## nerf_synthetic
#python main.py data/nerf_synthetic/chair     --workspace workspace/MGL_T2_32dB/nerf_synthetic/chair     -O -MGL --data_format nerf --target 32 --T2
#python main.py data/nerf_synthetic/drums     --workspace workspace/MGL_T2_32dB/nerf_synthetic/drums     -O -MGL --data_format nerf --target 32 --T2
#python main.py data/nerf_synthetic/lego      --workspace workspace/MGL_T2_32dB/nerf_synthetic/lego      -O -MGL --data_format nerf --target 32 --T2
#python main.py data/nerf_synthetic/ship      --workspace workspace/MGL_T2_32dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --T2
#python main.py data/nerf_synthetic/ficus     --workspace workspace/MGL_T2_32dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --target 32 --T2
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/MGL_T2_32dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --T2
#python main.py data/nerf_synthetic/materials --workspace workspace/MGL_T2_32dB/nerf_synthetic/materials -O -MGL --data_format nerf --target 32 --T2
#python main.py data/nerf_synthetic/mic       --workspace workspace/MGL_T2_32dB/nerf_synthetic/mic       -O -MGL --data_format nerf --target 32 --T2

## RTMV
#python main.py data/RTMV/V8                 --workspace workspace/MGL_T2_32dB/RTMV/V8                  -O -MGL --data_format nerf --selfbound --bound 0.5 --target 32 --T2

## mip-nerf 360
#python main.py data/mipnerf360/bonsai/      --workspace workspace/MGL_T2_32dB/mipnerf360/bonsai         -O -MGL --data_format colmap --downscale 4 --target 32 --T2
#python main.py data/mipnerf360/bicycle/     --workspace workspace/MGL_T2_32dB/mipnerf360/bicycle        -O -MGL --data_format colmap --downscale 4 --target 32 --T2
#python main.py data/mipnerf360/garden/      --workspace workspace/MGL_T2_32dB/mipnerf360/garden         -O -MGL --data_format colmap --downscale 4 --target 32 --T2
#python main.py data/mipnerf360/room/        --workspace workspace/MGL_T2_32dB/mipnerf360/room           -O -MGL --data_format colmap --downscale 4 --target 32 --T2

# ================================== LSQ+ ========================
# nerf_synthetic
#python main.py data/nerf_synthetic/chair     --workspace workspace/LSQp_8bit/nerf_synthetic/chair     -O -MDL --data_format nerf --T1
#python main.py data/nerf_synthetic/drums     --workspace workspace/LSQp_8bit/nerf_synthetic/drums     -O -MDL --data_format nerf --T1
#python main.py data/nerf_synthetic/lego      --workspace workspace/LSQp_8bit/nerf_synthetic/lego      -O -MDL --data_format nerf --T1
#python main.py data/nerf_synthetic/ship      --workspace workspace/LSQp_8bit/nerf_synthetic/ship      -O -MDL --data_format nerf --selfbound --bound 1.3 --T1
#python main.py data/nerf_synthetic/ficus     --workspace workspace/LSQp_8bit/nerf_synthetic/ficus     -O -MDL --data_format nerf --T1
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/LSQp_8bit/nerf_synthetic/hotdog    -O -MDL --data_format nerf --selfbound --bound 1.3 --T1
#python main.py data/nerf_synthetic/materials --workspace workspace/LSQp_8bit/nerf_synthetic/materials -O -MDL --data_format nerf --T1
#python main.py data/nerf_synthetic/mic       --workspace workspace/LSQp_8bit/nerf_synthetic/mic       -O -MDL --data_format nerf --T1

# RTMV
#python main.py data/RTMV/V8                 --workspace workspace/LSQp_8bit/RTMV/V8                  -O -MDL --data_format nerf --selfbound --bound 0.5 --T1

# mip-nerf 360
#python main.py data/mipnerf360/bonsai/      --workspace workspace/LSQp_8bit/mipnerf360/bonsai         -O -MDL --data_format colmap --downscale 4 --T1
#python main.py data/mipnerf360/bicycle/     --workspace workspace/LSQp_8bit/mipnerf360/bicycle        -O -MDL --data_format colmap --downscale 4 --T1
#python main.py data/mipnerf360/garden/      --workspace workspace/LSQp_8bit/mipnerf360/garden         -O -MDL --data_format colmap --downscale 4 --T1
#python main.py data/mipnerf360/room/        --workspace workspace/LSQp_8bit/mipnerf360/room           -O -MDL --data_format colmap --downscale 4 --T1
#python main.py data/mipnerf360/counter/       --workspace workspace/LSQp_8bit/mipnerf360/counter        -O -MDL --data_format colmap --downscale 4 --T1
#python main.py data/mipnerf360/kitchen/      --workspace workspace/LSQp_8bit/mipnerf360/kitchen         -O -MDL --data_format colmap --downscale 4 --T1
#python main.py data/mipnerf360/stump/        --workspace workspace/LSQp_8bit/mipnerf360/stump           -O -MDL --data_format colmap --downscale 4 --T1
