# =========================================================== MGL =======================================================
## synthetic NeRF
#python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/chair     -O -MGL --data_format nerf --weight_penalty 1e-4
#python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/drums     -O -MGL --data_format nerf --weight_penalty 1e-4
#python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/lego      -O -MGL --data_format nerf --weight_penalty 1e-4
#python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-4
#python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --weight_penalty 1e-4
#python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-4
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/materials -O -MGL --data_format nerf --weight_penalty 1e-4
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_10_4/MGL_30dB/nerf_synthetic/mic       -O -MGL --data_format nerf --weight_penalty 1e-4

## ===================================================== MGL2 target 32 ==================================================
## synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/chair     -O -MGL --data_format nerf --target 32 --weight_penalty 1e-4
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/drums     -O -MGL --data_format nerf --target 32 --weight_penalty 1e-4
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/lego      -O -MGL --data_format nerf --target 32 --weight_penalty 1e-4
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --weight_penalty 1e-4
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --target 32 --weight_penalty 1e-4
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --weight_penalty 1e-4
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/materials -O -MGL --data_format nerf --target 32 --weight_penalty 1e-4
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_10_4/MGL_32dB/nerf_synthetic/mic       -O -MGL --data_format nerf --target 32 --weight_penalty 1e-4

## ========================================================== MDL ========================================================
## synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_10_4/MDL/nerf_synthetic/chair     -O -MDL --data_format nerf --weight_penalty 1e-4
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_10_4/MDL/nerf_synthetic/drums     -O -MDL --data_format nerf --weight_penalty 1e-4
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_10_4/MDL/nerf_synthetic/lego      -O -MDL --data_format nerf --weight_penalty 1e-4
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_10_4/MDL/nerf_synthetic/ship      -O -MDL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-4
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_10_4/MDL/nerf_synthetic/ficus     -O -MDL --data_format nerf --weight_penalty 1e-4
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_10_4/MDL/nerf_synthetic/hotdog    -O -MDL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-4
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_10_4/MDL/nerf_synthetic/materials -O -MDL --data_format nerf --weight_penalty 1e-4
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_10_4/MDL/nerf_synthetic/mic       -O -MDL --data_format nerf --weight_penalty 1e-4

# ========================================================= 5 * 10^-4 ===================================================
# =========================================================== MGL =======================================================
# synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/chair     -O -MGL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/drums     -O -MGL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/lego      -O -MGL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --weight_penalty 5e-4
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --weight_penalty 5e-4
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/materials -O -MGL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_5_10_4/MGL_30dB/nerf_synthetic/mic       -O -MGL --data_format nerf --weight_penalty 5e-4

## ===================================================== MGL2 target 32 ==================================================
## synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/chair     -O -MGL --data_format nerf --target 32 --weight_penalty 5e-4
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/drums     -O -MGL --data_format nerf --target 32 --weight_penalty 5e-4
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/lego      -O -MGL --data_format nerf --target 32 --weight_penalty 5e-4
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --weight_penalty 5e-4
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --target 32 --weight_penalty 5e-4
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --weight_penalty 5e-4
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/materials -O -MGL --data_format nerf --target 32 --weight_penalty 5e-4
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_5_10_4/MGL_32dB/nerf_synthetic/mic       -O -MGL --data_format nerf --target 32 --weight_penalty 5e-4

## ========================================================== MDL ========================================================
## synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/chair     -O -MDL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/drums     -O -MDL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/lego      -O -MDL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/ship      -O -MDL --data_format nerf --selfbound --bound 1.3 --weight_penalty 5e-4
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/ficus     -O -MDL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/hotdog    -O -MDL --data_format nerf --selfbound --bound 1.3 --weight_penalty 5e-4
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/materials -O -MDL --data_format nerf --weight_penalty 5e-4
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_5_10_4/MDL/nerf_synthetic/mic       -O -MDL --data_format nerf --weight_penalty 5e-4


# ========================================================= 1 * 10^-5 ===================================================
# =========================================================== MGL =======================================================
# synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/chair     -O -MGL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/drums     -O -MGL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/lego      -O -MGL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-5
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-5
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/materials -O -MGL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_10_5/MGL_30dB/nerf_synthetic/mic       -O -MGL --data_format nerf --weight_penalty 1e-5

## ===================================================== MGL2 target 32 ==================================================
## synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/chair     -O -MGL --data_format nerf --target 32 --weight_penalty 1e-5
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/drums     -O -MGL --data_format nerf --target 32 --weight_penalty 1e-5
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/lego      -O -MGL --data_format nerf --target 32 --weight_penalty 1e-5
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/ship      -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --weight_penalty 1e-5
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/ficus     -O -MGL --data_format nerf --target 32 --weight_penalty 1e-5
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/hotdog    -O -MGL --data_format nerf --selfbound --bound 1.3 --target 32 --weight_penalty 1e-5
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/materials -O -MGL --data_format nerf --target 32 --weight_penalty 1e-5
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_10_5/MGL_32dB/nerf_synthetic/mic       -O -MGL --data_format nerf --target 32 --weight_penalty 1e-5

## ========================================================== MDL ========================================================
## synthetic NeRF
python main.py data/nerf_synthetic/chair     --workspace workspace/penalty_10_5/MDL/nerf_synthetic/chair     -O -MDL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/drums     --workspace workspace/penalty_10_5/MDL/nerf_synthetic/drums     -O -MDL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/lego      --workspace workspace/penalty_10_5/MDL/nerf_synthetic/lego      -O -MDL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/ship      --workspace workspace/penalty_10_5/MDL/nerf_synthetic/ship      -O -MDL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-5
python main.py data/nerf_synthetic/ficus     --workspace workspace/penalty_10_5/MDL/nerf_synthetic/ficus     -O -MDL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/hotdog    --workspace workspace/penalty_10_5/MDL/nerf_synthetic/hotdog    -O -MDL --data_format nerf --selfbound --bound 1.3 --weight_penalty 1e-5
python main.py data/nerf_synthetic/materials --workspace workspace/penalty_10_5/MDL/nerf_synthetic/materials -O -MDL --data_format nerf --weight_penalty 1e-5
python main.py data/nerf_synthetic/mic       --workspace workspace/penalty_10_5/MDL/nerf_synthetic/mic       -O -MDL --data_format nerf --weight_penalty 1e-5
