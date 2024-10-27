import os
import torch
import numpy as np

def calBitOps_CAQ(
        model,
        num_samples_each_image,
        skip_rate,
        hash_size,
        image_size=(800, 800),
):
    """
    Only counting multiplication
    """
    model_size = 0

    bitset = []
    for name, module in model.named_modules():
        if ('qo' in name) or ('qi' in name) or ('qw' in name):
            bitset.append(module.hard_bit)
    T = 1e12
    MB = 1024 * 1024 * 8
    # sum = 0
    ray_num = image_size[0] * image_size[1]
    # per_ray_bitOps = 0  # 每条光线需要的bitops
    per_point_bitOps = 0  # 每个采样点需要的bitops

    # (1)------Interpolation------
    # ------Trilinear Interpolation------
    encoder = model.qencoder
    L = encoder.encoder.num_levels
    F = encoder.encoder.level_dim
    hash_bit = bitset[1]
    cord_bit = 32
    TriInterOps = 0
    TriInterOps += 3 * cord_bit * cord_bit
    TriInterOps += L * F * ((4 + 8) * cord_bit * cord_bit + 8 * cord_bit * hash_bit)
    per_point_bitOps += TriInterOps
    # print('TriInterOps per point:',TriInterOps)
    model_size += hash_bit * hash_size

    # # (1-2)------Bilinear Interpolation------
    # L = 4
    # F = 2
    # hash_bit = 1
    # cord_bit = 32
    # BiInterOps = 0
    # BiInterOps += 2 * cord_bit * cord_bit
    # BiInterOps += L * F * ((2 + 4) * cord_bit * cord_bit + 4 * cord_bit * hash_bit)
    # per_point_bitOps += BiInterOps * 3
    # # print('BiInterOps per point:',BiInterOps)

    # (2)------SigmaNet------
    encoder_qo_bit = bitset[0]
    sigmanet0_qw_bit = bitset[3]
    sigmanet1_q0_bit = bitset[4]
    sigmanet2_qw_bit = bitset[6]
    sigma_in_dim = 32
    sigma0_dim = 64
    sigma2_dim = 16
    sigma_net_BitOps = 0
    # layer 0
    sigma_net_BitOps += (sigma_in_dim * sigma0_dim) * (encoder_qo_bit * sigmanet0_qw_bit)
    # layer 2
    sigma_net_BitOps += (sigma0_dim * sigma2_dim) * (sigmanet1_q0_bit * sigmanet2_qw_bit)
    per_point_bitOps += sigma_net_BitOps
    # print('sigma_net_BitOps per point:',sigma_net_BitOps)
    model_size += sigma_in_dim * sigma0_dim * sigmanet0_qw_bit + \
                  sigma0_dim * sigma2_dim * sigmanet2_qw_bit

    # (3)------ColorNet------
    # ColorNet：
    sigma_out_bit = bitset[5]
    dir_bit = bitset[7]
    color0_qw_bit = bitset[10]
    color1_qo_bit = bitset[11]
    color2_qw_bit = bitset[12]
    color3_qo_bit = bitset[13]
    color4_qw_bit = bitset[14]
    color_act_in = bitset[8]

    sigma_out_dim = 15
    dir_dim = 16
    color0_dim = 64
    color2_dim = 64
    color4_dim = 3

    color_net_BitOps = 0
    # layer 0
    color_net_BitOps += (sigma_out_dim * color0_dim) * (sigma_out_bit * color0_qw_bit) + \
                        dir_dim * color0_dim * dir_bit * color0_qw_bit

    # layer 2
    color_net_BitOps += (1-skip_rate) * (color0_dim * color2_dim) * (color1_qo_bit * color2_qw_bit)
    # layer 4
    color_net_BitOps += (color2_dim * color4_dim) * (color3_qo_bit * color4_qw_bit)
    # color activation
    color_net_BitOps += 3 * color_act_in
    per_point_bitOps += color_net_BitOps
    # print('color_net_BitOps per point:',color_net_BitOps)
    model_size += (sigma_out_dim + dir_dim) * color0_dim * color0_qw_bit + \
                  color0_dim * color2_dim * color2_qw_bit + \
                  color2_dim * color4_dim * color4_qw_bit

    # (4)------volume rendering------
    sigma_bit = bitset[2]
    color_bit = bitset[9]
    volRenderOps = 0
    num_samples_per_ray = num_samples_each_image / ray_num
    # print('samples per ray:',num_samples_per_ray)
    volRenderOps += sigma_bit**2 * num_samples_per_ray * (num_samples_per_ray - 1) / 2
    volRenderOps += num_samples_per_ray * sigma_bit * color_bit
    # print('volRenderOps per ray:',volRenderOps)

    # ------Sum------
    per_ray_bitOps = per_point_bitOps * num_samples_per_ray + volRenderOps
    sum = per_ray_bitOps * ray_num
    # print('bitOps per image:',sum / T, 'T')
    return sum / T, model_size / MB

def calBitOps_CA(
        model,
        num_samples_each_image,
        skip_rate,
        hash_size,
        image_size=(800, 800),
):
    """
    Only counting multiplication
    """
    model_size = 0

    bitset = torch.ones(15, device='cuda') * 32

    T = 1e12
    MB = 1024 * 1024 * 8
    # sum = 0
    ray_num = image_size[0] * image_size[1]
    # per_ray_bitOps = 0  # 每条光线需要的bitops
    per_point_bitOps = 0  # 每个采样点需要的bitops

    # (1)------Interpolation------
    # ------Trilinear Interpolation------
    encoder = model.qencoder
    L = encoder.encoder.num_levels
    F = encoder.encoder.level_dim
    hash_bit = bitset[1]
    cord_bit = 32
    TriInterOps = 0
    TriInterOps += 3 * cord_bit * cord_bit
    TriInterOps += L * F * ((4 + 8) * cord_bit * cord_bit + 8 * cord_bit * hash_bit)
    per_point_bitOps += TriInterOps
    # print('TriInterOps per point:',TriInterOps)
    model_size += hash_bit * hash_size

    # # (1-2)------Bilinear Interpolation------
    # L = 4
    # F = 2
    # hash_bit = 1
    # cord_bit = 32
    # BiInterOps = 0
    # BiInterOps += 2 * cord_bit * cord_bit
    # BiInterOps += L * F * ((2 + 4) * cord_bit * cord_bit + 4 * cord_bit * hash_bit)
    # per_point_bitOps += BiInterOps * 3
    # # print('BiInterOps per point:',BiInterOps)

    # (2)------SigmaNet------
    encoder_qo_bit = bitset[0]
    sigmanet0_qw_bit = bitset[3]
    sigmanet1_q0_bit = bitset[4]
    sigmanet2_qw_bit = bitset[6]
    sigma_in_dim = 32
    sigma0_dim = 64
    sigma2_dim = 16
    sigma_net_BitOps = 0
    # layer 0
    sigma_net_BitOps += (sigma_in_dim * sigma0_dim) * (encoder_qo_bit * sigmanet0_qw_bit)
    # layer 2
    sigma_net_BitOps += (sigma0_dim * sigma2_dim) * (sigmanet1_q0_bit * sigmanet2_qw_bit)
    per_point_bitOps += sigma_net_BitOps
    # print('sigma_net_BitOps per point:',sigma_net_BitOps)
    model_size += sigma_in_dim * sigma0_dim * sigmanet0_qw_bit + \
                  sigma0_dim * sigma2_dim * sigmanet2_qw_bit

    # (3)------ColorNet------
    # ColorNet：
    sigma_out_bit = bitset[5]
    dir_bit = bitset[7]
    color0_qw_bit = bitset[10]
    color1_qo_bit = bitset[11]
    color2_qw_bit = bitset[12]
    color3_qo_bit = bitset[13]
    color4_qw_bit = bitset[14]
    color_act_in = bitset[8]

    sigma_out_dim = 15
    dir_dim = 16
    color0_dim = 64
    color2_dim = 64
    color4_dim = 3

    color_net_BitOps = 0
    # layer 0
    color_net_BitOps += (sigma_out_dim * color0_dim) * (sigma_out_bit * color0_qw_bit) + \
                        dir_dim * color0_dim * dir_bit * color0_qw_bit

    # layer 2
    color_net_BitOps += (1-skip_rate) * (color0_dim * color2_dim) * (color1_qo_bit * color2_qw_bit)
    # layer 4
    color_net_BitOps += (color2_dim * color4_dim) * (color3_qo_bit * color4_qw_bit)
    # color activation
    color_net_BitOps += 3 * color_act_in
    per_point_bitOps += color_net_BitOps
    # print('color_net_BitOps per point:',color_net_BitOps)
    model_size += (sigma_out_dim + dir_dim) * color0_dim * color0_qw_bit + \
                  color0_dim * color2_dim * color2_qw_bit + \
                  color2_dim * color4_dim * color4_qw_bit

    # (4)------volume rendering------
    sigma_bit = bitset[2]
    color_bit = bitset[9]
    volRenderOps = 0
    num_samples_per_ray = num_samples_each_image / ray_num
    # print('samples per ray:',num_samples_per_ray)
    volRenderOps += sigma_bit**2 * num_samples_per_ray * (num_samples_per_ray - 1) / 2
    volRenderOps += num_samples_per_ray * sigma_bit * color_bit
    # print('volRenderOps per ray:',volRenderOps)

    # ------Sum------
    per_ray_bitOps = per_point_bitOps * num_samples_per_ray + volRenderOps
    sum = per_ray_bitOps * ray_num
    # print('bitOps per image:',sum / T, 'T')
    return sum / T, model_size / MB


def calBitOps_ngp(
        model,
        num_samples_each_image,
        image_size=(800, 800),
):
    """
    Only counting multiplication
    """
    model_size = 0

    bitset = torch.ones(15, device='cuda') * 32

    hash_size = 0
    max_params = 2 ** 19
    for i in range(model.encoder.num_levels):
        resolution = int(np.ceil(model.encoder.base_resolution * model.encoder.per_level_scale ** i))
        max_params_in_level = resolution ** model.encoder.input_dim
        params_in_level = min(max_params, max_params_in_level)  # limit max number
        params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
        hash_size += params_in_level
    hash_size = hash_size * model.encoder.level_dim
    T = 1e12
    MB = 1024 * 1024 * 8
    # sum = 0
    ray_num = image_size[0] * image_size[1]
    # per_ray_bitOps = 0  # 每条光线需要的bitops
    per_point_bitOps = 0  # 每个采样点需要的bitops

    # (1)------Interpolation------
    # ------Trilinear Interpolation------
    encoder = model.qencoder
    L = encoder.encoder.num_levels
    F = encoder.encoder.level_dim
    hash_bit = bitset[1]
    cord_bit = 32
    TriInterOps = 0
    TriInterOps += 3 * cord_bit * cord_bit
    TriInterOps += L * F * ((4 + 8) * cord_bit * cord_bit + 8 * cord_bit * hash_bit)
    per_point_bitOps += TriInterOps
    # print('TriInterOps per point:',TriInterOps)
    model_size += hash_bit * hash_size

    # # (1-2)------Bilinear Interpolation------
    # L = 4
    # F = 2
    # hash_bit = 1
    # cord_bit = 32
    # BiInterOps = 0
    # BiInterOps += 2 * cord_bit * cord_bit
    # BiInterOps += L * F * ((2 + 4) * cord_bit * cord_bit + 4 * cord_bit * hash_bit)
    # per_point_bitOps += BiInterOps * 3
    # # print('BiInterOps per point:',BiInterOps)

    # (2)------SigmaNet------
    encoder_qo_bit = bitset[0]
    sigmanet0_qw_bit = bitset[3]
    sigmanet1_q0_bit = bitset[4]
    sigmanet2_qw_bit = bitset[6]
    sigma_in_dim = 32
    sigma0_dim = 64
    sigma2_dim = 16
    sigma_net_BitOps = 0
    # layer 0
    sigma_net_BitOps += (sigma_in_dim * sigma0_dim) * (encoder_qo_bit * sigmanet0_qw_bit)
    # layer 2
    sigma_net_BitOps += (sigma0_dim * sigma2_dim) * (sigmanet1_q0_bit * sigmanet2_qw_bit)
    per_point_bitOps += sigma_net_BitOps
    # print('sigma_net_BitOps per point:',sigma_net_BitOps)
    model_size += sigma_in_dim * sigma0_dim * sigmanet0_qw_bit + \
                  sigma0_dim * sigma2_dim * sigmanet2_qw_bit

    # (3)------ColorNet------
    # ColorNet：
    sigma_out_bit = bitset[5]
    dir_bit = bitset[7]
    color0_qw_bit = bitset[10]
    color1_qo_bit = bitset[11]
    color2_qw_bit = bitset[12]
    color3_qo_bit = bitset[13]
    color4_qw_bit = bitset[14]
    color_act_in = bitset[8]

    sigma_out_dim = 15
    dir_dim = 16
    color0_dim = 64
    color2_dim = 64
    color4_dim = 3

    color_net_BitOps = 0
    # layer 0
    color_net_BitOps += (sigma_out_dim * color0_dim) * (sigma_out_bit * color0_qw_bit) + \
                        dir_dim * color0_dim * dir_bit * color0_qw_bit

    # layer 2
    color_net_BitOps += (color0_dim * color2_dim) * (color1_qo_bit * color2_qw_bit)
    # layer 4
    color_net_BitOps += (color2_dim * color4_dim) * (color3_qo_bit * color4_qw_bit)
    # color activation
    color_net_BitOps += 3 * color_act_in
    per_point_bitOps += color_net_BitOps
    # print('color_net_BitOps per point:',color_net_BitOps)
    model_size += (sigma_out_dim + dir_dim) * color0_dim * color0_qw_bit + \
                  color0_dim * color2_dim * color2_qw_bit + \
                  color2_dim * color4_dim * color4_qw_bit

    # (4)------volume rendering------
    sigma_bit = bitset[2]
    color_bit = bitset[9]
    volRenderOps = 0
    num_samples_per_ray = num_samples_each_image / ray_num
    # print('samples per ray:',num_samples_per_ray)
    volRenderOps += sigma_bit ** 2 * num_samples_per_ray * (num_samples_per_ray - 1) / 2
    volRenderOps += num_samples_per_ray * sigma_bit * color_bit
    # print('volRenderOps per ray:',volRenderOps)

    # ------Sum------
    per_ray_bitOps = per_point_bitOps * num_samples_per_ray + volRenderOps
    sum = per_ray_bitOps * ray_num
    # print('bitOps per image:',sum / T, 'T')
    return sum / T, model_size / MB

def quan_analysis(model):
    print(model)
    print('\n Quantization Results: \n')
    print('------------------------------------')
    # Qencoder
    print(f"Qencoder    : qw--scale({model.qencoder.qw.scale}); qw--zero_point({model.qencoder.qw.soft_zero.data}); \n")
    print(f"Qencoder    : qo--scale({model.qencoder.qo.scale}); qo--zero_point({model.qencoder.qo.soft_zero.data}); \n")
    print(f"Qencoder    : qo--max({model.qencoder.qo.max}); qo--min({model.qencoder.qo.min}); \n")
    # QsigmaNet
    print(f"QsigmaNet(0): qw--scale({model.qsigma_net.Qsigma_net[0].qw.scale}); "
          f"qw--zero_point({model.qsigma_net.Qsigma_net[0].qw.soft_zero.data}); \n")
    print(f"QsigmaNet(0): qo--scale({model.qsigma_net.Qsigma_net[0].qo.scale}); "
          f"qo--zero_point({model.qsigma_net.Qsigma_net[0].qo.soft_zero.data}); \n")
    print(f"QsigmaNet(0): qo--max({model.qsigma_net.Qsigma_net[0].qo.max}); "
          f"qo--min({model.qsigma_net.Qsigma_net[0].qo.min}); \n")
    print(f"QsigmaNet(1): qi--scale({model.qsigma_net.Qsigma_net[1].qi.scale}); "
          f"qi--zero_point({model.qsigma_net.Qsigma_net[1].qi.soft_zero.data}); \n")
    print(f"QsigmaNet(1): qi--max({model.qsigma_net.Qsigma_net[1].qi.max}); "
          f"qi--min({model.qsigma_net.Qsigma_net[1].qi.min}); \n")
    print(f"QsigmaNet(2): qw--scale({model.qsigma_net.Qsigma_net[2].qw.scale}); "
          f"qw--zero_point({model.qsigma_net.Qsigma_net[2].qw.soft_zero.data}); \n")
    print(f"QsigmaNet(2): qo--scale({model.qsigma_net.Qsigma_net[2].qo.scale}); "
          f"qo--zero_point({model.qsigma_net.Qsigma_net[2].qo.soft_zero.data}); \n")
    print(f"QsigmaNet(2): qo--max({model.qsigma_net.Qsigma_net[2].qo.max}); "
          f"qo--min({model.qsigma_net.Qsigma_net[2].qo.min}); \n")
    # QencoderDir
    # print(f'QencoderDir : ', 'qw: scale({model.qencoder_dir.qw.scale}); '
    #                          'qw: scale({model.qencoder_dir.qw.zero_point}); \n')
    print(f"QencoderDir : qo--scale({model.qencoder_dir.qo.scale}); "
          f"qo--zero_point({model.qencoder_dir.qo.soft_zero.data}); \n")
    print(f"QencoderDir : qo--max({model.qencoder_dir.qo.max}); "
          f"qo--min({model.qencoder_dir.qo.min}); \n")
    #QcolorNet
    print(f"QcolorNet(0): qw--scale({model.qcolor_net.Qcolor_net[0].qw.scale}); "
          f"qw--zero_point({model.qcolor_net.Qcolor_net[0].qw.soft_zero.data}); \n")
    print(f"QcolorNet(0): qo--scale({model.qcolor_net.Qcolor_net[0].qo.scale}); "
          f"qo--zero_point({model.qcolor_net.Qcolor_net[0].qo.soft_zero.data}); \n")
    print(f"QcolorNet(0): qo--max({model.qcolor_net.Qcolor_net[0].qo.max}); "
          f"qo--min({model.qcolor_net.Qcolor_net[0].qo.min}); \n")

    print(f"QcolorNet(1): qi--scale({model.qcolor_net.Qcolor_net[1].qi.scale}); "
          f"qi--zero_point({model.qcolor_net.Qcolor_net[1].qi.soft_zero.data}); \n")
    print(f"QcolorNet(1): qi--max({model.qcolor_net.Qcolor_net[1].qi.max}); "
          f"qi--min({model.qcolor_net.Qcolor_net[1].qi.min}); \n")

    print(f"QcolorNet(2): qw--scale({model.qcolor_net.Qcolor_net[2].qw.scale}); "
          f"qw--zero_point({model.qcolor_net.Qcolor_net[2].qw.soft_zero.data}); \n")
    print(f"QcolorNet(2): qo--scale({model.qcolor_net.Qcolor_net[2].qo.scale}); "
          f"qo--zero_point({model.qcolor_net.Qcolor_net[2].qo.soft_zero.data}); \n")
    print(f"QcolorNet(2): qo--max({model.qcolor_net.Qcolor_net[2].qo.max}); "
          f"qo--min({model.qcolor_net.Qcolor_net[2].qo.min}); \n")

    print(f"QcolorNet(3): qi--scale({model.qcolor_net.Qcolor_net[3].qi.scale}); "
          f"qi--zero_point({model.qcolor_net.Qcolor_net[3].qi.soft_zero.data}); \n")
    print(f"QcolorNet(3): qi--max({model.qcolor_net.Qcolor_net[3].qi.max}); "
          f"qi--min({model.qcolor_net.Qcolor_net[3].qi.min}); \n")

    print(f"QcolorNet(4): qw--scale({model.qcolor_net.Qcolor_net[4].qw.scale}); "
          f"qw--zero_point({model.qcolor_net.Qcolor_net[4].qw.soft_zero.data}); \n")
    print(f"QcolorNet(4): qo--scale({model.qcolor_net.Qcolor_net[4].qo.scale}); "
          f"qo--zero_point({model.qcolor_net.Qcolor_net[4].qo.soft_zero.data}); \n")
    print(f"QcolorNet(4): qo--max({model.qcolor_net.Qcolor_net[4].qo.max}); "
          f"qo--min({model.qcolor_net.Qcolor_net[4].qo.min}); \n")


def quan_analysis2(model, workspace=None, filename=None, valid_loader=None, trainer_qat=None):
    for name, module in model.named_modules():
        if ('qo' in name) or ('qi' in name) or ('qw' in name):
            print(name, module)

    log_ptr = None
    if workspace is not None:
        os.makedirs(workspace, exist_ok=True)
        log_path = os.path.join(workspace, f"log_{filename}.txt")
        log_ptr = open(log_path, "a+")

    # 将要写入的信息写入到 log_ptr 文件中
    if log_ptr is not None:
        for name, module in model.named_modules():
            if ('qo' in name) or ('qi' in name) or ('qw' in name):
                log_ptr.write(f"{name}: {module}\n")

    # 计算渲染一张图像的BitOps
    if trainer_qat is not None and valid_loader is not None:
        loss_val_q, result, sample_points_per_image_ave = trainer_qat.evaluate(valid_loader,
                                                                               name='qat with quan. learning')
        bitops_per_image_caq, model_size_caq = calBitOps_CAQ(model=model,
                                     num_samples_each_image=sample_points_per_image_ave,
                                     skip_rate = model.skip_rate,
                                     hash_size = model.encoder.embeddings.shape[0] * model.encoder.embeddings.shape[1],
                                     image_size=(valid_loader._data.W, valid_loader._data.H))

        bitops_per_image_ca, model_size_ca = calBitOps_CA(model=model,
                                     num_samples_each_image=sample_points_per_image_ave,
                                     skip_rate = model.skip_rate,
                                     hash_size = model.encoder.embeddings.shape[0] * model.encoder.embeddings.shape[1],
                                     image_size=(valid_loader._data.W, valid_loader._data.H))

        bitops_per_image_ngp, model_size_ngp = calBitOps_ngp(model=model,
                                     num_samples_each_image=sample_points_per_image_ave,
                                     image_size=(valid_loader._data.W, valid_loader._data.H))



        if log_ptr is not None:
            log_ptr.write(f"loss val: {loss_val_q}\n")
            log_ptr.write(f"CAQ BitOps for each image (T): {bitops_per_image_caq.item()}\n")
            log_ptr.write(f"CAQ model size (MB): {model_size_caq.item()}\n")
            log_ptr.write(f"CA BitOps for each image (T): {bitops_per_image_ca.item()}\n")
            log_ptr.write(f"CA model size (MB): {model_size_ca.item()}\n")
            log_ptr.write(f"NGP BitOps for each image (T): {bitops_per_image_ngp.item()}\n")
            log_ptr.write(f"NGP model size (MB): {model_size_ngp.item()}\n")

    # 关闭文件对象
    if log_ptr is not None:
        log_ptr.close()
