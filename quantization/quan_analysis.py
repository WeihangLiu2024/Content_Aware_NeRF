import os


def calBitOps(model, num_samples_each_image, image_size=(800, 800)):
    """
    Only counting multiplication
    """
    bitset = []
    for name, module in model.named_modules():
        if ('qo' in name) or ('qi' in name) or ('qw' in name):
            bitset.append(int(module.hard_bit))
    T = 1e12
    sum = 0
    ray_num = image_size[0] * image_size[1]
    per_ray_bitOps = 0  # 每条光线需要的bitops
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

    # (3)------ColorNet------
    # ColorNet：
    sigma_out_bit = bitset[5]
    dir_bit = bitset[7]
    color0_qw_bit = bitset[10]
    color1_qo_bit = bitset[11]
    color2_qw_bit = bitset[12]
    color3_qo_bit = bitset[13]
    color4_qw_bit = bitset[14]

    sigma_out_dim = 15
    dir_dim = 16
    color0_dim = 64
    color2_dim = 64
    color4_dim = 3

    color_net_BitOps = 0
    # layer 0
    color_net_BitOps += (sigma_out_dim * color0_dim) * (sigma_out_bit * color0_qw_bit)
    color_net_BitOps += (dir_dim * color0_dim) * (dir_bit * color0_qw_bit)
    # layer 2
    color_net_BitOps += (color0_dim * color2_dim) * (color1_qo_bit * color2_qw_bit)
    # layer 4
    color_net_BitOps += (color2_dim * color4_dim) * (color3_qo_bit * color4_qw_bit)
    per_point_bitOps += color_net_BitOps
    # print('color_net_BitOps per point:',color_net_BitOps)

    # (4)------volume rendering------
    sigma_bit = bitset[2]
    color_bit = bitset[9]
    volRenderOps = 0
    num_samples_per_ray = num_samples_each_image / ray_num
    # print('samples per ray:',num_samples_per_ray)
    volRenderOps += sigma_bit * num_samples_per_ray * (num_samples_per_ray - 1) / 2
    volRenderOps += num_samples_per_ray * sigma_bit * color_bit
    # print('volRenderOps per ray:',volRenderOps)

    # ------Sum------
    per_ray_bitOps = per_point_bitOps * num_samples_per_ray + volRenderOps
    sum = per_ray_bitOps * ray_num
    # print('bitOps per image:',sum / T, 'T')
    return sum / T


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
        loss_val_q, result, sample_points_per_image_ave = trainer_qat.evaluate(valid_loader, name='qat with quan. learning')
        bitops_per_image = calBitOps(model=model, 
                                    num_samples_each_image=sample_points_per_image_ave,
                                    image_size=(valid_loader._data.W, valid_loader._data.H))
        if log_ptr is not None:
            log_ptr.write(f"loss val: {loss_val_q}\n")
            log_ptr.write(f"BitOps for each image (T): {bitops_per_image}\n")

    # 关闭文件对象
    if log_ptr is not None:
        log_ptr.close()