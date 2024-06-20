import os

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


def quan_analysis2(model, workspace=None, filename=None):
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

    # 关闭文件对象
    if log_ptr is not None:
        log_ptr.close()