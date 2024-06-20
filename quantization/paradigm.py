from nerf.utils import *
from .quan_analysis import *
from .utils import *


def ptq(model, train_loader, device):
    print('PTQ starts \n')
    # direct pass calibration data w/o training
    # use QParam.update to update quan. parameter,
    # in case of BN layer changes in model.train(), use model.eval() instead
    model.to(device)
    model.eval()

    progress_bar = tqdm.tqdm(total=100, desc=f'PTQ')
    for i, data in enumerate(train_loader):
        rays_o = data['rays_o'] # [B(batch), N(sampled rays), 3(rgb)]
        rays_d = data['rays_d'] # [B, N, 3]
        _ = model.calibration_render(rays_o, rays_d,
                                     staged=False,
                                     bg_color=None,
                                     perturb=True,
                                     force_all_rays=True)
        progress_bar.update(1)
        if i >= 100:
            break
    progress_bar.close()

    # set manually
    offset1 = (model.qcolor_net.Qcolor_net[1].qo.var_max - 50.) * model.qcolor_net.Qcolor_net[1].qo.hard_bit / 32
    offset2 = (model.qcolor_net.Qcolor_net[3].qo.var_max - 22.) * model.qcolor_net.Qcolor_net[3].qo.hard_bit / 32
    # offset3_min = (model.qcolor_net.Qcolor_act.qi.var_min + 40.) * model.qcolor_net.Qcolor_act.qi.hard_bit / 32
    # offset3_max = (model.qcolor_net.Qcolor_act.qi.var_max - 30.) * model.qcolor_net.Qcolor_act.qi.hard_bit / 32
    # offset1 = 0
    # offset2 = 0
    # offset3_min = 0
    # offset3_max = 0

    model.qcolor_net.Qcolor_net[1].qo.manual_min_max(var_max=50.+offset1)
    model.qcolor_net.Qcolor_net[3].qo.manual_min_max(var_max=22.+offset2)
    # model.qcolor_net.Qcolor_act.qi.manual_min_max(-40. + offset3_min, 30. + offset3_max)

    # model.qcolor_net.Qcolor_net[1].qo.manual_min_max(var_max=50.)
    # model.qcolor_net.Qcolor_net[3].qo.manual_min_max(var_max=22.)
    # model.qcolor_net.Qcolor_act.qi.manual_min_max(-40., 30.)

    print('direct quantization finish')
    quan_analysis2(model, filename="ptq_result")


def qat(trainer_qat, train_loader, valid_loader, qat_epoch):
    # model can be:
    #     1) w/o any initialization;
    #     2) initialized by ptq
    trainer_qat.train(train_loader, valid_loader, qat_epoch)
