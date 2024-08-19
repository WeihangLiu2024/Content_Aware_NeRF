import os
import subprocess
from nerf.utils import *

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
# print(f'GPU selected: {os.environ["CUDA_VISIBLE_DEVICES"]}')

import torch
import argparse
import torch.optim as optim

from nerf.gui import NeRFGUI
from config.GetConfig import get_config
from nerf.network import NeRFNetwork

from quantization.paradigm import *

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    # 1) ===== fully configuration of running condition =====
    opt = get_config()
    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':  # [not use yet]
        from nerf.dtu_provider import NeRFDataset
    elif opt.data_format == 'nsvf' or opt.data_format == 'tank':
        from nerf.NsvfAndTanks.provider import NeRFDataset
    else: # nerf
        from nerf.provider import NeRFDataset
    seed_everything(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    # 2) ===== get NeRF network definition =====
    model = NeRFNetwork(opt)

    # 3) ===== main body =====
    if opt.test:  # loading a well-trained model, and run test & evaluation

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion,
                          fp16=opt.fp16, use_checkpoint=opt.ckpt)
        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        else:
            if not opt.test_no_video:
                test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

                if test_loader.has_gt:
                    trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)] # set up metrics
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True) # test and save video
            
            # if not opt.test_no_mesh:
            #     # need train loader to get camera poses for visibility test
            #     if opt.mesh_visibility_culling:
            #         train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
            #     trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
    else:  # training, evaluation and testing
        loss_fp = 0  # final loss with full-precision model; loss on test_dataset (if not exist, on val_dataset)
                     # this value serves as reference for bit-width learning in QAT
        # optimizer = optim.Adam(model.get_params(opt.lr), eps=1e-15)
        optimizer = lambda param: optim.Adam(params=param, lr=opt.lr, eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
        batch_num = len(train_loader)

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // max(1, opt.save_cnt)) # save ~50 times during the training
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        # colmap can estimate a more compact AABB
        if not opt.contract and opt.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)
        # nsvf and tank dataset contain bbox.txt
        if opt.data_format == 'nsvf' or opt.data_format == 'tank':
            model.update_aabb(train_loader._data.aabb)
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval,
                          save_interval=save_interval)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()

            trainer.metrics = [PSNRMeter(),]
            trainer.train(train_loader, valid_loader, max_epoch)

            loss_fp = trainer.evaluate_on_trainset(train_loader)
            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
            ################################
            loss_val_fp = trainer.evaluate(valid_loader)
            if opt.alpha:
                trainer.log(f"alpha skip rate:{1 - trainer.model.alpha_complex/trainer.model.alpha_sum}")

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            #
            # if test_loader.has_gt:
            #     trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            #################################

            # trainer.test(test_loader, write_video=True) # test and save video
            # trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)

            # 4) ===== quantization procedure =====
            if opt.quantization:
                # ======= Quantization 1: generate quan. model by inserting fake quantization layer =======
                model.quantize(bit_width_init=opt.bit_width)
                if opt.T1 and opt.bit_set is not None:
                    model.manual_bit(opt.bit_set)

                # ======= Quantization 2: pass calibration data to get scale/offset =======
                # use PTQ as the starting point of QAT
                ptq(model, train_loader, device, opt)
                # trainer_qat.evaluate(test_loader, name='PTQ')
                print("PTQ completed")

                # optimizer = lambda param: optim.Adam(params=param, lr=1e-2, weight_decay=5e-4)  # naive adam
                optimizer_qat = lambda param: optim.Adam(params=param, lr=opt.lr * opt.qat_lr,
                                                         betas=(0.9, 0.99), eps=1e-15)
                # optimizer_qat = lambda param: optim.Adam(params=param, lr=opt.lr * opt.qat_lr, eps=1e-15)

                scheduler_qat = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                              lambda iter: 0.1 ** min(
                                                                                           iter / (opt.qat_iteration-opt.lr_schedule),1))
                # model.load_state_dict(torch.load('model_params.pth'))
                metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
                trainer_qat = QatTrainer('ngp_qat', opt, model,
                                         device=device,
                                         workspace=opt.workspace,
                                         optimizer=optimizer_qat,
                                         criterion=criterion,
                                         ema_decay=0.95,
                                         fp16=opt.fp16,
                                         lr_scheduler=scheduler_qat,
                                         scheduler_update_every_step=True,
                                         metrics=metrics,
                                         weight_wrt_loss=opt.weight_penalty,
                                         loss_fp=loss_fp,
                                         target=opt.target)

                # density_bitfield_tmp = copy.deepcopy(trainer_qat.model.density_bitfield)
                loss_val_q, result, sample_points_per_image_ave = trainer_qat.evaluate(valid_loader, name='PTQ')
                # print(f'PTQ MSE degradation: {loss_val_q-loss_val_fp}')

                # ======= Quantization 3: Fine-tuning =======
                # trainer is defined individually
                qat_epoch = opt.qat_iteration // batch_num
                qat(trainer_qat, train_loader, valid_loader, qat_epoch)

                # trainer_qat.model.density_bitfield = density_bitfield_tmp

                # # Save quantification status, Evaluate on the Val dataset, Cal and Save BitOps for each image 
                quan_analysis2(model, workspace=opt.workspace, filename="qat_result", valid_loader=valid_loader, trainer_qat=trainer_qat)
                
                # if qat_epoch % trainer_qat.eval_interval:
                #     loss_val_q, _ = trainer_qat.evaluate(valid_loader, name='qat with quan. learning')
                    # print(f'QAT MSE degradation: {loss_val_q - loss_val_fp}')
                # loss_val_q, _ = trainer_qat.evaluate(valid_loader, name='qat with quan. learning')
                # print(f'QAT MSE degradation: {loss_val_q - loss_val_fp}')
                # trainer_qat.test(test_loader, name='qat with quan. learning')

                if True:  # if use_tensorboardX
                    trainer_qat.writer.close()

                # trainer_qat.test(test_loader, write_video=True)  # test and save video
