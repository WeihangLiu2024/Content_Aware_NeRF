import torch


def pass_calibration_data(sim_model,
                          device,
                          data_loader,
                          trainer_qat,
                          percentage_of_dataloader=0.5):
    # sim_model: Quantized sim - pytorch model
    # device: torch.device('cuda') or torch.device('cpu')
    # percentage_of_dataloader: using all the training samples to calibreation
    #             is unefficient and unnecessary, we choose part of it instead.
    batch_size = data_loader.batch_size

    sim_model.eval()
    num_samples = torch.fx.len(data_loader) * percentage_of_dataloader

    batch_cntr = 0
    with torch.no_grad():
        for input_data in data_loader:

            # inputs_batch = input_data.to(device)
            # 这里使用eval是因为pass data的过程不涉及参数的更新；
            # But! train_loader有像素的采样过程，(H, W) -> sample(H*W)，程序跑不通
            # TODO： 针对此进行修改，可以考虑在不使用eval_step这个函数
            trainer_qat.eval_step(input_data)

            batch_cntr += 1
            if (batch_cntr * batch_size) >= num_samples:
                break
