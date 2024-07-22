# definition of the quantization module

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy

from activation import trunc_exp
# from .utils import QParam
from .QParam import *

import scipy.io as sio

class QModule(nn.Module):
    # 后续改写量化模块的模板
    # 要改写的模块有：
    #     encoder,
    #     sigma_net,
    #     encoder_dir,
    #     color_net
    # 每个模块可选的量化位置：
    #     qw (parameter quan.),
    #     qi (input quan.),
    #     qo (output quan.)
    # 量化方式：
    #     对称量化 (for weight)
    #     非对称量化 (for activation)
    # 量化范式：
    #     PTQ (QParam, model.parameters() frozen)
    #     QAT (QParam frozen)
    #     QAT with range learning
    #     QAT with bit-width & range learning
    # Two forward path:
    # function calibration: passing typical data to get statistical val_min & val_max
    # function forward: inference & training forward
    def __init__(self, qi: str or bool=False, qo: str or bool=False, **kwargs):
        # kwargs['bw_qi'] and kwargs[‘bw_qo’]
        super(QModule, self).__init__()
        if len(kwargs) == 1:
            key1 = 'bit_width_init'
            key2 = 'bit_width_init'
        else:
            key1 = 'bw_qi'
            key2 = 'bw_qo'

        if qi == 'ReLU':
            self.qi = QParamReLUExp(bit_width_init=kwargs[key1])
        elif qi == 'other':
            self.qi = QParamOther(bit_width_init=kwargs[key1])
        elif qi == 'ReLU_cons':
            self.qi = QParamReLUExp_cons(bit_width_init=kwargs[key1])
        elif qi == 'other_cons':
            self.qi = QParamOther_cons(bit_width_init=kwargs[key1])

        if qo == 'ReLU':
            self.qo = QParamReLUExp(bit_width_init=kwargs[key2])
        elif qo == 'other':
            self.qo = QParamOther(bit_width_init=kwargs[key2])
        elif qo == 'ReLU_cons':
            self.qo = QParamReLUExp_cons(bit_width_init=kwargs[key1])
        elif qo == 'other_cons':
            self.qo = QParamOther_cons(bit_width_init=kwargs[key1])
        # layers with parameters are surly to quantized
        # layers w/o parameters are not.

    def calibration(self, x):
        # freeze parameters to avoid updating during fine-tune
        pass


# 1) Fundamental module derives from QModule: QLinear, QReLU, Qencoder, Qencoder_dir
# 2) High-level module derives from nn.Module. Because they are composed of several fundamental modules, which results
#    in their not having any independent quan. parameters: Qsigma_net, Qcolor_net
class QLinear(QModule):

    def __init__(self, fc_module, qi: str or bool = False, qo: str or bool = False, bit_width_init=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, bit_width_init=bit_width_init)
        self.fc_module = copy.deepcopy(fc_module)  # trained fc layer (the FP model & LBQ version can be saved independently.)
        self.qw = QParamWeight(bit_width_init=bit_width_init)
        self.qw.min_max(self.fc_module.weight.data)
        self.fc_module.weight.data = self.qw.fake_quantize(self.fc_module.weight.data)
        # self.qw = QParamOther(bit_width_init=bit_width_init)

    def calibration(self, x):
        # potential weakness: before min & max converging, collecting wrong distribution of activations
        if hasattr(self, 'qi'):  # input quan.
            self.qi.min_max(x)   # update range & zero-point
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        x = F.linear(x, self.fc_module.weight)  # bias is neglected

        if hasattr(self, 'qo'):
            self.qo.min_max(x)
            x = self.qo.fake_quantize(x)

        return x

    def forward(self, x):
        if hasattr(self, 'qi'):  # input quan.
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        quan_weight = self.qw.fake_quantize(self.fc_module.weight)
        x = F.linear(x, quan_weight)  # bias is neglected

        if hasattr(self, 'qo'):
            x = self.qo.fake_quantize(x)

        return x


class QLinear_cons(QModule):

    def __init__(self, fc_module, qi: str or bool = False, qo: str or bool = False, bit_width_init=8):
        super(QLinear_cons, self).__init__(qi=qi, qo=qo, bit_width_init=bit_width_init)
        self.fc_module = copy.deepcopy(fc_module)  # trained fc layer
        self.qw = QParamWeight_cons(bit_width_init=bit_width_init)
        # self.qw = QParamWeight(bit_width_init=bit_width_init)
        self.qw.min_max(self.fc_module.weight.data)
        self.fc_module.weight.data = self.qw.fake_quantize(self.fc_module.weight.data)
        # self.qw = QParamOther(bit_width_init=bit_width_init)

    def calibration(self, x):
        # potential weakness: before min & max converging, collecting wrong distribution of activations
        if hasattr(self, 'qi'):  # input quan.
            self.qi.min_max(x)   # update range & zero-point
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        x = F.linear(x, self.fc_module.weight)  # bias is neglected

        if hasattr(self, 'qo'):
            self.qo.min_max(x)
            x = self.qo.fake_quantize(x)

        return x

    def forward(self, x):
        if hasattr(self, 'qi'):  # input quan.
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        quan_weight = self.qw.fake_quantize(self.fc_module.weight)
        x = F.linear(x, quan_weight)  # bias is neglected

        if hasattr(self, 'qo'):
            x = self.qo.fake_quantize(x)

        return x


class Qexp(QModule):
    def __init__(self, qi=False, qo='ReLU', bw_qi=8, bw_qo=32):
        super(Qexp, self).__init__(qi=qi, qo=qo, bw_qi=bw_qi, bw_qo=bw_qo)

    def calibration(self, x):
        # potential weakness: before min & max converging, collecting wrong distribution of activations
        if hasattr(self, 'qi'):  # input quan.
            # self.qi.min_max(x, 'act_exp_qo')   # update range & zero-point
            self.qi.min_max(x)  # update range & zero-point
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        x = torch.exp(x)

        if hasattr(self, 'qo'):
            # self.qo.min_max(x, 'act_exp_qo')
            self.qo.min_max(x)
            x = self.qo.fake_quantize(x)

        return x

    def forward(self, x):
        if hasattr(self, 'qi'):
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        x = torch.exp(x)

        if hasattr(self, 'qo'):
            x = self.qo.fake_quantize(x)

        return x


class QSigmoid(QModule):
    def __init__(self, qi='other', qo='other', bit_width_init=8):
        super(QSigmoid, self).__init__(qi=qi, qo=qo, bit_width_init=bit_width_init)

    def calibration(self, x):
        # potential weakness: before min & max converging, collecting wrong distribution of activations
        if hasattr(self, 'qi'):  # input quan.
            self.qi.min_max(x)   # update range & zero-point
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        x = torch.sigmoid(x)

        if hasattr(self, 'qo'):
            self.qo.min_max(x)
            x = self.qo.fake_quantize(x)

        return x

    def forward(self, x):
        if hasattr(self, 'qi'):
            x = self.qi.fake_quantize(x)  # quan. & dequan. with qi's parameters

        x = torch.sigmoid(x)

        if hasattr(self, 'qo'):
            x = self.qo.fake_quantize(x)

        return x


class QReLU(QModule):

    def __init__(self, qi=False, qo='ReLU', bit_width_init=8):
        super(QReLU, self).__init__(qi=qi, qo=qo, bit_width_init=bit_width_init)

    def calibration(self, x):
        if hasattr(self, 'qi'):
            self.qi.min_max(x)
            x = self.qi.fake_quantize(x)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.min_max(x)
            x = self.qo.fake_quantize(x)

        return x

    def forward(self, x):
        if hasattr(self, 'qi'):
            x = self.qi.fake_quantize(x)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            x = self.qo.fake_quantize(x)
        return x


class QReLU_cons(QModule):

    def __init__(self, qi=False, qo='ReLU_cons', bit_width_init=8):
        super(QReLU_cons, self).__init__(qi=qi, qo=qo, bit_width_init=bit_width_init)

    def calibration(self, x):
        if hasattr(self, 'qi'):
            self.qi.min_max(x)
            x = self.qi.fake_quantize(x)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.min_max(x)
            x = self.qo.fake_quantize(x)

        return x

    def forward(self, x):
        if hasattr(self, 'qi'):
            x = self.qi.fake_quantize(x)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            x = self.qo.fake_quantize(x)
        return x


class Qencoder(QModule):
    def __init__(self, encoder, qi=False, qo='other', bit_width_init=8):
        super(Qencoder, self).__init__(qi=qi, qo=qo, bit_width_init=bit_width_init)
        self.encoder = copy.deepcopy(encoder)
        # self.qw = QParamWeight(bit_width_init=bit_width_init)
        self.qw = QParamOther(bit_width_init=bit_width_init)
        # sio.savemat('./quan_res/hash.mat',{'hash': self.encoder.embeddings.cpu().detach().numpy()})

    def calibration(self, x, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]
        self.qw.min_max(self.encoder.embeddings.data)
        quan_embeddings = self.qw.fake_quantize(self.encoder.embeddings)
        x = self.encoder(x, bound=bound, out_embeddings=quan_embeddings)
        if hasattr(self, 'qo'):
            self.qo.min_max(x)
            x = self.qo.fake_quantize(x)
        return x

    def forward(self, x, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]
        quan_embeddings = self.qw.fake_quantize(self.encoder.embeddings)
        x = self.encoder(x, bound=bound, out_embeddings=quan_embeddings)
        if hasattr(self, 'qo'):
            x = self.qo.fake_quantize(x)
        return x


class QencoderDir(QModule):
    def __init__(self, encoder_dir, qi=False, qo='other', bit_width_init=8):
        super(QencoderDir, self).__init__(qi=qi, qo=qo, bit_width_init=bit_width_init)
        # super(QencoderDir, self).__init__(qi=qi, qo=qo, bit_width_init=6)
        self.encoder_dir = copy.deepcopy(encoder_dir)

    def calibration(self, d):
        # no parameters
        d = self.encoder_dir(d)
        if hasattr(self, 'qo'):
            self.qo.min_max(d)
            d = self.qo.fake_quantize(d)
        return d

    def forward(self, d):
        # inputs: [..., input_dim], nomalized in [-1, 1]
        # return: [..., num_levels * level_dim]
        # self.qw.min_max(self.encoder_dir.embeddings.data)
        d = self.encoder_dir(d)
        if hasattr(self, 'qo'):
            d = self.qo.fake_quantize(d)
        return d


class QsigmaNet(nn.Module):
    def __init__(self, sigma_net, bit_width_init=8):
        super(QsigmaNet, self).__init__()
        self.bit_width_init = bit_width_init
        self.sigma_net = copy.deepcopy(sigma_net)
        self.Qsigma_net = []
        self.Qsigma_act = Qexp(bw_qo=32)   # TODO: exp is hard to quantized with low bit-width
        # self.Qsigma_act = Qexp(bw_qo=bit_width_init)  # TODO: exp is hard to quantized with low bit-width
        self.add_module("Qsigma_act", self.Qsigma_act)
        self.case_num = 0

        Qsigma_net = []
        # define quan. network and quantized the parameters
        for l in range(len(self.sigma_net)):
            if isinstance(self.sigma_net[l], nn.Linear):
                if l == 2:
                    Qsigma_net.append(QLinear(self.sigma_net[l], qo='other', bit_width_init=self.bit_width_init))
                    # Qsigma_net.append(QLinear(self.sigma_net[l], bit_width_init=6))
                else:
                    Qsigma_net.append(QLinear(self.sigma_net[l], bit_width_init=self.bit_width_init))
                    # Qsigma_net.append(QLinear(self.sigma_net[l], bit_width_init=6))
                # save data for MATLAB
                # sio.savemat(f'./quan_res/Wsigma_{l}.mat', {f'Wsigma_{l}': self.sigma_net[l].weight.data.cpu().detach().numpy()})
            elif isinstance(self.sigma_net[l], nn.ReLU):
                Qsigma_net.append(QReLU(bit_width_init=self.bit_width_init))
        self.Qsigma_net = nn.ModuleList(Qsigma_net)

    def calibration(self, x):
        # save data for MATLAB
        # sio.savemat(f'./quan_res/{self.case_num}sigma_in.mat', {'sigma_in': x.data.cpu().detach().numpy()})

        for l in range(len(self.Qsigma_net)):
            x = self.Qsigma_net[l].calibration(x)
            # save data for MATLAB
            # sio.savemat(f'./quan_res/{self.case_num}sigma_{l}.mat', {f'sigma_{l}': x.data.cpu().detach().numpy()})

        # sigma = self.sigma_act.quantize_inference(x)  # Qexp is invalid
        sigma = self.Qsigma_act.calibration(x[..., 0])
        # save data for MATLAB
        # sio.savemat(f'./quan_res/{self.case_num}sigma_out.mat', {'sigma_out': sigma.data.cpu().detach().numpy()})

        self.case_num = self.case_num + 1

        return x, sigma

    def forward(self, x):
        for l in range(len(self.Qsigma_net)):
            x = self.Qsigma_net[l](x)
        sigma = self.Qsigma_act(x[..., 0])
        return x, sigma


class QcolorNet(nn.Module):
    def __init__(self, color_net, bit_width_init=8):
        super(QcolorNet, self).__init__()
        self.color_net = copy.deepcopy(color_net)
        self.Qcolor_net = []  # in order to distinguish from the class objct "Qsigma_net"
        self.Qcolor_act = QSigmoid(bit_width_init=bit_width_init)
        # self.Qcolor_act = QSigmoid(qo='other_cons', bit_width_init=bit_width_init)
        self.add_module("Qcolor_act", self.Qcolor_act)
        self.case_num = 0

        Qcolor_net = []
        # define quan. network and quantized the parameters
        for l in range(len(self.color_net)):
            if isinstance(self.color_net[l], nn.Linear):
                # if l == 4:
                #     Qcolor_net.append(QLinear(self.color_net[l], bit_width_init=6))
                # else:
                #     Qcolor_net.append(QLinear(self.color_net[l], bit_width_init=bit_width_init))
                Qcolor_net.append(QLinear(self.color_net[l], bit_width_init=bit_width_init))
                # Qcolor_net.append(QLinear_cons(self.color_net[l], bit_width_init=bit_width_init))
                # save data for MATLAB
                # sio.savemat(f'./quan_res/Wcolor_{l}.mat', {f'Wcolor_{l}': self.color_net[l].weight.data.cpu().detach().numpy()})
            elif isinstance(self.color_net[l], nn.ReLU):
                # if l == 3:
                #     Qcolor_net.append(QReLU(bit_width_init=6))
                # else:
                #     Qcolor_net.append(QReLU(bit_width_init=bit_width_init))
                Qcolor_net.append(QReLU(bit_width_init=bit_width_init))
                # Qcolor_net.append(QReLU_cons(bit_width_init=bit_width_init))
        self.Qcolor_net = nn.ModuleList(Qcolor_net)

    def calibration(self, d):
        # save data for MATLAB
        # sio.savemat(f'./quan_res/{self.case_num}color_in.mat', {'color_in': d.data.cpu().detach().numpy()})

        d = self.Qcolor_net[0].calibration(d)
        # uncertainty calculation
        alpha = d[:, 0].unsqueeze(dim=1)  # uncertainty
        d_res = d[:, 1:]
        d_res = self.Qcolor_net[1].calibration(d_res)  # ReLU
        d = d_res
        alpha = torch.sigmoid(alpha)
        complex_mask = (alpha >= 0.5).squeeze()
        d = d[complex_mask, :]
        for l in range(2, len(self.Qcolor_net)-1):  # 5 layerss
            if len(d) == 0:
                break
            d = self.Qcolor_net[l].calibration(d)
        d_res[complex_mask] = d
        d = self.Qcolor_net[-1].calibration(d_res)

        # save data for MATLAB
        # sio.savemat(f'./quan_res/{self.case_num}color_{l}.mat', {f'color_{l}': d.data.cpu().detach().numpy()})
        rgb = self.Qcolor_act.calibration(d)

        # save data for MATLAB
        # sio.savemat(f'./quan_res/{self.case_num}color_out.mat', {'color_out': rgb.data.cpu().detach().numpy()})

        self.case_num = self.case_num + 1

        return rgb

    def forward(self, d):
        d = self.Qcolor_net[0](d)
        alpha = d[:, 0].unsqueeze(dim=1)  # uncertainty
        d_res = d[:, 1:]
        d_res = self.Qcolor_net[1](d_res)
        d = d_res
        alpha = torch.sigmoid(alpha)

        complex_mask = (alpha >= 0.5).squeeze()  # pre-defined threshold, this is a conservative value
        # complex point processing
        d = d[complex_mask, :]
        for l in range(2, len(self.color_net) - 1):
            d = self.Qcolor_net[l](d)
        d_res[complex_mask, :] = d
        d = self.Qcolor_net[-1](d_res)

        rgb = self.Qcolor_act(d)

        return rgb
