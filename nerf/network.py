import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from quantization.quan_module import *

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 num_layers=2, hidden_dim=64, geo_feat_dim=15,  # sigma net
                 num_layers_color=3, hidden_dim_color=64,  # color net
                 ):

        super().__init__(opt)
        self.alpha = opt.alpha

        # ======== position encoder ========
        self.encoder, self.in_dim = get_encoder(encoding="hashgrid", desired_resolution=2048 * self.bound, log2_hashmap_size=opt.log2_hashmap_size)

        # ======== sigma network ========
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
                # sigma is activated by exp (see forward)
            else:
                out_dim = hidden_dim
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
                sigma_net.append(nn.ReLU())
        self.sigma_net = nn.ModuleList(sigma_net)

        # ======== direction encoder =========
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding="sh")

        # ======== color network ========
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
                color_net.append(nn.Linear(in_dim, out_dim, bias=False))
                # rgb is activated by sigmoid (see forward)
            else:
                if self.alpha and l == 0:
                    self.uncertainty_net = nn.Linear(in_dim, 1, bias=False)
                    # out_dim = hidden_dim_color + 1 # uncertainty, early termination indicator
                out_dim = hidden_dim_color
                color_net.append(nn.Linear(in_dim, out_dim, bias=False))
                color_net.append(nn.ReLU())
        self.color_net = nn.ModuleList(color_net)

        # ======== quantization network ========
        self.qencoder = None
        self.qsigma_net = None
        self.qencoder_dir = None
        self.qcolor_net = None
        #
        # # ======= alpha statistical =======
        if self.alpha:
            self.target = 35 # [dB]
            self.uncertainty_metric = 0.5 * 10 ** (-1 * self.target / 10)
            self.alpha_sum = 0
            self.alpha_complex = 0  # number of simple points
        else:
            self.uncertainty_metric = 0

        # proposal network  (??? Weihang Liu)
        # if not self.opt.cuda_ray:
        #     self.prop_encoders = nn.ModuleList()
        #     self.prop_mlp = nn.ModuleList()
        #
        #     # hard coded 2-layer prop network
        #     prop0_encoder, prop0_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=128)
        #     prop0_mlp = MLP(prop0_in_dim, 1, 16, 2, bias=False)
        #     self.prop_encoders.append(prop0_encoder)
        #     self.prop_mlp.append(prop0_mlp)
        #
        #     prop1_encoder, prop1_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=256)
        #     prop1_mlp = MLP(prop1_in_dim, 1, 16, 2, bias=False)
        #     self.prop_encoders.append(prop1_encoder)
        #     self.prop_mlp.append(prop1_mlp)

    # ===== full precision part =====
    def forward(self, x, d, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # ======== position encoder ========
        x = self.encoder(x, bound=self.bound)
        # ======== sigma net ========
        h = x
        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)
        # exp activation for sigma
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # ======== direction encoder ========
        d = self.encoder_dir(d)
        # ======== color net ========
        if self.alpha:
            h_in = torch.cat([d, geo_feat], dim=-1)
            # content-aware uncertainty
            h = self.color_net[0](h_in)
            alpha = self.uncertainty_net(h_in)  # uncertainty
            alpha = torch.sigmoid(alpha) # when this value is low, it means the point is quite simple and the further calculation is redundant

            # # complex point processing
            # for l in range(1, len(self.color_net) - 1):
            #     h = self.color_net[l](h)
            # # final blend
            # h_new = alpha * h + (1 - alpha) * h_res
            # h = self.color_net[-1](h_new)

            if self.training and kwargs["step"] < 10000:
            # if False: # debug
                # complex point processing
                h_relu = self.color_net[1](h)
                for l in range(2, len(self.color_net)-1):
                    h = self.color_net[l](h_relu)
                # final blend
                # h = alpha * h + (1-alpha) * h_relu
                # residual connection
                h = 0.5 * h + 0.5 * h_relu
                h = self.color_net[-1](h)  # final decode
                # sigmoid activation for rgb
                color = torch.sigmoid(h)
                return {
                    'sigma': sigma,
                    'color': color,
                    'alpha': alpha
                    # 'alpha': 0
                }
            else:
                complex_mask = (alpha >= self.uncertainty_metric).squeeze() # pre-defined threshold, this is a conservative value
                if not self.training:
                    self.alpha_sum += torch.sum(~torch.isnan(alpha))
                    self.alpha_complex += complex_mask.sum()
                h_relu = self.color_net[1](h).clone()  # here clone() is necessary to avoid "gradient computation error due tp inplace operation", but why?
                # complex point processing
                h = h_relu[complex_mask,:]
                for l in range(2, len(self.color_net)-1):
                    h = self.color_net[l](h)
                h_relu[complex_mask,:] = h
                h = self.color_net[-1](h_relu)  # final decode
                # sigmoid activation for rgb
                color = torch.sigmoid(h)
                return {
                    'sigma': sigma,
                    'color': color,
                    # 'alpha': alpha.mean()
                    'alpha': alpha
                }
        else:
            h = torch.cat([d, geo_feat], dim=-1)
            for l in range(len(self.color_net)):
                h = self.color_net[l](h)
            # sigmoid activation for rgb
            color = torch.sigmoid(h)

            return {
                'sigma': sigma,
                'color': color,
                'alpha': torch.zeros(color.shape[0],1, device=color.device)
            }

    def density(self, x, **kwargs):

        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def apply_total_variation(self, w):
        self.grid_encoder.grad_total_variation(w)

    def apply_weight_decay(self, w):
        self.grid_encoder.grad_weight_decay(w)

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]

        return params

    # def get_params_hash(self, lr):
    #
    #     params = [
    #         {'params': self.encoder.parameters(), 'lr': lr},
    #         # {'params': self.sigma_net.parameters(), 'lr': lr},
    #         # {'params': self.encoder_dir.parameters(), 'lr': lr},
    #         # {'params': self.color_net.parameters(), 'lr': lr},
    #     ]
    #
    #     return params

    # ===== quantization part =====
    # define quantized model
    def quantize(self, bit_width_init=8):
        self.qencoder = Qencoder(self.encoder, bit_width_init=bit_width_init)
        self.qsigma_net = QsigmaNet(self.sigma_net, bit_width_init=bit_width_init)
        self.qencoder_dir = QencoderDir(self.encoder_dir, bit_width_init=bit_width_init)
        self.qcolor_net = QcolorNet(self.color_net, bit_width_init=bit_width_init,
                                    uncertainty_weight=self.uncertainty_net.weight if self.opt.alpha else None,
                                    uncertainty_metric=self.uncertainty_metric)

    # pass calibration data
    def calibration(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # ======== position encoder ========
        x = self.qencoder.calibration(x, bound=self.bound)

        # ======== sigma net ========
        h, sigma = self.qsigma_net.calibration(x)
        geo_feat = h[..., 1:]

        # ======== direction encoder ========
        d = self.qencoder_dir.calibration(d)

        # ======== color net ========
        h = torch.cat([d, geo_feat], dim=-1)
        color = self.qcolor_net.calibration(h, self.uncertainty_metric)

        return {
            'sigma': sigma,
            'color': color,
        }

    def quantize_forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # ======== position encoder ========
        # start_time = time.time()
        qx = self.qencoder(x, bound=self.bound)
        # end_time = time.time()
        # print("time (qencoder)：", end_time - start_time, "s")

        # ======== sigma net ========
        # start_time = time.time()
        qh, sigma = self.qsigma_net(qx)
        geo_feat = qh[..., 1:]
        # end_time = time.time()
        # print("time (qsigma_net)：", end_time - start_time, "s")

        # ======== direction encoder ========
        d = self.qencoder_dir(d)

        # ======== color net ========
        # start_time = time.time()
        qh = torch.cat([d, geo_feat], dim=-1)
        color = self.qcolor_net(qh, self.uncertainty_metric)
        # end_time = time.time()
        # print("time (qcolor_net)：", end_time - start_time, "s")

        return {
            'sigma': sigma,
            'color': color,
        }

    def manual_bit(self, bit_setting: list):
        # Note:
        #     len(bit_setting) = 15
        for name, module in self.named_modules():
            if 'qencoder.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[0]]))
            if 'qencoder.qw' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[1]]))
            if 'Qsigma_act.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[2]]))
            if 'Qsigma_net.0.qw' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[3]]))
            if 'Qsigma_net.1.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[4]]))
            if 'Qsigma_net.2.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[5]]))
            if 'Qsigma_net.2.qw' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[6]]))
            if 'qencoder_dir.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[7]]))
            if 'Qcolor_act.qi' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[8]]))
            if 'Qcolor_act.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[9]]))
            if 'Qcolor_net.0.qw' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[10]]))
            if 'Qcolor_net.1.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[11]]))
            if 'Qcolor_net.2.qw' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[12]]))
            if 'Qcolor_net.3.qo' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[13]]))
            if 'Qcolor_net.4.qw' in name:
                module.soft_bit = nn.Parameter(torch.tensor([bit_setting[14]]))

        self.to('cuda')

    def adjust_hash(self, gradient_sum, threshold_up=8e-7, threshold_down=2e-7):
        # gradient: accumulated gradients. [s0, C] -> [number of params, feature dim]
        # threshold: pre-defined threshold. [up, down] -> [scale up threshold, scale down threshold]
        # gradient_mean = torch.mean(gradient)
        dim = self.encoder.embeddings.shape
        gradient_mean = gradient_sum / (dim[0] * dim[1])
        if gradient_mean > threshold_up:
            self.encoder.size_up()
        elif gradient_mean < threshold_down:
            self.encoder.size_down()
