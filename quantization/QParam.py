# definition of the quantization scheme

from .utils import *


class QParamWeight(nn.Module):
    # Z = 0;
    # qmin = -2**(B-1);  qmax = 2**(B-1)-1
    def __init__(self, bit_width_init=8):
        super().__init__()
        # trainable parameters
        self.var_range = nn.Parameter(torch.tensor([0.]))
        self.soft_bit = nn.Parameter(torch.tensor([float(bit_width_init)]))

        # intermediate paramaters
        self.var_max = torch.tensor([], requires_grad=False)
        self.var_min = torch.tensor([], requires_grad=False)  # -var_max
        self.hard_bit = torch.tensor([], requires_grad=False)
        self.zero = torch.tensor([0], device='cuda', requires_grad=False)  # fixed to 0
        self.scale = torch.tensor([], requires_grad=False)

        # other functional variable
        self.fakeQuantizer = FakeQuantizeWeight

    def manual_min_max(self, var_max):  # only used for initialization
        # self.var_min.data = torch.tensor([var_min], device=self.var_min.device, requires_grad=False)
        self.var_max.data = torch.tensor([var_max], device=self.var_max.device,requires_grad=False)
        self.var_min.data = -1 * self.var_max.data
        self.var_range.data = (self.var_max.data - self.var_min.data).view(torch.Size([1]))

    def min_max(self, tensor):  # only used for initialization
        # name: used to collect histogram of quantized data
        flag = 0
        if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data or self.var_max.data < torch.abs(tensor.min().data):
            self.var_max.data = max(tensor.max().data, torch.abs(tensor.min().data))
            flag = 1

        # if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data:
        #     self.var_max.data = tensor.max().data
        #     flag = 1

        if flag:
            self.var_min.data = -1 * self.var_max.data
            self.var_range.data = (self.var_max.data - self.var_min.data).view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def __str__(self):  # used for print(QParam)
        self.var_max.data = self.var_range / 2
        self.var_min.data = -1 * self.var_max.data
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(2., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamWeight._calc_step(self.var_range, self.hard_bit)
        info = ' var_range: %.10f ' % self.var_range
        info += ' min: %.6f ' % self.var_min
        info += ' max: %.6f ' % self.var_max
        info += ' zero point: %d ' % self.zero
        info += ' soft_bit: %.2f ' % self.soft_bit
        info += ' hard_bit: %.2f ' % self.hard_bit
        return info

    def fake_quantize(self, tensor):
        # return FakeQuantize.apply(tensor, self.var_range, self.soft_zero, self.soft_bit, self.min, self.max)
        # 1) update intermediate parameters
        self.var_max.data = self.var_range / 2
        self.var_min.data = -1 * self.var_max.data
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(2., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamWeight._calc_step(self.var_range, self.hard_bit)
        # 2) fake quantize
        return self.fakeQuantizer.apply(tensor, self.var_range, self.var_max, self.soft_bit,
                                        self.var_min, self.scale, self.zero.data)

    @staticmethod
    def _calc_step(var_range, hard_bit):

        qmin = torch.tensor([0.], device=var_range.device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(var_range.device)
        # qmax = torch.tensor(2. ** hard_bit - 1., device=soft_bit.device)

        scale_data = var_range / (qmax - qmin)

        return scale_data.data
        # return var_range.data, zero_point_data.data, scale_data.data


# constrain bitwidth to a flexible range
class QParamWeight_cons(nn.Module):
    # Z = 0;
    # qmin = -2**(B-1);  qmax = 2**(B-1)-1
    def __init__(self, bit_width_init=8):
        super().__init__()
        # trainable parameters
        self.var_range = nn.Parameter(torch.tensor([0.]))
        self.soft_bit = nn.Parameter(torch.tensor([float(bit_width_init)]))

        # intermediate paramaters
        self.var_max = torch.tensor([], requires_grad=False)
        self.var_min = torch.tensor([], requires_grad=False)  # -var_max
        self.hard_bit = torch.tensor([], requires_grad=False)
        self.zero = torch.tensor([0], device='cuda', requires_grad=False)  # fixed to 0
        self.scale = torch.tensor([], requires_grad=False)

        # other functional variable
        self.fakeQuantizer = FakeQuantizeWeight

    def manual_min_max(self, var_max):  # only used for initialization
        # self.var_min.data = torch.tensor([var_min], device=self.var_min.device, requires_grad=False)
        self.var_max.data = torch.tensor([var_max], device=self.var_max.device,requires_grad=False)
        self.var_min.data = -1 * self.var_max.data
        self.var_range.data = (self.var_max.data - self.var_min.data).view(torch.Size([1]))

    def min_max(self, tensor):  # only used for initialization
        # name: used to collect histogram of quantized data
        flag = 0
        if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data or self.var_max.data < torch.abs(tensor.min().data):
            self.var_max.data = max(tensor.max().data, torch.abs(tensor.min().data))
            flag = 1

        # if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data:
        #     self.var_max.data = tensor.max().data
        #     flag = 1

        if flag:
            self.var_min.data = -1 * self.var_max.data
            self.var_range.data = (self.var_max.data - self.var_min.data).view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def __str__(self):  # used for print(QParam)
        self.var_max.data = self.var_range / 2
        self.var_min.data = -1 * self.var_max.data
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(4., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamWeight._calc_step(self.var_range, self.hard_bit)
        info = ' var_range: %.10f ' % self.var_range
        info += ' min: %.6f ' % self.var_min
        info += ' max: %.6f ' % self.var_max
        info += ' zero point: %d ' % self.zero
        info += ' soft_bit: %.2f ' % self.soft_bit
        info += ' hard_bit: %.2f ' % self.hard_bit
        return info

    def fake_quantize(self, tensor):
        # return FakeQuantize.apply(tensor, self.var_range, self.soft_zero, self.soft_bit, self.min, self.max)
        # 1) update intermediate parameters
        self.var_max.data = self.var_range / 2
        self.var_min.data = -1 * self.var_max.data
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(4., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamWeight._calc_step(self.var_range, self.hard_bit)
        # 2) fake quantize
        return self.fakeQuantizer.apply(tensor, self.var_range, self.var_max, self.soft_bit,
                                        self.var_min, self.scale, self.zero.data)

    @staticmethod
    def _calc_step(var_range, hard_bit):

        qmin = torch.tensor([0.], device=var_range.device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(var_range.device)
        # qmax = torch.tensor(2. ** hard_bit - 1., device=soft_bit.device)

        scale_data = var_range / (qmax - qmin)

        return scale_data.data
        # return var_range.data, zero_point_data.data, scale_data.data


class QParamReLUExp(nn.Module):
    # Z = 0;
    # qmin = 0;  qmax = 2**B-1
    def __init__(self, bit_width_init=8):
        super().__init__()
        # trainable parameters
        self.var_range = nn.Parameter(torch.tensor([0.]))
        self.soft_bit = nn.Parameter(torch.tensor([float(bit_width_init)]))

        # intermediate paramaters
        self.var_max = torch.tensor([], requires_grad=False)
        self.var_min = torch.tensor([0], device='cuda', requires_grad=False)  # fixed to 0
        self.hard_bit = torch.tensor([], requires_grad=False)
        self.zero = torch.tensor([0], device='cuda', requires_grad=False)  # fixed to 0
        self.scale = torch.tensor([], requires_grad=False)

        # other functional variable
        self.fakeQuantizer = FakeQuantizeReLU

    def manual_min_max(self, var_max):  # only used for initialization
        # self.var_min.data = torch.tensor([var_min], device=self.var_min.device, requires_grad=False)
        self.var_max.data = torch.tensor([var_max], device=self.var_max.device,requires_grad=False)
        self.var_range.data = self.var_max.data.view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max.data, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def min_max(self, tensor):  # only used for initialization
        # name: used to collect histogram of quantized data
        flag = 0
        if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data:
            self.var_max.data = tensor.max().data
            flag = 1

        if flag:
            self.var_range.data = self.var_max.data.view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def __str__(self):  # used for print(QParam)
        # self.var_min = QParamReLUExp._calc_min(self.var_max, self.var_range)
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(2., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamReLUExp._calc_step(self.var_range, self.hard_bit)
        info = ' var_range: %.10f ' % self.var_range
        info += ' min: %.6f ' % self.var_min
        info += ' max: %.6f ' % self.var_max
        info += ' zero point: %d ' % self.zero
        info += ' soft_bit: %.2f ' % self.soft_bit
        info += ' hard_bit: %.2f ' % self.hard_bit
        return info

    def fake_quantize(self, tensor):
        # return FakeQuantize.apply(tensor, self.var_range, self.soft_zero, self.soft_bit, self.min, self.max)
        # 1) update intermediate parameters
        # self.var_min = QParamReLUExp._calc_min(self.var_max, self.var_range)
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(2., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamReLUExp._calc_step(self.var_range, self.hard_bit)
        # 2) fake quantize
        return self.fakeQuantizer.apply(tensor, self.var_range, self.var_max, self.soft_bit,
                                        self.var_min, self.scale, self.zero.data)

    @staticmethod
    def _calc_step(var_range, hard_bit):

        qmin = torch.tensor([0.], device=hard_bit.device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(hard_bit.device)
        # qmax = torch.tensor(2. ** hard_bit - 1., device=soft_bit.device)

        scale_data = var_range / (qmax - qmin)

        return scale_data.data
        # return var_range.data, zero_point_data.data, scale_data.data


# constrain bitwidth to a flexible range
class QParamReLUExp_cons(nn.Module):
    # Z = 0;
    # qmin = 0;  qmax = 2**B-1
    def __init__(self, bit_width_init=8):
        super().__init__()
        # trainable parameters
        self.var_range = nn.Parameter(torch.tensor([0.]))
        self.soft_bit = nn.Parameter(torch.tensor([float(bit_width_init)]))

        # intermediate paramaters
        self.var_max = torch.tensor([], requires_grad=False)
        self.var_min = torch.tensor([0], device='cuda', requires_grad=False)  # fixed to 0
        self.hard_bit = torch.tensor([], requires_grad=False)
        self.zero = torch.tensor([0], device='cuda', requires_grad=False)  # fixed to 0
        self.scale = torch.tensor([], requires_grad=False)

        # other functional variable
        self.fakeQuantizer = FakeQuantizeReLU

    def manual_min_max(self, var_max):  # only used for initialization
        # self.var_min.data = torch.tensor([var_min], device=self.var_min.device, requires_grad=False)
        self.var_max.data = torch.tensor([var_max], device=self.var_max.device,requires_grad=False)
        self.var_range.data = self.var_max.data.view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max.data, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def min_max(self, tensor):  # only used for initialization
        # name: used to collect histogram of quantized data
        flag = 0
        if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data:
            self.var_max.data = tensor.max().data
            flag = 1

        if flag:
            self.var_range.data = self.var_max.data.view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def __str__(self):  # used for print(QParam)
        # self.var_min = QParamReLUExp._calc_min(self.var_max, self.var_range)
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(4., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamReLUExp._calc_step(self.var_range, self.hard_bit)
        info = ' var_range: %.10f ' % self.var_range
        info += ' min: %.6f ' % self.var_min
        info += ' max: %.6f ' % self.var_max
        info += ' zero point: %d ' % self.zero
        info += ' soft_bit: %.2f ' % self.soft_bit
        info += ' hard_bit: %.2f ' % self.hard_bit
        return info

    def fake_quantize(self, tensor):
        # return FakeQuantize.apply(tensor, self.var_range, self.soft_zero, self.soft_bit, self.min, self.max)
        # 1) update intermediate parameters
        # self.var_min = QParamReLUExp._calc_min(self.var_max, self.var_range)
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(4., 64.))
        # self.zero.data = QParamReLUExp._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamReLUExp._calc_step(self.var_range, self.hard_bit)
        # 2) fake quantize
        return self.fakeQuantizer.apply(tensor, self.var_range, self.var_max, self.soft_bit,
                                        self.var_min, self.scale, self.zero.data)

    @staticmethod
    def _calc_step(var_range, hard_bit):

        qmin = torch.tensor([0.], device=hard_bit.device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(hard_bit.device)
        # qmax = torch.tensor(2. ** hard_bit - 1., device=soft_bit.device)

        scale_data = var_range / (qmax - qmin)

        return scale_data.data
        # return var_range.data, zero_point_data.data, scale_data.data


class QParamOther(nn.Module):
    # Z != 0;
    # qmin = 0;  qmax = 2**B-1
    def __init__(self, bit_width_init=8):
        super().__init__()
        # trainable parameters
        self.var_range = nn.Parameter(torch.tensor([0.]))
        self.soft_bit = nn.Parameter(torch.tensor([float(bit_width_init)]))
        self.var_max = nn.Parameter(torch.tensor([0.]))

        # intermediate paramaters
        self.var_min = torch.tensor([], requires_grad=False)
        self.hard_bit = torch.tensor([], requires_grad=False)
        self.zero = torch.tensor([], requires_grad=False)
        self.scale = torch.tensor([], requires_grad=False)

        # other functional variable
        self.fakeQuantizer = FakeQuantizeOther

        # temp variable
        self.var_min_init = torch.tensor([], requires_grad=False)

    def manual_min_max(self, var_min, var_max):  # only used for initialization
        self.var_min.data = torch.tensor([var_min], device=self.var_min.device, requires_grad=False)
        self.var_max.data = torch.tensor([var_max], device=self.var_max.device,requires_grad=True)
        self.var_range.data = (self.var_max.data - self.var_min.data).view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max.data, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def min_max(self, tensor, name=None):  # only used for initialization
        # name: used to collect histogram of quantized data
        flag = 0
        if torch.isinf(tensor.max().data) or torch.isneginf(tensor.min().data):
            return
        if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data:
            self.var_max.data = torch.tensor([tensor.max().data], device=tensor.device)
            flag = 1

        if self.var_min_init.nelement() == 0 or self.var_min_init.data > tensor.min().data:
            self.var_min_init.data = torch.tensor([tensor.min().data], device=tensor.device)
            flag = 1

        if flag:
            self.var_range.data = (self.var_max.data - self.var_min_init.data).view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def __str__(self):  # used for print(QParam)
        self.var_min = QParamOther._calc_min(self.var_max, self.var_range)
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(2., 64.))
        self.zero.data = QParamOther._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamOther._calc_step(self.var_range, self.hard_bit)
        info = ' var_range: %.10f ' % self.var_range
        info += ' min: %.6f ' % self.var_min
        info += ' max: %.6f ' % self.var_max
        info += ' zero point: %d ' % self.zero
        info += ' soft_bit: %.2f ' % self.soft_bit
        info += ' hard_bit: %.2f ' % self.hard_bit
        return info

    def fake_quantize(self, tensor):
        # return FakeQuantize.apply(tensor, self.var_range, self.soft_zero, self.soft_bit, self.min, self.max)
        # 1) update intermediate parameters
        self.var_min.data = QParamOther._calc_min(self.var_max, self.var_range).clone()
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(2., 64.))
        self.zero.data = QParamOther._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamOther._calc_step(self.var_range, self.hard_bit)
        # 2) fake quantize
        return self.fakeQuantizer.apply(tensor, self.var_range, self.var_max, self.soft_bit,
                                        self.var_min, self.scale, self.zero.data)
        # return FakeQuantize.apply(tensor, self.var_range, self.var_max, self.soft_bit,
        #                           self.var_min, self.scale, self.zero.data)

    @staticmethod
    def _calc_step(var_range, hard_bit):

        qmin = torch.tensor([0.], device=hard_bit.device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(hard_bit.device)
        # qmax = torch.tensor(2. ** hard_bit - 1., device=soft_bit.device)

        scale_data = var_range / (qmax - qmin)

        return scale_data.data
        # return var_range.data, zero_point_data.data, scale_data.data

    @staticmethod
    def _calc_zero(var_max, var_range, hard_bit):
        if var_range:
            return ((1 - var_max / var_range) * (2 ** hard_bit - 1)).round_()
        else:
            return torch.tensor(0, device=var_max.device)

    @staticmethod
    def _calc_min(var_max, var_range):
        return var_max.data - var_range.data


# constrain bitwidth to a flexible range
class QParamOther_cons(nn.Module):
    # Z != 0;
    # qmin = 0;  qmax = 2**B-1
    def __init__(self, bit_width_init=8):
        super().__init__()
        # trainable parameters
        self.var_range = nn.Parameter(torch.tensor([0.]))
        self.soft_bit = nn.Parameter(torch.tensor([float(bit_width_init)]))
        self.var_max = nn.Parameter(torch.tensor([0.]))

        # intermediate paramaters
        self.var_min = torch.tensor([], requires_grad=False)
        self.hard_bit = torch.tensor([], requires_grad=False)
        self.zero = torch.tensor([], requires_grad=False)
        self.scale = torch.tensor([], requires_grad=False)

        # other functional variable
        self.fakeQuantizer = FakeQuantizeOther

        # temp variable
        self.var_min_init = torch.tensor([], requires_grad=False)

    def manual_min_max(self, var_min, var_max):  # only used for initialization
        self.var_min.data = torch.tensor([var_min], device=self.var_min.device, requires_grad=False)
        self.var_max.data = torch.tensor([var_max], device=self.var_max.device,requires_grad=True)
        self.var_range.data = (self.var_max.data - self.var_min.data).view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max.data, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def min_max(self, tensor, name=None):  # only used for initialization
        # name: used to collect histogram of quantized data
        flag = 0
        if torch.isinf(tensor.max().data) or torch.isneginf(tensor.min().data):
            return
        if self.var_max.nelement() == 0 or self.var_max.data < tensor.max().data:
            self.var_max.data = torch.tensor([tensor.max().data], device=tensor.device)
            flag = 1

        if self.var_min_init.nelement() == 0 or self.var_min_init.data > tensor.min().data:
            self.var_min_init.data = torch.tensor([tensor.min().data], device=tensor.device)
            flag = 1

        if flag:
            self.var_range.data = (self.var_max.data - self.var_min_init.data).view(torch.Size([1]))
        # self.var_range.data, self.scale = _calc_range(self.var_min, self.var_max, self.soft_bit.data)
        # self.var_range.data, self.soft_zero.data, self.scale = _calc_range(self.min, self.max, self.soft_bit.data)

    def __str__(self):  # used for print(QParam)
        self.var_min = QParamOther._calc_min(self.var_max, self.var_range)
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(7., 64.))
        self.zero.data = QParamOther._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamOther._calc_step(self.var_range, self.hard_bit)
        info = ' var_range: %.10f ' % self.var_range
        info += ' min: %.6f ' % self.var_min
        info += ' max: %.6f ' % self.var_max
        info += ' zero point: %d ' % self.zero
        info += ' soft_bit: %.2f ' % self.soft_bit
        info += ' hard_bit: %.2f ' % self.hard_bit
        return info

    def fake_quantize(self, tensor):
        # return FakeQuantize.apply(tensor, self.var_range, self.soft_zero, self.soft_bit, self.min, self.max)
        # 1) update intermediate parameters
        self.var_min.data = QParamOther._calc_min(self.var_max, self.var_range).clone()
        self.hard_bit.data = torch.round(self.soft_bit.data.clamp_(7., 64.))
        self.zero.data = QParamOther._calc_zero(self.var_max, self.var_range, self.hard_bit)
        self.scale.data = QParamOther._calc_step(self.var_range, self.hard_bit)
        # 2) fake quantize
        return self.fakeQuantizer.apply(tensor, self.var_range, self.var_max, self.soft_bit,
                                        self.var_min, self.scale, self.zero.data)
        # return FakeQuantize.apply(tensor, self.var_range, self.var_max, self.soft_bit,
        #                           self.var_min, self.scale, self.zero.data)

    @staticmethod
    def _calc_step(var_range, hard_bit):

        qmin = torch.tensor([0.], device=hard_bit.device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(hard_bit.device)
        # qmax = torch.tensor(2. ** hard_bit - 1., device=soft_bit.device)

        scale_data = var_range / (qmax - qmin)

        return scale_data.data
        # return var_range.data, zero_point_data.data, scale_data.data

    @staticmethod
    def _calc_zero(var_max, var_range, hard_bit):
        if var_range:
            return ((1 - var_max / var_range) * (2 ** hard_bit - 1)).round_()
        else:
            return torch.tensor(0, device=var_max.device)

    @staticmethod
    def _calc_min(var_max, var_range):
        return var_max.data - var_range.data
