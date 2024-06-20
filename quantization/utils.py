import os
import torch
import torch.nn as nn
import scipy.io as io

from torch.autograd import Function


def _quantize_tensor(x, scale, hard_zero, qmin, qmax):
    if scale:
        q_x = hard_zero + x / scale
        q_x.clamp_(qmin, qmax).round_()
    else:
        q_x = torch.zeros_like(x)

    return q_x


def _dequantize_tensor(q_x, scale, hard_zero):
    return scale * (q_x - hard_zero)


def _grad_range(grad_output, x, var_range, hard_zero, hard_bit,
                mask_mid, mask_small, mask_large, var_max):
    grad_range = torch.zeros_like(x)
    q_range = 2**hard_bit - 1
    grad_range[mask_mid] = torch.round(x[mask_mid] * q_range / var_range) / q_range - x[mask_mid] / var_range
    grad_range[mask_small] = -1 * var_max / var_range - hard_zero / q_range
    grad_range[mask_large] = 1 - 1 * var_max / var_range - hard_zero / q_range
    grad_range = grad_range * grad_output
    return grad_range.sum().view(torch.Size([1]))


def _grad_vmax(grad_output, x, mask_mid):
    grad_vmax = torch.ones_like(x)
    grad_vmax[mask_mid] = 0
    grad_vmax = grad_vmax * grad_output
    return grad_vmax.sum().view(torch.Size([1]))


def _grad_bit(grad_output, x, var_range, hard_zero, hard_bit, var_max, mask_mid, mask_small, mask_large):
    # if x.dtype==torch.float16 or grad_output.dtype==torch.float16:
    #     x = x.to(torch.float32)
    #     grad_output = grad_output.to(torch.float32)

    grad_bit = torch.zeros_like(x)
    q_max = 2 ** hard_bit - 1 # r_q
    tmp = (q_max+1) * torch.log(torch.tensor(2))  # 2^B * ln2
    factor_tmp = tmp * var_range / (q_max**2)  # blue
    grad_adj = hard_zero * factor_tmp + tmp * (var_max - var_range) / q_max

    grad_bit[mask_mid] = -1 * factor_tmp * torch.round(x[mask_mid] * q_max / var_range) + x[mask_mid] * tmp / q_max
    # grad_bit[mask_mid] = tmp / q_max * torch.abs(x[mask_mid] - var_range * torch.round(x[mask_mid] * q_max / var_range) / q_max)
    grad_bit[mask_small] = grad_adj
    grad_bit[mask_large] = grad_adj
    grad_bit = grad_bit * grad_output

    return grad_bit.sum().view(torch.Size([1]))


def _grad_range_weight(grad_output, x, var_range, hard_zero, hard_bit,
                       mask_mid, mask_small, mask_large, var_max):
    grad_range = torch.zeros_like(x)
    q_range = 2**hard_bit - 1
    grad_range[mask_mid] = torch.round(x[mask_mid] * q_range / var_range) / q_range - x[mask_mid] / var_range
    grad_range[mask_large] = (2**(hard_bit-1) - 1) / q_range
    grad_range[mask_small] = (-1 * 2**(hard_bit-1)) / q_range
    grad_range = grad_range * grad_output
    return grad_range.sum().view(torch.Size([1]))


def _grad_bit_weight(grad_output, x, var_range, hard_zero, hard_bit, var_max, mask_mid, mask_small, mask_large):
    # if x.dtype==torch.float16 or grad_output.dtype==torch.float16:
    #     x = x.to(torch.float32)
    #     grad_output = grad_output.to(torch.float32)

    grad_bit = torch.zeros_like(x)
    q_range = 2 ** hard_bit - 1  # r_q
    tmp = (q_range+1) * torch.log(torch.tensor(2))  # 2^B * ln2
    factor_tmp = tmp * var_range / (q_range**2)  # blue

    grad_bit[mask_mid] = -1 * factor_tmp * torch.round(x[mask_mid] * q_range / var_range) + x[mask_mid] * tmp / q_range
    # grad_bit[mask_mid] = tmp / q_range * torch.abs(x[mask_mid] - var_range * torch.round(x[mask_mid] * q_range / var_range) / q_range)
    grad_bit[mask_large] = -1 * (2**(hard_bit-1)-1) * factor_tmp + factor_tmp * q_range / 2
    grad_bit[mask_small] = 2**(hard_bit-1) * factor_tmp - factor_tmp * q_range / 2
    grad_bit = grad_bit * grad_output

    return grad_bit.sum().view(torch.Size([1]))


class FakeQuantizeOther(Function):

    @staticmethod
    def forward(ctx,
                x, var_range, var_max, soft_bit,
                var_min, scale, zero):
    # def forward(ctx, x, var_range, soft_zero, soft_bit, var_min, var_max):
        # x: [Batch, Ndim]
        # range, soft_zero, soft_bit:
        device = var_range.device

        # # ============ used for debug =============
        # x_backup = x.clone().detach().to(device)

        hard_bit = torch.round(soft_bit.data.clamp_(1., 64.))
        # zero = _calc_zero(var_max, var_range, hard_bit)
        # hard_zero = torch.round(soft_zero.data)
        ctx.save_for_backward(x, var_range, zero, hard_bit, var_min, var_max)
        # ctx.save_for_backward(x, var_range, hard_zero, hard_bit, var_min, var_max)
        qmin = torch.tensor(0., device=device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(device)
        # # qmax = torch.tensor(2. ** hard_bit - 1., device=var_min.device)
        # scale = var_range / (qmax - qmin)
        # x = _quantize_tensor(x, scale, hard_zero, qmin, qmax)
        # x = _dequantize_tensor(x, scale, hard_zero)
        output = _quantize_tensor(x, scale, zero, qmin, qmax)
        output = _dequantize_tensor(output, scale, zero)

        # if torch.sum((x - x_backup)**2):
        #     pass

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # for SGD: need to sum the grad of all samples in one mini-batch,
        # (no need to average which has been done in loss function)
        # grad_output: [Nx, Ny]
        x, var_range, hard_zero, hard_bit, var_min, var_max = ctx.saved_tensors
        mask_mid = (x >= var_min) & (x <= var_max)
        mask_small = (x < var_min)
        mask_large = (x > var_max)
        # grad_x:
        grad_x = torch.zeros_like(x, dtype=grad_output.dtype)
        grad_x[mask_mid] = grad_output[mask_mid]
        # grad_x = grad_output.clone()
        # grad_x = grad_output

        # grad_range:
        grad_range = _grad_range(grad_output, x, var_range, hard_zero,
                                 hard_bit, mask_mid, mask_small, mask_large, var_max)
        # print(grad_range.shape)

        # grad_vmax:
        grad_vmax = _grad_vmax(grad_output, x, mask_mid)
        # print(grad_zero.shape)

        # grad_bit:
        grad_bit = _grad_bit(grad_output, x, var_range, hard_zero,
                             hard_bit, var_max,
                             mask_mid, mask_small, mask_large)
        # print(grad_bit.shape)

        # print(grad_output[1000000][1])

        return grad_x, grad_range, grad_vmax, grad_bit, None, None, None


class FakeQuantizeReLU(Function):

    @staticmethod
    def forward(ctx,
                x, var_range, var_max, soft_bit,
                var_min, scale, zero):
    # def forward(ctx, x, var_range, soft_zero, soft_bit, var_min, var_max):
        # x: [Batch, Ndim]
        # range, soft_zero, soft_bit:
        device = var_range.device

        # # ============ used for debug =============
        # x_backup = x.clone().detach().to(device)

        hard_bit = torch.round(soft_bit.data.clamp_(1., 64.))
        # zero = _calc_zero(var_max, var_range, hard_bit)
        # hard_zero = torch.round(soft_zero.data)
        ctx.save_for_backward(x, var_range, zero, hard_bit, var_min, var_max)
        # ctx.save_for_backward(x, var_range, hard_zero, hard_bit, var_min, var_max)
        qmin = torch.tensor(0., device=device)
        qmax = (2. ** hard_bit - 1.).clone().detach().to(device)
        # # qmax = torch.tensor(2. ** hard_bit - 1., device=var_min.device)
        # scale = var_range / (qmax - qmin)
        # x = _quantize_tensor(x, scale, hard_zero, qmin, qmax)
        # x = _dequantize_tensor(x, scale, hard_zero)
        output = _quantize_tensor(x, scale, zero, qmin, qmax)
        output = _dequantize_tensor(output, scale, zero)

        # if torch.sum((x - x_backup) ** 2):
        #     pass

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # for SGD: need to sum the grad of all samples in one mini-batch,
        # (no need to average which has been done in loss function)
        # grad_output: [Nx, Ny]
        x, var_range, hard_zero, hard_bit, var_min, var_max = ctx.saved_tensors
        mask_mid = (x >= var_min) & (x <= var_max)
        mask_small = (x < var_min)
        mask_large = (x > var_max)
        # grad_x:
        grad_x = torch.zeros_like(x, dtype=grad_output.dtype)
        grad_x[mask_mid] = grad_output[mask_mid]
        # grad_x = grad_output.clone()
        # grad_x = grad_output

        # grad_range:
        grad_range = _grad_range(grad_output, x, var_range, hard_zero,
                                 hard_bit, mask_mid, mask_small, mask_large, var_max)

        # grad_bit:
        grad_bit = _grad_bit(grad_output, x, var_range, hard_zero,
                             hard_bit, var_max,
                             mask_mid, mask_small, mask_large)
        # print(grad_bit.shape)

        # print(grad_output[1000000][1])

        return grad_x, grad_range, None, grad_bit, None, None, None


class FakeQuantizeWeight(Function):

    @staticmethod
    def forward(ctx,
                x, var_range, var_max, soft_bit,
                var_min, scale, zero):
    # def forward(ctx, x, var_range, soft_zero, soft_bit, var_min, var_max):
        # x: [Batch, Ndim]
        # range, soft_zero, soft_bit:
        device = var_range.device

        # # ============ used for debug =============
        # x_backup = x.clone().detach().to(device)

        hard_bit = torch.round(soft_bit.data.clamp_(1., 64.))
        # zero = _calc_zero(var_max, var_range, hard_bit)
        # hard_zero = torch.round(soft_zero.data)
        ctx.save_for_backward(x, var_range, zero, hard_bit, var_min, var_max)
        # ctx.save_for_backward(x, var_range, hard_zero, hard_bit, var_min, var_max)
        qmin = (-2. ** (hard_bit - 1.)).clone().detach().to(device)
        qmax = (2. ** (hard_bit - 1.) - 1.).clone().detach().to(device)
        # # qmax = torch.tensor(2. ** hard_bit - 1., device=var_min.device)
        # scale = var_range / (qmax - qmin)
        # x = _quantize_tensor(x, scale, hard_zero, qmin, qmax)
        # x = _dequantize_tensor(x, scale, hard_zero)
        output = _quantize_tensor(x, scale, zero, qmin, qmax)
        output = _dequantize_tensor(output, scale, zero)

        # if torch.sum((x - x_backup) ** 2):
        #     pass

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # for SGD: need to sum the grad of all samples in one mini-batch,
        # (no need to average which has been done in loss function)
        # grad_output: [Nx, Ny]
        x, var_range, hard_zero, hard_bit, var_min, var_max = ctx.saved_tensors
        mask_mid = (x >= var_min) & (x <= var_max)
        mask_small = (x < var_min)
        mask_large = (x > var_max)
        # grad_x:
        grad_x = torch.zeros_like(x, dtype=grad_output.dtype)
        grad_x[mask_mid] = grad_output[mask_mid]
        # grad_x = grad_output.clone()
        # grad_x = grad_output

        # grad_range:
        grad_range = _grad_range_weight(grad_output, x, var_range, hard_zero,
                                        hard_bit, mask_mid, mask_small, mask_large, var_max)

        # grad_bit:
        grad_bit = _grad_bit_weight(grad_output, x, var_range, hard_zero,
                                    hard_bit, var_max,
                                    mask_mid, mask_small, mask_large)
        # print(grad_bit.shape)

        # print(grad_output[1000000][1])

        return grad_x, grad_range, None, grad_bit, None, None, None
