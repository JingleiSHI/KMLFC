import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from fb import *

class Conv_DCF(nn.Module):
    r"""Pytorch implementation for 2D DCF Convolution operation.
    Link to ICML paper:
    https://arxiv.org/pdf/1802.04145.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, optional): Zero-padding added to both sides of
            the input. Default: 0
        num_bases (int, optional): Number of basis elements for decomposition.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        mode (optional): Either `mode0` for two-conv or `mode1` for reconstruction + conv.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 num_bases=-1, bases_grad=False, dilation=1):
        super(Conv_DCF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bases_grad = bases_grad
        self.dilation = dilation

        if kernel_size % 2 == 0:
            raise Exception('Kernel size for FB initialization only supports odd number for now.')
        base_np, _, _ = calculate_FB_bases(int((kernel_size-1)/2))
        if num_bases > base_np.shape[1]:
            raise Exception('The maximum number of bases for kernel size = %d is %d' %(kernel_size, base_np.shape[1]))
        elif num_bases == -1:
            num_bases = base_np.shape[1]
        else:
            base_np = base_np[:, :num_bases]
        base_np = base_np.reshape(kernel_size, kernel_size, num_bases)
        base_np = np.array(np.expand_dims(base_np.transpose(2,0,1), 1), np.float32)

        self.bases = Parameter(torch.tensor(base_np).float(), requires_grad=bases_grad)
        self.weight = Parameter(torch.Tensor(out_channels, in_channels*num_bases, 1, 1))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        self.num_bases = num_bases

        if self.mode == 'mode1':
            self.weight.data = self.weight.data.view(out_channels, in_channels, num_bases)
            self.bases.data = self.bases.data.view(num_bases, kernel_size, kernel_size)
            self.forward = self.forward_mode1
        else:
            self.forward = self.forward_mode0

    def forward_mode0(self, input):
        N, C, H, W = input.size()
        input = input.view(N*C, 1, H, W)
        
        feature = F.conv2d(input, self.bases, None, self.stride, self.padding, dilation=self.dilation)
        
        H = int((H-self.kernel_size+2*self.padding)/self.stride+1)
        W = int((W-self.kernel_size+2*self.padding)/self.stride+1)

        feature = feature.view(N, C*self.num_bases, H, W)

        feature_out = F.conv2d(feature, self.weight, self.bias, 1, 0)

        return feature_out

    def forward_mode1(self, input):
        rec_kernel = torch.einsum('abc,cdf->abdf', self.weight, self.bases)

        feature = F.conv2d(input, rec_kernel, self.bias, self.stride, self.padding, dilation=self.dilation)
        
        return feature
