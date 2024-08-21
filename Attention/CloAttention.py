"""
Author:  Qi tong Chen
Date: 2024.08.20
Attention module
"""
import torch
import torch.nn as nn
from efficientnet_pytorch.model import MemoryEfficientSwish
from torch import Tensor


class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            PConv(in_channels=dim, n_div=8, kernel_size=3, stride=1, forward='split_cat'),  # FC
            MemoryEfficientSwish(),
            PConv(in_channels=dim, n_div=8, kernel_size=3, stride=1, forward='split_cat'),  # FC
        )

    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):  # 16  [4 2]
    def __init__(self, dim, num_heads=8, group_split=[4, 4], kernel_sizes=[5], window_size=4,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(PConv(in_channels=3 * self.dim_head * group_head, n_div=8, kernel_size=kernel_size, stride=1, forward='split_cat'))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))  # FC

        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = PConv(in_channels=dim, n_div=8, kernel_size=3, stride=1, forward='split_cat')
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


class PConv(nn.Module):
    """
    Partial convolution (PConv)
    """
    def __init__(self, in_channels: int, n_div: int, kernel_size: int = 3, stride: int = 1, forward: str = 'split_cat'):
        """
        Construct a PConv layer.
        Args:
            in_channels: Number of input/output channels
            n_div: Reciprocal of the partial ratio
            kernel_size: default = 3 in paper
            stride: default=1 in paper
            forward: Forward type，can be either ' split_cat' or 'slicing', default=' split_cat'
        """
        super().__init__()
        self.dim_conv3 = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv3
        self.stride = stride
        self.partial_conv3 = nn.Conv2d(in_channels=self.dim_conv3,
                                       out_channels=self.dim_conv3,
                                       kernel_size=kernel_size,
                                       stride=self.stride,
                                       padding=(kernel_size-1) // 2,
                                       bias=False)
        self.averagePool = nn.AvgPool2d(3, 2, 1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        """
        for training/inference
        :param x:
        :return:
        """
        # Split the features to obtain the features of the partial channel x1 that needs convolution
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)  # Convolution for partial features
        if self.stride == 2:
            x2 = self.averagePool(x2)
        else:
            x2 = x2
        # Concatenate the features obtained by partial convolution with the remaining (original) features
        x = torch.cat((x1, x2), 1)

        return x


if __name__ == '__main__':
    block = EfficientAttention(96).cuda()
    input = torch.rand(1, 96, 4, 4).cuda()
    output = block(input)
    print(input.size(), output.size())

    print(torch.__version__)