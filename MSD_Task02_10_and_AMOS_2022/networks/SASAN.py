from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from operator import itemgetter
from torch.cuda.amp import autocast 

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]
    

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
        

    return permutations


class Permute(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for index in range(0, num_dimensions):
            for permutation in calculate_permutations(num_dimensions, dim_index + index):
                attentions.append(Permute(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions)) * 0.2

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


class SpatialAttention(torch.nn.Module):
    def __init__(self, spatial_dims, in_channels):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.conv = torch.nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        out = x * attention
        return out


class ASEM(nn.Module):
    def __init__(self, axial_dim, spatial_dims, in_channels, head, dim_heads=None):
        super().__init__()
        self.axial_attention = AxialAttention(dim=axial_dim, dim_index=2, heads=head, dim_heads=dim_heads,
                                              num_dimensions=3)
        self.spatial_attention = SpatialAttention(spatial_dims, in_channels)

    def forward(self, x):
        axial_output = self.axial_attention(x)
        spatial_output = self.spatial_attention(x)
        merged_output = axial_output * spatial_output
        return merged_output

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            feat: int = 96,
            dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            pre_conv: Optional[Union[nn.Module, str]] = "default",
            interp_mode: str = "linear",
            align_corners: Optional[bool] = True,
            halves: bool = True,
            dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  
        else:
            x = self.convs(x_0)

        return x


class filter_trans(nn.Module):
    def __init__(self, mode='low'):
        super(filter_trans, self).__init__()
        
        self.mode = mode
        

    def forward(self, x):
        f = torch.fft.fftn(x, dim=(2, 3, 4))
        fshift = torch.fft.fftshift(f)
        if self.mode == 'high':
            fshift = torch.fft.fftshift(f)

        return fshift















        








class FINE(nn.Module):
    def __init__(self, rate, cutoff, feat):  
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate), requires_grad=True)
        self.cutoff = cutoff
        self.feat = feat
        self.mask = nn.Parameter(torch.ones(1, self.feat, self.cutoff, self.cutoff, self.cutoff), requires_grad=True)

    def forward(self, x, fier):
        col = fier.shape[2]
        start_cut = (col - self.cutoff) // 2
        end_cut = start_cut + self.cutoff
        
        channel = fier.shape[1]
        fier2 = fier[:, :, start_cut:end_cut, start_cut:end_cut, start_cut:end_cut].repeat(1, int(self.feat / channel), 1, 1,
                                                                                           1) * self.mask
        
        original_dtype = x.dtype

        
        if x.dtype == torch.float16:
            x = x.float()
            fier2 = fier2.float()
        x_fft = torch.fft.fftn(x, dim=(2, 3, 4))
        x_fft = torch.fft.fftshift(x_fft)
        y = x_fft * self.rate + fier2 * (1 - self.rate)
        y = torch.fft.fftshift(y)
        y_ifft = torch.fft.ifftn(y, dim=(2, 3, 4))
        y_ifft_real = y_ifft.real

        
        if original_dtype == torch.float16:
            y_ifft_real = y_ifft_real.half()

        return y_ifft_real


class SASAN(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,

            features: Sequence[int] = (32, 64, 128, 256, 512, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            depths=[2, 2, 2, 2],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 512,
            conv_block: bool = True,
            res_block: bool = True,
            dimensions: Optional[int] = None,
    ):
        super().__init__()

        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, feat=12)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        self.fine1 = FINE(0.5, 96, 32)
        self.fine2 = FINE(0.5, 48, 64)
        self.fine3 = FINE(0.5, 24, 128)
        self.fine4 = FINE(0.5, 12, 256)
        self.fine5 = FINE(0.5, 6, 512)
        
        self.filter_trans = filter_trans(mode = 'low')

        self.asem = ASEM(axial_dim=96, spatial_dims=3, in_channels=out_channels, head=16)

    def forward(self, x: torch.Tensor):
        filter_low = self.filter_trans(x)
        x0 = self.conv_0(x)
        x0 = self.fine1(x0, filter_low) * x0
        x1 = self.down_1(x0)
        x1 = self.fine2(x1, filter_low) * x1
        x2 = self.down_2(x1)
        x2 = self.fine3(x2, filter_low) * x2
        x3 = self.down_3(x2)
        x3 = self.fine4(x3, filter_low) * x3
        x4 = self.down_4(x3)
        x4 = self.fine5(x4, filter_low) * x4
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        logits = self.asem(logits)

        return logits

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand((1, 4, 96, 96, 96)).to(device)
    model = SASAN(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        features=(32, 64, 128, 256, 512, 32),
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        bias=True,
        dropout=0.0
    )
    model.to(device=device)
    out = model(x)
    print(out.shape)
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print("FLOPs: {:.2f} M".format(flops / 1e6))
    print("Params: {:.2f} M".format(params / 1e6))
    