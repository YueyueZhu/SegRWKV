

"""
Created on Tue Sep 10 13:15:47 2024

@author: Md Mostafijur Rahman
"""

import sys
from typing import Tuple
import numpy as np
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.UXNet_3D.uxnet_encoder import uxnet_conv

import logging
logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class ModifiedUnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size,
        upsample_kernel_size ,
        norm_name,
        res_block = False,
        skip_aggregation = 'concatenation', 
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
            skip_aggregation: type of skip aggregation, addition or concatenation 
        """

        super().__init__()
        self.skip_aggregation = skip_aggregation
        in_out_channels = out_channels
        if self.skip_aggregation =='concatenation':
            in_out_channels = out_channels + out_channels
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  
                spatial_dims,
                in_out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        
        out = self.transp_conv(inp)
        if self.skip_aggregation=='concatenation':
            out = torch.cat((out, skip), dim=1)
        else:
            out = out + skip
        out = self.conv_block(out)
        return out

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)





















































class UXNET(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        
        
        
        
        
        
        
        
        
        self.hidden_size = hidden_size
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        
        
        
        
        
        enc1 = self.encoder1(x_in)
        
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        
        
        enc_hidden = self.encoder5(outs[3])
        
        dec3 = self.decoder5(enc_hidden, enc4)
        
        dec2 = self.decoder4(dec3, enc3)
        
        dec1 = self.decoder3(dec2, enc2)
        
        dec0 = self.decoder2(dec1, enc1)
        
        out = self.decoder1(dec0)
        
        
        
        return self.out(out)

class UXNET_EffiDec3D(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        n_decoder_channels=48,
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        skip_aggregation: str = 'concatenation',
        resolution_factor: int = 2,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        
        
        
        
        
        
        
        
        
        self.hidden_size = hidden_size
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.n_decoder_channels = n_decoder_channels
        self.resolution_factor = resolution_factor
        self.cls_head_in_channels = n_decoder_channels
        self.n_channels_enc2_dec3 = min(self.n_decoder_channels,self.feat_size[0])
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        if self.resolution_factor <= 1:
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.in_chans,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 2:
            self.encoder2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 4:
            self.encoder3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 8:
            self.encoder4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 16:
            self.encoder5 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.cls_head_in_channels = self.n_decoder_channels
        if self.resolution_factor <= 8:
            self.decoder5 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation,
            )
            self.cls_head_in_channels = self.n_decoder_channels
        if self.resolution_factor <= 4:
            self.decoder4 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation,
            )
            self.cls_head_in_channels = self.n_decoder_channels
        if self.resolution_factor <= 2:
            self.decoder3 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation
            )
            self.cls_head_in_channels = self.n_channels_enc2_dec3
        if self.resolution_factor <= 1:
            self.decoder2 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_channels_enc2_dec3,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation
            )
            self.decoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_channels_enc2_dec3,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.cls_head_in_channels = self.n_channels_enc2_dec3
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.cls_head_in_channels, out_channels=self.out_chans)
        

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):

        
        if self.resolution_factor > 16:
            print("Invalid resolution_factor for this model. Must be <= 16.")
            return sys.exit() 

        outs = self.uxnet_3d(x_in)
        
        
        
        
        
        
        
        
        enc1, enc2, enc3, enc4, enc_hidden, result = None, None, None, None, None, None

        if self.resolution_factor <= 1:
            enc1 = self.encoder1(x_in)
        if self.resolution_factor <= 2:
            x2 = outs[0]  
            enc2 = self.encoder2(x2) if hasattr(self, 'encoder2') else x2
        if self.resolution_factor <= 4:
            x3 = outs[1]
            enc3 = self.encoder3(x3) if hasattr(self, 'encoder3') else x3
        if self.resolution_factor <= 8:
            x4 = outs[2]
            enc4 = self.encoder4(x4) if hasattr(self, 'encoder4') else x4
        if self.resolution_factor <= 16:
            enc_hidden = self.encoder5(outs[3]) 
            result = enc_hidden  

        

        if self.resolution_factor <= 8:
            dec3 = self.decoder5(enc_hidden, enc4 if hasattr(self, 'encoder4') else None)
            result = dec3
        if self.resolution_factor <= 4:
            dec2 = self.decoder4(dec3, enc3 if hasattr(self, 'encoder3') else None)
            result = dec2
        if self.resolution_factor <= 2:
            dec1 = self.decoder3(dec2, enc2 if hasattr(self, 'encoder2') else None)
            result = dec1
        if self.resolution_factor <= 1:
            dec0 = self.decoder2(dec1, enc1 if hasattr(self, 'encoder1') else None)
            result = self.decoder1(dec0)

        
        
        return self.out(result)
