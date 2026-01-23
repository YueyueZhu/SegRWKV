

""""""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Tuple
import torch

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from torch.nn.functional import interpolate
import functools
import os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from torch.nn.functional import interpolate


class ModuleHelper(object):

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == 'torchbn':
            return nn.Sequential(
                nn.BatchNorm3d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'torchsyncbn':
            return nn.Sequential(
                nn.SyncBatchNorm(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return nn.Sequential(
                BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return nn.Sequential(
                SwitchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'gn':
            return nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'fn':
            
            exit(1)
        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver in ('1.0', '1.1'):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)

        else:
            
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type='torch', ret_cls=False):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d

        elif bn_type == 'torchsyncbn':
            return nn.SyncBatchNorm

        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return BatchNorm2d

        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return SwitchNorm2d

        elif bn_type == 'gn':
            return functools.partial(nn.GroupNorm, num_groups=32)

        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='none')

            elif torch_ver in ('1.0', '1.1'):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='none')

            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='identity')

        else:
            
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network='resnet101'):
        if pretrained is None:
            return model

        if all_match:
            
            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'resinit.{}'.format(k) in model_dict:
                    load_dict['resinit.{}'.format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:
            
            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)

            
            if network == "wide_resnet":
                pretrained_dict = pretrained_dict['state_dict']

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                
                
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pvt':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                pretrained_dict['pos_embed1'] =\
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed2'] =\
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed3'] =\
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                pretrained_dict['pos_embed4'] =\
                    interpolate(pretrained_dict['pos_embed4'].unsqueeze(dim=0), size=[256, 512])[0]
                pretrained_dict['pos_embed7'] =\
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed6'] =\
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed5'] =\
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pcpvt' or network == 'svt':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                

            elif network == 'transunet_swin':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                for item in list(pretrained_dict.keys()):
                    if item.startswith('layers.0') and not item.startswith('layers.0.downsample'):
                        pretrained_dict['dec_layers.2' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.1') and not item.startswith('layers.1.downsample'):
                        pretrained_dict['dec_layers.1' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.2') and not item.startswith('layers.2.downsample'):
                        pretrained_dict['dec_layers.0' + item[15:]] = pretrained_dict[item]

                for item in list(pretrained_dict.keys()):
                    if 'relative_position_index' in item:
                        pretrained_dict[item] =\
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[256, 256])[0][0]
                    if 'relative_position_bias_table' in item:
                        pretrained_dict[item] =\
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[961, pretrained_dict[item].size(1)])[0][0]
                    if 'attn_mask' in item:
                        pretrained_dict[item] =\
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[pretrained_dict[item].size(0), 256, 256])[0][0]

            elif network == "hrnet" or network == "xception" or network == 'resnest':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                

            elif network == "dcnet" or network == "resnext":
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if 'resinit.{}'.format(k) in model_dict:
                        load_dict['resinit.{}'.format(k)] = v
                    else:
                        if k in model_dict:
                            load_dict[k] = v
                        else:
                            pass

            elif network == "wide_resnet":
                load_dict = {'.'.join(k.split('.')[1:]): v\
                             for k, v in pretrained_dict.items()\
                             if '.'.join(k.split('.')[1:]) in model_dict}
            else:
                load_dict = {'.'.join(k.split('.')[1:]): v\
                             for k, v in pretrained_dict.items()\
                             if '.'.join(k.split('.')[1:]) in model_dict}

            
            if int(os.environ.get("debug_load_model", 0)):
                
                for key in load_dict.keys():
                    
                    pass
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.PyTorchCV', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            
            urlretrieve(url, cached_file)

        
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)












class LayerNorm(nn.Module):
    ''
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class ux_block(nn.Module):
    ''

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) 
        x = self.norm(x)
        x = x .permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x


class uxnet_conv(nn.Module):
    """"""
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() 
        
        
        
        
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ux_block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        

    def forward_features(self, x):
        outs = []
        for i in range(4):
            
            
            x = self.downsample_layers[i](x)
            
            x = self.stages[i](x)
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x



class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

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
        """"""

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

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UXNET(
            in_chans=4,
            out_chans=4,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3)
    x = torch.randn(1, 4, 128, 128, 128).to(device)
    model.to(device=device)

    
    size_rates = [1]
    images = x
    res1 = model(images)
    print("input: {}".format(images.shape))
    print("output: {}".format(res1.shape))
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print("FLOPs: {:.2f} M".format(flops / 1e6))
    print("Params: {:.2f} M".format(params / 1e6))