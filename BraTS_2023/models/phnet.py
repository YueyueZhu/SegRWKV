from torch import nn
import torch
from monai.networks.layers import DropPath, trunc_normal_
import torch.nn.functional as F


class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, p=0.2):
        super(_ConvINReLU3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(p=p, inplace=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_chns, out_chns, k=1, p=0, dropout=0.2):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=k,padding=p,p=dropout),
            _ConvIN3D(in_channels=out_chns,out_channels=out_chns,kernel_size=k,padding=p),
        )
        self.conv2 = nn.Sequential(
            _ConvINReLU3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p,p=dropout),
            _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p),
        )
        self.conv3 = nn.Conv3d(in_channels=in_chns, out_channels=out_chns, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv3(x)
        x1 = self.relu(x1+x)
        x2 = self.conv2(x1)
        x2 = self.relu(x2+x1)
        return x2



class Encoder_s(nn.Module):
    def __init__(self, in_chns, out_chns, k=1, p=0, dropout=0.2):
        super(Encoder_s, self).__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=k,padding=p,p=dropout),
            _ConvIN3D(in_channels=out_chns,out_channels=out_chns,kernel_size=k,padding=p),
        )
        self.conv2 = nn.Conv3d(in_channels=in_chns, out_channels=out_chns, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        x1 = self.relu(x1+x)
        return x1


class Decoder(nn.Module):  
    def __init__(self, in_chns,out_chns,dropout):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(   
             _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=(3,3,1),padding=(1,1,0),p=dropout),
             _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=(1,1,3), padding=(0,0,1)),
        )
        self.conv2 = nn.Conv3d(in_channels=in_chns,out_channels=out_chns,kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        x1 = self.relu(x1 + x)  
        return x1


class Up_cat(nn.Module):  
    def __init__(self, in_chns, cat_chns, out_chns, kernel,stride,dropout,attention_block=None, halves=True):
        super(Up_cat, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=kernel,stride=stride)
        self.attention = attention_block
        self.convs = Decoder(cat_chns+up_chns ,out_chns,dropout)

    def forward(self,x1,x2):   
        x_1 = self.up(x1)
        
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_1 = F.pad(x_1, sp, "replicate")
            x = torch.cat([x2,x_1],dim=1)   
            if self.attention is not None:   
                x,w = self.attention(x)
                x = self.convs(x)
                return x,w
            else:
                x = self.convs(x)
                return x

        else:
            x = self.convs(x_1)
            return x


class Up_sum(nn.Module):  
    def __init__(self, in_chns, out_chns, kernel,stride,dropout,attention_block=None, halves=True):
        super(Up_sum, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=kernel,stride=stride)
        self.attention = attention_block
        self.convs = Decoder(up_chns ,out_chns,dropout)

    def forward(self,x1,x2):   
        x_1 = self.up(x1)
        
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            if sp != [0] * (dimensions * 2):
                x_1 = F.pad(x_1, sp, "replicate")
            x = x_1+x2  
            if self.attention is not None:   
                x,w = self.attention(x)
                x = self.convs(x)
                return x,w
            else:
                x = self.convs(x)
                return x

        else:
            x = self.convs(x)
            return x


class conv_layer(nn.Module):
    def __init__ (self, dim, res_ratio, dropout_rate):
        super(conv_layer, self).__init__()
        self.network = []
        for i in range(3):
            if res_ratio < 2**i+ 2**(i-1):  
                self.network.append(Encoder(in_chns=dim[i],out_chns=dim[i+1],k=3,p=1, dropout=dropout_rate))
                if i < 2:
                    self.network.append(nn.MaxPool3d(kernel_size=2,stride=2))
            else:  
                self.network.append(Encoder(in_chns=dim[i],out_chns=dim[i+1],k=(3,3,1),p=(1,1,0), dropout=dropout_rate))
                if i < 2:
                    self.network.append(nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)))
        
        self.network = nn.Sequential(*self.network)
        
    def forward(self,x):
        output = []
        for i in range(len(self.network)):
            x = self.network[i](x)
            if i % 2 ==0:
                output.append(x)
        
        return output


class deconv_layer(nn.Module):
    def __init__ (self, embed_dims, res_ratio, dropout_rate):
        super(deconv_layer, self).__init__()
        self.network = []
        for i in range(len(embed_dims)-1, 0, -1):
            is_half = True if embed_dims[i] == 2*embed_dims[i-1] else False
            if i <= 3 and res_ratio >= 2**(i-1)+ 2**(i-2): 
                self.network.append(Up_sum(in_chns=embed_dims[i], out_chns=embed_dims[i-1], kernel=(2, 2, 1), stride=(2, 2, 1), dropout=dropout_rate,
                                    halves=is_half))
            else:
                self.network.append(Up_sum(in_chns=embed_dims[i], out_chns=embed_dims[i-1], kernel=2, stride=2, dropout=dropout_rate,
                    halves=is_half))
        self.network = nn.Sequential(*self.network)   
    
    def forward(self, hidden_states):
        for i in range(0, len(self.network)):
            if i == 0:
                x = self.network[i](hidden_states[0], hidden_states[1])
            else:
                x = self.network[i](x, hidden_states[i+1])
        
        return x



class MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            hidden_size_2: int,
            dropout_rate: float = 0.0,
    ):
        super(MLP, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size_2)
        self.fn = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size: tuple, hidden_size, dropout_rate=0.0):
        super(PatchEmbedding, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.patch_embeddings = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.embed_dim = hidden_size

    def forward(self, x):
        x2 = self.patch_embeddings(x)
        x_shape = x2.size()
        x2 = x2.flatten(2).transpose(1, 2)
        x2 = self.norm(x2)
        d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
        x2 = x2.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
        x2 = self.dropout(x2)
        return x2


class PatchMerging3d(nn.Module):
    def __init__(self, dim, kernel_size=2, double=True):
        super(PatchMerging3d, self).__init__()
        if double:
            self.pool = nn.Conv3d(dim, 2 * dim, kernel_size=kernel_size, stride=kernel_size)
        else:
            self.pool = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        x = self.pool(x)
        return x


class IP_MLP(nn.Module):
    def __init__(self, dim, segment_dim=14, qkv_bias=False, proj_drop=0.0):
        super(IP_MLP, self).__init__()
        self.segment_dim = segment_dim
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)  
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = MLP(dim, dim // 4, dim * 3)
        self.attention_reweight = nn.Linear(segment_dim*segment_dim,segment_dim*segment_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, D, C = x.shape
        S = C // self.segment_dim  
        
        m = x.reshape(B,H//self.segment_dim, self.segment_dim, W//self.segment_dim,self.segment_dim,D,C).permute(0,1,3,5,6,2,4).reshape(B,H//self.segment_dim,W//self.segment_dim,D,C,self.segment_dim*self.segment_dim)
        m = self.attention_reweight(m).reshape(B,H//self.segment_dim,W//self.segment_dim,D,C,self.segment_dim,self.segment_dim).permute(0,1,5,2,6,3,4).reshape(B, H, W, D, C) 

        h = x.transpose(2,1).reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H*W//self.segment_dim, self.segment_dim, D,
                                                                                          self.segment_dim* S)
        h = self.mlp_h(h).reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, W, H, D, C).transpose(2,1)

        w = x.reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H*W//self.segment_dim, self.segment_dim, D,
                                                                                         self.segment_dim * S)
        w = self.mlp_w(w).reshape(B, H*W//self.segment_dim, self.segment_dim, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H, W, D, C)

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        x = x+m*x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TP_MLP(nn.Module):
    def __init__(self, dim, segment_dim=14, qkv_bias=False, proj_drop=0.0):
        super(TP_MLP, self).__init__()
        self.segment_dim = segment_dim
        self.mlp_d = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, D, C = x.shape
        S = C // self.segment_dim
        d = x.reshape(B, H, W, D//self.segment_dim,self.segment_dim, self.segment_dim, S).permute(0, 1, 2, 3, 5,4,6).reshape(B, H, W, D//self.segment_dim, self.segment_dim, 
                                                                                      self.segment_dim * S)
        x = self.mlp_d(d).reshape(B, H, W, D//self.segment_dim, self.segment_dim,self.segment_dim , S).permute(0, 1, 2, 3, 5,4,6).reshape(B, H, W, D, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim, mlp_ratio=3.0, qkv_bias=False,
                 drop_path=0.0, skip_lam=1.0):
        super(PermutatorBlock, self).__init__()
        self.s_norm = nn.LayerNorm(dim)
        self.t_norm = nn.LayerNorm(dim)
        self.attn1 = IP_MLP(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, proj_drop=drop_path)
        self.attn2 = TP_MLP(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, proj_drop=drop_path)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(hidden_size=dim, mlp_dim=mlp_hidden_dim, hidden_size_2=dim, dropout_rate=drop_path)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = x + self.attn1(self.s_norm(x))/self.skip_lam
        x = x + self.drop_path(self.attn2(self.t_norm(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        x = x.permute(0, 4, 1, 2, 3)
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3.0, qkv_bias=False, drop_path_rate=0.0, skip_lam=1.0):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop_path=block_dpr, skip_lam=skip_lam))
    blocks = nn.Sequential(*blocks)
    return blocks


class MLPP(nn.Module):
    def __init__(self, res_ratio, layers, in_channels=1,
                 embed_dims=None, segment_dim=None, mlp_ratios=3.0, skip_lam=1.0,
                 qkv_bias=False, dropout_rate=0.2,
                 ):
        super(MLPP, self).__init__()

        if res_ratio > 6 : 
            patch_size=(2, 2, 1)
        else:
            patch_size=(2, 2, 2)
        self.patch_embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, hidden_size=embed_dims[0],
                                          dropout_rate=dropout_rate)

        self.network = []
        for i in range(len(layers)):
            self.network.append(
                basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                             drop_path_rate=dropout_rate, skip_lam=skip_lam))

            if i >= len(layers) - 1:
                break
            elif embed_dims[i + 1] == 2 * embed_dims[i]:
                self.network.append(PatchMerging3d(embed_dims[i]))
            else:
                self.network.append(PatchMerging3d(embed_dims[i], double=False))
        self.network = nn.Sequential(*self.network)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        hidden_states_out = []
        x1 = self.patch_embed(x)
        for i in range(len(self.network) // 2 + 1):
            if i >= len(self.network) // 2:
                x1 = self.network[2 * i](x1)
            else:
                hidden = self.network[2 * i](x1)  
                hidden_states_out.append(hidden)
                x1 = self.network[2 * i + 1](x1)  

        hidden_states_out.append(x1)
        return hidden_states_out



class PHNet(nn.Module):
    def __init__(self, res_ratio, layers, in_channels, out_channels, embed_dims, segment_dim, mlp_ratio, dropout_rate):
        super(PHNet,self).__init__()
        conv_dim = [in_channels,embed_dims[0],embed_dims[1],embed_dims[2]]
        self.conv = conv_layer(conv_dim, res_ratio, dropout_rate=dropout_rate)
        self.deconv = deconv_layer(embed_dims, res_ratio,dropout_rate=dropout_rate)
        self.mlpp = MLPP(res_ratio,layers,in_channels=embed_dims[-3],embed_dims=embed_dims[-2:],segment_dim=segment_dim,
                                    mlp_ratios=mlp_ratio,dropout_rate=dropout_rate)
        self.final_conv = nn.Conv3d(embed_dims[0],out_channels=out_channels,kernel_size=1)

    def forward(self, x):
        conv_hidden_states = self.conv(x)
        mlpp_hidden_states = self.mlpp(conv_hidden_states[-1])
        hidden_states = (conv_hidden_states + mlpp_hidden_states)[::-1]
        u0 = self.deconv(hidden_states)
        logits = self.final_conv(u0)
        return logits

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PHNet(
            res_ratio=5 / 0.74,
            layers=(15, 4),
            in_channels=4,  
            out_channels=4,  
            embed_dims=(42, 84, 168, 168, 336),
            segment_dim=(8, 8),
            mlp_ratio=4.0,
            dropout_rate=0.2
        )
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