import os, math, gc, importlib
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 65535 
STOP_TOKEN_INDEX = 261

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
os.environ["RWKV_JIT_ON"] = "1"

if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

from torch.utils.cpp_extension import load

os.environ["RWKV_HEAD_SIZE_A"] = "40"

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

CHUNK_LEN = 16

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}",
         "--use_fast_math", "-O3", "-Xptxas", "-v", "-Xptxas", "-O3",  "--extra-device-vectorization"]

sources = [
    './FPG_WKV.cu',
    './FPG_WKV.cpp'
]

so_path = os.path.expanduser("~/.cache/torch_extensions/py310_cu121/wind_backstepping/FPG_WKV.so")

try:
    if os.path.exists(so_path):
        torch.ops.load_library(so_path)
    else:
        load(name="FPG_WKV", sources=sources,
             is_python_module=False, verbose=True,
             extra_cuda_cflags=flags)
except:
    
    time.sleep(2)
    torch.ops.load_library(so_path)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b, alpha):
        
        
        assert w.dtype == torch.bfloat16
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0

        device = w.device

        
        y = torch.empty_like(v)  
        
        s = torch.empty((B, H, T // CHUNK_LEN, C), dtype=torch.float32, device=device)
        sa = torch.empty((B, T, H, C), dtype=torch.float32, device=device)
        s_back = torch.empty_like(s)
        sa_back = torch.empty_like(sa)

        
        y_f = torch.empty_like(y)
        y_b = torch.empty_like(y)

        
        if not torch.is_tensor(alpha):
            alpha_t = torch.tensor(float(alpha), dtype=torch.float32, device=device)
        else:
            alpha_t = alpha.detach().to(device=device, dtype=torch.float32).contiguous()

        
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa, s_back, sa_back, y_f, y_b, alpha_t)

        
        
        ctx.save_for_backward(w, q, k, v, z, b, s, sa, s_back, sa_back, y_f, y_b, alpha_t)
        return y

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        w, q, k, v, z, b, s, sa, s_back, sa_back, y_f, y_b, alpha_t = saved

        
        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dz = torch.empty_like(z)
        db = torch.empty_like(b)

        device = w.device
        
        dalpha = torch.zeros(1, dtype=torch.float32, device=device)

        
        torch.ops.wind_backstepping.backward(
            w, q, k, v, z, b, dy, s, sa,
            dw, dq, dk, dv, dz, db,
            s_back, sa_back,
            y_f, y_b, dalpha, alpha_t
        )

        
        
        
        return dw, dq, dk, dv, dz, db, dalpha


def RUN_CUDA_RWKV7g(q, w, k, v, a, b, alpha):
    
    dev = q.device
    q,w,k,v,a,b = [t.contiguous().to(dev).to(torch.bfloat16) for t in (q,w,k,v,a,b)]
    
    B,T,HC = q.shape
    
    head_width = HC // (HC // HC)  
    
    
    C_local = HEAD_SIZE
    H = HC // C_local
    q,w,k,v,a,b = [i.view(B, T, H, C_local) for i in [q,w,k,v,a,b]]
    
    y = WindBackstepping.apply(w, q, k, v, a, b, alpha)
    
    return y.view(B, T, HC)







def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)

def x_shift(input, shift_pixel=1, gamma=1/6, patch_resolution=None):
    assert gamma <= 1/6, "gamma must be <= 1/6 for 6 direction groups"
    assert patch_resolution is not None and len(patch_resolution) == 3, "patch_resolution must be (D,H,W)"
    D, H, W = patch_resolution

    B, N, C = input.shape
    assert N == D * H * W, f"N({N}) must equal D*H*W({D*H*W})"
    assert 0 <= shift_pixel < min(D, H, W), "shift_pixel must be >=0 and < min(D,H,W)"

    
    x = input.transpose(1, 2).reshape(B, C, D, H, W)
    out = torch.zeros_like(x)

    g = int(C * gamma)  
    c0 = 0
    c1 = g
    c2 = 2 * g
    c3 = 3 * g
    c4 = 4 * g
    c5 = 5 * g
    c6 = 6 * g

    
    if g > 0:
        
        out[:, c0:c1, :, :, shift_pixel:W] = x[:, c0:c1, :, :, 0:W-shift_pixel]
        
        out[:, c1:c2, :, :, 0:W-shift_pixel] = x[:, c1:c2, :, :, shift_pixel:W]

        
        out[:, c2:c3, :, shift_pixel:H, :] = x[:, c2:c3, :, 0:H-shift_pixel, :]
        
        out[:, c3:c4, :, 0:H-shift_pixel, :] = x[:, c3:c4, :, shift_pixel:H, :]

        
        out[:, c4:c5, shift_pixel:D, :, :] = x[:, c4:c5, 0:D-shift_pixel, :, :]
        
        out[:, c5:c6, 0:D-shift_pixel, :, :] = x[:, c5:c6, shift_pixel:D, :, :]

    
    if c6 < C:
        out[:, c6:, ...] = x[:, c6:, ...]

    
    return out.flatten(2).transpose(1, 2)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, layer_id, n_embd = 512, head_size_a = 64, dim_att = 0, num_layer = 6, head_size_divisor = 8, shift_mode='x_shift'):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.head_size_a = head_size_a 
        self.dim_att = dim_att 
        self.num_layer = num_layer
        self.head_size_divisor = head_size_divisor
        self.shift_mode = shift_mode

        self.head_size = self.head_size_a 
        self.n_head = self.dim_att // self.head_size 
        assert self.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = self.n_embd

        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / (self.num_layer - 1)  
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.num_layer)  
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) 
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) 

            
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) 
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) 
            if self.layer_id != 0: 
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) 
            
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            if self.shift_mode == 'x_shift':
                self.time_shift = eval(self.shift_mode)
            else:
                self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(self.head_size_divisor**2)) 

            
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

            self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32)) 


    def forward(self, x, v_first, shift_pixel=1, gamma=1/6, patch_resolution=(128, 128, 128)):
        B, T, C = x.size()
        H = self.n_head
        if self.shift_mode == 'x_shift':
            xx = self.time_shift(x, shift_pixel=shift_pixel, gamma=gamma, patch_resolution=patch_resolution) - x
        else:    
            xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v 
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) 
        
        
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) 
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)

        
        k = k * (1 + (a-1) * self.k_a)

        
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a, self.alpha).to(torch.float32)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    


class RWKV_CMix_x070(nn.Module):
    def __init__(self, layer_id, n_embed = 512, num_layer = 6, shift_mode = 'x_shift'):
        super().__init__()
        self.layer_id = layer_id
        self.n_embed = n_embed
        self.num_layer = num_layer
        self.shift_mode = shift_mode
        if self.shift_mode == 'x_shift':
            self.time_shift = eval(self.shift_mode)
        else:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.num_layer)  
            ddd = torch.ones(1, 1, self.n_embed)
            for i in range(self.n_embed):
                ddd[0, 0, i] = i / self.n_embed
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(self.n_embed, self.n_embed * 4, bias=False)
        self.value = nn.Linear(self.n_embed * 4, self.n_embed, bias=False)

        
        self.key.weight.data.uniform_(-0.5/(self.n_embed**0.5), 0.5/(self.n_embed**0.5))
        self.value.weight.data.zero_()

    def forward(self, x, shift_pixel=1, gamma=1/6, patch_resolution=(128, 128, 128)):
        if self.shift_mode == 'x_shift':
            xx = self.time_shift(x, shift_pixel=shift_pixel, gamma=gamma, patch_resolution=patch_resolution) - x
        else:    
            xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)



class Block(nn.Module):
    def __init__(self, layer_id, n_embed = 512, num_layer = 6, head_size_a = 64, dim_att = 0, head_size_divisor = 8):
        super().__init__()
        self.n_embed = n_embed
        self.layer_id = layer_id
        self.num_layer = num_layer
        self.head_size_a = head_size_a
        self.dim_att = dim_att
        self.head_size_divisor = head_size_divisor
        if self.dim_att <= 0:
            self.dim_att = self.n_embed

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(self.n_embed) 
        self.ln1 = nn.LayerNorm(self.n_embed)
        self.ln2 = nn.LayerNorm(self.n_embed)
        
        self.att = RWKV_Tmix_x070(shift_mode='x_shift',layer_id = self.layer_id, n_embd = self.n_embed, head_size_a = self.head_size_a, dim_att = self.dim_att, num_layer=self.num_layer, head_size_divisor = self.head_size_divisor)
        
        self.ffn = RWKV_CMix_x070(shift_mode='x_shift',layer_id = self.layer_id, n_embed = self.n_embed, num_layer = self.num_layer)
        
    def forward(self, x, v_first, shift_pixel=1, gamma=1/6, patch_resolution=(128, 128, 128)):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first, shift_pixel=shift_pixel, gamma=gamma, patch_resolution=patch_resolution)
        x = x + xx
        x = x + self.ffn(self.ln2(x), shift_pixel=shift_pixel, gamma=gamma, patch_resolution=patch_resolution)
        return x, v_first

class RWKV(pl.LightningModule):
    def __init__(self, num_layer = 6, n_embd=1, hidhde_size=512, head_size_a = HEAD_SIZE, dropout=0.0):
        super().__init__()
        self.n_embd = n_embd
        self.hidhde_size = hidhde_size
        self.head_size_a = head_size_a
        self.dropout = dropout
        self.num_layer = num_layer
        
        self.blocks = nn.ModuleList([Block(n_embed = hidhde_size, layer_id = i, num_layer = self.num_layer, head_size_a = self.head_size_a) for i in range(self.num_layer)])
        self.ln_out = nn.LayerNorm(self.hidhde_size)
        

        if self.dropout > 0:
            self.drop0 = nn.Dropout(p = self.dropout)


    def unpad(self, x, num_tokens_to_pad):
        
        if num_tokens_to_pad > 0:
            x = x[:, num_tokens_to_pad:]
        return x

    def forward(self, x, shift_pixel=1, gamma=1/6, patch_resolution=(128, 128, 128)):

        if self.dropout > 0:
            x = self.drop0(x)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first, shift_pixel=shift_pixel, gamma=gamma, patch_resolution=patch_resolution)

        x = self.ln_out(x)
        return x


class ThreeAxisScannerRWKV7(nn.Module):
    def __init__(self,
                 dim,
                 num_layer=6,
                 input_layout='BCDHW',    
                 final_combine='sum'      
                 ):
        super().__init__()
        assert final_combine in ('sum', 'concat')
        assert len(input_layout) == 5 and set(input_layout) == set("BCDHW")
        self.in_ch = dim
        self.input_layout = input_layout.upper()
        self.final_combine = final_combine

        self.rwkv = RWKV(num_layer = num_layer, hidhde_size = dim)
        if self.final_combine == 'sum':
            self.outlinear = nn.Linear(dim, dim)
        else:
            self.outlinear = nn.Linear(dim*3, dim)


    def _layout_index(self):
        L = list(self.input_layout)
        return {ch: L.index(ch) for ch in L}

    def _flatten(self, x, target):  
        idx = self._layout_index()
        spatials = [c for c in self.input_layout if c in 'DHW' and c != target]
        perm = [idx['B'], idx[target], idx[spatials[0]], idx[spatials[1]], idx['C']]
        y = x.permute(*perm).contiguous()            
        B, A1, A2, A3, C = y.shape
        return y.view(B, A1*A2*A3, C), (A1, A2, A3)

    def _unflatten_to_BCHWD(self, seq, target, meta):
        A1, A2, A3 = meta
        B, L, M = seq.shape
        v = seq.view(B, A1, A2, A3, M)  

        spatials = [c for c in self.input_layout if c in 'DHW' and c != target]
        v_names = ['B', target, spatials[0], spatials[1], 'M']

        desired = list(self.input_layout)
        desired_with_M = ['M' if d == 'C' else d for d in desired]

        perm = [v_names.index(d) for d in desired_with_M]

        v = v.permute(*perm).contiguous()
        return v

    def forward(self, x):
        assert x.dim() == 5
        x_res = x
        seqs = []
        shapes = []
        for t in ['D', 'H', 'W']:
            seq, shape = self._flatten(x, t)
            seqs.append(seq); shapes.append((t, shape))

        outs = [self.rwkv(x = s, patch_resolution = shape) for s, (t,shape) in zip(seqs, shapes)]

        vols = [self._unflatten_to_BCHWD(o, t, m) for o, (t, m) in zip(outs, shapes)]

        if self.final_combine == 'sum':
            fused = vols[0] + vols[1] + vols[2]
        else:
            fused = torch.cat(vols, dim=1) 
        B, C = fused.shape[:2]
        img_dims = fused.shape[2:]
        n_tokens = fused.shape[2:].numel()
        fused_flat = fused.reshape(B, C, n_tokens).transpose(-1, -2)
        out = self.outlinear(fused_flat)
        out = out.transpose(-1, -2).reshape(B, C, *img_dims)
        return out + x_res

class LayerNorm(nn.Module):
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

class CNA3d(nn.Module):  
    def __init__(self, in_channels, out_channels, kSize, stride, padding=(1, 1, 1), bias=True, norm_args=None,
                 activation_args=None):
        super().__init__()
        self.norm_args = norm_args
        self.activation_args = activation_args

        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias, groups=in_channels)

        self.pwconv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        if norm_args is not None:
            self.norm = nn.InstanceNorm3d(out_channels, **norm_args)

        if activation_args is not None:
            self.activation = nn.LeakyReLU(**activation_args)

    def forward(self, x):
        
        x = self.dwconv(x)
        
        x = self.pwconv(x)

        if self.norm_args is not None:
            x = self.norm(x)

        if self.activation_args is not None:
            x = self.activation(x)
        return x


class CB3d(nn.Module):  
    def __init__(self, in_channels, out_channels, kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1), bias=True,
                 norm_args: tuple = (None, None), activation_args: tuple = (None, None)):
        super().__init__()

        self.conv1 = CNA3d(in_channels, out_channels, kSize=kSize[0], stride=stride[0],
                           padding=padding, bias=bias, norm_args=norm_args[0], activation_args=activation_args[0])

    def forward(self, x):
        x = self.conv1(x)
        return x


class BasicNet(nn.Module):
    norm_kwargs = {'affine': True}
    activation_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    def __init__(self):
        super(BasicNet, self).__init__()

    def parameter_count(self):
        print("model have {} paramerters in total".format(sum(x.numel() for x in self.parameters()) / 1e6))


def FMU(x1, x2, mode='sub'):
    if mode == 'sum':
        return torch.add(x1, x2)
    elif mode == 'sub':
        return torch.abs(x1 - x2)
    elif mode == 'cat':
        return torch.cat((x1, x2), dim=1)
    else:
        raise Exception('Unexpected mode')

class LECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.channel = channel
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.ln = nn.LayerNorm(channel)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.shape

        u = self.gap(x).view(b, c)        

        u = u.unsqueeze(1)                

        u = self.conv1(u)                 

        u = self.ln(u.squeeze(1))         
        u = self.prelu(u)
        u = self.sigmoid(u)

        u = u.view(b, c, 1, 1, 1)
        return u


class CSA_Inter_Module(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel

        self.spatial_conv = nn.Conv3d(in_channels=channel, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.leca = LECA(channel)
        self.conv_x2 = nn.Conv3d(in_channels=channel, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x1, x2):
        b, c, t, h, w = x1.shape

        z2 = self.spatial_conv(x2)
        z2 = self.sigmoid(z2)
        z2_expand = z2.expand(b, c, t, h, w)
        u1 = self.leca(x1)
        u1_expand = u1.expand(b, c, t, h, w)

        x1_refined = x1 * z2_expand
        x2_refined = self.conv_x2(x2) * u1_expand

        out = x1_refined + x2_refined   

        return out


class Down(BasicNet):
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub', downsample=True, min_z=8):
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.downsample = downsample
        self.FMU = FMU
        self.min_z = min_z
        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)
        self.IIM = CSA_Inter_Module(channel=in_channels)
        if self.mode_out == "both":
            self.attn1 = ActivationAttentionBlock(out_channels)   
            self.attn2 = ActivationAttentionBlock(out_channels)   

    def forward(self, x):
        if self.downsample:
            if self.mode_in == 'both':
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))

                
                x = self.IIM(p2d, p3d)

            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.mode_out == '2d':
            return self.CB2d(x)
        elif self.mode_out == '3d':
            return self.CB3d(x)
        elif self.mode_out == 'both':
            x2d = self.CB2d(x)
            x3d = self.CB3d(x)
            attn1 = self.attn1(x2d)
            attn2 = self.attn2(x3d)
            return x2d * attn2 + x2d, x3d * attn1 + x3d

class ActivationAttentionBlock(nn.Module):
    def __init__(self, in_channels, poolkernel_size=(1, 2, 2), poolstride=(1, 2, 2)):
        super(ActivationAttentionBlock, self).__init__()
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ln = nn.LayerNorm([in_channels])
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        batch_size, channels, height, width, depth = x.size()
        max_out = self.global_max_pool(x)  
        avg_out = self.global_avg_pool(x)  
        pool_sum2 = max_out + avg_out  
        
        pool_sum_flat = pool_sum2.view(batch_size, -1)  
        fc_out = self.fc(pool_sum_flat)               
        ln_out = self.ln(fc_out)                      
        fc_out = ln_out.view(batch_size, channels, 1, 1, 1)  
        prelu_out = self.prelu(fc_out)                
        attn = self.sigmoid(prelu_out)                
        
        return attn


class Up(BasicNet):
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub'):
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.FMU = FMU
        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)
        self.attenchannel = in_channels - out_channels
        self.attn1 = ActivationAttentionBlock(self.attenchannel)   
        self.attn2 = ActivationAttentionBlock(self.attenchannel)

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x

        tarSize = xskip2d.shape[2:]
        up2d = F.interpolate(x2d, size=tarSize, mode='trilinear', align_corners=False)
        up3d = F.interpolate(x3d, size=tarSize, mode='trilinear', align_corners=False)
        attn1 = self.attn1(up2d)
        attn2 = self.attn2(up3d)
        up2d = up2d * attn2 + up2d
        up3d = up3d * attn1 + up3d
        cat = torch.cat([FMU(xskip2d, xskip3d, self.FMU), FMU(up2d, up3d, self.FMU)], dim=1)

        if self.mode_out == '2d':
            return self.CB2d(cat)
        elif self.mode_out == '3d':
            return self.CB3d(cat)
        elif self.mode_out == 'both':
            return self.CB2d(cat), self.CB3d(cat)


class SegRWKV(BasicNet):
    def __init__(self, in_channels, out_channels, kn=(32, 48, 64, 80, 96), FMU='sub', evaluation = False):
        super().__init__()
        self.evaluation = evaluation
        channel_factor = {'sum': 1, 'sub': 1, 'cat': 2}
        fct = channel_factor[FMU]

        self.down11 = Down(in_channels, kn[0], ('/', 'both'), downsample=False)
        self.down12 = Down(kn[0], kn[1], ('2d', 'both'))
        self.down13 = Down(kn[1], kn[2], ('2d', 'both'))
        self.down14 = Down(kn[2], kn[3], ('2d', 'both'))
        self.bottleneck1 = Down(kn[3], kn[4], ('2d', '2d'))
        self.up11 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '2d'), FMU)
        self.up12 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '2d'), FMU)
        self.up13 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '2d'), FMU)
        self.up14 = Up(fct * (kn[0] + kn[1]), kn[0], ('both', 'both'), FMU)

        self.down21 = Down(kn[0], kn[1], ('3d', 'both'))
        self.down22 = Down(fct * kn[1], kn[2], ('both', 'both'), FMU)
        self.down23 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck2 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up21 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU)
        self.up22 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', 'both'), FMU)
        self.up23 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '3d'), FMU)

        self.down31 = Down(kn[1], kn[2], ('3d', 'both'))
        self.down32 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck3 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up31 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU)
        self.up32 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '3d'), FMU)

        self.down41 = Down(kn[2], kn[3], ('3d', 'both'), FMU)
        self.bottleneck4 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up41 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '3d'), FMU)

        self.bottleneck5 = Down(kn[3], kn[4], ('3d', '3d'))

        self.outputs = nn.ModuleList(
            [nn.Conv3d(c, out_channels, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
             for c in [kn[0], kn[1], kn[2], kn[3]]]
        )

        self.sigmoid = nn.Sigmoid()

        self.RWKV1 = ThreeAxisScannerRWKV7(dim=kn[4], num_layer=2) 
        self.RWKV2 = ThreeAxisScannerRWKV7(dim=kn[4], num_layer=2) 
        self.RWKV3 = ThreeAxisScannerRWKV7(dim=kn[4], num_layer=2) 
        self.RWKV4 = ThreeAxisScannerRWKV7(dim=kn[4], num_layer=2) 
        self.RWKV5 = ThreeAxisScannerRWKV7(dim=kn[4], num_layer=2) 

    def forward(self, x):
        down11 = self.down11(x)
        down12 = self.down12(down11[0])
        down13 = self.down13(down12[0])
        down14 = self.down14(down13[0])
        bottleNeck1 = self.bottleneck1(down14[0])

        bottleNeck1 = self.RWKV1(bottleNeck1)

        down21 = self.down21(down11[1])
        down22 = self.down22([down21[0], down12[1]])
        down23 = self.down23([down22[0], down13[1]])
        bottleNeck2 = self.bottleneck2([down23[0], down14[1]])

        bottleNeck2 = tuple(self.RWKV2(x) for x in bottleNeck2)

        down31 = self.down31(down21[1])
        down32 = self.down32([down31[0], down22[1]])
        bottleNeck3 = self.bottleneck3([down32[0], down23[1]])

        bottleNeck3 = tuple(self.RWKV3(x) for x in bottleNeck3)

        down41 = self.down41(down31[1])
        bottleNeck4 = self.bottleneck4([down41[0], down32[1]])

        bottleNeck4 = tuple(self.RWKV4(x) for x in bottleNeck4)

        bottleNeck5 = self.bottleneck5(down41[1])

        bottleNeck5 = self.RWKV5(bottleNeck5)

        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5, down41[1]])

        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[1]])
        up32 = self.up32([up31[0], down31[0], up41, down31[1]])

        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1], down23[1]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[1]])
        up23 = self.up23([up22[0], down21[0], up32, down21[1]])

        up11 = self.up11([bottleNeck1, down14[0], bottleNeck2[1], down14[1]])
        up12 = self.up12([up11, down13[0], up21[1], down13[1]])
        up13 = self.up13([up12, down12[0], up22[1], down12[1]])
        up14 = self.up14([up13, down11[0], up23, down11[1]])

        if self.evaluation == True:
            return self.outputs[0](up14[0] + up14[1])

        return self.outputs[0](up14[0] + up14[1]), self.outputs[1](up23), self.outputs[2](up32), self.outputs[3](up41)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegRWKV(1, 1, kn=(28, 36, 48, 64, 80), FMU='sub')
    x = torch.randn(1, 1, 128, 128, 128).to(device).contiguous()
    model.to(device=device)

    size_rates = [1]
    images = x
    res1, res2, res3, res4 = model(images)
    print("input: {}".format(images.shape))
    print("output: {}".format(res1.shape))
    print("output: {}".format(res2.shape))
    print("output: {}".format(res3.shape))
    print("output: {}".format(res4.shape))
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print("FLOPs: {:.2f} M".format(flops / 1e6))
    print("Params: {:.2f} M".format(params / 1e6))
