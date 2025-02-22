__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import numpy as np
from einops import rearrange
#from collections import OrderedDict
# from models.layers.PatchTST_layers import *
from src.models.blocks.PatchTSTMha import MultiheadAttention
from src.models.blocks.RevIN import RevIN
from src.models.blocks.Encoding import positional_encoding
# from models.ts2vec.ncca import TSEncoderTime, GlobalLocalMultiscaleTSEncoder
# from models.ts2vec.fsnet import TSEncoder, GlobalLocalMultiscaleTSEncoder
import importlib
import copy
# Cell


class PositionwiseFeedForward(nn.Module):
    def __init__(self, args, d_model, d_ff, dropout=0., bias = True, activation = "GELU"):
        super().__init__()
        
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.activation = getattr(nn, activation)() 
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        
    def forward(self, x):

        out = self.layer1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out
    
    


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, configs, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="GELU", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False, channel_cross=False,
                 verbose:bool=False, mix_tcn=False,**kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(configs, c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, mix_tcn=mix_tcn, **kwargs)

        # Head
        self.channel_cross = channel_cross
        if channel_cross:
            self.head_nf = d_model * c_in # use the represetations of the last patch for forecasting
            target_window = c_in * target_window
        else:
            self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):    
        # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        bsz, nvars, seq_len = z.shape # 32, 321, 96
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z) # 32, 321, 104
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # num_seq: 96//16                 
        # z: [bs x nvars x patch_num x patch_len] 32, 321, 12, 16
        z = z.permute(0,1,3,2) # z: [bs x nvars x patch_len x patch_num] 32, 321, 16, 12
        
        # model
        z = self.backbone(z) # z: [bs x nvars x d_model x patch_num] 32, 321, 128, 12
        if self.channel_cross:
            z = z.permute(0, 3, 1, 2)                             # x: [bs x patch_len x nvars x patch_num]     
            z = z[:,-1,:,:]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        if self.channel_cross:
            z = z.view(bsz, nvars, -1)
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )



class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num] = 32, 321, 256, 2
        # import ipdb; ipdb.set_trace()
        if self.individual: # default: False
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x) # 32, 321, 512
            x = self.linear(x)
            x = self.dropout(x)
        return x
    
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, configs, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="GELU", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, mix_tcn=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(configs, q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, mix_tcn=mix_tcn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    

# Cell
class TSTEncoder(nn.Module):
    def __init__(self, configs, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='GELU',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, mix_tcn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(configs, q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention
        
    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None

        outputs = []

        for mod in self.layers:
            output, scores = mod(output, prev = scores if self.res_attention else None,
                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            
        
        return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, args, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="GELU", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        
        self.self_attn = MultiheadAttention(args, d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)

        self.ff = PositionwiseFeedForward(args, d_model, d_ff, dropout, bias, activation)
        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        
        
        Normalization = {
            'batchnorm': nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model, affine=args.affine ), Transpose(1,2)),
            'layernorm': nn.LayerNorm([q_len, d_model], elementwise_affine = args.affine),
            'instancenorm': nn.InstanceNorm1d(q_len, affine=args.affine),
        }[args.norm]
        
        
        self.norm_attn = Normalization
        self.norm_ffn = copy.deepcopy(Normalization)
        # import ipdb; ipdb.set_trace()
        

        self.args = args        
        # import ipdb; ipdb.set_trace()
        
        '''
        self_attn.W_Q 16384
        self_attn.W_K 16384
        self_attn.W_V 16384
        self_attn.to_out.0 16384
        norm_attn.1 128
        ff.0 32768
        ff.3 32768
        norm_ffn.1 128
        '''

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        '''
        src2 = residual attention QKV
        src2 = batchnorm
        src2 = dropout
        src = src + src2
        src = batchnorm(ff layer)
        
        src2 = ff layer (src)
        src = src + dropout(src2)
        src = batchnorm(src)
        '''
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
            
            
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # (321*32) x 12 x 128
        # import ipdb; ipdb.set_trace()
        if self.store_attn:
            self.attn = attn
        
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        
        # import ipdb; ipdb.set_trace()
        if not self.pre_norm: # batch norm along 128 (each timestep) (patchlen 16 -> 128)
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
                
            
        ## Position-wise Feed-Forward
        src2 = self.ff(src) # 128 -> 256 -> 128
        
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm: # batch norm
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src, None

