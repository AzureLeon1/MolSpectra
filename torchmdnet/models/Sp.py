from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")
from torchmdnet.models.CBAM import CBAMBlock
from torchmdnet.models.SpecFormer_layers import *


class ConvolutionalModule(nn.Module):
    def __init__(
        self,
        input_dim,
        kernel_size=5,
        stride=1,
        padding=0,
        pool_kernel_size=2,
        pool_stride=2,
        last_layer="cbam",
    ):
        super(ConvolutionalModule, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.maxpool1d = nn.MaxPool1d(
            kernel_size=pool_kernel_size, stride=pool_stride
        )  # 2, 2
        self.relu = nn.ReLU()

        self.last_layer_name = last_layer.lower()
        reduction = 32
        if self.last_layer_name == "mlp":
            self.last_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=input_dim // reduction,
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=input_dim // reduction,
                    out_channels=input_dim,
                    kernel_size=1,
                ),
            )
        elif self.last_layer_name == "cbam":
            self.last_layer = CBAMBlock(
                channel=input_dim
            )
            
    def reset_parameters(self) -> None:
        self.conv1d.reset_parameters()
        if self.last_layer_name == "mlp":
            self.last_layer.reset_parameters()
        elif self.last_layer_name == "cbam":
            self.last_layer.init_weights()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.maxpool1d(x)
        x = self.relu(x)
        x = self.last_layer(x)
        return x


class CNN_AM(nn.Module):
    def __init__(
        self,
        input_dim=702,
        hidden_channel=None,
        in_channel=1,
        kernel_size=5,
        stride=1,
        padding=0,
        output_channel=256,
        pool_kernel_size=2,
        pool_stride=2,
    ):
        super(CNN_AM, self).__init__()
        self.index = 1

        self.uv_proj = nn.Linear(701, 500)
        self.ir_proj = nn.Linear(3501, 500)
        self.raman_proj = nn.Linear(3501, 500)

        hidden_channel = 512 if hidden_channel == None else hidden_channel
        self.conv1d_input = nn.Conv1d(
            in_channels=in_channel,
            out_channels=hidden_channel,
            kernel_size=5,
            stride=1,
            padding=0,
        )
        self.conv_module = nn.ModuleList(
            [
                ConvolutionalModule(
                    input_dim=hidden_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride,
                    last_layer="cbam",
                )
                for _ in range(5)
            ]
        )
        self.adp_pool = nn.AdaptiveAvgPool2d((1, None))

        hidden_dim = (input_dim - kernel_size + 2 * padding) // stride + 1
        for i in range(5):
            hidden_dim = (hidden_dim - kernel_size + 2 * padding) // stride + 1
            hidden_dim = (hidden_dim - pool_kernel_size) // pool_stride + 1
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=output_channel)
        self.out_norm = nn.LayerNorm(output_channel)

        self.reset_parameters()

    def reset_parameters(self):
        self.uv_proj.reset_parameters()
        self.ir_proj.reset_parameters()
        self.raman_proj.reset_parameters()
        self.conv1d_input.reset_parameters()

        for conv_layer in self.conv_module:
            conv_layer.reset_parameters()

        self.fc1.reset_parameters()

    def forward(self, x):
        uv = self.uv_proj(x[0])
        ir = self.ir_proj(x[1])
        raman = self.raman_proj(x[2])

        x = torch.concat([uv, ir, raman], dim=1).unsqueeze(1)  # [batch, channel, feature_dim]

        if torch.isnan(x).sum() != 0:
            print("input vector x in CNN-AM has nan!")

        x = self.conv1d_input(x)

        for module in self.conv_module:
            if torch.isnan(x).sum() != 0:
                print('middle vector x in CNN-AM has nan!')
            x = module(x)

        x = self.adp_pool(x).squeeze(1)
        x = self.fc1(x)

        x = self.out_norm(x)

        if torch.isnan(x).sum() != 0:
            print('output vector x in CNN-AM has nan!')
        return x


class SpecFormer(nn.Module):
    def __init__(
        self,
        patch_len=[20,50,50], stride=[10,25,25], output_dim=256, mask_ratios=[0.1,0.1,0.1],
        
        input_norm_type:str='minmax',  # or 'log10', 'log'
        
        n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256, 
        attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto', padding_var:Optional[int]=None, 
        attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
        pe:str='zeros', learn_pe:bool=True, verbose:bool=False,
        
        fc_dropout:float=0., head_dropout = 0,
        pretrain_head:bool=False, head_type = 'flatten', individual = False,
        **kwargs
    ):
        super(SpecFormer, self).__init__()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratios = mask_ratios

        # uv ir raman
        list_len_spectrum = [701, 3501, 3501]

        patch_nums = [int((list_len_spectrum[i] - self.patch_len[i])/self.stride[i] + 1) for i in range(len(list_len_spectrum))]
        self.patch_nums = patch_nums
        all_patch_num = sum(patch_nums)

        self.input_norm_type = input_norm_type.lower()
        # Minmax_norm
        self.norm_eps = 1e-8
        self.spectra_min_vals = [0.0, 7.871089474065229e-5, 3.300115713500418e-5]                          # [uv, ir, raman]
        self.spectra_max_vals = [2.1593494415283203, 2029.77783203125, 19843.4453125]                      # [uv, ir, raman]

        # Backbone
        self.backbone = TSTiEncoder(patch_nums=patch_nums, patch_len=self.patch_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * all_patch_num
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        self.head = Flatten_Head(self.individual, self.head_nf, output_dim, head_dropout=head_dropout)

        self.reconstruct_heads = nn.ModuleList([nn.Linear(d_model, self.patch_len[i]) for i in range(len(self.patch_len))])

        self.out_norm = nn.LayerNorm(output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.backbone.reset_parameters()
        self.head.reset_parameters()

        for h in self.reconstruct_heads:
            nn.init.xavier_uniform_(h.weight)
            h.bias.data.fill_(0)

        self.out_norm.reset_parameters()

    def forward(self, spectra):  # spectra is a list

        # uv, ir, raman = x[0], x[1], x[2]

        # patching
        patched_spectra = []
        patched_spectra_masked = []
        masks = []
        for i, spec in enumerate(spectra):

            if self.input_norm_type == 'minmax':
                spec = (spec - self.spectra_min_vals[i] + self.norm_eps) / (self.spectra_max_vals[i] - self.spectra_min_vals[i] + self.norm_eps)
            elif self.input_norm_type == 'log10':
                spec = torch.log10(spec + 1)
            elif self.input_norm_type == 'log':
                spec = torch.log(spec + 1)

            spec = spec.unfold(dimension=-1, size=self.patch_len[i], step=self.stride[i])

            spec_masked, _, mask, _ = random_masking(spec, self.mask_ratios[i])

            masks.append(mask)

            spec = spec.permute(0,2,1)   
            patched_spectra.append(spec)  

            spec_masked = spec_masked.permute(0,2,1)   
            patched_spectra_masked.append(spec_masked)  

        # model
        z = self.backbone(patched_spectra_masked)          # list -> z: [bs x patch_num x d_model]

        # reconstruct
        z_reconstruct = z.clone()
        loss_reconstruct = 0
        start_idx = 0
        for i in range(len(spectra)):
            cur_reconstructed_patch = self.reconstruct_heads[i](z_reconstruct[:, start_idx:start_idx+self.patch_nums[i], :])
            start_idx += self.patch_nums[i]
            cur_orginal_patch = patched_spectra[i].permute(0,2,1)   
            loss_reconstruct += compute_reconstruct_loss(cur_reconstructed_patch, cur_orginal_patch, masks[i])

        # flatten and linear to get representations
        z = self.head(z)              # z: [bs x patch_num x d_model] -> z: [bs x output_dim]
        z = self.out_norm(z)
        return z, loss_reconstruct


def compute_reconstruct_loss(preds, target, mask):        
        """
        preds:   [bs x num_patch x patch_len]
        targets: [bs x num_patch x patch_len] 
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L,device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, patch_nums, patch_len,
                    n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256, 
                    norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", key_padding_mask='auto', padding_var=None, 
                    attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                    pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()

        self.patch_nums = patch_nums
        self.patch_len = patch_len

        # Input encoding
        self.W_P = nn.ModuleList([nn.Linear(patch_len[i], d_model) for i in range(len(patch_nums))])     # Eq 1: projection of feature vectors onto a d-dim vector space
        # Positional encoding
        # self.W_pos = [positional_encoding(pe, learn_pe, q_len, d_model) for q_len in patch_nums]
        self.W_pos_uv = positional_encoding(pe, learn_pe, patch_nums[0], d_model)
        self.W_pos_ir = positional_encoding(pe, learn_pe, patch_nums[1], d_model)
        self.W_pos_raman = positional_encoding(pe, learn_pe, patch_nums[2], d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        all_patch_nums = sum(patch_nums)
        self.encoder = TSTEncoder(all_patch_nums, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def reset_parameters(self):
        for w in self.W_P:
            nn.init.xavier_uniform_(w.weight)
            w.bias.data.fill_(0)

        self.encoder.reset_parameters()

    def forward(self, patched_spectra) -> Tensor:                                              # x: [bs x patch_len x patch_num]

        # Input encoding
        encoded_spectra = []
        for i, patched_spec in enumerate(patched_spectra):
            patched_spec = patched_spec.permute(0,2,1)         # x: [bs x patch_num x patch_len]
            patched_spec = self.W_P[i](patched_spec)           # x: [bs x patch_num x d_model]
            # patched_spec = self.dropout(patched_spec + self.W_pos[i])
            if i==0:
                patched_spec = self.dropout(patched_spec + self.W_pos_uv)
            elif i==1:
                patched_spec = self.dropout(patched_spec + self.W_pos_ir)
            elif i==2:
                patched_spec = self.dropout(patched_spec + self.W_pos_raman)

            encoded_spectra.append(patched_spec)

        # merge all spectra
        z = torch.cat(encoded_spectra, dim=1)

        # Encoder
        z = self.encoder(z)                                                      # z: [bs x patch_num x d_model] -> [bs x patch_num x d_model]
        # z = z.permute(0,2,1)                                                   # z: [bs x patch_num x d_model] -> [bs x d_model x patch_num]

        return z   


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                        attn_dropout=attn_dropout, dropout=dropout,
                                                        activation=activation, res_attention=res_attention,
                                                        pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                    norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if isinstance(self.norm_attn, nn.LayerNorm):
            self.norm_attn.reset_parameters()
        for name, param in self.ff.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, val=0)
        if isinstance(self.norm_ffn, nn.LayerNorm):
            self.norm_ffn.reset_parameters()

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        self.W_Q.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_K.weight)
        self.W_K.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_V.weight)
        self.W_V.bias.data.fill_(0)
        
    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class Flatten_Head(nn.Module):
    def __init__(self, individual, nf, target_window, head_dropout=0,  n_vars=1):
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

    def reset_parameters(self):
        if self.individual:
            for l in self.linears:
                nn.init.xavier_uniform_(l.weight)
                l.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.linear.weight)
            self.linear.bias.data.fill_(0)

    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

if __name__ == "__main__":
    input_dim = 3502
    channel = 3
    batch_size = 128

    model = CNN_AM(input_dim=input_dim, in_channel=channel, output_channel=256)

    input_data = torch.randn(batch_size, channel, input_dim)
    print(input_data.shape)

    output = model(input_data)
    print(output.shape)
