import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from typing import Optional, Tuple, List
from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention

class PoolingAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super(PoolingAttention, self).__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, batch_first=batch_first, device=device, dtype=dtype
        )

        self.pool_ratios = [4, 8, 16, 32]
        self.head_dim = embed_dim // num_heads

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if not is_batched:
            raise NotImplementedError("PoolingAttention only supports batched inputs")
        if self.batch_first:
            raise NotImplementedError("PoolingAttention only supports batch_first=False")
        
        if not self._qkv_same_embed_dim:
            raise NotImplementedError("PoolingAttention only supports q, k, v with same feature dim")
         
        attn_output, attn_output_weights = self._single_forward(
            query, key, value, key_padding_mask, need_weights, attn_mask, 
            average_attn_weights, is_causal,
        )

        return attn_output, attn_output_weights
    
    def _single_forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:        
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        if is_causal:
            raise NotImplementedError("PoolingAttention does not support causal attention")
        
        assert key is value, "PoolingAttention only supports key is value"

        assert key.shape[0] % self.pool_ratios[-1] == 0, \
            f"PoolingAttention only supports sequence length divisible by {self.pool_ratios[-1]}"
        
        key = key.permute(1, 2, 0)
        value = key = self.get_pyinput(key)
        
        q, k, v = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The size of the 2D attn_mask is {attn_mask.size()}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_2d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The size of the 3D attn_mask is {attn_mask.size()}, but should be {correct_2d_size}."
                    )
            else:
                raise RuntimeError("attn_mask's dimension must be 2 or 3")
        
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
            "key_padding_mask shape is not correct with size {}, but the correct shape should be {}".format(key_padding_mask.shape, (bsz, src_len))
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if self.dropout > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

            attn_output = torch.bmm(attn_output_weights, v) # [bsz * num_heads, tgt_len, embed_dim]
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            return attn_output, attn_output_weights
        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)
            
            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, self.num_heads, src_len, self.head_dim)
            v = v.view(bsz, self.num_heads, src_len, self.head_dim)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            return attn_output, None

    def _in_projection_packed(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            w: Tensor,
            b: Optional[Tensor] = None,
    ) -> List[Tensor]:
        # add support for q is k
        E = q.size(-1)
        if k is v:
            if q is k:
                # self-attention
                proj = F.linear(q, w, b)
                proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                return proj[0], proj[1], proj[2]
            else:
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                q_proj = F.linear(q, w_q, b_q)
                kv_proj = F.linear(k, w_kv, b_kv)
                kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                return (q_proj, kv_proj[0], kv_proj[1])
        else:
            raise NotImplementedError("PoolingAttention only supports k is v ")
        
    
    def get_pyinput(self, key):
        pools = [key]

        for i, ratio in enumerate(self.pool_ratios):
            if i == 0:
                pool = F.avg_pool1d(key, ratio, ratio)
                pools.append(pool)
            else:
                pool = F.avg_pool1d(pools[-1], 2, 2)
                pools.append(pool)
        
        key = torch.cat(pools, dim=2).permute(2, 0, 1)

        return key


class ConvLayer(nn.Module):
    def __init__(self, in_channels, window_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=window_size,
            stride=window_size,
        )
        self.norm = nn.BatchNorm1d(in_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    

class ConvConstruct(nn.Module):
    def __init__(self, d_model, window_size):
        super(ConvConstruct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                ]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size[0]),
                    ConvLayer(d_model, window_size[1]),
                    ConvLayer(d_model, window_size[2]),
                    ConvLayer(d_model, window_size[3]),
                ]
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        all_input = [x]
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            all_input.append(x)

        all_input = torch.cat(all_input, dim=2).transpose(1, 2)
        all_input = self.norm(all_input)

        return all_input
    

class ConvConstructReverse(ConvConstruct):
    def __init__(self, d_model, window_size):
        super().__init__(d_model, window_size)

    def forward(self, x):
        all_input = [x]

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            all_input.insert(0, x)

        all_input = torch.cat(all_input, dim=2).transpose(1, 2)
        all_input = self.norm(all_input)

        return all_input


class PoolingAttentionConv(PoolingAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super(PoolingAttentionConv, self).__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, batch_first=batch_first, device=device, dtype=dtype
        )

        self.conv_construct = ConvConstruct(embed_dim, window_size=[self.pool_ratios[0], 2, 2, 2])


    def get_pyinput(self, key):        
        conv_keys = self.conv_construct(key)
        return conv_keys.permute(1, 0, 2).contiguous()
    

class PoolingAttentionConvEmbedFirst(PoolingAttentionConv):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super(PoolingAttentionConvEmbedFirst, self).__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, batch_first=batch_first, device=device, dtype=dtype
        )

    def _single_forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:        
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        if is_causal:
            raise NotImplementedError("PoolingAttention does not support causal attention")
        
        assert key is value, "PoolingAttention only supports key is value"

        assert key.shape[0] % self.pool_ratios[-1] == 0, \
            f"PoolingAttention only supports sequence length divisible by {self.pool_ratios[-1]}"
        
        q, k, v = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        k = k.permute(1, 2, 0)
        v = k = self.get_pyinput(k)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The size of the 2D attn_mask is {attn_mask.size()}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_2d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The size of the 3D attn_mask is {attn_mask.size()}, but should be {correct_2d_size}."
                    )
            else:
                raise RuntimeError("attn_mask's dimension must be 2 or 3")
        
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
            "key_padding_mask shape is not correct with size {}, but the correct shape should be {}".format(key_padding_mask.shape, (bsz, src_len))
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if self.dropout > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

            attn_output = torch.bmm(attn_output_weights, v) # [bsz * num_heads, tgt_len, embed_dim]
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            return attn_output, attn_output_weights
        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)
            
            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, self.num_heads, src_len, self.head_dim)
            v = v.view(bsz, self.num_heads, src_len, self.head_dim)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            return attn_output, None
        