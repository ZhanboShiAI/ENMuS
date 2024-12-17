import torch.nn as nn
import torch

from torch import Tensor
from typing import Optional

from sen_baselines.enmus.models.pooling_attn import PoolingAttentionConvEmbedFirst


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs):
        return inputs * inputs.sigmoid()
    

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)

class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be a multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)
    

class PointwiseConv1d(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            stride: int = 1,
            padding: int = 0,
            bias: bool = True
    ):
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            stride=stride,
            padding=padding, 
            bias=bias
    )

    def forward(self, x):
        return self.conv(x)


class MSMTConvModule(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ):
        super(MSMTConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be an odd number"
        assert expansion_factor == 2, "expansion_factor should be 2"

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)


class MultiHeadCrossAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super(MultiHeadCrossAttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_p, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,  
    ) -> Tensor:
        x = self.layer_norm(x)
        x, _ = self.attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout(x)
        

class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int = 512, expansion_factor: int = 4, dropout_p: float = 0.1):
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            Swish(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.sequential(x)
    

class MSMTDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_attention_heads: int,
            feed_forward_expansion_factor: int,
            conv_expansion_factor: int,
            feed_forward_dropout_p: float,
            attention_dropout_p: float,
            conv_dropout_p: float,
            conv_kernel_size: int,
            half_step_residual: bool = True,
            norm_first: bool = True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MSMTDecoderLayer, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0

        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_attention_heads,
            dropout=attention_dropout_p,
            bias = True,
            **factory_kwargs,
        )

        self.multi_head_attn = PoolingAttentionConvEmbedFirst(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=attention_dropout_p,
            bias=True,
        )

        self.self_attn_dropout = nn.Dropout(attention_dropout_p)
        self.multi_head_attn_dropout = nn.Dropout(attention_dropout_p)

        self.linear1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.linear2 = nn.Sequential(
            nn.Linear(d_model*2, d_model*4),
            Swish(),
            nn.Dropout(feed_forward_dropout_p),
            nn.Linear(d_model*4, d_model*2),
            nn.Dropout(feed_forward_dropout_p),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(True),
        )

        self.conv = MSMTConvModule(
            in_channels=d_model,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
        )

        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm4 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm5 = nn.LayerNorm(d_model, **factory_kwargs)

        if not norm_first:
            self.norm5 = nn.LayerNorm(d_model*2)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        x = tgt
        if self.norm_first:
            x = x + self._ff1_block(self.norm1(x)) * self.feed_forward_residual_factor
            x = x + self._sa_block(self.norm2(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x1 = x

            x = x + self._mha_block(self.norm3(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x1 = x1 + self._conv_block(self.norm4(x1))

            x = torch.cat([x, x1], dim=-1)
            x = x + self._ff2_block(self.norm5(x)) * self.feed_forward_residual_factor

            x = self.linear3(x)
        else:
            x = self.norm1(x + self._ff1_block(x) * self.feed_forward_residual_factor)
            x = self.norm2(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x1 = x

            x = self.norm3(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x1 = self.norm4(x1 + self._conv_block(x1))

            x = torch.cat([x, x1], dim=-1)
            x = self.norm5(x + self._ff2_block(x) * self.feed_forward_residual_factor)

            x = self.linear3(x)

        return x
            
            
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(x, x, x, 
                           attn_mask=attn_mask, 
                           key_padding_mask=key_padding_mask, 
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.self_attn_dropout(x)
    
    def _mha_block(
            self,
            x: Tensor,
            memory: Tensor,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
    ) -> Tensor:
        x = self.multi_head_attn(x, memory, memory,
                                 attn_mask=attn_mask, 
                                 key_padding_mask=key_padding_mask,
                                 is_causal=is_causal,
                                 need_weights=False)[0]
        return self.multi_head_attn_dropout(x)
    
    def _ff1_block(self, x: Tensor) -> Tensor:
        return self.linear1(x)
    
    def _ff2_block(self, x: Tensor) -> Tensor:
        return self.linear2(x)
    
    def _conv_block(self, x: Tensor) -> Tensor:
        return self.conv(x)


class MSMTDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MSMTDecoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None, 
                tgt_is_causal: bool = False, 
                memory_is_causal: bool = False
    ) -> Tensor:
        

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, 
                         memory_mask=memory_mask, 
                         tgt_key_padding_mask=tgt_key_padding_mask, 
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal = tgt_is_causal,
                         memory_is_causal=memory_is_causal
                    )
            
        if self.norm is not None:
            output = self.norm(output)
        
        return output