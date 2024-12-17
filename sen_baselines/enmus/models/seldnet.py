import numpy as np
import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        self.fc_o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        Q = self.fc_q(q)
        K = self.fc_k(k)
        V = self.fc_v(v)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)
        x = self.fc_o(x)
        
        return x
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class CRNN(nn.Module):
    def __init__(self, in_feat_shape, output_size) -> None:
        super().__init__()
        self.conv_block_list = nn.ModuleList()
        self.f_pool_size = [4, 4, 2]
        self.t_pool_size = [5, 1, 1]
        for conv_cnt in range(len(self.f_pool_size)):
            self.conv_block_list.extend([
                ConvBlock(
                    in_channels=64 if conv_cnt else in_feat_shape[1],
                    out_channels=64,
                ),
                nn.MaxPool2d((
                    self.t_pool_size[conv_cnt], 
                    self.f_pool_size[conv_cnt]
                )),
                nn.Dropout2d(
                    p=0.05,
                )
            ])
        
        self.gru = nn.GRU(
            input_size=64 * int(np.floor(in_feat_shape[-1] / np.prod(self.f_pool_size))),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.05,
            bidirectional=True,
        )

        self.attn = MultiHeadAttentionLayer(
            hidden_size=128,
            n_heads=4,
            dropout=0.05,
        )

        self.fnn_list = nn.ModuleList()
        self.fnn_list.extend([
            nn.Linear(128, 128, bias=True),
            nn.Linear(128, output_size[-1], bias=True),
        ])

    def forward(self, x):
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            x = torch.tanh(x)

        for fnn in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa