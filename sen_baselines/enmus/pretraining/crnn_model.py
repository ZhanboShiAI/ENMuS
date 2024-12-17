from turtle import forward
import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.div(torch.matmul(Q, K.permute(0, 1, 3, 2)), np.sqrt(self.head_dim))
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        
        x = self.fc_o(x)
        
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, key_channels, 1, bias=False)
        self.key_conv = nn.Conv1d(in_channels, key_channels, 1, bias=False)
        self.value_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        sim_map = torch.matmul(key.permute(0, 2, 1), query)
        sim_map = torch.softmax(sim_map, dim=-1)
        x = torch.matmul(value, sim_map).permute(0, 2, 1)
        # output dim: [batch_size, seq_len, out_channels]
        
        return x
    
    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.query_conv.in_channels,
            self.value_conv.out_channels,
            self.key_conv.out_channels
            )
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu_(x)
        return x
    
class CRNN(nn.Module):
    def __init__(self, in_feat_shape, out_dim, params) -> None:
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            # if downsample:
            #     self.in_gru_size = params['nb_cnn2d_filt'] * int( np.floor( np.ceil(in_feat_shape[-1] / 4) / np.prod(params['f_pool_size'])))
            # else:
            self.in_gru_size = params['nb_cnn2d_filt'] * int( np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        
        self.attn = None
        if params['self_attn']:
            self.attn = MultiHeadAttentionLayer(
                hid_dim=params['rnn_size'],
                n_heads=params['nb_heads'],
                dropout=params['dropout_rate']
            )

        self.fnn_list = nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(
                        in_features=params['fnn_size'] if fc else params['rnn_size'],
                        out_features=params['fnn_size'],
                        bias=True
                    )
                )
        self.fnn_list.append(
            nn.Linear(
                in_features=params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'],
                out_features=out_dim,
                bias=True
            )
        )

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device='cuda:0').unsqueeze(0)
            
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

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
