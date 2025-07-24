import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import LSTM

class MultiDomainAttention(nn.Module):
    def __init__(self, feature_dim, temporal_window):
        super(MultiDomainAttention, self).__init__()
        self.feature_dim = feature_dim
        self.temporal_window = temporal_window
        
        self.Wq = nn.Linear(feature_dim, feature_dim)
        self.Wk = nn.Linear(feature_dim, feature_dim)
        self.Wv = nn.Linear(feature_dim, feature_dim)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, F_block):
        # F_block: [batch_size, temporal_window, feature_dim]
        
        Q = self.Wq(F.normalize(F_block))   # [batch_size, temporal_window, feature_dim]
        K = self.Wk(F.normalize(F_block))
        V = self.Wv(F.normalize(F_block))
        
        beta_t = torch.bmm(Q, K.transpose(1, 2)) / (self.feature_dim ** 0.5)  # [batch_size, temporal_window, temporal_window]

        A_t = F.softmax(beta_t, dim=1)  # Temporal attention map
        
        beta_f = torch.bmm(K.transpose(1, 2), Q) / (self.temporal_window ** 0.5)  # [batch_size, feature_dim, feature_dim]

        A_f = F.softmax(beta_f, dim=0)  # Feature attention map
        
        V_t = torch.bmm(A_t, V)  # Temporal attention applied
        V_tf = torch.bmm(V_t, A_f)  # Feature attention applied
        
        return V_tf


class MWTF(nn.Module):
    def __init__(self, feature_dim, temporal_window_list):
        super(MWTF, self).__init__()
        # temporal_window_list: [4, 8, 16]
        self.temporal_window_list = temporal_window_list
        self.feature_dim = feature_dim
        self.extract = LSTM(feature_dim * len(temporal_window_list), 512, 2, 0.1, residual_embeddings=True) # output: (batch, sequence, features)
        
        self.attention_blocks = nn.ModuleList([
            MultiDomainAttention(feature_dim, window_length)
            for window_length in temporal_window_list
        ])

    def forward(self, Feat):
        batch_size, T, _ = Feat.size()
        
        outputs = []
        
        for attention_block, window_length in zip(self.attention_blocks, self.temporal_window_list):
            # Split
            num_blocks = T // window_length
            F_split = Feat.view(batch_size, num_blocks, window_length, -1).contiguous()  # [batch_size, num_blocks, window_length, feature_dim]
            
            block_results = []
            for i in range(num_blocks):
                block_result = attention_block(F_split[:, i, :, :])  # [batch_size, window_length, feature_dim]
                block_results.append(block_result)
            
            F_fused = torch.cat(block_results, dim=1)  # [batch_size, total_temporal_steps, feature_dim]
            outputs.append(F_fused)
            
        output = torch.cat(outputs, dim=2)  # [batch_size, T, total_feature_dim]
        output = self.extract(output)
        
        return output