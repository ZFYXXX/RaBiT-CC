import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size_tuple):

    B, H, W, C = x.shape
    Wh, Ww = window_size_tuple 
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C) 
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Wh, Ww, C) 
    return windows


def window_reverse(windows, window_size_tuple, H, W):
    
    Wh, Ww = window_size_tuple 
    B = int(windows.shape[0] / (H * W / Wh / Ww)) 
    x = windows.view(B, H // Wh, W // Ww, Wh, Ww, -1) 
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class RaBiT_Fusion(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, num_mediators=4, 
                 local_radius=9, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.window_size = to_2tuple(window_size)
        self.num_mediators = num_mediators  
        self.local_radius = local_radius      
        
        self.attn_pool_psi = nn.Linear(dim * 2, num_mediators) 
        self.attn_pool_phi = nn.Linear(dim * 2, dim * 2)       
        self.norm_pool = norm_layer(dim * 2)
        self.norm_Z = norm_layer(dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) 
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        self.norm_Z_ffn = norm_layer(dim)
        self.ffn_Z = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                         act_layer=act_layer, drop=drop)
        
        self.norm_R_out = norm_layer(dim)
        self.norm_T_out = norm_layer(dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate_g = nn.Sequential(
            nn.Linear(2, 8), 
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def _get_relative_position_bias(self):
        return self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], 
            self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous().unsqueeze(0)

    def _get_local_attention_mask(self, H, W, device):
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  
        coords_flatten = torch.flatten(coords, 1)  
        
        relative_coords_dist = torch.max(
            torch.abs(coords_flatten[:, :, None] - coords_flatten[:, None, :]),
            dim=0
        )[0] 
        
        mask = torch.where(relative_coords_dist <= self.local_radius, 0.0, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0) 

    def forward(self, F_R, F_T, r_R, r_T,):
        B, C, H, W = F_R.shape
        Wh, Ww = self.window_size
        
        F_R = F_R.permute(0, 2, 3, 1).contiguous()
        F_T = F_T.permute(0, 2, 3, 1).contiguous()
        
        F_R_windows = window_partition(F_R, self.window_size) 
        F_T_windows = window_partition(F_T, self.window_size) 
        
        F_R_windows = F_R_windows.view(-1, Wh * Ww, C)
        F_T_windows = F_T_windows.view(-1, Wh * Ww, C)
        B_nW = F_R_windows.shape[0] 

        F_hat = torch.cat([F_R_windows, F_T_windows], dim=2)
        F_hat_norm = self.norm_pool(F_hat)

        A_m = F.softmax(self.attn_pool_psi(F_hat_norm), dim=1) 
        Z_m = torch.bmm(A_m.transpose(1, 2), self.attn_pool_phi(F_hat_norm)) 
        
        Z_m = (Z_m[:, :, :C] + Z_m[:, :, C:]) / 2.0
        Z_m = self.norm_Z(Z_m) 

        relative_position_bias = self._get_relative_position_bias() 
        local_mask = self._get_local_attention_mask(Wh, Ww, F_R.device)
        attn_mask = relative_position_bias + local_mask
        
        r_R_windows = window_partition(r_R.permute(0, 2, 3, 1), self.window_size).view(B_nW, -1, 1)
        r_T_windows = window_partition(r_T.permute(0, 2, 3, 1), self.window_size).view(B_nW, -1, 1)

        Q_Z = self.q_proj(Z_m).reshape(B_nW, self.num_mediators, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        K_R = self.k_proj(F_R_windows).reshape(B_nW, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) 
        V_R = self.v_proj(F_R_windows).reshape(B_nW, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        K_T = self.k_proj(F_T_windows).reshape(B_nW, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        V_T = self.v_proj(F_T_windows).reshape(B_nW, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn_R_Z = (Q_Z @ K_R) * self.scale 
        
        attn_R_Z = attn_R_Z * r_R_windows.permute(0, 2, 1).unsqueeze(1) 
        attn_R_Z = F.softmax(attn_R_Z, dim=-1)
        attn_R_Z = self.attn_drop(attn_R_Z)
        Z_hat_R = (attn_R_Z @ V_R).transpose(1, 2).contiguous().reshape(B_nW, self.num_mediators, C) 

        
        attn_T_Z = (Q_Z @ K_T) * self.scale
        
        attn_T_Z = attn_T_Z * r_T_windows.permute(0, 2, 1).unsqueeze(1)
        attn_T_Z = F.softmax(attn_T_Z, dim=-1)
        attn_T_Z = self.attn_drop(attn_T_Z)
        Z_hat_T = (attn_T_Z @ V_T).transpose(1, 2).contiguous().reshape(B_nW, self.num_mediators, C) 

        Z_m_prime = Z_m + self.proj_drop(Z_hat_R) + self.proj_drop(Z_hat_T)
        Z_m_prime = self.norm_Z_ffn(Z_m_prime)
        Z_m_prime = self.ffn_Z(Z_m_prime) 

        Q_R = self.q_proj(F_R_windows).reshape(B_nW, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        Q_T = self.q_proj(F_T_windows).reshape(B_nW, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        K_Zp = self.k_proj(Z_m_prime).reshape(B_nW, self.num_mediators, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1) 
        V_Zp = self.v_proj(Z_m_prime).reshape(B_nW, self.num_mediators, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        
        attn_Z_R = (Q_R @ K_Zp) * self.scale 
        
        attn_Z_R = attn_Z_R * r_T_windows.unsqueeze(1) 
        attn_Z_R = F.softmax(attn_Z_R, dim=-1)
        attn_Z_R = self.attn_drop(attn_Z_R)
        F_R_hat = (attn_Z_R @ V_Zp).transpose(1, 2).contiguous().reshape(B_nW, Wh * Ww, C) 

        
        attn_Z_T = (Q_T @ K_Zp) * self.scale
        
        attn_Z_T = attn_Z_T * r_R_windows.unsqueeze(1) 
        attn_Z_T = F.softmax(attn_Z_T, dim=-1)
        attn_Z_T = self.attn_drop(attn_Z_T)
        F_T_hat = (attn_Z_T @ V_Zp).transpose(1, 2).contiguous().reshape(B_nW, Wh * Ww, C)

        F_R_prime = self.norm_R_out(F_R_windows + self.proj_drop(F_R_hat)) 
        F_T_prime = self.norm_T_out(F_T_windows + self.proj_drop(F_T_hat)) 
        
        F_R_prime = F_R_prime.view(-1, Wh, Ww, C)
        F_T_prime = F_T_prime.view(-1, Wh, Ww, C)
        F_R_prime = window_reverse(F_R_prime, self.window_size, H, W) 
        F_T_prime = window_reverse(F_T_prime, self.window_size, H, W) 
        
        F_R_prime = F_R_prime.permute(0, 3, 1, 2).contiguous() 
        F_T_prime = F_T_prime.permute(0, 3, 1, 2).contiguous() 

        r_R_pooled = self.pool(r_R) 
        r_T_pooled = self.pool(r_T) 
        gamma = self.gate_g(torch.cat([r_R_pooled, r_T_pooled], dim=1).squeeze(-1).squeeze(-1)) 
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) 

        F_J = gamma * F_R_prime + (1.0 - gamma) * F_T_prime

        return F_J