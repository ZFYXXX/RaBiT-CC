import torch
import torch.nn as nn
import torch.nn.functional as F

class BPL_Loss(nn.Module):
    def __init__(self, window_size=16, stride=16, delta=0.5, epsilon=1e-8,downsample_ratio=8.0):
    
        super(BPL_Loss, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.delta = delta
        self.epsilon = epsilon
        self.downsample_ratio = downsample_ratio
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.window_size, 
            stride=self.stride, 
            padding=0
        )

    def forward(self, D_R, D_T, r_R, r_T, points_list):
    
        C_R_j = self.avg_pool(D_R) * (self.window_size ** 2)
        C_T_j = self.avg_pool(D_T) * (self.window_size ** 2)
        
        B, _, H_w, W_w = C_R_j.shape
        
        N_GT_j = torch.zeros_like(C_R_j) 
        for b in range(B):
            points = points_list[b] 
            if points.numel() > 0:
                
                downsample_ratio = 4.0 
                
                points_ds = points / downsample_ratio
                
                bin_x = (points_ds[:, 0] / self.stride).long()
                bin_y = (points_ds[:, 1] / self.stride).long()
                
                mask = (bin_x < W_w) & (bin_y < H_w) & (bin_x >= 0) & (bin_y >= 0)
                bin_x = bin_x[mask]
                bin_y = bin_y[mask]
                
                indices = bin_y * W_w + bin_x
                counts = torch.bincount(indices, minlength=H_w * W_w).float()
                N_GT_j[b, 0, :, :] = counts.view(H_w, W_w)
        
        e_R_j = torch.abs(C_R_j - N_GT_j)
        e_T_j = torch.abs(C_T_j - N_GT_j)
        
        y_j = torch.where(e_R_j < e_T_j, 1.0, 0.0)
        
        ignore_mask = (torch.abs(e_R_j - e_T_j) < self.delta)
        
        r_R_j = self.avg_pool(r_R) 
        r_T_j = self.avg_pool(r_T)
        
        w_R_j = r_R_j / (r_R_j + r_T_j + self.epsilon) 
        
        loss = F.binary_cross_entropy(w_R_j, y_j, reduction='none')
        
        loss[ignore_mask] = 0.0
        
        num_valid_windows = torch.sum(~ignore_mask).float()
        if num_valid_windows == 0:
            return torch.tensor(0.0, device=D_R.device, requires_grad=True)
            
        return loss.sum() / num_valid_windows