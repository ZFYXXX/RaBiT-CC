import torch
import torch.nn as nn
from utils.tensor_ops import cus_sample, upsample_add
from models.Modules import BasicConv2d, DenseFusion, FeatureFusionAndPrediction,ReliabilityEstimator
from models.pvt_v2_encoders import pvt_v2_b3
from models.RaBiT_Fusion import RaBiT_Fusion
import torch.nn.functional as F

def pad_to_window_size(x, window_size=8):
    B, C, H, W = x.shape
    
    target_H = (H + window_size - 1) // window_size * window_size
    target_W = (W + window_size - 1) // window_size * window_size
    
    if target_H == H and target_W == W:
        return x, (H, W) 

    pad_h = target_H - H
    pad_w = target_W - W

    padded_x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
    original_size = (H, W)
    
    return padded_x, original_size

class RaBiT_CC(nn.Module):
    def __init__(self, construct_method='threshold', k_interact=4, thresh_interact=0.8):
        super(RaBiT_CC, self).__init__()
        
        self.upsample_add = upsample_add   
        self.upsample = cus_sample      
        
        # -------------------- Backbone (PVT) --------------------
        self.pvt_backbone_rgb = pvt_v2_b3 (pretrained="pretrained_weights/pvt_v2_b3.pth")
        self.pvt_backbone_t   = pvt_v2_b3 (pretrained="pretrained_weights/pvt_v2_b3.pth")
        
        # Adjust the number of output channels
        self.trans_rgb_32x = BasicConv2d(512, 64, 1)
        self.trans_rgb_16x = BasicConv2d(320, 64, 1)
        self.trans_rgb_8x  = BasicConv2d(128, 64, 1)
        self.trans_rgb_4x  = BasicConv2d(64,  64, 1)
        
        self.trans_t_32x = BasicConv2d(512, 64, 1)
        self.trans_t_16x = BasicConv2d(320, 64, 1)
        self.trans_t_8x  = BasicConv2d(128, 64, 1)
        self.trans_t_4x  = BasicConv2d(64,  64, 1)
        
        # ------------ Reliability Estimator Heads ------------
        self.reli_head_32x = ReliabilityEstimator(in_channels=64)
        self.reli_head_16x = ReliabilityEstimator(in_channels=64)
        self.reli_head_8x  = ReliabilityEstimator(in_channels=64)
        self.reli_head_4x  = ReliabilityEstimator(in_channels=64)

        # ------------ RaBiT-Fusion Modules ------------
        rabbit_kwargs = dict(dim=64, num_heads=2, window_size=8, num_mediators=4)
        self.rabbit_32x = RaBiT_Fusion(**rabbit_kwargs)
        self.rabbit_16x = RaBiT_Fusion(**rabbit_kwargs)
        self.rabbit_8x  = RaBiT_Fusion(**rabbit_kwargs)
        self.rabbit_4x  = RaBiT_Fusion(**rabbit_kwargs)

        # ------------ Pyramid Cascade Modules ------------
        self.cascade_16x = BasicConv2d(in_planes=64 + 64, out_planes=64, kernel_size=3, padding=1)
        self.cascade_8x  = BasicConv2d(in_planes=64 + 64, out_planes=64, kernel_size=3, padding=1)
        self.cascade_4x  = BasicConv2d(in_planes=64 + 64, out_planes=64, kernel_size=3, padding=1)

        # -------------------- Feature Fusion and Prediction --------------------
        self.fdm = FeatureFusionAndPrediction()  
        self.fdm_R = FeatureFusionAndPrediction() 
        self.fdm_T = FeatureFusionAndPrediction() 
    
    def forward(self, rgbt_inputs):

        rgb, t_img = rgbt_inputs[0], rgbt_inputs[1]

        # --------------- Backbone Feature Extraction ---------------
        rgb_4x  = self.pvt_backbone_rgb.forward_stage1(rgb)
        rgb_8x  = self.pvt_backbone_rgb.forward_stage2(rgb_4x)
        rgb_16x = self.pvt_backbone_rgb.forward_stage3(rgb_8x)
        rgb_32x = self.pvt_backbone_rgb.forward_stage4(rgb_16x)

        t_4x  = self.pvt_backbone_t.forward_stage1(t_img)
        t_8x  = self.pvt_backbone_t.forward_stage2(t_4x)
        t_16x = self.pvt_backbone_t.forward_stage3(t_8x)
        t_32x = self.pvt_backbone_t.forward_stage4(t_16x)

        # --------------- Channel Adjustment ---------------
        trans_rgb_4x  = self.trans_rgb_4x(rgb_4x)
        trans_rgb_8x  = self.trans_rgb_8x(rgb_8x)
        trans_rgb_16x = self.trans_rgb_16x(rgb_16x)
        trans_rgb_32x = self.trans_rgb_32x(rgb_32x)

        trans_t_4x  = self.trans_t_4x(t_4x)
        trans_t_8x  = self.trans_t_8x(t_8x)
        trans_t_16x = self.trans_t_16x(t_16x)
        trans_t_32x = self.trans_t_32x(t_32x)

        # --------------- RaBiT-CC ---------------
        window_size = 8
        
        # (32x)
        rgb_32x_padded, size_32x = pad_to_window_size(trans_rgb_32x, window_size)
        t_32x_padded, _          = pad_to_window_size(trans_t_32x, window_size)
        r_rgb_32x_padded = self.reli_head_32x(rgb_32x_padded)
        r_t_32x_padded   = self.reli_head_32x(t_32x_padded)
        
        # (16x)
        rgb_16x_padded, size_16x = pad_to_window_size(trans_rgb_16x, window_size)
        t_16x_padded, _          = pad_to_window_size(trans_t_16x, window_size)
        r_rgb_16x_padded = self.reli_head_16x(rgb_16x_padded)
        r_t_16x_padded   = self.reli_head_16x(t_16x_padded)
        
        # (8x)
        rgb_8x_padded, size_8x = pad_to_window_size(trans_rgb_8x, window_size)
        t_8x_padded, _         = pad_to_window_size(trans_t_8x, window_size)
        r_rgb_8x_padded = self.reli_head_8x(rgb_8x_padded)
        r_t_8x_padded   = self.reli_head_8x(t_8x_padded)
        
        # (4x)
        rgb_4x_padded, size_4x = pad_to_window_size(trans_rgb_4x, window_size)
        t_4x_padded, _         = pad_to_window_size(trans_t_4x, window_size)
        r_rgb_4x_padded = self.reli_head_4x(rgb_4x_padded)
        r_t_4x_padded   = self.reli_head_4x(t_4x_padded)

        fused_32x_padded = self.rabbit_32x(rgb_32x_padded, t_32x_padded, r_rgb_32x_padded, r_t_32x_padded)
        fused_16x_padded = self.rabbit_16x(rgb_16x_padded, t_16x_padded, r_rgb_16x_padded, r_t_16x_padded)
        fused_8x_padded  = self.rabbit_8x(rgb_8x_padded, t_8x_padded, r_rgb_8x_padded, r_t_8x_padded)
        fused_4x_padded  = self.rabbit_4x(rgb_4x_padded, t_4x_padded, r_rgb_4x_padded, r_t_4x_padded)
        
        H_32, W_32 = size_32x
        H_16, W_16 = size_16x
        H_8, W_8   = size_8x
        H_4, W_4   = size_4x
        
        fused_32x = fused_32x_padded[:, :, :H_32, :W_32]
        fused_16x = fused_16x_padded[:, :, :H_16, :W_16]
        fused_8x  = fused_8x_padded[:, :, :H_8, :W_8]
        fused_4x  = fused_4x_padded[:, :, :H_4, :W_4]

        f_cascaded_32x = fused_32x

        f_up_32x = self.upsample(f_cascaded_32x, size=fused_16x.shape[2:])
        f_concat_16x = torch.cat([f_up_32x, fused_16x], dim=1)
        f_cascaded_16x = self.cascade_16x(f_concat_16x)

        f_up_16x = self.upsample(f_cascaded_16x, size=fused_8x.shape[2:])
        f_concat_8x = torch.cat([f_up_16x, fused_8x], dim=1)
        f_cascaded_8x = self.cascade_8x(f_concat_8x)

        f_up_8x = self.upsample(f_cascaded_8x, size=fused_4x.shape[2:])
        f_concat_4x = torch.cat([f_up_8x, fused_4x], dim=1)
        f_cascaded_4x = self.cascade_4x(f_concat_4x)

        final_pred = self.fdm(f_cascaded_4x, f_cascaded_8x, f_cascaded_16x, f_cascaded_32x)

        if self.training:
            
            pred_R = self.fdm_R(trans_rgb_4x, trans_rgb_8x, trans_rgb_16x, trans_rgb_32x)

            pred_T = self.fdm_T(trans_t_4x, trans_t_8x, trans_t_16x, trans_t_32x)

            r_rgb_4x = r_rgb_4x_padded[:, :, :H_4, :W_4]
            r_rgb_8x = r_rgb_8x_padded[:, :, :H_8, :W_8]
            r_rgb_16x = r_rgb_16x_padded[:, :, :H_16, :W_16]
            r_rgb_32x = r_rgb_32x_padded[:, :, :H_32, :W_32]

            r_t_4x = r_t_4x_padded[:, :, :H_4, :W_4]
            r_t_8x = r_t_8x_padded[:, :, :H_8, :W_8]
            r_t_16x = r_t_16x_padded[:, :, :H_16, :W_16]
            r_t_32x = r_t_32x_padded[:, :, :H_32, :W_32]
            
            reli_maps_R = [r_rgb_4x, r_rgb_8x, r_rgb_16x, r_rgb_32x]
            reli_maps_T = [r_t_4x, r_t_8x, r_t_16x, r_t_32x]

            return final_pred, pred_R, pred_T, reli_maps_R, reli_maps_T
        else:
            H_4, W_4 = size_4x
            r_rgb_vis = r_rgb_4x_padded[:, :, :H_4, :W_4]
            r_t_vis   = r_t_4x_padded[:, :, :H_4, :W_4]
            final_pred = final_pred[:, :, :H_4, :W_4]
            return final_pred, r_rgb_vis, r_t_vis
        
def fusion_model(construct_method='threshold', k_interact=4, thresh_interact=0.8):
    model = RaBiT_CC(construct_method, k_interact, thresh_interact)
    return model