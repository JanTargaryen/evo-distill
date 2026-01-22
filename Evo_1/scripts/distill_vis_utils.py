import matplotlib.pyplot as plt
import numpy as np
import torch
import io
import PIL.Image
import os
from matplotlib.gridspec import GridSpec

def fig2img(fig):
    """将 Matplotlib Figure 转换为 PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

def visualize_trajectory_batch(model, batch_data, device="cuda", step=0):
    """
    升级版可视化：使用 get_action 进行真实的 1-Step 推理
    """
    # 兼容性处理：获取真正的 Action Head
    if hasattr(model, "action_head"):
        predictor = model.action_head
    else:
        predictor = model
    
    predictor.eval()
    
    # 1. 数据解包
    b_z0, b_z1, b_ft, b_state, b_mask, b_eid = [t.to(device) for t in batch_data]
    
    # 2. 筛选有效样本
    if b_mask.sum() == 0:
        return None
        
    valid_indices = torch.nonzero(b_mask[:, 0, 0]).squeeze()
    if valid_indices.numel() == 0: 
        idx = 0 
    else:
        idx = valid_indices[0] if valid_indices.numel() > 1 else valid_indices.item()
    
    # 3. 提取单个样本 (保持 Batch 维度为 1)
    z0 = b_z0[idx:idx+1]          # [1, 50, 24]
    z1_gt = b_z1[idx:idx+1]       # [1, 50, 24]
    ft = b_ft[idx:idx+1]
    state = b_state[idx:idx+1]
    eid = b_eid[idx:idx+1]
    mask = b_mask[idx:idx+1]      # [1, 50, 24]

    # 4. 推理 (使用 get_action 模拟真实生成)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 注意：get_action 需要的 mask 是 [B, Dim]，取第一个时间步即可
            # z0 需要展平传入 [B, Horizon*Dim]
            z1_pred_flat = predictor.get_action(
                fused_tokens=ft, 
                state=state, 
                embodiment_id=eid, 
                action_mask=mask[:, 0, :],  # [1, 24]
                init_noise=z0.flatten(1)    # [1, 1200]
            )
            
            # 将结果还原回 [1, 50, 24]
            z1_pred = z1_pred_flat.view_as(z0)

    # 5. 转 Numpy
    T_seq = z1_gt[0].float().cpu().numpy()   # Teacher
    S_seq = z1_pred[0].float().cpu().numpy() # Student (Pred)
    
    horizon = T_seq.shape[0]
    time_steps = np.arange(horizon)

    # ================= 绘图逻辑 (保持不变) =================
    fig = plt.figure(figsize=(20, 9)) 
    gs = GridSpec(2, 5, figure=fig)
    
    # Area 1: 3D 空间轨迹
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_3d.set_title(f"3D Spatial Trajectory (Step {step})", fontsize=12, fontweight='bold')
    ax_3d.plot(T_seq[:, 0], T_seq[:, 1], T_seq[:, 2], 'g.-', label='Teacher', linewidth=2, alpha=0.6)
    ax_3d.plot(S_seq[:, 0], S_seq[:, 1], S_seq[:, 2], 'r.-', label='Student', linewidth=2, alpha=0.6)
    ax_3d.scatter(T_seq[0,0], T_seq[0,1], T_seq[0,2], c='g', marker='o', s=50, label='Start')
    ax_3d.scatter(T_seq[-1,0], T_seq[-1,1], T_seq[-1,2], c='g', marker='x', s=50, label='End')
    ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z'); ax_3d.legend()

    # Area 2: 时序曲线
    dims_cfg = [(0, "X"), (1, "Y"), (2, "Z"), (3, "Rx"), (4, "Ry"), (5, "Rz"), (6, "Gripper")]
    for i, (dim_idx, name) in enumerate(dims_cfg):
        if i < 4: ax = fig.add_subplot(gs[0, i+1])
        else: ax = fig.add_subplot(gs[1, i-3])
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.plot(time_steps, T_seq[:, dim_idx], 'g.-', label='GT', alpha=0.7)
        ax.plot(time_steps, S_seq[:, dim_idx], 'r.-', label='Pred', alpha=0.7)
        ax.grid(True, linestyle=':', alpha=0.3)
        if i == 0: ax.legend(fontsize=8)

    plt.suptitle(f"Distillation Analysis (Epoch={step})", fontsize=16)
    plt.tight_layout()
    img = fig2img(fig)
    plt.close(fig)
    return img

# ================= 自测代码 =================
if __name__ == "__main__":
    print("🚀 Running visualization test...")
    
    class MockModel:
        def eval(self): pass
        def train(self): pass
        def action_head(self, **kwargs):
            z0 = kwargs.get('z0')
            device = z0.device if z0 is not None else "cpu"
            z1_gt = kwargs.get('z1')
            v_real = z1_gt - z0
            # 模拟：Student 预测稍微有点误差
            v_pred = v_real * 0.8 + torch.randn_like(v_real) * 0.05
            return v_pred.to(device), None

    # 生成模拟数据
    H, B, D = 16, 1, 7
    t = torch.linspace(0, 6.28, H)
    z1_gt = torch.zeros(B, H, D)
    z1_gt[0, :, 0] = torch.sin(t) 
    z1_gt[0, :, 1] = torch.cos(t)
    z1_gt[0, :, 2] = t / 6.28
    z1_gt[0, :, 6] = torch.cat([torch.ones(8), -torch.ones(8)])
    z0 = z1_gt + torch.randn_like(z1_gt) * 0.2
    
    # 辅助数据
    dummy_ft = torch.randn(B, 10, 768)
    dummy_state = torch.randn(B, 7)
    dummy_mask = torch.ones(B, H, D)
    dummy_eid = torch.zeros(B, dtype=torch.long)
    
    batch_data = [z0, z1_gt, dummy_ft, dummy_state, dummy_mask, dummy_eid]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    try:
        img = visualize_trajectory_batch(MockModel(), batch_data, device=device, step=100)
        save_path = "test_trajectory_vis.png"
        img.save(save_path)
        print(f"✅ Success! Image saved to: {os.path.abspath(save_path)}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Failed: {e}")