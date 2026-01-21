# Evo_1/scripts/distill_vis_utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import io
import PIL.Image
from matplotlib.gridspec import GridSpec

def fig2img(fig):
    """将 Matplotlib Figure 转换为 PIL Image，用于 SwanLab/WandB 记录"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

def visualize_distill_batch(model, batch_data, device="cuda", step=0):
    """
    输入一个 Batch 的数据，随机抽取一个样本，生成对比图。
    """
    model.eval()
    
    # 1. 解包数据
    # batch_data 顺序参考 StreamingOfflineDataset yield 的顺序:
    # z0, z1, ft, state, mask, eid
    b_z0, b_z1, b_ft, b_state, b_mask, b_eid = [t.to(device) for t in batch_data]
    
    # 2. 随机选一个有动作的样本 (避免选到 mask=0 的 padding)
    # 简单的策略：选第一个 mask 为 1 的
    valid_indices = torch.nonzero(b_mask[:, 0, 0]).squeeze()
    if valid_indices.numel() == 0:
        return None # 全是 Mask 掉的数据，不画了
    
    idx = valid_indices[0] if valid_indices.numel() > 1 else valid_indices.item()
    
    # 提取单个样本
    z0 = b_z0[idx:idx+1]
    z1_gt = b_z1[idx:idx+1] # Teacher
    ft = b_ft[idx:idx+1]
    state = b_state[idx:idx+1]
    eid = b_eid[idx:idx+1]
    
    # 3. 模型推理 (Student)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Student 预测 Velocity
            pred_vel, _ = model.action_head(
                fused_tokens=ft, state=state, embodiment_id=eid,
                z0=z0, z1=z1_gt, is_reflow=True
            )
            z1_pred = z0 + pred_vel # 还原 Action
    
    # 4. 转换数据为 Numpy (只取第一步 Horizon=0)
    # 假设动作维度是 [B, Horizon, Dim]
    teacher_act = z1_gt[0, 0, :].float().cpu().numpy() # [Dim]
    student_act = z1_pred[0, 0, :].float().cpu().numpy() # [Dim]
    
    # ================= 绘图逻辑 =================
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig)
    
    # --- 区域 A: 罗盘 (Compass) - XY 平面方向 ---
    ax_compass = fig.add_subplot(gs[0, 0])
    ax_compass.set_title("Compass (XY Plane Direction)", fontsize=12, fontweight='bold')
    ax_compass.set_xlim(-1.2, 1.2)
    ax_compass.set_ylim(-1.2, 1.2)
    ax_compass.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_compass.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 画原点
    ax_compass.scatter(0, 0, color='black', s=50)
    
    # Teacher 箭头 (绿)
    ax_compass.arrow(0, 0, teacher_act[0], teacher_act[1], 
                     head_width=0.05, head_length=0.1, fc='lime', ec='green', label='Teacher', linewidth=2, alpha=0.7)
    
    # Student 箭头 (红)
    ax_compass.arrow(0, 0, student_act[0], student_act[1], 
                     head_width=0.05, head_length=0.1, fc='red', ec='maroon', label='Student', linewidth=2, alpha=0.7)
    
    ax_compass.legend(loc='upper right')
    ax_compass.grid(True, linestyle=':', alpha=0.3)
    ax_compass.set_aspect('equal')

    # --- 区域 B: 均衡器 (Equalizer) - 7维动作对比 ---
    ax_bar = fig.add_subplot(gs[1, :]) # 占据底部整行
    ax_bar.set_title("Action Dimensions Equalizer (GT vs Pred)", fontsize=12, fontweight='bold')
    
    dims = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'Gripper']
    x = np.arange(len(dims))
    width = 0.35
    
    rects1 = ax_bar.bar(x - width/2, teacher_act[:7], width, label='Teacher (GT)', color='lime', alpha=0.7)
    rects2 = ax_bar.bar(x + width/2, student_act[:7], width, label='Student (Pred)', color='red', alpha=0.7)
    
    ax_bar.set_ylabel('Normalized Action Value')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(dims)
    ax_bar.set_ylim(-1.2, 1.2)
    ax_bar.legend()
    ax_bar.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- 区域 C: 夹爪状态与指标 (Gripper & Info) ---
    ax_info = fig.add_subplot(gs[0, 1:])
    ax_info.axis('off')
    
    # 计算指标
    mse = np.mean((teacher_act - student_act)**2)
    cosine = np.dot(teacher_act, student_act) / (np.linalg.norm(teacher_act) * np.linalg.norm(student_act) + 1e-8)
    
    # 夹爪判断 (假设 > 0 为 Open, < 0 为 Close，具体看你的数据定义，这里只对比符号一致性)
    g_teacher = teacher_act[6]
    g_student = student_act[6]
    gripper_match = (g_teacher * g_student) > 0 # 符号相同即匹配
    
    info_text = (
        f"Step: {step}\n\n"
        f"METRICS:\n"
        f"  - MSE Loss: {mse:.5f}\n"
        f"  - Cosine Sim: {cosine:.4f}\n\n"
        f"GRIPPER STATUS (Dim 6):\n"
        f"  - Teacher: {g_teacher:.2f}\n"
        f"  - Student: {g_student:.2f}\n"
        f"  - Status: "
    )
    
    ax_info.text(0.1, 0.4, info_text, fontsize=14, fontfamily='monospace', va='center')
    
    # 绘制夹爪状态的大字
    status_text = "MATCH ✅" if gripper_match else "MISMATCH ❌"
    status_color = "green" if gripper_match else "red"
    ax_info.text(0.5, 0.4, status_text, fontsize=20, fontweight='bold', color=status_color, va='center')

    plt.tight_layout()
    
    # 转为图片对象
    img = fig2img(fig)
    plt.close(fig)
    return img

if __name__ == "__main__":
    print("Test run...")
    
    class MockModel:
        def eval(self): pass
        def action_head(self, **kwargs): 
            # 获取 z0 的 device，确保返回的 Tensor 也在同一个设备上
            z0 = kwargs.get('z0')
            device = z0.device if z0 is not None else "cpu"
            return torch.zeros(1,1,7, device=device), None
    
    # 创建 dummy batch (在 CPU 上)
    dummy_batch = [torch.randn(1,1,7) for _ in range(6)]
    
    # 自动检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    try:
        # 运行可视化测试
        img = visualize_distill_batch(MockModel(), dummy_batch, device=device)
        print("✅ Visualization logic passed! Image object created.")
        
        # 可选：保存图片看看效果
        img.save("test_vis.png")
        print("Saved test_vis.png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Test failed: {e}")