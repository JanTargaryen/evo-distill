import sys
import os
import time
import glob
import json
import torch
import swanlab
import gzip
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Evo1 import EVO1
try:
    from scripts.distill_vis_utils import visualize_trajectory_batch
except ImportError:
    from distill_vis_utils import visualize_trajectory_batch

CKPT_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline"
DATA_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/dataset/offline_distillation_data"
CONFIG_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/config.json"

swanlab.init(
    project="Evo-1-Reflow-Offline",
    experiment_name="Satellite-Monitor",
    description="Independent process monitoring checkpoints and visualizing trajectories"
)

def load_student_model(ckpt_path):
    """从 Checkpoint 加载 Student 模型"""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    config["num_inference_timesteps"] = 1 
    
    model = EVO1(config).cuda().to(torch.bfloat16)
    
    weight_path = os.path.join(ckpt_path, "student_head_only.pt")
    if not os.path.exists(weight_path):
        print(f"⚠️ 权重文件未就绪: {weight_path}")
        return None
        
    try:
        state_dict = torch.load(weight_path, map_location="cuda")
        model.action_head.load_state_dict(state_dict, strict=True)
        model.eval()
        return model.action_head
    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        return None

def get_random_batch():
    """随机读取一个数据文件并提取 Batch"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt.gz")))
    if not files: return None
    
    import random
    f_path = random.choice(files)
    
    try:
        with gzip.open(f_path, 'rb') as f:
            data = torch.load(f, map_location="cpu")
            
        total = len(data["state"])
        for _ in range(10): # 尝试 10 次找有效数据
            idx = random.randint(0, total-1)
            mask = data["action_mask"][idx]
            if mask.sum() > 0:
                break
        
        batch = [
            data["z0"][idx:idx+1].cuda().to(torch.bfloat16),
            data["z1"][idx:idx+1].cuda().to(torch.bfloat16),
            data["fused_tokens"][idx:idx+1].cuda().to(torch.bfloat16),
            data["state"][idx:idx+1].cuda().to(torch.bfloat16),
            data["action_mask"][idx:idx+1].cuda().to(torch.bfloat16),
            data["embodiment_id"][idx:idx+1].cuda()
        ]
        return batch
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

def main_loop():
    print("🛰️ Satellite Monitor 启动！正在监视 Checkpoint...")
    processed_ckpts = set()
    
    while True:
        subdirs = glob.glob(os.path.join(CKPT_DIR, "checkpoint_epoch_*"))
        subdirs.sort(key=os.path.getmtime)
        
        if not subdirs:
            print("⏳ 暂无 Checkpoint... (每 600 秒检查一次)")
            time.sleep(600)
            continue
            
        latest_ckpt = subdirs[-1]
        ckpt_name = os.path.basename(latest_ckpt)
        
        if ckpt_name in processed_ckpts:
            time.sleep(600)
            continue
            
        print(f"\n🔎 发现新 Checkpoint: {ckpt_name}")
        print(f"   时间: {datetime.now().strftime('%H:%M:%S')}")
        
        student_head = load_student_model(latest_ckpt)
        batch_data = get_random_batch()
        
        if student_head and batch_data:
            print("🎨 正在生成轨迹可视化...")
            try:
                epoch_num = int(ckpt_name.split("_")[-1])
                
                vis_img = visualize_trajectory_batch(
                    model=student_head,
                    batch_data=batch_data,
                    device="cuda",
                    step=epoch_num
                )
                
                swanlab.log({
                    "Satellite/Trajectory_Curve": swanlab.Image(vis_img),
                    "Monitor_Epoch": epoch_num
                })
                print(f"✅ 可视化已上传！Epoch: {epoch_num}")
                
                processed_ckpts.add(ckpt_name)
                
            except Exception as e:
                print(f"⚠️ 绘图失败: {e}")
                import traceback
                traceback.print_exc()
        
        del student_head
        del batch_data
        torch.cuda.empty_cache()
        
        print("💤 等待下一个 Checkpoint...")
        time.sleep(600)

if __name__ == "__main__":
    main_loop()