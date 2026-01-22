import sys
import os
import json
import time
import torch
import numpy as np
from types import SimpleNamespace
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__), ".."))))
from scripts.Evo1 import EVO1

CKPT_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline/checkpoint_epoch_30" 
DEVICE = "cuda"
NUM_WARMUP = 5    
NUM_REPEAT = 20   

class Profiler:
    def __init__(self, name):
        self.name = name
        self.times = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()

    def end(self):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event) # 毫秒
        self.times.append(elapsed)

    def report(self):
        avg_time = np.mean(self.times)
        std_time = np.std(self.times)
        min_time = np.min(self.times)
        max_time = np.max(self.times)
        print(f"⏱️  [{self.name}]")
        print(f"    Avg: {avg_time:.2f} ms ± {std_time:.2f}")
        print(f"    Min: {min_time:.2f} ms | Max: {max_time:.2f} ms")
        return avg_time

def load_model(ckpt_dir):
    print(f"正在加载模型: {ckpt_dir} ...")
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)

    # 强制设置推理参数，与 server 保持一致
    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    # Flow Matching 的推理步数，通常影响 Action Head 的耗时
    config["num_inference_timesteps"] = config.get("num_inference_timesteps", 32) 

    model = EVO1(config).eval()
    
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["module"], strict=True)
    model = model.to(DEVICE)
    
    # 打印关键参数
    print(f"模型加载完成。")
    print(f" - Horizon: {model.horizon}")
    print(f" - Inference Timesteps: {config['num_inference_timesteps']}")
    
    return model

def generate_dummy_inputs(device):
    """生成符合模型输入的假数据"""
    # 模拟 3 张 448x448 的图像 (RGB)
    dummy_images = []
    for _ in range(3):
        img_array = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
        dummy_images.append(Image.fromarray(img_array))
    
    # 模拟 Prompt
    prompt = "Pick up the red cube."
    
    # 模拟 State (假设是归一化后的状态)
    # 通常 state_dim 是 24 (包含 padding)
    state = torch.randn(1, 24, device=device, dtype=torch.float32)
    
    # 模拟 Masks
    image_mask = torch.tensor([1, 1, 0], dtype=torch.int32, device=device) # 假设第三张图无效
    action_mask = torch.tensor([[1]*7 + [0]*17], dtype=torch.int32, device=device)
    
    return dummy_images, prompt, state, image_mask, action_mask

def run_profile(model):
    images, prompt, state, image_mask, action_mask = generate_dummy_inputs(DEVICE)
    
    # 定义计时器
    prof_vlm = Profiler("VLM Embedding (InternVL3)")
    prof_act = Profiler("Action Head (Flow Matching)")
    prof_total = Profiler("End-to-End Inference")

    print(f"\n开始预热 ({NUM_WARMUP} 次) ...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(NUM_WARMUP):
            # 运行一次完整的推理流程
            fused_tokens = model.get_vl_embeddings(images, image_mask, prompt)
            _ = model.predict_action(fused_tokens, state, action_mask=action_mask)

    print(f"开始测试 ({NUM_REPEAT} 次) ...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(NUM_REPEAT):
            torch.cuda.synchronize()
            
            # --- 总时间开始 ---
            prof_total.start()
            
            # 1. 测试 VLM 部分
            prof_vlm.start()
            fused_tokens = model.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=prompt,
                return_cls_only=False
            )
            prof_vlm.end()
            
            # 2. 测试 Action Head 部分
            # 注意：这里包含了 Flow Matching 的循环去噪步骤
            prof_act.start()
            action = model.predict_action(
                fused_tokens, 
                state, 
                action_mask=action_mask
            )
            prof_act.end()
            
            # --- 总时间结束 ---
            prof_total.end()
            
            print(f"\r进度: {i+1}/{NUM_REPEAT}", end="")
    
    print("\n\n" + "="*40)
    print(f"📊 性能测试报告 (Device: {DEVICE})")
    print("="*40)
    
    t_vlm = prof_vlm.report()
    t_act = prof_act.report()
    t_total = prof_total.report()
    
    print("-" * 40)
    print(f"VLM 占比: {t_vlm/t_total*100:.1f}%")
    print(f"Action Head 占比: {t_act/t_total*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    try:
        model = load_model(CKPT_DIR)
        run_profile(model)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("请检查 CKPT_DIR 路径是否正确，以及是否安装了必要的依赖库。")