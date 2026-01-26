import sys
import os
import torch
import copy
import json
import asyncio
import websockets
import signal
import numpy as np
from torch.optim import AdamW
'''
start the server for DAgger online training, with a frozen teacher model and a trainable student head
'''

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1_server import load_model_and_normalizer, handle_request, Normalizer
TEACHER_CKPT_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld"
STUDENT_CKPT_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline_forever/final_checkpoint/student_head_only.pt" 
SAVE_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/online_dagger_output"
PORT = 9099

LEARNING_RATE = 1e-5
BATCH_SIZE = 8
BUFFER_SIZE = 2000
TEACHER_STEPS = 32
STUDENT_STEPS = 4   
SAVE_INTERVAL = 50

class DAggerModelWrapper:
    def __init__(self, teacher_model, save_dir, student_ckpt_path=None):
        self.device = "cuda"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 1. teacher frozen
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        
        # 2. student head
        print("Creating Student Head structure...")
        self.student_head = copy.deepcopy(self.teacher.action_head)
        
        if student_ckpt_path and os.path.exists(student_ckpt_path):
            print(f"🔄 Loading Student Checkpoint from: {student_ckpt_path}")
            checkpoint = torch.load(student_ckpt_path, map_location="cpu")
            
            state_dict = checkpoint
            if "module" in checkpoint:
                state_dict = checkpoint["module"]
            
            if any(k.startswith("action_head.") for k in state_dict.keys()):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("action_head."):
                        new_state_dict[k.replace("action_head.", "")] = v
                state_dict = new_state_dict
            
            msg = self.student_head.load_state_dict(state_dict, strict=False)
            print(f"✅ Student weights loaded. {msg}")
        else:
            print("⚠️ No Student Checkpoint provided or file not found. Initializing from Teacher copy.")

        self.student_head.train()
        
        self.optimizer = AdamW(self.student_head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        self.buffer = []
        self.train_step_count = 0

    def parameters(self):
        return self.student_head.parameters()

    def run_inference(self, images, image_mask, prompt, state_input, return_cls_only=None, action_mask=None):
        # 1. VLM feature extraction
        with torch.no_grad():
            fused_tokens = self.teacher.get_vl_embeddings(
                images=images, image_mask=image_mask, prompt=prompt, return_cls_only=False
            ).to(dtype=torch.bfloat16)

        # 2. Student inference
        self.student_head.config.num_inference_timesteps = STUDENT_STEPS
        with torch.no_grad():
            student_action = self.student_head.get_action(
                fused_tokens, state=state_input, action_mask=action_mask, embodiment_id=None
            )

        # 3. Teacher inference
        self.teacher.action_head.config.num_inference_timesteps = TEACHER_STEPS
        with torch.no_grad():
            teacher_target = self.teacher.action_head.get_action(
                fused_tokens, state=state_input, action_mask=action_mask, embodiment_id=None
            )

        # 4. 存入 Buffer
        self.buffer.append({
            "fused_tokens": fused_tokens.clone(),
            "state": state_input.clone(),
            "action_mask": action_mask.clone(),
            "target_action": teacher_target.clone()
        })
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)

        # 5. train
        if len(self.buffer) >= BATCH_SIZE:
            self.train_step()

        return student_action

    def train_step(self):
        self.student_head.train()
        self.optimizer.zero_grad()
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        b_ft = torch.cat([s["fused_tokens"] for s in batch], dim=0)
        b_state = torch.cat([s["state"] for s in batch], dim=0)
        b_mask = torch.cat([s["action_mask"] for s in batch], dim=0)
        b_target = torch.cat([s["target_action"] for s in batch], dim=0)
        b_z0 = torch.randn_like(b_target)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_v, target_v = self.student_head(
                fused_tokens=b_ft, state=b_state, action_mask=b_mask,
                z0=b_z0, z1=b_target, is_reflow=True
            )
            target_v = target_v.flatten(1)
            mask_flat = b_mask.flatten(1)
            pred_v = pred_v * mask_flat
            target_v = target_v * mask_flat
            loss = ((pred_v - target_v) ** 2).sum() / (mask_flat.sum() + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_head.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.train_step_count += 1
        if self.train_step_count % SAVE_INTERVAL == 0:
            self.save_model(f"step_{self.train_step_count}")

    def save_model(self, tag):
        path = os.path.join(self.save_dir, f"student_head_{tag}.pt")
        torch.save(self.student_head.state_dict(), path)
        print(f"\n💾 Saved: {path}")

if __name__ == "__main__":
    print("🚀 Initializing DAgger Server (Wrapper Mode)...")
    
    print(f"Loading Teacher from {TEACHER_CKPT_DIR}...")
    teacher_model, normalizer = load_model_and_normalizer(TEACHER_CKPT_DIR)
    
    dagger_wrapper = DAggerModelWrapper(
        teacher_model=teacher_model, 
        save_dir=SAVE_DIR, 
        student_ckpt_path=STUDENT_CKPT_PATH 
    )
    
    def signal_handler(sig, frame):
        print("\n🛑 Saving before exit...")
        dagger_wrapper.save_model("final")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    async def main():
        print(f"🔥 Server running at ws://0.0.0.0:{PORT}")
        print(f"   Student Path: {STUDENT_CKPT_PATH}")
        async with websockets.serve(
            lambda ws: handle_request(ws, dagger_wrapper, normalizer),
            "0.0.0.0", PORT, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())