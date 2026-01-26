import sys
import os
import torch
import json
import glob
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader,IterableDataset
from torch.optim import AdamW
import swanlab
import subprocess
import uuid
import shutil
import math
import random
import torch
import atexit
import signal
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Evo1 import EVO1
from scripts.train_utils.distill_vis_utils import visualize_trajectory_batch

# --- Configuration ---
CONFIG_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/config.json"
NORM_STATS_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "norm_stats.json")
CKPT_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/mp_rank_00_model_states.pt"
DATA_SAVE_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/dataset/offline_distillation_data"
SAVE_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline_forever"
# STUDENT_RESUME_PATH = None  # Start training from scratch
STUDENT_RESUME_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline/checkpoint_epoch_70/mp_rank_00_model_states.pt"  

# Training Settings
TRAIN_EPOCHS = 5000
BATCH_SIZE_TRAIN = 256
LR = 5e-5
SAVE_INTERVAL = 20
VIS_INTERVAL = 500

class StreamingOfflineDataset(IterableDataset):
    def __init__(self, data_dir, ratio=1.0):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "distill_data_part_*.pt.gz")))
        if not self.files:
            raise FileNotFoundError(f"No .pt.gz files found in {data_dir}")
        
        if ratio < 1.0:
            random.seed(42)
            random.shuffle(self.files)
            target_len = int(len(self.files) * ratio)
            self.files = self.files[:target_len]
            print(f"✂️  [Fast Mode] Using {len(self.files)} files ({ratio*100}%)")

        base_shm = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
        
        main_pid = os.getpid()
        run_id = uuid.uuid4().hex[:8]
        self.job_temp_dir = os.path.join(base_shm, f"evo_train_tmp_{main_pid}_{run_id}")
        
        os.makedirs(self.job_temp_dir, exist_ok=True)
        print(f"🛡️  Sandbox Created: {self.job_temp_dir}")
        print(f"   (All temp files will be confined here and auto-cleaned on exit)")

        def global_cleanup():
            if os.path.exists(self.job_temp_dir):
                try:
                    print(f"\n🧹 Cleaning up resources in {self.job_temp_dir}...")
                    shutil.rmtree(self.job_temp_dir, ignore_errors=True)
                    print(f"✅ Cleanup Done. Bye!")
                except Exception as e:
                    print(f"⚠️ Cleanup failed: {e}")

        atexit.register(global_cleanup)

        def signal_handler(sig, frame):
            print("\n🛑 Interrupt received (Ctrl+C). Exiting safely...")
            global_cleanup() 
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = self.files[:]
        
        if worker_info is not None:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.files))
            files_to_read = self.files[start_idx:end_idx]
        
        random.shuffle(files_to_read)

        for f_path in files_to_read:
            temp_name = f"worker_{uuid.uuid4().hex}.pt"
            temp_path = os.path.join(self.job_temp_dir, temp_name)
            
            try:
                cmd = f"gunzip -c {f_path} > {temp_path}"
                subprocess.check_call(cmd, shell=True)
                
                data = torch.load(temp_path, map_location="cpu")
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                num_samples = len(data["state"])
                indices = torch.randperm(num_samples).tolist()
                
                for idx in indices:
                    yield (
                        data["z0"][idx],
                        data["z1"][idx],
                        data["fused_tokens"][idx],
                        data["state"][idx],
                        data["action_mask"][idx],
                        data["embodiment_id"][idx]
                    )
                
                del data
                
            except Exception as e:
                pass 
                
            finally:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

    def __len__(self):
        return len(self.files) * 2048

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("🦢 Initializing SwanLab...")
    swanlab.init(
        project="Evo-1-Reflow-Offline",
        experiment_name="Offline-Training",
        config={"batch_size": BATCH_SIZE_TRAIN, "lr": LR, "epochs": TRAIN_EPOCHS}
    )

    # 1. Initialize Student
    print("🔧 Initializing Student Model...")
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
    
    temp_model = EVO1(config_dict).cuda().to(torch.bfloat16)
    student_head = copy.deepcopy(temp_model.action_head)
    
    if STUDENT_RESUME_PATH and os.path.exists(STUDENT_RESUME_PATH):
        print(f"🔄 Resuming from Student Checkpoint: {STUDENT_RESUME_PATH}")
        student_ckpt = torch.load(STUDENT_RESUME_PATH, map_location="cpu")
        state_dict = {
            k.replace("action_head.", ""): v 
            for k, v in student_ckpt["module"].items() 
            if k.startswith("action_head.")
        }
        student_head.load_state_dict(state_dict, strict=True)
        print("✅ Student weights loaded successfully.")
        ckpt_teacher = torch.load(CKPT_PATH, map_location="cpu")
        if "module" in ckpt_teacher: ckpt_teacher = ckpt_teacher["module"]
        base_state_dict = copy.deepcopy(ckpt_teacher)
        del ckpt_teacher

    else:
        print("⚠️ No Student Checkpoint found, initializing from Teacher...")
        ckpt_teacher = torch.load(CKPT_PATH, map_location="cpu")
        if "module" in ckpt_teacher: ckpt_teacher = ckpt_teacher["module"]
        
        temp_model.load_state_dict(ckpt_teacher, strict=True)
        student_head = copy.deepcopy(temp_model.action_head)
        base_state_dict = copy.deepcopy(ckpt_teacher)
        del ckpt_teacher

    del temp_model
    torch.cuda.empty_cache()
    
    student_head.train()
    for p in student_head.parameters():
        p.requires_grad = True
        
    optimizer = AdamW(student_head.parameters(), lr=LR, weight_decay=1e-4)
    print("✅ Student Initialized.")

    print("🌊 Using High-Speed Streaming Dataset...")
    
    full_dataset = StreamingOfflineDataset(DATA_SAVE_DIR, ratio=1.0) 
    
    train_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=6, 
        pin_memory=True,
        drop_last=True
    )
    
    steps_per_epoch = len(full_dataset) // BATCH_SIZE_TRAIN
    print(f"   Approx. Steps per Epoch: {steps_per_epoch}")

    # 3. Training Loop
    print(f"🚀 Start Offline Training for {TRAIN_EPOCHS} Epochs...")
    
    
    total_steps = 0
    loss_history = []
    
    def compute_masked_loss(pred, target, mask):
        error = (pred - target) ** 2
        error = error * mask
        loss = error.sum() / (mask.sum() + 1e-8)
        return loss

    for epoch in range(TRAIN_EPOCHS):
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}", total=steps_per_epoch)
        
        for batch in batch_pbar:
            b_z0, b_z1, b_ft, b_state, b_mask, b_eid = [t.cuda().to(torch.bfloat16) for t in batch[:-1]] + [batch[-1].cuda()]
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_v, target_v = student_head(
                    fused_tokens=b_ft,
                    state=b_state,
                    embodiment_id=b_eid,
                    action_mask=b_mask,
                    z0=b_z0,
                    z1=b_z1,
                    is_reflow=True
                )
                target_v = target_v.flatten(1)
                mask_flat = b_mask.flatten(1)
                pred_v = pred_v * mask_flat
                target_v = target_v * mask_flat
                loss = compute_masked_loss(pred_v, target_v, mask_flat)
            
            if not torch.isfinite(loss):
                print(f"⚠️ Warning: Non-finite loss {loss.item()}. Skipping.")
                continue
                
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student_head.parameters(), max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred_v, target_v, dim=1, eps=1e-8)
                
                valid_sample_mask = (mask_flat.sum(dim=1) > 0).float()
                
                avg_cos_sim = (cos_sim * valid_sample_mask).sum() / (valid_sample_mask.sum() + 1e-8)
                avg_cos_sim_val = avg_cos_sim.item()

            loss_val = loss.item()
            total_steps += 1
            if total_steps % VIS_INTERVAL == 0:
                gpu_batch = [b_z0, b_z1, b_ft, b_state, b_mask, b_eid]
                
                vis_img = visualize_trajectory_batch(
                    model=student_head, 
                    batch_data=gpu_batch, 
                    device="cuda", 
                    step=total_steps
                )
                
                if vis_img is not None:
                    swanlab.log({"Eval/Trajectory_Curve": swanlab.Image(vis_img)}, step=total_steps)


                student_head.train()

            batch_pbar.set_postfix({
                "Loss": f"{loss_val:.5f}", 
                "Grad": f"{grad_norm:.2f}",
                "Cos": f"{avg_cos_sim_val:.4f}"
            })
            swanlab.log({
                "train/loss": loss_val, 
                "train/grad_norm": grad_norm,
                "train/cosine_similarity": avg_cos_sim_val
            }, step=total_steps)

        if (epoch + 1) % SAVE_INTERVAL == 0:
            print(f"\n💾 Saving checkpoint at Epoch {epoch+1}...")
            ckpt_name = f"checkpoint_epoch_{epoch+1}"
            ckpt_subdir = os.path.join(SAVE_DIR, ckpt_name)
            os.makedirs(ckpt_subdir, exist_ok=True)
            
            torch.save(student_head.state_dict(), os.path.join(ckpt_subdir, "student_head_only.pt"))
            print("   Merging weights into full checkpoint...")
            full_state_dict = copy.deepcopy(base_state_dict) 
            student_state = student_head.state_dict()
            for k, v in student_state.items():
                full_key = f"action_head.{k}"
                if full_key in full_state_dict:
                    full_state_dict[full_key] = v.cpu()
                else:
                    print(f"⚠️ Warning: Key mismatch {full_key}")

            torch.save({"module": full_state_dict}, os.path.join(ckpt_subdir, "mp_rank_00_model_states.pt"))
            save_config = config_dict.copy()
            save_config["num_inference_timesteps"] = 1
            with open(os.path.join(ckpt_subdir, "config.json"), "w") as f:
                json.dump(save_config, f, indent=2)

            shutil.copy(NORM_STATS_PATH, os.path.join(ckpt_subdir, "norm_stats.json"))

            checkpoint_meta = {
                "type": "ds_model",
                "version": 0.0,
                "checkpoints": "mp_rank_00_model_states.pt"
            }
            with open(os.path.join(ckpt_subdir, "checkpoint.json"), "w") as f:
                json.dump(checkpoint_meta, f, indent=2)
                
            print(f"✅ Saved to {ckpt_subdir}")

    # 4. Save Final Checkpoint
    print("💾 Saving final checkpoint...")
    ckpt_subdir = os.path.join(SAVE_DIR, "final_checkpoint")
    os.makedirs(ckpt_subdir, exist_ok=True)
    
    # Re-load config to save it
    save_config = config_dict.copy()
    save_config["num_inference_timesteps"] = 1
    
    torch.save(student_head.state_dict(), os.path.join(ckpt_subdir, "student_head_only.pt"))
    with open(os.path.join(ckpt_subdir, "config.json"), "w") as f:
        json.dump(save_config, f, indent=2)

    if os.path.exists(NORM_STATS_PATH):
        shutil.copy(NORM_STATS_PATH, os.path.join(ckpt_subdir, "norm_stats.json"))
    
    checkpoint_meta = {
        "type": "ds_model",
        "version": 0.0,
        "checkpoints": "mp_rank_00_model_states.pt"
    }
    with open(os.path.join(ckpt_subdir, "checkpoint.json"), "w") as f:
        json.dump(checkpoint_meta, f, indent=2)
    print(f"✅ Training Complete. Saved to {ckpt_subdir}")

if __name__ == "__main__":
    main()