import sys
import os
import torch
import json
import gc
import copy
import time 
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import swanlab 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.train import (
    prepare_dataset, 
    custom_collate_fn, 
    build_param_groups,
    get_lr_lambda,        
    check_numerical_stability 
)
from Evo1 import EVO1

CONFIG_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/config.json"
CKPT_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/mp_rank_00_model_states.pt"
SAVE_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_online"

# teacher
TEACHER_STEPS = 50      
BUFFER_SIZE = 2048       
BATCH_SIZE_GEN = 32
SAVE_EVERY_CYCLES = 50 

# students
TRAIN_EPOCHS = 5             
BATCH_SIZE_TRAIN = 128  
LR = 1e-5                    

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("🦢 Initializing SwanLab...")
    swanlab.init(
        project="Evo-1-Reflow",
        experiment_name=f"Online-Single-Batch-{BATCH_SIZE_GEN}",
        description="Clean Single Model Online Distillation",
        config={
            "teacher_steps": TEACHER_STEPS,
            "buffer_size": BUFFER_SIZE,
            "batch_size_gen": BATCH_SIZE_GEN,
            "batch_size_train": BATCH_SIZE_TRAIN,
            "lr": LR,
            "model": "InternVL3-1B"
        }
    )

    # load teacher
    print("🔧 [1/5] Loading Teacher Model...")
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
    
    config_dict["num_inference_timesteps"] = TEACHER_STEPS 
    
    teacher_model = EVO1(config_dict).cuda().to(torch.bfloat16)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if "module" in ckpt: ckpt = ckpt["module"]
    teacher_model.load_state_dict(ckpt, strict=True)
    
    # freeze teacher
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
        
    print(f"   Teacher Loaded. Horizon: {teacher_model.action_head.horizon}, Steps: {TEACHER_STEPS}")

    # init students
    print("🔧 [2/5] Initializing Student Head...")
    
    # student use the same vlm as teacher
    student_head = copy.deepcopy(teacher_model.action_head)
    student_head.train()
    
    # open Student gradient
    for p in student_head.parameters():
        p.requires_grad = True
        
    optimizer = AdamW(student_head.parameters(), lr=LR, weight_decay=1e-4)

    # data prepare
    print("📂 [3/5] Loading Dataset...")
    dataset = prepare_dataset(config_dict)
    
    gen_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_GEN,
        shuffle=True, 
        num_workers=8,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    gen_iterator = iter(gen_loader)

    # main loop , online distillation
    print(f"🚀 [4/5] Start Online Distillation")
    print(f"   Strategy: Fill {BUFFER_SIZE} -> Train {TRAIN_EPOCHS} Epoch -> Flush")
    
    total_train_steps = 0
    cycle_idx = 0
    
    while True:
        cycle_idx += 1
        print(f"\n🔄 === Cycle {cycle_idx} Start ===")
        
        # fill buffer
        buffer_data = {
            "z0": [], "z1": [], "fused_tokens": [], 
            "state": [], "action_mask": [], "embodiment_id": []
        }
        current_count = 0
        
        pbar_gen = tqdm(total=BUFFER_SIZE, desc="Phase 1: Generating (Teacher)", unit="samples")
        
        start_time_gen = time.time()
        
        while current_count < BUFFER_SIZE:
            try:
                batch = next(gen_iterator)
            except StopIteration:
                gen_iterator = iter(gen_loader)
                batch = next(gen_iterator)
            
            with torch.no_grad():
                images = batch["images"]
                prompts = batch["prompts"]
                states = batch["states"].cuda().to(torch.bfloat16)
                action_mask = batch["action_mask"].cuda().to(torch.bfloat16)
                embodiment_ids = torch.zeros(states.shape[0], dtype=torch.long, device="cuda")
                
                fused_tokens_list = []
                img_masks = batch.get("image_masks", [None]*len(images))
                for img, prompt, mask in zip(images, prompts, img_masks):
                    ft = teacher_model.get_vl_embeddings(img, mask, prompt)
                    fused_tokens_list.append(ft)
                fused_tokens = torch.cat(fused_tokens_list, dim=0).to(torch.bfloat16)
                
                B = states.shape[0]
                HORIZON = teacher_model.action_head.horizon
                PER_ACT = teacher_model.action_head.per_action_dim
                z0 = (torch.rand(B, HORIZON, PER_ACT, device="cuda") * 2 - 1).to(torch.bfloat16)
                
                action_mask_input = action_mask[:, 0, :]
                
                z1_flat = teacher_model.action_head.get_action(
                    fused_tokens, 
                    state=states, 
                    embodiment_id=embodiment_ids, 
                    action_mask=action_mask_input, 
                    init_noise=z0.flatten(1)
                )
                z1 = z1_flat.view(B, HORIZON, PER_ACT)
                if torch.isnan(z1).any() or torch.isinf(z1).any():
                    print(f"⚠️ Warning: Teacher generated NaN/Inf at batch count {current_count}. Skipping this batch.")
                    continue
                
                buffer_data["z0"].append(z0.cpu())
                buffer_data["z1"].append(z1.cpu())
                buffer_data["fused_tokens"].append(fused_tokens.cpu())
                buffer_data["state"].append(states.cpu())
                buffer_data["action_mask"].append(action_mask.cpu())
                buffer_data["embodiment_id"].append(embodiment_ids.cpu())
                
                current_count += B
                pbar_gen.update(B)
        
        pbar_gen.close()
        
        time_gen_elapsed = time.time() - start_time_gen
        gen_speed = current_count / time_gen_elapsed
        swanlab.log({"perf/gen_speed": gen_speed}, step=total_train_steps)
        print(f"   ⚡ Gen Speed: {gen_speed:.2f} samples/s")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # train student
        temp_dataset = TensorDataset(
            torch.cat(buffer_data["z0"]),
            torch.cat(buffer_data["z1"]),
            torch.cat(buffer_data["fused_tokens"]),
            torch.cat(buffer_data["state"]),
            torch.cat(buffer_data["action_mask"]),
            torch.cat(buffer_data["embodiment_id"])
        )
        
        train_loader = DataLoader(
            temp_dataset, 
            batch_size=BATCH_SIZE_TRAIN, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Phase 2: Training Student on {len(temp_dataset)} samples...")
        
        student_head.train()
        loss_history = []

        def compute_masked_loss(pred, target, mask):
            """
            pred, target: [B, Horizon, Dim]
            mask: [B, Horizon, Dim] (0 or 1)
            """
            error = (pred - target) ** 2
            error = error * mask
            loss = error.sum() / (mask.sum() + 1e-8)
            return loss
        
        for epoch in range(TRAIN_EPOCHS):
            batch_pbar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{TRAIN_EPOCHS}", leave=False)
            
            for t_batch in batch_pbar:
                b_z0, b_z1, b_ft, b_state, b_mask, b_eid = [t.cuda().to(torch.bfloat16) for t in t_batch[:-1]] + [t_batch[-1].cuda()]
                
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
                    print(f"⚠️ Warning: Loss is {loss.item()} at step {total_train_steps}. Skipping step.")
                    optimizer.zero_grad() # 清空梯度
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(student_head.parameters(), max_norm=1.0)
                optimizer.step()
                
                loss_val = loss.item()
                loss_history.append(loss_val)
                batch_pbar.set_postfix({
                "Loss": f"{loss_val:.5f}", 
                "Grad": f"{grad_norm:.2f}"
            })
                
                total_train_steps += 1
                swanlab.log({
                "train/loss": loss_val, 
                "train/grad_norm": grad_norm
                }, step=total_train_steps)
            
        avg_loss = np.mean(loss_history)
        print(f"   Avg Loss: {avg_loss:.5f}")
        
        swanlab.log({"train/avg_loss_cycle": avg_loss}, step=total_train_steps)

        # after train saving
        if cycle_idx % SAVE_EVERY_CYCLES == 0:
            ckpt_subdir = os.path.join(SAVE_DIR, f"checkpoint_step_{total_train_steps}")
            os.makedirs(ckpt_subdir, exist_ok=True)
            
            print(f"💾 Saving full checkpoint to {ckpt_subdir}...")
            
            final_dict = teacher_model.state_dict() 
            student_dict = student_head.state_dict()
            
            for k, v in student_dict.items():
                final_dict[f"action_head.{k}"] = v
            
            torch.save({"module": final_dict}, os.path.join(ckpt_subdir, "mp_rank_00_model_states.pt"))
            del final_dict 

            save_config = config_dict.copy()
            save_config["num_inference_timesteps"] = 1 
            with open(os.path.join(ckpt_subdir, "config.json"), "w") as f:
                json.dump(save_config, f, indent=2)

            if hasattr(dataset, "arm2stats_dict"):
                with open(os.path.join(ckpt_subdir, "norm_stats.json"), "w") as f:
                    json.dump(dataset.arm2stats_dict, f, indent=2)
            
            checkpoint_meta = {
                "type": "ds_model",
                "version": 0.0,
                "checkpoints": "mp_rank_00_model_states.pt"
            }
            with open(os.path.join(ckpt_subdir, "checkpoint.json"), "w") as f:
                json.dump(checkpoint_meta, f, indent=2)

            print(f"✅ Saved successfully.")

        del temp_dataset, train_loader, buffer_data, loss_history
        gc.collect() 
        print("🧹 Buffer flushed.")

if __name__ == "__main__":
    main()