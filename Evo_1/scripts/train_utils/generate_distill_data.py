import sys
import os
import torch
import json
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import swanlab
import gzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__), ".."))))
from scripts.train import prepare_dataset, custom_collate_fn
from Evo1 import EVO1

# --- Configuration ---
CONFIG_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/config.json"
CKPT_PATH = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/mp_rank_00_model_states.pt"
DATA_SAVE_DIR = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/dataset/offline_distillation_data"

# Generation Settings
TEACHER_STEPS = 50
BATCH_SIZE_GEN = 32
SAVE_CHUNK_SIZE = 2048  # Save to disk every 2048 samples to prevent OOM

def save_chunk(data_buffer, chunk_id, save_dir):
    """Helper to save accumulated data to disk with GZIP compression"""
    if len(data_buffer["state"]) == 0:
        return

    save_path = os.path.join(save_dir, f"distill_data_part_{chunk_id:04d}.pt.gz")
    
    # Concatenate lists
    save_dict = {k: torch.cat(v) for k, v in data_buffer.items()}
    
    print(f"💾 Saving chunk {chunk_id} ({save_dict['state'].shape[0]} samples) to {save_path} (Compressed)...")
    
    with gzip.open(save_path, 'wb', compresslevel=3) as f:
        torch.save(save_dict, f)
    
    # Clear buffer memory
    del save_dict
    for k in data_buffer:
        data_buffer[k] = []
    gc.collect()

def main():
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    
    # 1. Load Teacher Model
    print("🔧 Loading Teacher Model...")
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
    
    config_dict["num_inference_timesteps"] = TEACHER_STEPS
    
    teacher_model = EVO1(config_dict).cuda().to(torch.bfloat16)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if "module" in ckpt: ckpt = ckpt["module"]
    teacher_model.load_state_dict(ckpt, strict=True)
    
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
        
    print(f"   Teacher Loaded. Horizon: {teacher_model.action_head.horizon}")

    # 2. Prepare Dataset (Full Dataset)
    print("📂 Loading Full Dataset...")
    dataset = prepare_dataset(config_dict)
    
    gen_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_GEN,
        shuffle=False, 
        num_workers=8,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    total_samples = len(dataset)
    print(f"   Total samples to process: {total_samples}")

    # 3. Generation Loop (Iterate through whole dataset)
    buffer_data = {
        "z0": [], "z1": [], "fused_tokens": [], 
        "state": [], "action_mask": [], "embodiment_id": []
    }
    
    chunk_counter = 0
    current_buffer_count = 0
    
    print("🚀 Start Full Data Generation...")
    
    # Iterate over the entire loader once
    for batch in tqdm(gen_loader, desc="Generating", unit="batch"):
        
        with torch.no_grad():
            images = batch["images"]
            prompts = batch["prompts"]
            states = batch["states"].cuda().to(torch.bfloat16)
            action_mask = batch["action_mask"].cuda().to(torch.bfloat16)
            
            embodiment_ids = torch.zeros(states.shape[0], dtype=torch.long, device="cuda")
            
            # Get embeddings
            fused_tokens_list = []
            img_masks = batch.get("image_masks", [None]*len(images))
            for img, prompt, mask in zip(images, prompts, img_masks):
                ft = teacher_model.get_vl_embeddings(img, mask, prompt)
                fused_tokens_list.append(ft)
            fused_tokens = torch.cat(fused_tokens_list, dim=0).to(torch.bfloat16)
            
            # Prepare Noise
            B = states.shape[0]
            HORIZON = teacher_model.action_head.horizon
            PER_ACT = teacher_model.action_head.per_action_dim
            z0 = (torch.rand(B, HORIZON, PER_ACT, device="cuda") * 2 - 1).to(torch.bfloat16)
            # Teacher Inference
            z1_flat = teacher_model.action_head.get_action(
                fused_tokens, 
                state=states, 
                embodiment_id=embodiment_ids, 
                action_mask=action_mask[:, 0, :], 
                init_noise=z0.flatten(1)
            )
            z1 = z1_flat.view(B, HORIZON, PER_ACT)

            # Collect Data
            buffer_data["z0"].append(z0.cpu())
            buffer_data["z1"].append(z1.cpu())
            buffer_data["fused_tokens"].append(fused_tokens.cpu())
            buffer_data["state"].append(states.cpu())
            buffer_data["action_mask"].append(action_mask.cpu())
            buffer_data["embodiment_id"].append(embodiment_ids.cpu())
            
            current_buffer_count += B

            # Check if we need to save a chunk
            if current_buffer_count >= SAVE_CHUNK_SIZE:
                save_chunk(buffer_data, chunk_counter, DATA_SAVE_DIR)
                chunk_counter += 1
                current_buffer_count = 0
                torch.cuda.empty_cache()

    # Save remaining data after loop finishes
    if current_buffer_count > 0:
        print("💾 Saving final remaining chunk...")
        save_chunk(buffer_data, chunk_counter, DATA_SAVE_DIR)
    
    print("✅ All data generated and saved.")

if __name__ == "__main__":
    main()