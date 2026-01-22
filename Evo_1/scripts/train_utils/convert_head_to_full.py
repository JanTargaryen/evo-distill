import torch
import os
import sys
import copy
import argparse

TEACHER_CKPT = "/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/metaworld/mp_rank_00_model_states.pt"

def merge_checkpoint(head_ckpt_path, output_dir):
    print(f"Loading Base Teacher: {TEACHER_CKPT}")
    base_ckpt = torch.load(TEACHER_CKPT, map_location="cpu")
    if "module" in base_ckpt:
        base_state = base_ckpt["module"]
    else:
        base_state = base_ckpt

    print(f"Loading Student Head: {head_ckpt_path}")
    head_state = torch.load(head_ckpt_path, map_location="cpu")

    print("Merging...")
    merged_state = copy.deepcopy(base_state)
    
    updated_keys = 0
    for k, v in head_state.items():
        # 补全前缀
        full_key = f"action_head.{k}"
        if full_key in merged_state:
            merged_state[full_key] = v
            updated_keys += 1
        else:
            print(f"Warning: {full_key} not found in base model")

    print(f"Updated {updated_keys} keys in action_head.")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mp_rank_00_model_states.pt")
    torch.save({"module": merged_state}, save_path)
    print(f"✅ Saved full model to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=str, default="/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline/checkpoint_epoch_50/student_head_only.pt")
    parser.add_argument("--out", type=str, default="/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline/checkpoint_epoch_50")
    args = parser.parse_args()

    merge_checkpoint(args.head, args.out)