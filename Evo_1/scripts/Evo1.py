import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from types import SimpleNamespace
from typing import List, Union, Tuple
from PIL import Image
import torch
import torch.nn as nn
import time
from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.flow_matching import FlowmatchingActionHead
import logging
class EVO1(nn.Module):
    def __init__(self, config: dict):
        super().__init__() 
        self.config = config
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device)

        action_head_type = config.get("action_head", "flowmatching").lower()
        
        if action_head_type == "flowmatching":
           
            horizon = config.get("action_horizon", config.get("horizon", 16))
            per_action_dim = config.get("per_action_dim", 7)
            action_dim = horizon * per_action_dim
            
            config["horizon"] = horizon
            config["per_action_dim"] = per_action_dim
            config["action_dim"] = action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            self.action_head = FlowmatchingActionHead(config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896),    
                hidden_dim=config.get("hidden_dim", 1024),
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                state_dim=config.get("state_dim", 7),
                state_hidden_dim=config.get("state_hidden_dim", 1024),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 8),
                dropout=config.get("dropout", 0.0),
                num_inference_timesteps=config.get("num_inference_timesteps", 50),
                num_categories=config.get("num_categories", 1)
            )).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,  
        prompt: str = "",
        return_cls_only: Union[bool, None] = None
    ) -> torch.Tensor:

        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:

        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
        steps: int = None,
        layer_idx: int = -1
    ):
        if actions_gt is None:
            # inference
            # Input fused_tokens is [B, Num_Exits, N, D]. Select specific layer for inference.
            # Default to -1 (last layer/Deepest)
            if fused_tokens.dim() == 4: 
                fused_tokens = fused_tokens[:, layer_idx, :, :]
            elif fused_tokens.dim() == 3 and fused_tokens.shape[1] == 4: 
                fused_tokens = fused_tokens[:, layer_idx, :].unsqueeze(1)
            
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids, steps=steps)
        else:
            # training
            # Input fused_tokens is [B, Num_Exits, N, D].
            # We iterate over all exits and compute predictions for each.
            # We assume Shared Head (same action_head weights for all layers).
            
            pred_velocities = []
            noises = []
            
            # fused_tokens shape: [B, Num_Exits, Seq, Dim]
            num_exits = fused_tokens.shape[1]
            
            for i in range(num_exits):
                tokens_i = fused_tokens[:, i, :, :] # [B, Seq, Dim]
                pred_v, noise = self.action_head(tokens_i, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids)
                pred_velocities.append(pred_v)
                noises.append(noise)
                
            # Stack results: [B, Num_Exits, ...]
            return torch.stack(pred_velocities, dim=1), torch.stack(noises, dim=1)

    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None,
        steps: int = None,
        layer_idx: int = -1
    ):
        t0 = time.time()
        fused_tokens = self.get_vl_embeddings(
                        images=images,
                        image_mask=image_mask,
                        prompt=prompt,
                        return_cls_only=return_cls_only
                    )

        state_tensor = self.prepare_state(state_input)  
        action, metadata = self.predict_action(fused_tokens, state_tensor, action_mask=action_mask, steps=steps, layer_idx=layer_idx)
        
        t1 = time.time()
        latency = (t1 - t0) * 1000
        
        return action, latency, metadata
    

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
   

        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _freeze_module(self, module: nn.Module, name: str):
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    def set_finetune_flags(self):
        config = self.config  
        if not config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
        else:
            print("Finetuning VLM (InternVL3)...")

        if not config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")
