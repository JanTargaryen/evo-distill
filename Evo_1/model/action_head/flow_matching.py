import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.randn(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.randn(num_categories, out_dim))

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):

        if self.num_categories <= 1:
            return self.linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        if category_id.dim() == 0:
       
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
           
            category_id = category_id.view(-1)  
            weight_selected = self.weight[category_id]        
            bias_selected = self.bias[category_id]        
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module):

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
   
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):

        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"
       
        x = action_seq.reshape(B * H, D) 
      
        if category_id.dim() == 0:
           
            cat_ids = category_id.repeat(H * B)
        else:
            cat_ids = category_id.unsqueeze(1).repeat(1, H).reshape(B * H)
        out = self.activation(self.W1(x, cat_ids))            
    
        pos_enc = self.pos_encoding(H).to(out.device)       
        pos_enc = pos_enc.repeat(B, 1, 1).reshape(B * H, -1) 
        out = out + pos_enc
        out = self.activation(self.W2(out, cat_ids))         
        out = self.W3(out, cat_ids)                        
        out = out.view(B, H, self.embed_dim)
        return out

class BasicTransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, time_emb: torch.Tensor):

        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)

        x = action_tokens + attn_out

        x2 = self.norm2(x)

        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

class FlowmatchingActionHead(nn.Module):

    def __init__(self, config=None,
                 embed_dim: int = 896, 
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 20,
                 num_categories: int = 1):
        super().__init__()

        if config is not None:
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            self.config = config
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                          action_dim=action_dim, horizon=horizon,
                                          num_heads=num_heads, num_layers=num_layers,
                                          dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                                          num_categories=num_categories)
        print(f"num_inference_timesteps {num_inference_timesteps}")
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = config.per_action_dim
        self.action_dim = config.action_dim


        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                   hidden_dim=embed_dim*4, dropout=dropout)
            for _ in range(num_layers)
        ])
       
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)

        self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                            output_dim=action_dim, num_categories=num_categories)

        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
        
            self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                    hidden_dim=state_hidden,
                                                    output_dim=embed_dim,
                                                    num_categories=num_categories)

        self.action_encoder = None
        if horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,  
                                                               horizon=horizon,
                                                               num_categories=num_categories)

    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None, 
                action_mask: torch.Tensor = None,
                ): 

        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id)
        B = fused_tokens.size(0)
        device = fused_tokens.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = fused_tokens 
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id)  
            state_emb = state_emb.unsqueeze(1) 
            context_tokens = torch.cat([context_tokens, state_emb], dim=1) 

        # Original Flow Matching logic
        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype) # Beta distribution , trick
        time_index = (t * 1000).long()  
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0) 

        actions_gt_seq = actions_gt  
        noise = torch.rand_like(actions_gt) * 2 - 1   # rand_like * 2 - 1 : [0,1] -> [-1,1]

        if action_mask is not None: # action mask: clear abundant 0 padding , avoid model to learn useless pattern
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            assert action_mask.shape == noise.shape, f"action_mask shape {action_mask.shape} != noise shape {noise.shape}"
            noise = noise * action_mask

        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
        else:
            noise_seq = noise.unsqueeze(1)

        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        # (1 - t) * noise + t * actions_gt(ground )
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq
        

        if self.horizon > 1 and self.action_encoder is not None:
            action_tokens = self.action_encoder(action_intermediate_seq, embodiment_id)  
        else:
            if not hasattr(self, "single_action_proj"):
                self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim).to(device)
            action_tokens = self.single_action_proj(action_intermediate_seq) 

        x = action_tokens  
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)

        x = self.norm_out(x)  

        if self.horizon > 1:
            x_flat = x.reshape(B, -1)  
            if not hasattr(self, "seq_pool_proj"):
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
            x_pooled = self.seq_pool_proj(x_flat)  
        else:
            x_pooled = x.squeeze(1) 

        pred_velocity = self.mlp_head(x_pooled, embodiment_id) 

        return pred_velocity, noise

    def eval_velocity(self, action, t_val, context_tokens, embodiment_id, action_mask_seq, per_action_dim):
        """
        calulate v(x, t)
        """
        B = action.shape[0]
        device = action.device
        
        time_index = int(t_val * 1000)
        time_index = min(time_index, 999) 
        
        time_emb = self.time_pos_enc(1000)[:, time_index, :].to(device).squeeze(0)
        time_emb = time_emb.unsqueeze(0).repeat(B, 1)

        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)
        else:
            action_seq = action.view(B, 1, per_action_dim)

        if self.horizon > 1 and self.action_encoder is not None:
            action_seq = action_seq * action_mask_seq
            action_tokens = self.action_encoder(action_seq, embodiment_id)
        else:
            if hasattr(self, "single_action_proj"):
                action_tokens = self.single_action_proj(action_seq)
            else:
                self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)
                action_tokens = self.single_action_proj(action_seq)

        x = action_tokens
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)
        x = self.norm_out(x)

        if self.horizon > 1:
            x_flat = x.reshape(B, -1)
            if hasattr(self, "seq_pool_proj"):
                x_pooled = self.seq_pool_proj(x_flat)
            else:
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
                x_pooled = self.seq_pool_proj(x_flat)
        else:
            x_pooled = x.squeeze(1)

        pred_velocity = self.mlp_head(x_pooled, embodiment_id)
        return pred_velocity

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, 
                   embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None, 
                   verbose: bool = False,
                   steps: int = None):
        
        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None: embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)
            context_tokens = torch.cat([context_tokens, state_emb], dim=1) 
            
        action_dim_total = getattr(self.config, "action_dim", self.action_dim)
        if self.horizon > 1: per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else: per_action_dim = action_dim_total
        
        action = (torch.rand(B, action_dim_total, device=device) * 2 - 1)
        
        action_mask_seq = None

        if action_mask is not None:
            expected_full_size = B * self.horizon * per_action_dim
            
            if action_mask.numel() == expected_full_size:
                action_mask_seq = action_mask.view(B, self.horizon, per_action_dim)
            else:
                action_mask_seq = action_mask.view(B, 1, per_action_dim).repeat(1, self.horizon, 1)
                
            action_mask_seq = action_mask_seq.to(dtype=action.dtype, device=device)

        if not hasattr(self, "single_action_proj") and (self.horizon == 1 or self.action_encoder is None):
            self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)

        # Strategy Selection: Linearity-Aware Quantization (Dynamic) vs Direct Solver (Fixed)
        if steps is None:
            #  Linearity-Aware Adaptive Sampling ===
            
            dt_probe = 0.5 
            t = 0.0
            
            # 1. Lookahead Probe 
            v_start = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
            x_mid = action + v_start * dt_probe
            v_mid = self.eval_velocity(x_mid, t + dt_probe, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
            
            # 2. Compute Metrics
            flat_v_start = v_start.view(B, -1)
            flat_v_mid = v_mid.view(B, -1)
            
            # Metric: Cosine Similarity (Linearity)
            cos_sim = F.cosine_similarity(flat_v_start, flat_v_mid, dim=1)
            sim_score = cos_sim.min().item()

            # Metric: Magnitude Stability (Optional, for monitoring)
            mag_start = flat_v_start.norm(dim=1)
            mag_mid = flat_v_mid.norm(dim=1)
            mag_ratio = torch.minimum(mag_start, mag_mid) / (torch.maximum(mag_start, mag_mid) + 1e-6)
            mag_score = mag_ratio.min().item()

            # 3. Apply Quantization Formula
            # Formula: N = clip(N_min + 2 * floor((1 - sim) / epsilon), N_min, N_max)
            # epsilon = 0.008 corresponds to the sensitivity threshold derived from empirical tuning.
            epsilon = 0.008 
            curvature = 1.0 - sim_score
            
            # Math: map curvature error to step increments
            raw_steps = 2 + 2 * math.floor(curvature / epsilon)
            target_steps = int(min(max(raw_steps, 2), 8))

            if verbose:
                print(f"[Adaptive] Sim: {sim_score:.4f} (Err: {curvature:.4f}) -> Steps: {target_steps}")

        else:
            # fix steps
            target_steps = steps
            t = 0.0
            v_start = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
            
            # Dummy metadata
            sim_score = 0.0
            mag_score = 0.0

        
        # Optimization: If Dynamic Mode selected 2 steps, re-use x_mid/v_mid from the probe phase.
        if steps is None and target_steps == 2:
            action = x_mid + v_mid * dt_probe
        else:
            # Standard Euler Integration
            # Note: v_start is already computed in both branches above.
            dt = 1.0 / target_steps
            action = action + v_start * dt
            t += dt
            
            # Iterate for remaining steps
            for _ in range(target_steps - 1):
                v = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
                action = action + v * dt
                t += dt

        metadata = {
            "steps": target_steps,
            "sim": sim_score,
            "mag": mag_score 
        }
        
        return action, metadata
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype