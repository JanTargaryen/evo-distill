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
                state_mask: torch.Tensor = None, action_mask: torch.Tensor = None,
                z0: torch.Tensor = None, z1: torch.Tensor = None, is_reflow: bool = False):  # start and end actions for reflow

        if actions_gt is None and not is_reflow:
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

        if is_reflow: # is_reflow: target is velocity z1-z0
            if action_mask is not None:
                action_mask = action_mask.to(dtype=z0.dtype, device=z0.device)
                z0 = z0 * action_mask
                z1 = z1 * action_mask
            t_float = torch.rand(B, device=device)
            time_index = (t_float * 1000).long().clamp(max=999)
            time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0)

            t = t_float.to(dtype=self.dtype)  # to bfloat16

            if self.horizon > 1:
                t_expand = t.view(B, 1, 1)
            else:
                t_expand = t.view(B, 1)
            
            action_intermediate_seq = t_expand * z1 + (1 - t_expand) * z0
            target = z1 - z0 # Target Velocity

        else:
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
            target = noise

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

        return pred_velocity, target

    def eval_velocity(self, action, t_val, context_tokens, embodiment_id, action_mask_seq, per_action_dim):
        """
        [新增辅助函数] 将原 get_action 循环中的模型前向推理逻辑剥离出来。
        用于计算给定动作和时间点下的速度场 v(x, t)。
        """
        B = action.shape[0]
        device = action.device
        
        # [cite_start]1. 时间编码 (对应原代码 [cite: 111-112])
        # 将 t [0, 1] 映射到 time index [0, 1000]
        time_index = int(t_val * 1000)
        time_index = min(time_index, 999) 
        
        # [1, embed_dim] -> [B, embed_dim]
        time_emb = self.time_pos_enc(1000)[:, time_index, :].to(device).squeeze(0)
        time_emb = time_emb.unsqueeze(0).repeat(B, 1)

        # [cite_start]2. 动作序列维度调整 (对应原代码 [cite: 119])
        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)
        else:
            action_seq = action.view(B, 1, per_action_dim)

        # [cite_start]3. 动作编码 (对应原代码 [cite: 112-113])
        if self.horizon > 1 and self.action_encoder is not None:
            # 严格应用 mask
            action_seq = action_seq * action_mask_seq
            action_tokens = self.action_encoder(action_seq, embodiment_id)
        else:
            # 单步动作的线性投影 (Lazy init)
            if hasattr(self, "single_action_proj"):
                action_tokens = self.single_action_proj(action_seq)
            else:
                self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)
                action_tokens = self.single_action_proj(action_seq)

        # [cite_start]4. Transformer 主干 (对应原代码 [cite: 114])
        x = action_tokens
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)
        x = self.norm_out(x)

        # [cite_start]5. 池化与预测 (对应原代码 [cite: 115-116])
        if self.horizon > 1:
            x_flat = x.reshape(B, -1)
            if hasattr(self, "seq_pool_proj"):
                x_pooled = self.seq_pool_proj(x_flat)
            else:
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
                x_pooled = self.seq_pool_proj(x_flat)
        else:
            x_pooled = x.squeeze(1)

        # 输出速度预测
        pred_velocity = self.mlp_head(x_pooled, embodiment_id)
        return pred_velocity

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, 
                   embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None, 
                   init_noise: torch.Tensor = None): 
        
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
        
        if init_noise is not None: action = init_noise.to(device)
        else: action = (torch.rand(B, action_dim_total, device=device) * 2 - 1)
        
        if action_mask is not None:
            action_mask_seq = action_mask.view(B, 1, per_action_dim).repeat(1, self.horizon, 1)
            action_mask_seq = action_mask_seq.to(dtype=action.dtype, device=device)

        if not hasattr(self, "single_action_proj") and (self.horizon == 1 or self.action_encoder is None):
            self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)

        # ================== 诊断增强版自适应求解器 ==================
        
        t = 0.0
        dt = 0.1          
        step_count = 0
        
        # [策略调整]
        # 1. 基础容忍度调严：从 1e-2 -> 3e-3 (0.3% 误差)。这会把步数从 6 步增加到 10-15 步，但能保证精度。
        # 2. 末端强制收敛：在 t > 0.8 时，容忍度进一步降低。
        base_atol = 3e-3   
        rtol = 3e-3
        dt_min = 0.01      
        dt_max = 0.25      # 限制最大步长，防止跳过了关键的非线性区域

        v1 = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)

        # [诊断] 打印表头增加 MaxDim (出问题的维度索引)
        print(f"\n{'Step':<4} | {'Time':<5} | {'dt':<6} | {'Error':<8} | {'MaxDim':<6} | {'Status'}")

        while t < 1.0:
            if t + dt > 1.0: dt = 1.0 - t
            
            # --- 动态精度策略 ---
            # 如果接近终点 (t > 0.8)，我们希望精度极高，防止抓取时手抖
            # 如果是前期 (t < 0.5)，可以稍微粗糙一点
            if t > 0.8:
                current_atol = base_atol * 0.5  # 末端加倍严格 (1.5e-3)
            else:
                current_atol = base_atol

            # 1. Euler
            x_euler = action + v1 * dt
            
            # 2. Heun Corrector
            v2 = self.eval_velocity(x_euler, t + dt, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
            x_heun = action + 0.5 * dt * (v1 + v2)
            
            # [诊断] 计算逐维度的误差
            # error_dims: [Batch, ActionDim] -> 取 Batch max -> [ActionDim]
            error_dims = torch.abs(x_heun - x_euler).max(dim=0).values
            
            # 找到误差最大的那个维度索引
            max_err_val, max_err_idx = torch.max(error_dims, dim=0)
            max_err_val = max_err_val.item()
            max_err_idx = max_err_idx.item()
            
            # 自适应缩放
            current_abs_max = torch.max(torch.abs(x_heun)).item()
            tolerance = current_atol + rtol * current_abs_max
            
            if max_err_val > 0:
                scale = 0.9 * (tolerance / max_err_val) ** 0.5
            else:
                scale = 2.0

            # 决策
            if max_err_val < tolerance or dt <= dt_min:
                # === ACCEPT ===
                action = x_heun
                t += dt
                v1 = v2 
                print(f"{step_count:<4} | {t:<5.2f} | {dt:<6.3f} | {max_err_val:<8.5f} | {max_err_idx:<6} | ACC")
                step_count += 1
                dt = min(dt * scale, dt_max)
                if abs(t - 1.0) < 1e-5: break
            else:
                # === REJECT ===
                old_dt = dt
                dt = max(dt * scale, dt_min)
                print(f"{'REJ':<4} | {t:<5.2f} | {old_dt:<6.3f}->{dt:<6.3f} | {max_err_val:<8.5f} | {max_err_idx:<6} | RETRY")
        
        print(f"[DiagSolver] Total Steps: {step_count}")
        return action
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype