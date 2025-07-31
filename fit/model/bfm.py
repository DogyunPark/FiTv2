import torch
import torch.nn as nn
from functools import partial
from typing import Optional
from einops import rearrange, repeat
from fit.model.modules_lwd_bk import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder,
    FiTBlock, FinalLayer, RepresentationBlock
)
from fit.model.utils import make_grid_mask_size_online, make_grid_mask_size
from fit.utils.eval_utils import init_from_ckpt
from fit.model.rope import VisionRotaryEmbedding
#################################################################################
#                                 Core FiT Model                                #
#################################################################################


class FiT(nn.Module):
    """
    Flexible Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        context_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        use_sit: bool = False,
        use_checkpoint: bool=False,
        use_swiglu: bool = False,
        use_swiglu_large: bool = False,
        rel_pos_embed: Optional[str] = 'rope',
        norm_type: str = "layernorm",
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        adaln_bias: bool = True,
        adaln_type: str = "normal",
        adaln_lora_dim: int = None,
        rope_theta: float = 10000.0,
        custom_freqs: str = 'normal',
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        online_rope: bool = False,
        add_rel_pe_to_v: bool = False,
        pretrain_ckpt: str = None,
        ignore_keys: list = None,
        finetune: str = None,
        time_shifting: int = 1,
        number_of_perflow: int = 1,
        overlap: bool = False,
        max_cached_len: int = 256,
        number_of_representation_blocks: int = 1,
        global_cls: bool = False,
        n_patch_h: int = 16,
        n_patch_w: int = 16,
        finetune_representation: bool = False,
        concat_adaln: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size
        assert not (learn_sigma and use_sit)
        self.learn_sigma = learn_sigma
        self.use_sit = use_sit
        self.use_checkpoint = use_checkpoint
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = self.in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.adaln_type = adaln_type
        self.online_rope = online_rope
        self.time_shifting = time_shifting
        self.sigmas = torch.linspace(0, 1, number_of_perflow+1)
        #self.sigmas = linear_decrease_division(number_of_perflow)
        #self.sigmas = symmetric_segment_division(number_of_perflow)
        self.number_of_perflow = number_of_perflow
        self.number_of_layers_for_perflow = depth // number_of_perflow
        self.number_of_representation_blocks = number_of_representation_blocks
        self.global_cls = global_cls
        self.n_patch_h = n_patch_h
        self.n_patch_w = n_patch_w
        self.representation_align = kwargs.get('representation_align', False)

        self.representation_x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.representation_blocks = nn.ModuleList([RepresentationBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type, 
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim
        ) for _ in range(number_of_representation_blocks)])

        if self.representation_align:
            self.linear_projection = nn.Sequential(
                    nn.Linear(hidden_size, 2048),
                    nn.SiLU(),
                    nn.Linear(2048, 2048),
                    nn.SiLU(),
                    nn.Linear(2048, 768),
                )

        self.x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.rel_pos_embed = VisionRotaryEmbedding(
            head_dim=hidden_size//num_heads, theta=rope_theta, custom_freqs=custom_freqs, online_rope=online_rope,
            max_pe_len_h=max_pe_len_h, max_pe_len_w=max_pe_len_w, decouple=decouple, ori_max_pe_len=ori_max_pe_len,
            max_cached_len=max_cached_len,
        )
        
        if adaln_type == 'lora':
            self.global_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias)
            )
            if concat_adaln:
                self.global_adaLN_modulation2 = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size*2, 6 * hidden_size, bias=adaln_bias)
                )
            else:
                 self.global_adaLN_modulation2 = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias)
                )
        else:
            self.global_adaLN_modulation = None        
        
        self.blocks = nn.ModuleList([nn.ModuleList([FiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type, 
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias, 
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim, concat_adaln=concat_adaln
        ) for _ in range(self.number_of_layers_for_perflow)]) for _ in range(self.number_of_perflow)])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type, concat_adaln=concat_adaln)

        if finetune_representation:
            self.mid_block = nn.ModuleList([FiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
               rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type, 
               q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias, 
               adaln_bias=adaln_bias, adaln_type='normal', adaln_lora_dim=adaln_lora_dim) for _ in range(4)])
            self.finetune(type=finetune, unfreeze=['mid_block'])

        self.initialize_weights(pretrain_ckpt=pretrain_ckpt, ignore=ignore_keys)
        
        if finetune != None:
            self.finetune(type=finetune, unfreeze=ignore_keys)

    def initialize_weights(self, pretrain_ckpt=None, ignore=None):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.xavier_uniform_(self.representation_x_embedder.proj.weight.data)
        nn.init.constant_(self.representation_x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for blocks in self.blocks:
            for block in blocks:
                if self.adaln_type in ['normal', 'lora']:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                elif self.adaln_type == 'swiglu':
                    nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                    nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)

        for block in self.representation_blocks:
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)

        if self.adaln_type == 'lora':
            nn.init.constant_(self.global_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.global_adaLN_modulation2[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation2[-1].bias, 0)
        # Zero-out output layers:
        if self.adaln_type == 'swiglu':
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.bias, 0)
        else:   # adaln_type in ['normal', 'lora']
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        keys = list(self.state_dict().keys())
        ignore_keys = []
        if ignore != None:
            for ign in ignore:
                for key in keys:
                    if ign in key:
                        ignore_keys.append(key)
        ignore_keys = list(set(ignore_keys))
        if pretrain_ckpt != None:
            init_from_ckpt(self, pretrain_ckpt, ignore_keys, verbose=True)
        

    def unpatchify(self, x, hw):
        """
        args:
            x: (B, p**2 * C_out, N)
            N = h//p * w//p
        return: 
            imgs: (B, C_out, H, W)
        """
        h, w = hw
        p = self.patch_size
        if self.use_sit:
            x = rearrange(x, "b (h w) c -> b h w c", h=h//p, w=w//p) # (B, h//2 * w//2, 16) -> (B, h//2, w//2, 16)
            x = rearrange(x, "b h w (c p1 p2) -> b c (h p1) (w p2)", p1=p, p2=p) # (B, h//2, w//2, 16) -> (B, h, w, 4)
        else:
            x = rearrange(x, "b c (h w) -> b c h w", h=h//p, w=w//p) # (B, 16, h//2 * w//2) -> (B, 16, h//2, w//2)
            x = rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=p, p2=p) # (B, 16, h//2, w//2) -> (B, h, w, 4)
        return x

    def get_segment_index(self, t):
        """
        Return the index (0-based) of the segment in [0,1] that t belongs to,
        when [0,1] is divided into 6 uniform segments.
        """
        assert 0.0 <= t <= 1.0, "t must be in [0,1]"
        
        # Edge case: if t == 1.0, assign it to the last segment
        if t == 1.0:
            return self.number_of_perflow - 1
        
        segment_length = 1.0 / self.number_of_perflow
        return int(t // segment_length)
    
    def forward(self, x, t, y, target_layer_start=None, target_layer_end=None, 
                    target_representation_layer_start=None, target_representation_layer_end=None,
                    finetune_representation=False, t_next=None, xt_next=None, ratio=None, layer_idx=None):
        return self.forward_run_layer(x, t, y, target_layer_start, target_layer_end, target_representation_layer_start, target_representation_layer_end, layer_idx=layer_idx)
    
    def forward_run_layer(self, x, t, y, target_layer_start=None, target_layer_end=None, 
        target_representation_layer_start=None, target_representation_layer_end=None,
        t_next=None, representation_noise=None, layer_idx=None):
        # assert target_layer_start is not None, "target_layer_start must be provided"
        # assert target_layer_end is not None, "target_layer_end must be provided"
        # assert len(self.blocks) >= target_layer_end, "target_layer_end must be within the range of the number of blocks"
        
        # Add a dummy loss that touches every parameter of the model
        dummy_loss = 0.0
        for param in self.parameters():
            dummy_loss = dummy_loss + 0.0 * param.sum()        

        grid, mask, size = make_grid_mask_size_online(x, self.patch_size, x.device)
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
        t = t.float().to(x.dtype)
        t = self.t_embedder(t)        
        y = self.y_embedder(y, self.training)           # (B, D)
        c = t + y

        if self.global_adaLN_modulation != None:
            global_adaln = self.global_adaLN_modulation(c)
        else: 
            global_adaln = 0.0
        
        representation_noise = self.representation_x_embedder(x)
        for rep_block in self.representation_blocks:
            representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
        
        if self.representation_align:
            representation_linear = self.linear_projection(representation_noise)
        else:
            representation_linear = None

        c_repre = c.unsqueeze(1) + representation_noise

        if self.global_adaLN_modulation != None:
            global_adaln2 = self.global_adaLN_modulation2(c_repre)
        else: 
            global_adaln2 = 0.0

        x = self.x_embedder(x)

        #for i in range(target_layer_start, target_layer_end):
        for j, blocks in enumerate(self.blocks[layer_idx]):
            x = blocks(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)

        x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        return x, representation_linear, dummy_loss

    def forward_wo_cfg(self, x, y, number_of_step_perflow=1):
        """
        Forward pass of FiT.
        x: (B, p**2 * C_in, N), tensor of sequential inputs (flattened latent features of images, N=H*W/(p**2))
        t: (B,), tensor of diffusion timesteps
        y: (B,), tensor of class labels
        grid: (B, 2, N), tensor of height and weight indices that spans a grid
        mask: (B, N), tensor of the mask for the sequence
        size: (B, n, 2), tensor of the height and width, n is the number of the packed iamges
        --------------------------------------------------------------------------------------------
        return: (B, p**2 * C_out, N), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        """
        #cfg = cfg.float().to(x.dtype) 
        
        grid, mask, size = make_grid_mask_size(x.shape[0], self.n_patch_h, self.n_patch_w, self.patch_size, x.device)
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        for i in range(self.number_of_perflow):
            y_embed = self.y_embedder(y, self.training)           # (B, D)

            sigma_next = self.sigmas[i+1] 
            sigma_current = self.sigmas[i]
            sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)

            for step in range(number_of_step_perflow):
                t = sigma_list[step].expand(x.shape[0]).to(x.device)
                t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
                t = t.float().to(x.dtype)
                t = self.t_embedder(t)
                c = t + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0
                
                representation_noise = self.representation_x_embedder(x)
                for rep_block in self.representation_blocks:
                    representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                c_repre = c.unsqueeze(1) + representation_noise

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0

                residual = x.clone()
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                for block in self.blocks[i]:
                    x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
        return x
    
    def forward_maruyama(self, x, y, number_of_step_perflow=1):
        """
        Forward pass of FiT.
        x: (B, p**2 * C_in, N), tensor of sequential inputs (flattened latent features of images, N=H*W/(p**2))
        t: (B,), tensor of diffusion timesteps
        y: (B,), tensor of class labels
        grid: (B, 2, N), tensor of height and weight indices that spans a grid
        mask: (B, N), tensor of the mask for the sequence
        size: (B, n, 2), tensor of the height and width, n is the number of the packed iamges
        --------------------------------------------------------------------------------------------
        return: (B, p**2 * C_out, N), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        """
        #cfg = cfg.float().to(x.dtype) 
        y_embed = self.y_embedder(y, self.training)           # (B, D)

        grid, mask, size = make_grid_mask_size(x.shape[0], self.n_patch_h, self.n_patch_w, self.patch_size, x.device)
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        # get RoPE frequences in advance, then calculate attention.

        for i in range(self.number_of_perflow):

            if i == self.number_of_perflow - 1:
                sigma_next = (1 - 0.04)
                sigma_current = self.sigmas[i]
                sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow)
                sigma_list = torch.cat([sigma_list, torch.tensor([1.0])], dim=0)
            else:
                sigma_next = self.sigmas[i+1] 
                sigma_current = self.sigmas[i]
                sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)
            
            for step in range(number_of_step_perflow):
                t = sigma_list[step].expand(x.shape[0]).to(x.device)
                t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
                t = t.float().to(x.dtype)
                t = self.t_embedder(t)
                c = t + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                if self.number_of_representation_blocks > 1:
                    representation_noise = self.representation_x_embedder(x)
                    for rep_block in self.representation_blocks:
                        if not self.use_checkpoint:
                            representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                        else:
                            representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                c_repre = t.unsqueeze(1) + representation_noise

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0

                x_prev = x.clone()
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                
                for block in self.blocks[i]:
                    x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
                dt = sigma_list[step+1] - sigma_list[step]
                w_cur = torch.randn_like(x)
                t = torch.ones_like(x) * sigma_list[step]
                dw = w_cur * torch.sqrt(dt)
                
                score = (t * x - x_prev) / (1 - t)
                drift = x + (1-t) * score
                diffusion = (1-t)

                if (i == self.number_of_perflow - 1) and (step == number_of_step_perflow - 1):
                    x = x_prev + drift*dt
                else:
                    x = x_prev + drift*dt
                    x = x + torch.sqrt(2 * diffusion) * dw
        return x

    def forward_cfg(self, x, cfg, y, number_of_step_perflow=1,
        guidance_low=0.0, guidance_high=1.0, self_guidance=False):
        """
        Forward pass of FiT.
        x: (B, p**2 * C_in, N), tensor of sequential inputs (flattened latent features of images, N=H*W/(p**2))
        t: (B,), tensor of diffusion timesteps
        y: (B,), tensor of class labels
        grid: (B, 2, N), tensor of height and weight indices that spans a grid
        mask: (B, N), tensor of the mask for the sequence
        size: (B, n, 2), tensor of the height and width, n is the number of the packed iamges
        --------------------------------------------------------------------------------------------
        return: (B, p**2 * C_out, N), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        """

        assert cfg > 1, "cfg must be greater than 1"
        y_null = torch.tensor([self.num_classes] * x.shape[0], device=x.device)
        y_embed = torch.cat([y, y_null], dim=0)
        y_embed = self.y_embedder(y_embed, self.training)           # (B, D)

        grid, mask, size = make_grid_mask_size(x.shape[0], self.n_patch_h, self.n_patch_w, self.patch_size, x.device)
        size = torch.cat([size, size], dim=0)
        grid = torch.cat([grid, grid], dim=0)
        mask = torch.cat([mask, mask], dim=0)

        x_next = x
        _dtype = x.dtype
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        
        for i in range(self.number_of_perflow):
            if i == self.number_of_perflow - 1:
                sigma_next = (1 - 0.04)
                sigma_current = self.sigmas[i]
                sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow)
                sigma_list = torch.cat([sigma_list, torch.tensor([1.0])], dim=0)
            else:
                sigma_next = self.sigmas[i+1] 
                sigma_current = self.sigmas[i]
                sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)

            for step in range(number_of_step_perflow):
                t_cur = sigma_list[step]
                x = x_next
                x = torch.cat([x, x], dim=0).to(dtype=_dtype)

                t = sigma_list[step].expand(x.shape[0]).to(x.device, dtype=_dtype)
                t = self.t_embedder(t)
                c = t + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                representation_noise = self.representation_x_embedder(x)
                for rep_block in self.representation_blocks:
                    representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                c_repre = c.unsqueeze(1) + representation_noise

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0

                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                for block in self.blocks[i]:
                    x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)

                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
                x = x.to(x_next.dtype)
                
                x_cond, x_uncond = x.chunk(2, dim=0)
                if guidance_low <= t_cur and t_cur <= guidance_high:
                    x = x_uncond + cfg * (x_cond - x_uncond)
                else:
                    x = x_cond
                x_next = (sigma_list[step+1] - sigma_list[step]) * x + x_next
        return x_next

    #########################################################################################
    #########################################################################################

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
    
    
    def finetune(self, type, unfreeze):
        if type == 'full':
            return
        for name, param in self.named_parameters():
                param.requires_grad = False
        for unf in unfreeze:
            for name, param in self.named_parameters():
                if unf in name: # LN means Layer Norm
                    #print(f'{unf} in {name}')
                    param.requires_grad = True

    def set_blocks_grad(self, active_layer_idx=None, requires_grad=True):
        """
        Set requires_grad for self.blocks based on the active layer index.
        
        Args:
            active_layer_idx (int, optional): Index of the layer group to set as trainable.
                                             If None, sets all blocks according to requires_grad.
            requires_grad (bool): Whether to enable gradients for the active layer.
                                 Inactive layers are always set to requires_grad=False.
        """
        # First, set all blocks to not require gradients
        for i, layer_group in enumerate(self.blocks):
            for block in layer_group:
                for param in block.parameters():
                    param.requires_grad = False
        
        # If active_layer_idx is specified, enable gradients for that group
        if active_layer_idx is not None:
            if 0 <= active_layer_idx < len(self.blocks):
                for block in self.blocks[active_layer_idx]:
                    for param in block.parameters():
                        param.requires_grad = requires_grad
            else:
                raise ValueError(f"active_layer_idx {active_layer_idx} is out of range. "
                               f"Valid range is 0 to {len(self.blocks)-1}")
    
    def get_active_block_params(self, layer_idx):
        """
        Get parameters of the active block group for optimizer setup.
        
        Args:
            layer_idx (int): Index of the active layer group.
            
        Returns:
            Iterator of parameters that require gradients in the active layer group.
        """
        if 0 <= layer_idx < len(self.blocks):
            for block in self.blocks[layer_idx]:
                for param in block.parameters():
                    if param.requires_grad:
                        yield param
        else:
            raise ValueError(f"layer_idx {layer_idx} is out of range. "
                           f"Valid range is 0 to {len(self.blocks)-1}")
    
    def get_trainable_params(self, exclude_blocks=False):
        """
        Get all trainable parameters in the model.
        
        Args:
            exclude_blocks (bool): If True, exclude self.blocks parameters.
                                  Useful when you want other model components to remain trainable.
        
        Returns:
            Iterator of all trainable parameters.
        """
        for name, param in self.named_parameters():
            if exclude_blocks and name.startswith('blocks.'):
                continue
            if param.requires_grad:
                yield param

    def verify_grad_settings(self, verbose=True):
        """
        Verify and report which parameters have requires_grad=True.
        
        Args:
            verbose (bool): If True, print detailed information about gradient settings.
            
        Returns:
            dict: Summary of gradient settings per component.
        """
        summary = {
            'blocks': {},
            'other_components': {},
            'total_trainable_params': 0,
            'total_params': 0
        }
        
        # Check blocks
        for i, layer_group in enumerate(self.blocks):
            trainable_params = 0
            total_params = 0
            for j, block in enumerate(layer_group):
                for param in block.parameters():
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
            
            summary['blocks'][f'layer_group_{i}'] = {
                'trainable_params': trainable_params,
                'total_params': total_params,
                'is_trainable': trainable_params > 0
            }
        
        # Check other components
        other_components = [
            ('x_embedder', self.x_embedder),
            ('t_embedder', self.t_embedder), 
            ('y_embedder', self.y_embedder),
            ('representation_x_embedder', self.representation_x_embedder),
            ('representation_blocks', self.representation_blocks),
            ('final_layer', self.final_layer)
        ]
        
        for name, component in other_components:
            if component is not None:
                trainable_params = sum(p.numel() for p in component.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in component.parameters())
                summary['other_components'][name] = {
                    'trainable_params': trainable_params,
                    'total_params': total_params,
                    'is_trainable': trainable_params > 0
                }
        
        # Calculate totals
        summary['total_trainable_params'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary['total_params'] = sum(p.numel() for p in self.parameters())
        
        if verbose:
            print("=== Gradient Settings Verification ===")
            print(f"Total trainable parameters: {summary['total_trainable_params']:,}")
            print(f"Total parameters: {summary['total_params']:,}")
            print(f"Trainable ratio: {summary['total_trainable_params']/summary['total_params']:.4f}")
            
            print("\n--- Blocks ---")
            for layer_name, info in summary['blocks'].items():
                status = "✓ TRAINABLE" if info['is_trainable'] else "✗ FROZEN"
                print(f"{layer_name}: {info['trainable_params']:,}/{info['total_params']:,} params {status}")
            
            print("\n--- Other Components ---")
            for comp_name, info in summary['other_components'].items():
                status = "✓ TRAINABLE" if info['is_trainable'] else "✗ FROZEN"
                print(f"{comp_name}: {info['trainable_params']:,}/{info['total_params']:,} params {status}")
        
        return summary

    def count_trainable_params_by_component(self):
        """
        Quick count of trainable parameters by component.
        
        Returns:
            dict: Component name -> number of trainable parameters
        """
        counts = {}
        
        # Count blocks
        for i, layer_group in enumerate(self.blocks):
            counts[f'blocks_layer_{i}'] = sum(
                p.numel() for block in layer_group for p in block.parameters() if p.requires_grad
            )
        
        # Count other components
        components = {
            'x_embedder': self.x_embedder,
            't_embedder': self.t_embedder,
            'y_embedder': self.y_embedder,
            'representation_x_embedder': self.representation_x_embedder,
            'representation_blocks': self.representation_blocks,
            'final_layer': self.final_layer
        }
        
        for name, component in components.items():
            if component is not None:
                counts[name] = sum(p.numel() for p in component.parameters() if p.requires_grad)
        
        return counts

    def create_layer_optimizers(self, optimizer_class=torch.optim.AdamW, **optimizer_kwargs):
        """
        Create separate optimizers for each layer group to avoid DDP unused parameter issues.
        
        Args:
            optimizer_class: The optimizer class to use (default: AdamW)
            **optimizer_kwargs: Arguments to pass to the optimizer
            
        Returns:
            list: List of optimizers, one for each layer group
            dict: Shared parameters optimizer (embedders, final layer, etc.)
        """
        optimizers = []
        
        # Create optimizer for each layer group
        for i, layer_group in enumerate(self.blocks):
            layer_params = []
            for block in layer_group:
                layer_params.extend(block.parameters())
            
            if layer_params:  # Only create optimizer if there are parameters
                optimizer = optimizer_class(layer_params, **optimizer_kwargs)
                optimizers.append(optimizer)
            else:
                optimizers.append(None)
        
        # Create optimizer for shared components (always trainable)
        shared_components = [
            self.x_embedder,
            self.t_embedder, 
            self.y_embedder,
            self.representation_x_embedder,
            self.representation_blocks,
            self.final_layer
        ]
        
        shared_params = []
        for component in shared_components:
            if component is not None:
                shared_params.extend(component.parameters())
        
        # Add global modulation if exists
        if hasattr(self, 'global_adaLN_modulation') and self.global_adaLN_modulation is not None:
            shared_params.extend(self.global_adaLN_modulation.parameters())
        if hasattr(self, 'global_adaLN_modulation2') and self.global_adaLN_modulation2 is not None:
            shared_params.extend(self.global_adaLN_modulation2.parameters())
        
        # Add representation alignment if exists
        if hasattr(self, 'linear_projection') and self.linear_projection is not None:
            shared_params.extend(self.linear_projection.parameters())
        
        shared_optimizer = optimizer_class(shared_params, **optimizer_kwargs) if shared_params else None
        
        return optimizers, shared_optimizer

    def get_current_optimizers(self, layer_optimizers, shared_optimizer, active_layer_idx):
        """
        Get the optimizers that should be used for the current training step.
        
        Args:
            layer_optimizers: List of layer-specific optimizers
            shared_optimizer: Optimizer for shared components
            active_layer_idx: Index of the currently active layer
            
        Returns:
            list: List of optimizers to use for this step
        """
        current_optimizers = []
        
        # Add shared optimizer (always used)
        if shared_optimizer is not None:
            current_optimizers.append(shared_optimizer)
        
        # Add active layer optimizer
        if 0 <= active_layer_idx < len(layer_optimizers) and layer_optimizers[active_layer_idx] is not None:
            current_optimizers.append(layer_optimizers[active_layer_idx])
        
        return current_optimizers

    def apply_gradient_mask(self, active_layer_idx):
        """
        Alternative to requires_grad: manually zero out gradients for inactive layers.
        Call this after backward() but before optimizer.step().
        
        Args:
            active_layer_idx: Index of the layer group that should keep gradients
        """
        for i, layer_group in enumerate(self.blocks):
            if i != active_layer_idx:
                for block in layer_group:
                    for param in block.parameters():
                        if param.grad is not None:
                            param.grad.zero_()