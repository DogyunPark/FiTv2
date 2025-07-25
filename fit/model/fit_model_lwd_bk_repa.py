import torch
import torch.nn as nn
from functools import partial
from typing import Optional
from einops import rearrange, repeat
from fit.model.modules_lwd import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder,
    FiTBlock, FinalLayer, RepresentationBlock
)
from fit.model.utils import get_parameter_dtype, make_grid_mask_size, make_grid_mask_size_online
from fit.utils.eval_utils import init_from_ckpt
#from fit.model.sincos import get_2d_sincos_pos_embed_from_grid
from fit.model.rope import VisionRotaryEmbedding, get_2d_sincos_pos_embed
from fit.utils.utils import linear_decrease_division, symmetric_segment_division
#################################################################################
#                                 Core FiT Model                                #
#################################################################################


class FiTLwD_sharedenc_sepdec(nn.Module):
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
        fourier_basis: bool = False,
        perlayer_embedder: bool = False,
        max_cached_len: int = 256,
        number_of_shared_blocks: int = 1,
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
        self.perlayer_embedder = perlayer_embedder
        self.number_of_perflow = number_of_perflow
        self.number_of_layers_for_perflow = depth // number_of_perflow
        self.number_of_shared_blocks = number_of_shared_blocks
        self.number_of_representation_blocks = number_of_representation_blocks
        self.global_cls = global_cls
        self.n_patch_h = n_patch_h
        self.n_patch_w = n_patch_w

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
                nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias))
        else:
            self.global_adaLN_modulation = None        
        
        self.blocks = nn.ModuleList([FiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type, 
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias, 
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim
        ) for _ in range(depth)])

        final_layer_out_channels = self.out_channels*2 if fourier_basis else self.out_channels
        self.final_layer = FinalLayer(hidden_size, patch_size, final_layer_out_channels, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type)

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

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)
        if self.adaln_type == 'lora':
            nn.init.constant_(self.global_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation[-1].bias, 0)
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

    def forward_wo_cfg(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):

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

                residual = x.clone()
                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in

                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                    x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                x = self.final_layer(x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
        return x
    
    def forward_wo_cfg_int(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
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
                
                if self.number_of_representation_blocks > 1:
                    representation_noise = self.representation_x_embedder(x)
                    #import pdb; pdb.set_trace()
                    for rep_block in self.representation_blocks:
                        if not self.use_checkpoint:
                            representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                        else:
                            representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                c_repre = t.unsqueeze(1) + representation_noise
                #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0

                residual = x.clone()
                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in

                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
        return x
    
    def forward_wo_cfg_repre(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        repre_list = []
        grid, mask, size = make_grid_mask_size(x.shape[0], self.n_patch_h, self.n_patch_w, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
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
                
                if step == 0:
                    if self.number_of_representation_blocks > 1:
                        representation_noise = self.representation_x_embedder(x)
                        for rep_block in self.representation_blocks:
                            if not self.use_checkpoint:
                                representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                            else:
                                representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                residual = x.clone()

                x_mid = self.representation_x_embedder(x)
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  

                if step != 0:
                    #ratio = sigma_list_ratio[step].expand(x.shape[0]).to(x.device)
                    #ratio = ratio.float().to(x.dtype)
                    #ratio = self.t_embedder(ratio)

                    c_mid = t.unsqueeze(1) + representation_noise
                    coefficient = self.coefficient_layers[i](t)
                    #c_mid = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)
                    representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block[i](x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block[i](x_mid, c_mid)
                else:
                    representation_noise_t = representation_noise
                #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0
                
                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
        return x, repre_list
    
    def forward_wo_cfg_int_repre(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        repre_list = []
        grid, mask, size = make_grid_mask_size(x.shape[0], self.n_patch_h, self.n_patch_w, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
            y_embed = self.y_embedder(y, self.training)           # (B, D)

            sigma_next = self.sigmas[i+1] 
            sigma_current = self.sigmas[i]
            sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)
            sigma_list_ratio = (sigma_list - sigma_current) / (sigma_next - sigma_current)

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
                
                if step == 0:
                    if self.number_of_representation_blocks > 1:
                        representation_noise = self.representation_x_embedder(x)
                        for rep_block in self.representation_blocks:
                            if not self.use_checkpoint:
                                representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                            else:
                                representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                residual = x.clone()

                x_mid = self.representation_x_embedder(x)
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  

                if step != 0:
                    ratio = sigma_list_ratio[step].expand(x.shape[0]).to(x.device)
                    ratio = ratio.float().to(x.dtype)
                    ratio = self.t_embedder(ratio)

                    c_mid = t.unsqueeze(1) + representation_noise
                    coefficient = self.coefficient_layers[i](ratio)
                    #c_mid = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)
                    #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block[i](x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block[i](x_mid, c_mid)
                else:
                    representation_noise_t = representation_noise

                repre_list.append(representation_noise_t.detach().cpu())
                c_repre = t.unsqueeze(1) + representation_noise_t
                #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0

                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
        return x, repre_list
    
    def forward(self, x, t, cfg, y, target_layer_start=None, target_layer_end=None, 
                    target_representation_layer_start=None, target_representation_layer_end=None,
                    finetune_representation=False, t_next=None, xt_next=None, ratio=None):
        if finetune_representation:
            return self.forward_run_layer_finetune(x, t, cfg, y, target_layer_start, target_layer_end, t_next=t_next, xt_next=xt_next, ratio=ratio)
        else:
            return self.forward_run_layer(x, t, cfg, y, target_layer_start, target_layer_end, target_representation_layer_start, target_representation_layer_end)
    
    def forward_run_layer(self, x, t, cfg, y, target_layer_start=None, target_layer_end=None, 
        target_representation_layer_start=None, target_representation_layer_end=None,
        t_next=None, representation_noise=None):
        assert target_layer_start is not None, "target_layer_start must be provided"
        assert target_layer_end is not None, "target_layer_end must be provided"
        assert len(self.blocks) >= target_layer_end, "target_layer_end must be within the range of the number of blocks"

        
        grid, mask, size = make_grid_mask_size_online(x, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
         
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

        if not self.use_sit:
            x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in

        x = self.x_embedder(x)

        linear_idx = 0
        for i in range(target_layer_start, target_layer_end):
            linear_idx += 1
            x = self.blocks[i](x, c, mask, freqs_cos, freqs_sin, global_adaln)
            if linear_idx == 6:
                representation_linear = self.linear_projection(x)

        x = self.final_layer(x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

        x = x * mask[..., None]                         # mask the padding tokens
        if not self.use_sit:
            x = rearrange(x, 'B N C -> B C N')
        
        return x, representation_linear, None
    
    def forward_run_layer_finetune(self, x, t, cfg, y, target_layer_start=None, target_layer_end=None, 
        target_representation_layer_start=None, target_representation_layer_end=None,
        t_next=None, xt_next=None, ratio=None):
        assert target_layer_start is not None, "target_layer_start must be provided"
        assert target_layer_end is not None, "target_layer_end must be provided"
        assert len(self.blocks) >= target_layer_end, "target_layer_end must be within the range of the number of blocks"

        grid, mask, size = make_grid_mask_size_online(x, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
         
        with torch.no_grad():
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            y = self.y_embedder(y, self.training)           # (B, D)

            t = t.float().to(x.dtype)
            t = self.t_embedder(t)
            c = t + y

            t_next = t_next.float().to(x.dtype)
            t_next = self.t_embedder(t_next)        
            c_next = t_next + y

            if self.global_adaLN_modulation != None:
                global_adaln = self.global_adaLN_modulation(c_next)
            else: 
                global_adaln = 0.0

            if self.number_of_representation_blocks > 1:
                representation_noise = self.representation_x_embedder(xt_next)
                for rep_block in self.representation_blocks:
                    if not self.use_checkpoint:
                        representation_noise = rep_block(representation_noise, c_next, mask, freqs_cos, freqs_sin, global_adaln)
                    else:
                        representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c_next, mask, freqs_cos, freqs_sin, global_adaln)
        
        ### residual approximation ###
        #ratio = ratio.float().to(x.dtype)
        #ratio = self.t_embedder(ratio)
        #c_repre = ratio.unsqueeze(1) + representation_noise
        x_mid = self.representation_x_embedder(x).detach()
        c_mid = t.unsqueeze(1) + representation_noise
        #c_mid = t + y
        #coefficient = self.coefficient_layers[target_layer_start//self.number_of_layers_for_perflow](t) 
        coefficient = ratio[:, None, None]
        #coefficient = self.coefficient_layers(x_mid.detach(), c_mid.detach())

        #import pdb; pdb.set_trace()
        #representation_noise_t = representation_noise.detach() + coefficient.unsqueeze(1) * self.mid_block[target_layer_start//self.number_of_layers_for_perflow](x_mid.detach(), c_repre.detach(), mask, freqs_cos, freqs_sin, 0.0)
        #representation_noise_t = representation_noise.detach() + coefficient.unsqueeze(1) * self.mid_block(x_mid.detach(), c_mid.detach(), mask, freqs_cos, freqs_sin, 0.0)
        for mid_block in self.mid_block:
            x_mid = mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
        #representation_noise_mid = representation_noise.clone().detach()
        #for mid_block in self.mid_block:
        #    representation_noise_mid = mid_block(representation_noise_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
        #x_mid = self.mid_block[target_layer_start//self.number_of_layers_for_perflow](x_mid.detach(), c_mid.detach(), mask, freqs_cos, freqs_sin, 0.0)
        #representation_noise_t = representation_noise + coefficient * x_mid
        representation_noise_t = x_mid
        representation_linear = self.linear_projection(representation_noise_t)
        ### residual approximation ###

        #x_pred = None
        #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise_t], dim=-1)
        c_repre = t.unsqueeze(1) + representation_noise_t

        if self.global_adaLN_modulation != None:
            global_adaln2 = self.global_adaLN_modulation2(c_repre)
        else: 
            global_adaln2 = 0.0

        x_pred = self.x_embedder(x)

        if not self.use_checkpoint:
            for i in range(target_layer_start, target_layer_end):
                x_pred = self.blocks[i](x_pred, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
        else:
            for i in range(target_layer_start, target_layer_end):
                x_pred = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(self.blocks[i]), x_pred, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)

        x_pred = self.final_layer(x_pred, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        x_pred = x_pred * mask[..., None]                         # mask the padding tokens
        

        with torch.no_grad():
            if self.global_adaLN_modulation != None:
                global_adaln2 = self.global_adaLN_modulation(c)
            else: 
                global_adaln2 = 0.0

            if self.number_of_representation_blocks > 1:
                representation_noise2 = self.representation_x_embedder(x)
                for rep_block in self.representation_blocks:
                    if not self.use_checkpoint:
                        representation_noise2 = rep_block(representation_noise2, c, mask, freqs_cos, freqs_sin, global_adaln2)
                    else:
                        representation_noise2 = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise2, c, mask, freqs_cos, freqs_sin, global_adaln2)
            representation_linear2 = self.linear_projection(representation_noise2)
            #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise2], dim=-1)
            c_repre = t.unsqueeze(1) + representation_noise2

            if self.global_adaLN_modulation != None:
                global_adaln2 = self.global_adaLN_modulation2(c_repre)
            else: 
                global_adaln2 = 0.0

            x_target = self.x_embedder(x)

            if not self.use_checkpoint:
                for i in range(target_layer_start, target_layer_end):
                    x_target = self.blocks[i](x_target, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
            else:
                for i in range(target_layer_start, target_layer_end):
                    x_target = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(self.blocks[i]), x_target, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)

            x_target = self.final_layer(x_target, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

            x_target = x_target * mask[..., None] 

        return x_pred, x_target, representation_linear, representation_linear2
        #return representation_noise_t, representation_noise2, representation_linear, x_pred
    

    def forward_run_layer_finetune3(self, x, t, cfg, y, target_layer_start=None, target_layer_end=None, 
        target_representation_layer_start=None, target_representation_layer_end=None,
        t_next=None, xt_next=None, ratio=None):
        assert target_layer_start is not None, "target_layer_start must be provided"
        assert target_layer_end is not None, "target_layer_end must be provided"
        assert len(self.blocks) >= target_layer_end, "target_layer_end must be within the range of the number of blocks"

        grid, mask, size = make_grid_mask_size_online(x, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
         
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        y = self.y_embedder(y, self.training)           # (B, D)

        with torch.no_grad():
            t = t.float().to(x.dtype)
            t2 = self.t_embedder(t)
            c = t2 + y

            t_next = t_next.float().to(x.dtype)
            t1 = self.t_embedder(t_next)        
            c_next = t1 + y

            x_target = xt_next.clone()
            t_list = [t_next, (t+t_next)/2, t]
            for j in range(2):
                t_cur = t_list[j]
                t_cur_embed = self.t_embedder(t_cur)
                c_cur = t_cur_embed + y
                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c_cur)
                else: 
                    global_adaln = 0.0

                if self.number_of_representation_blocks > 1:
                    representation_noise = self.representation_x_embedder(x_target)
                    for rep_block in self.representation_blocks:
                        if not self.use_checkpoint:
                            representation_noise = rep_block(representation_noise, c_cur, mask, freqs_cos, freqs_sin, global_adaln)
                        else:
                            representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c_next, mask, freqs_cos, freqs_sin, global_adaln)
                
                c_next = t_cur_embed.unsqueeze(1) + representation_noise

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_next)
                else: 
                    global_adaln2 = 0.0

                residual = x_target.clone()
                x_target = self.x_embedder(x_target)

                for i in range(target_layer_start, target_layer_end):
                    x_target = self.blocks[i](x_target, c_next, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x_target = self.final_layer(x_target, c_next)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
                #import pdb; pdb.set_trace()
                x_target = (t_list[j+1][:, None, None] - t_list[j][:, None, None]) * x_target + residual
        

        ### residual approximation ###
        x_mid = self.representation_x_embedder(x_target)
        #c_mid = t.unsqueeze(1) + x_mid
        c_mid = t2 + y
        coefficient = self.coefficient_layers[target_layer_start//self.number_of_layers_for_perflow](x_mid.detach(), c_mid.detach())

        representation_noise_mid = self.mid_block[target_layer_start//self.number_of_layers_for_perflow](x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
        representation_noise_t = (1 - coefficient) * representation_noise + coefficient * representation_noise_mid
        representation_linear = self.linear_projection(representation_noise_t)
        ### residual approximation ###

        with torch.no_grad():
            if self.global_adaLN_modulation != None:
                global_adaln2 = self.global_adaLN_modulation(c)
            else: 
                global_adaln2 = 0.0

            if self.number_of_representation_blocks > 1:
                representation_noise2 = self.representation_x_embedder(x_target)
                for rep_block in self.representation_blocks:
                    if not self.use_checkpoint:
                        representation_noise2 = rep_block(representation_noise2, c, mask, freqs_cos, freqs_sin, global_adaln2)
                    else:
                        representation_noise2 = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise2, c, mask, freqs_cos, freqs_sin, global_adaln2)

        return representation_noise_t, representation_noise2, representation_linear
    
    def forward_run_layer_finetune2(self, x, t, cfg, y, target_layer_start=None, target_layer_end=None, 
        target_representation_layer_start=None, target_representation_layer_end=None,
        t_next=None, xt_next=None, ratio=None):
        assert target_layer_start is not None, "target_layer_start must be provided"
        assert target_layer_end is not None, "target_layer_end must be provided"
        assert len(self.blocks) >= target_layer_end, "target_layer_end must be within the range of the number of blocks"

        grid, mask, size = make_grid_mask_size_online(x, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
         
        with torch.no_grad():
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            y = self.y_embedder(y, self.training)           # (B, D)

            t = t.float().to(x.dtype)
            t = self.t_embedder(t)
            c = t + y

            t_next = t_next.float().to(x.dtype)
            t_next = self.t_embedder(t_next)        
            c_next = t_next + y

            if self.global_adaLN_modulation != None:
                global_adaln = self.global_adaLN_modulation(c_next)
            else: 
                global_adaln = 0.0

            if self.number_of_representation_blocks > 1:
                representation_noise = self.representation_x_embedder(xt_next)
                for rep_block in self.representation_blocks:
                    if not self.use_checkpoint:
                        representation_noise = rep_block(representation_noise, c_next, mask, freqs_cos, freqs_sin, global_adaln)
                    else:
                        representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c_next, mask, freqs_cos, freqs_sin, global_adaln)
        
        ### residual approximation ###
        #ratio = ratio.float().to(x.dtype)
        #ratio = self.t_embedder(ratio)
        #c_repre = ratio.unsqueeze(1) + representation_noise
        x_mid = self.representation_x_embedder(x).detach()
        #c_mid = t.unsqueeze(1) + x_mid
        c_mid = t + y
        coefficient = self.coefficient_layers[target_layer_start//self.number_of_layers_for_perflow](t) 
        #coefficient = ratio[:, None, None]

        #import pdb; pdb.set_trace()
        #representation_noise_t = representation_noise.detach() + coefficient.unsqueeze(1) * self.mid_block[target_layer_start//self.number_of_layers_for_perflow](x_mid.detach(), c_repre.detach(), mask, freqs_cos, freqs_sin, 0.0)
        #representation_noise_t = representation_noise.detach() + coefficient.unsqueeze(1) * self.mid_block(x_mid.detach(), c_mid.detach(), mask, freqs_cos, freqs_sin, 0.0)
        for mid_block in self.mid_block:
           x_mid = mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
        #representation_noise_mid = representation_noise.clone().detach()
        #for mid_block in self.mid_block:
        #    representation_noise_mid = mid_block(representation_noise_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
        #x_mid = self.mid_block[target_layer_start//self.number_of_layers_for_perflow](x_mid.detach(), c_mid.detach(), mask, freqs_cos, freqs_sin, 0.0)
        representation_noise_t = representation_noise.detach() + coefficient.unsqueeze(1) * x_mid
        representation_linear = self.linear_projection(representation_noise_t)
        ### residual approximation ###        

        with torch.no_grad():
            if self.global_adaLN_modulation != None:
                global_adaln2 = self.global_adaLN_modulation(c)
            else: 
                global_adaln2 = 0.0

            if self.number_of_representation_blocks > 1:
                representation_noise2 = self.representation_x_embedder(x)
                for rep_block in self.representation_blocks:
                    if not self.use_checkpoint:
                        representation_noise2 = rep_block(representation_noise2, c, mask, freqs_cos, freqs_sin, global_adaln2)
                    else:
                        representation_noise2 = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise2, c, mask, freqs_cos, freqs_sin, global_adaln2)
            representation_linear2 = self.linear_projection(representation_noise2)
            #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise2], dim=-1)
        return representation_linear, representation_linear2
    
    def forward_cfg(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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

        #assert cfg > 1, "cfg must be greater than 1"
        y_null = torch.tensor([self.num_classes] * x.shape[0], device=x.device)
        y = torch.cat([y, y_null], dim=0)

        y_embed = self.y_embedder(y, self.training)           # (B, D)
        grid, mask, size = make_grid_mask_size(x.shape[0], self.n_patch_h, self.n_patch_w, self.patch_size, x.device)
        size = torch.cat([size, size], dim=0)
        grid = torch.cat([grid, grid], dim=0)
        mask = torch.cat([mask, mask], dim=0)

        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        
        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
            sigma_next = self.sigmas[i+1]
            sigma_current = self.sigmas[i]
            sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)

            for step in range(number_of_step_perflow):
                t = sigma_list[step].expand(x.shape[0]*2).to(x.device)
                t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
                t = t.float().to(x.dtype)
                
                t = self.t_embedder(t)
                c = t + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                residual = x.clone()
                x = torch.cat([x, x], dim=0)

                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in

                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln)

                x = self.final_layer(x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                
                x_cond, x_uncond = x.chunk(2, dim=0)
                x = x_uncond + cfg * (x_cond - x_uncond)
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
        return x
    

    def forward_maruyama(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        # get RoPE frequences in advance, then calculate attention.

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
            if self.perlayer_embedder:
                y_embed = self.y_embedder[i](y, self.training)           # (B, D)
            else:
                y_embed = self.y_embedder(y, self.training)           # (B, D)

            if self.online_rope:    
                freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
                freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            else:
                freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
                freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

            if i == len(self.blocks) // self.number_of_layers_for_perflow - 1:
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
                if self.perlayer_embedder:
                    t = self.t_embedder[i](t)
                else:
                    t = self.t_embedder(t)
                c = t + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                if self.number_of_representation_blocks > 1:
                    representation_noise = self.representation_x_embedder(x)
                    #import pdb; pdb.set_trace()
                    for rep_block in self.representation_blocks:
                        if not self.use_checkpoint:
                            representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                        else:
                            representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                c_repre = t.unsqueeze(1) + representation_noise
                #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0

                x_prev = x.clone()
                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
                
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                
                dt = sigma_list[step+1] - sigma_list[step]
                w_cur = torch.randn_like(x)
                t = torch.ones_like(x) * sigma_list[step]
                dw = w_cur * torch.sqrt(dt)
                
                score = (t * x - x_prev) / (1 - t)
                drift = x + (1-t) * score
                diffusion = (1-t)

                if (i == len(self.blocks) // self.number_of_layers_for_perflow - 1) and (step == number_of_step_perflow - 1):
                    x = x_prev + drift*dt
                else:
                    x = x_prev + drift*dt
                    x = x + torch.sqrt(2 * diffusion) * dw
        return x
    
    def forward_maruyama_cfg(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        # get RoPE frequences in advance, then calculate attention.

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
            if self.perlayer_embedder:
                y_embed = self.y_embedder[i](y, self.training)           # (B, D)
            else:
                y_embed = self.y_embedder(y, self.training)           # (B, D)

            if self.online_rope:    
                freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
                freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            else:
                freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
                freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

            if i == len(self.blocks) // self.number_of_layers_for_perflow - 1:
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
                if self.perlayer_embedder:
                    t = self.t_embedder[i](t)
                else:
                    t = self.t_embedder(t)
                c = t + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                if self.number_of_representation_blocks > 1:
                    representation_noise = self.representation_x_embedder(x)
                    #import pdb; pdb.set_trace()
                    for rep_block in self.representation_blocks:
                        if not self.use_checkpoint:
                            representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                        else:
                            representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                c_repre = t.unsqueeze(1) + representation_noise
                #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0

                x_prev = x.clone()
                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
                
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                
                dt = sigma_list[step+1] - sigma_list[step]
                w_cur = torch.randn_like(x)
                t = torch.ones_like(x) * sigma_list[step]
                dw = w_cur * torch.sqrt(dt)
                
                score = (t * x - x_prev) / (1 - t)
                drift = x + (1-t) * score
                diffusion = (1-t)

                if (i == len(self.blocks) // self.number_of_layers_for_perflow - 1) and (step == number_of_step_perflow - 1):
                    x = x_prev + drift*dt
                else:
                    x = x_prev + drift*dt
                    x = x + torch.sqrt(2 * diffusion) * dw
        return x
    
    def forward_maruyama_int(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        # get RoPE frequences in advance, then calculate attention.

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
            if self.perlayer_embedder:
                y_embed = self.y_embedder[i](y, self.training)           # (B, D)
            else:
                y_embed = self.y_embedder(y, self.training)           # (B, D)

            if self.online_rope:    
                freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
                freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            else:
                freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
                freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

            if i == len(self.blocks) // self.number_of_layers_for_perflow - 1:
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
                t_rep = self.t_embedder(t)
                c = t_rep + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                if step == 0:
                    if self.number_of_representation_blocks > 1:
                        representation_noise = self.representation_x_embedder(x)
                        for rep_block in self.representation_blocks:
                            if not self.use_checkpoint:
                                representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                            else:
                                representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)

                x_prev = x.clone()
                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
                
                x_mid = self.representation_x_embedder(x)
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  

                if step != 0:
                #if 0:
                    c_mid = t_rep.unsqueeze(1) + x_mid
                    coefficient = self.coefficient_layers[i](t_rep)
                    #coefficient = self.coefficient_layers[i](x_mid.detach(), c_mid.detach())
                    #c_mid = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)
                    #for mid_block in self.mid_block:
                    #    x_mid = mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    representation_noise_mid = self.mid_block[i](representation_noise, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    representation_noise_t = representation_noise + coefficient.unsqueeze(1) * representation_noise_mid
                    #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * representation_noise_mid
                    #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block[i](x_mid, c_mid)
                    # for mid_block in self.mid_block:
                    #     x_mid = mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    # representation_noise_t = representation_noise + coefficient.unsqueeze(1) * x_mid
                else:
                    representation_noise_t = representation_noise
                
                c_repre = t_rep.unsqueeze(1) + representation_noise_t
                #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0
                
                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                
                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                
                dt = sigma_list[step+1] - sigma_list[step]
                w_cur = torch.randn_like(x)
                t = torch.ones_like(x) * sigma_list[step]
                dw = w_cur * torch.sqrt(dt)
                
                score = (t * x - x_prev) / (1 - t)
                drift = x + (1-t) * score
                diffusion = (1-t)

                if (i == len(self.blocks) // self.number_of_layers_for_perflow - 1) and (step == number_of_step_perflow - 1):
                    x = x_prev + drift*dt
                else:
                    x = x_prev + drift*dt
                    x = x + torch.sqrt(2 * diffusion) * dw
        return x

    def forward_cfg_int(self, x, t, cfg, y, number_of_step_perflow=1, noise=None, representation_noise=None, int=2):
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

        #assert cfg > 1, "cfg must be greater than 1"
        y_null = torch.tensor([self.num_classes] * x.shape[0], device=x.device)
        y = torch.cat([y, y_null], dim=0)

        y_embed = self.y_embedder(y, self.training)           # (B, D)
        #y_embed_rep = self.y_embedder_rep(y, self.training)           # (B, D)
        grid, mask, size = make_grid_mask_size(x.shape[0], self.n_patch_h, self.n_patch_w, self.patch_size, x.device)
        size = torch.cat([size, size], dim=0)
        grid = torch.cat([grid, grid], dim=0)
        mask = torch.cat([mask, mask], dim=0)

        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        
        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
            sigma_next = self.sigmas[i+1]
            sigma_current = self.sigmas[i]
            sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)
            sigma_list_ratio = (sigma_list - sigma_current) / (sigma_next - sigma_current)

            for step in range(number_of_step_perflow):
                t = sigma_list[step].expand(x.shape[0]*2).to(x.device)
                t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
                t = t.float().to(x.dtype)

                t = self.t_embedder(t)
                c = t + y_embed

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                residual = x.clone()
                x = torch.cat([x, x], dim=0)

                if step == 0:
                    if self.number_of_representation_blocks > 1:
                        representation_noise = self.representation_x_embedder(x)
                        for rep_block in self.representation_blocks:
                            if not self.use_checkpoint:
                                representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                            else:
                                representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)

                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
                
                x_mid = self.representation_x_embedder(x)
                x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  

                if step != 0:
                #if 0:
                    ratio = sigma_list_ratio[step].expand(x.shape[0]).to(x.device)
                    #ratio = ratio.float().to(x.dtype)
                    #ratio = self.t_embedder(ratio)

                    c_mid = t.unsqueeze(1) + representation_noise
                    #c_mid = t + y_embed
                    #coefficient = self.coefficient_layers[i](t)
                    #coefficient = self.coefficient_layers(x_mid.detach(), c_mid.detach())
                    coefficient = ratio[:, None, None]
                    #c_mid = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)
                    for mid_block in self.mid_block:
                       x_mid = mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    #representation_noise_mid = representation_noise.clone().detach()
                    #for mid_block in self.mid_block:
                    #    representation_noise_mid = mid_block(representation_noise_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    #representation_noise_mid = self.mid_block[i](x_mid.detach(), c_mid.detach(), mask, freqs_cos, freqs_sin, 0.0)
                    #x_mid = self.mid_block[i](x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
                    #representation_noise_t = representation_noise + coefficient * x_mid
                    representation_noise_t = x_mid
                else:
                    representation_noise_t = representation_noise

                #c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise_t], dim=-1)
                c_repre = t.unsqueeze(1) + representation_noise_t

                if self.global_adaLN_modulation != None:
                    global_adaln2 = self.global_adaLN_modulation2(c_repre)
                else: 
                    global_adaln2 = 0.0
                
                if self.use_checkpoint:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)
                else:
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c_repre, mask, freqs_cos, freqs_sin, global_adaln2)

                x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                
                x_cond, x_uncond = x.chunk(2, dim=0)
                x = x_uncond + cfg * (x_cond - x_uncond)
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
        return x

    def forward_run_layer_test(self, x, t, cfg, y, target_layer_start=None, target_layer_end=None, 
        target_representation_layer_start=None, target_representation_layer_end=None,
        t_next=None, representation_noise=None, return_dict=False):
        assert target_layer_start is not None, "target_layer_start must be provided"
        assert target_layer_end is not None, "target_layer_end must be provided"
        assert len(self.blocks) >= target_layer_end, "target_layer_end must be within the range of the number of blocks"
        
        grid, mask, size = make_grid_mask_size_online(x, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
         
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

        if self.number_of_representation_blocks > 1:
            representation_noise = self.representation_x_embedder(x)
            for rep_block in self.representation_blocks:
                if not self.use_checkpoint:
                    representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
                else:
                    representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
            
        return representation_noise
    
    def forward_run_layer_int_test(self, x, t, cfg, y, t_next=None, first_index=False, layer_index=None, 
                                  first_input=None, t_first_input=None):
        grid, mask, size = make_grid_mask_size_online(x, self.patch_size, x.device)
        # get RoPE frequences in advance, then calculate attention.
         
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        
        y = self.y_embedder(y, self.training)
        t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
        t = t.float().to(x.dtype)
        t = self.t_embedder(t)        
        c = t + y
        
        if not first_index:
            t_first_input = t_first_input.float().to(x.dtype)
            t_first_input = self.t_embedder(t_first_input)
            c = t_first_input + y
            if self.global_adaLN_modulation != None:
                global_adaln = self.global_adaLN_modulation(c)
            else: 
                global_adaln = 0.0
            representation_noise = self.representation_x_embedder(first_input)
        else:
            if self.global_adaLN_modulation != None:
                global_adaln = self.global_adaLN_modulation(c)
            else: 
                global_adaln = 0.0
            representation_noise = self.representation_x_embedder(x)
        
        for rep_block in self.representation_blocks:
            if not self.use_checkpoint:
                representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
            else:
                representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin, global_adaln)
        
        if not first_index:
            x_mid = self.representation_x_embedder(x)
            #t_next = t_next.float().to(x.dtype)
            #ratio = self.t_embedder(t_next)
            c_mid = t.unsqueeze(1) + representation_noise
            coefficient = self.coefficient_layers[layer_index](x_mid.detach(), c_mid.detach())
            #c_mid = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)
            representation_noise_t = representation_noise + coefficient * self.mid_block[layer_index](x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
            #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block(x_mid, c_mid, mask, freqs_cos, freqs_sin, 0.0)
            #representation_noise_t = representation_noise + coefficient.unsqueeze(1) * self.mid_block[i](x_mid, c_mid)
        else:
            representation_noise_t = representation_noise
        
        return representation_noise_t

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