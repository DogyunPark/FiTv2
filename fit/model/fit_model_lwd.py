import torch
import torch.nn as nn
from functools import partial
from typing import Optional
from einops import rearrange, repeat
from fit.model.modules_lwd import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder,
    FiTBlock, FinalLayer, RepresentationBlock
)
from fit.model.utils import get_parameter_dtype
from fit.utils.eval_utils import init_from_ckpt
#from fit.model.sincos import get_2d_sincos_pos_embed_from_grid
from fit.model.rope import VisionRotaryEmbedding
#from fit.utils.utils import linear_increase_division
#################################################################################
#                                 Core FiT Model                                #
#################################################################################



class FiTLwD(nn.Module):
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
        self.sigmas_overlap = self.sigmas - 1/(number_of_perflow*5)
        self.overlap = overlap
        self.perlayer_embedder = perlayer_embedder
        self.number_of_perflow = number_of_perflow
        self.number_of_layers_for_perflow = depth // number_of_perflow
        self.number_of_shared_blocks = number_of_shared_blocks
        self.number_of_representation_blocks = number_of_representation_blocks

        if number_of_representation_blocks > 1:
            self.representation_x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
            self.representation_blocks = nn.ModuleList([RepresentationBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
                rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type, 
                q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
                adaln_bias=adaln_bias, adaln_type='lora', adaln_lora_dim=adaln_lora_dim
            ) for _ in range(number_of_representation_blocks)])
            # self.linear_projection = nn.Sequential(
            #         nn.Linear(hidden_size, 2048),
            #         nn.SiLU(),
            #         nn.Linear(2048, 2048),
            #         nn.SiLU(),
            #         nn.Linear(2048, 1024),
            #     )
            self.linear_projection_cls = nn.Sequential(
                    nn.Linear(hidden_size, 2048),
                    nn.SiLU(),
                    nn.Linear(2048, 2048),
                    nn.SiLU(),
                    nn.Linear(2048, 1024),
                )
            # self.linear_projection_jepa = nn.Sequential(
            #         nn.Linear(hidden_size, 2048),
            #         nn.SiLU(),
            #         nn.Linear(2048, 2048),
            #         nn.SiLU(),
            #         nn.Linear(2048, 1280),
            #     )

        if number_of_shared_blocks > 0:
            self.start_shared_blocks = nn.ModuleList([FiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
                rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type, 
                q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias, 
                adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim
            ) for _ in range(number_of_shared_blocks)])

        if perlayer_embedder:
            self.x_embedder = nn.ModuleList([PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True) for _ in range(number_of_perflow)])
        else:
            self.x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        
        # if perlayer_embedder:
        #     self.t_embedder = nn.ModuleList([TimestepEmbedder(hidden_size) for _ in range(number_of_perflow)])
        # else:
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # if perlayer_embedder:
        #     self.y_embedder = nn.ModuleList([LabelEmbedder(num_classes, hidden_size, class_dropout_prob) for _ in range(number_of_perflow)])
        # else:
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        if fourier_basis:
            self.fourier_basis = TimestepEmbedder(patch_size*patch_size*self.out_channels*2)
        else:
            self.fourier_basis = None
        
        self.rel_pos_embed = VisionRotaryEmbedding(
            head_dim=hidden_size//num_heads, theta=rope_theta, custom_freqs=custom_freqs, online_rope=online_rope,
            max_pe_len_h=max_pe_len_h, max_pe_len_w=max_pe_len_w, decouple=decouple, ori_max_pe_len=ori_max_pe_len,
            max_cached_len=max_cached_len,
        )
        
        if adaln_type == 'lora':
            self.global_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size*2, 6 * hidden_size, bias=adaln_bias)
            )
        else:
            self.global_adaLN_modulation = None        
        
        self.blocks = nn.ModuleList([FiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type, 
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias, 
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim
        ) for _ in range(depth)])

        final_layer_out_channels = self.out_channels*2 if fourier_basis else self.out_channels
        if perlayer_embedder:
            self.final_layer = nn.ModuleList([FinalLayer(hidden_size, patch_size, final_layer_out_channels, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type) for _ in range(number_of_perflow)])
        else:
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

        if self.number_of_representation_blocks > 1:
            nn.init.xavier_uniform_(self.representation_x_embedder.proj.weight.data)
            nn.init.constant_(self.representation_x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.perlayer_embedder:
            for i, embedder in enumerate(self.x_embedder):
                w = embedder.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(embedder.proj.bias, 0)
        else:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.perlayer_embedder:
        #     for i, embedder in enumerate(self.y_embedder):
        #         nn.init.normal_(embedder.embedding_table.weight, std=0.02)
        # else:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # if self.perlayer_embedder:
        #     for i, embedder in enumerate(self.t_embedder):
        #         nn.init.normal_(embedder.mlp[0].weight, std=0.02)
        #         nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        # else:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.fourier_basis is not None:
            nn.init.normal_(self.fourier_basis.mlp[0].weight, std=0.02)
            nn.init.normal_(self.fourier_basis.mlp[2].weight, std=0.02)

        #nn.init.normal_(self.c_embedder.mlp[0].weight, std=0.02)
        #nn.init.normal_(self.c_embedder.mlp[2].weight, std=0.02)

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
        if self.perlayer_embedder:
            for i, final_layer in enumerate(self.final_layer):
                nn.init.constant_(final_layer.linear.weight, 0)
                nn.init.constant_(final_layer.linear.bias, 0)
        else:
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

    def forward(self, x, t, cfg, y, grid, mask, size=None, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        cfg = cfg.float().to(x.dtype) 
        
        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        for i in range(len(self.blocks) // self.number_of_layers_for_perflow):
            # i_mod = i // number_of_layers_for_perflow
            # i_drop = i % number_of_layers_for_perflow
            # i_finish = (i+1) % number_of_layers_for_perflow
            # if self.perlayer_embedder:
            #     y_embed = self.y_embedder[i](y, self.training)           # (B, D)
            # else:
            y_embed = self.y_embedder(y, self.training)           # (B, D)

            sigma_next = self.sigmas[i+1] 
            if self.overlap:
                if i == 0:
                    sigma_current = self.sigmas[i]
                else:
                    sigma_current = self.sigmas_overlap[i]
            else:
                sigma_current = self.sigmas[i]
            sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)

            for step in range(number_of_step_perflow):
                #if i_drop == 0:
                t = sigma_list[step].expand(x.shape[0]).to(x.device)
                t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
                t = t.float().to(x.dtype)
                # if self.perlayer_embedder:
                #     t = self.t_embedder[i](t)
                # else:
                t = self.t_embedder(t)
                c = t + y_embed

                if self.fourier_basis is not None:
                    t_next = self.sigmas[i+1].expand(x.shape[0]).to(x.device)
                    t_next = torch.clamp(self.time_shifting * t_next / (1  + (self.time_shifting - 1) * t_next), max=1.0)        
                    t_next = t_next.float().to(x.dtype)
                    basis = self.fourier_basis(t_next)
                    cos_basis, sin_basis = basis.chunk(2, dim=-1)
                    cos_basis = cos_basis.unsqueeze(1).expand(-1, x.shape[1], -1)
                    sin_basis = sin_basis.unsqueeze(1).expand(-1, x.shape[1], -1)
                
                if self.number_of_representation_blocks > 1:
                    #assert representation_noise is not None, "representation_noise must be provided when representation_blocks > 1"
                    representation_noise = self.representation_x_embedder(x)
                    for rep_block in self.representation_blocks:
                        if not self.use_checkpoint:
                            representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin)
                        else:
                            representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin)
                    
                    
                    representation_noise_mean = torch.mean(representation_noise, dim=1)
                    c = torch.cat([c, representation_noise_mean], dim=1)
                    #c = c + representation_noise_mean

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                residual = x.clone()
                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
                if self.perlayer_embedder:
                    x = self.x_embedder[i](x)                          # (B, N, C) -> (B, N, D)  
                else:
                    x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  

                # if self.number_of_representation_blocks > 1:
                #     x += representation_noise
                
                if self.use_checkpoint:
                    if self.number_of_shared_blocks > 0:
                        for block in self.start_shared_blocks:
                            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
                else:
                    if self.number_of_shared_blocks > 0:
                        for block in self.start_shared_blocks:
                            x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln)
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                #if i_finish == 0:
                if self.perlayer_embedder:
                    x = self.final_layer[i](x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
                else:
                    x = self.final_layer(x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                if self.fourier_basis is not None:
                    coeff_cos, coeff_sin = x.chunk(2, dim=-1)
                    x = coeff_cos * cos_basis + coeff_sin * sin_basis

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual
            
            if self.overlap and i != len(self.blocks) // self.number_of_layers_for_perflow - 1:
                if noise is not None:
                    add_noise = noise[i]
                else:
                    add_noise = torch.randn_like(x)
                x = (self.sigmas_overlap[i+1] / self.sigmas[i+1]) * x + (((1-self.sigmas_overlap[i+1])**2 - ((1-self.sigmas[i+1])**2 * (self.sigmas_overlap[i+1])**2 / (self.sigmas[i+1])**2))).sqrt() * add_noise
        return x
    
    def forward_run_layer(self, x, t, cfg, y, grid, mask, size=None, target_layer_start=None, target_layer_end=None, t_next=None, representation_noise=None):
        assert target_layer_start is not None, "target_layer_start must be provided"
        assert target_layer_end is not None, "target_layer_end must be provided"
        assert len(self.blocks) >= target_layer_end, "target_layer_end must be within the range of the number of blocks"

        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
        t = t.float().to(x.dtype)
        cfg = cfg.float().to(x.dtype)
        # if self.perlayer_embedder:
        #     t = self.t_embedder[target_layer_start // self.number_of_layers_for_perflow](t)        
        # else:
        t = self.t_embedder(t)        
        #cfg = self.c_embedder(cfg)
        # if self.perlayer_embedder:
        #     y = self.y_embedder[target_layer_start // self.number_of_layers_for_perflow](y, self.training)           # (B, D)
        # else:
        y = self.y_embedder(y, self.training)           # (B, D)
        c = t + y

        if self.fourier_basis is not None:
            assert t_next is not None, "t_next must be provided when fourier_basis is True"
            basis = self.fourier_basis(t_next)
            cos_basis, sin_basis = basis.chunk(2, dim=-1)
            cos_basis = cos_basis.unsqueeze(1).expand(-1, x.shape[1], -1)
            sin_basis = sin_basis.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        if self.number_of_representation_blocks > 1:
            #assert representation_noise is not None, "representation_noise must be provided when representation_blocks > 1"
            representation_noise = self.representation_x_embedder(x)
            for rep_block in self.representation_blocks:
                if not self.use_checkpoint:
                    representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin)
                else:
                    representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin)
            
            #if 1:
            #representation_linear = self.linear_projection(representation_noise)
            #representation_linear_jepa = self.linear_projection_jepa(representation_noise)
            representation_noise_mean = torch.mean(representation_noise, dim=1)
            representation_linear_cls = self.linear_projection_cls(representation_noise_mean)
            drop_ids = torch.rand(x.shape[0], device=x.device) < 0.1
            # Replace drop_ids of representation_noise_mean with zeros
            representation_noise_mean = torch.where(drop_ids[:, None], 0, representation_noise_mean)
            c = torch.cat([c, representation_noise_mean], dim=1)
            #c = c + representation_noise_mean
        
        if self.global_adaLN_modulation != None:
            global_adaln = self.global_adaLN_modulation(c)
        else: 
            global_adaln = 0.0

        if not self.use_sit:
            x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
        if self.perlayer_embedder:
            x = self.x_embedder[target_layer_start // self.number_of_layers_for_perflow](x)                          # (B, N, C) -> (B, N, D)  
        else:
            x = self.x_embedder(x)
        
        # if self.number_of_representation_blocks > 1:
        #     x += representation_noise

        if not self.use_checkpoint:
            if self.number_of_shared_blocks > 0:
                for block in self.start_shared_blocks:
                    x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln)
            for i in range(target_layer_start, target_layer_end):
                x = self.blocks[i](x, c, mask, freqs_cos, freqs_sin, global_adaln)
        else:
            if self.number_of_shared_blocks > 0:
                for block in self.start_shared_blocks:
                    x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
            for i in range(target_layer_start, target_layer_end):
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(self.blocks[i]), x, c, mask, freqs_cos, freqs_sin, global_adaln)
        
        if self.perlayer_embedder:
            x = self.final_layer[target_layer_start // self.number_of_layers_for_perflow](x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        else:
            x = self.final_layer(x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

        if self.fourier_basis is not None:
            coeff_cos, coeff_sin = x.chunk(2, dim=-1)
            x = coeff_cos * cos_basis + coeff_sin * sin_basis

        x = x * mask[..., None]                         # mask the padding tokens
        if not self.use_sit:
            x = rearrange(x, 'B N C -> B C N')
        
        if self.number_of_representation_blocks > 1:
            return x, representation_linear_cls, representation_linear_cls, representation_linear_cls
        else:
            return x, None, None, None
    
    def forward_run_layer_from_target_layer(self, x, t_input, cfg, y, grid, mask, size=None, 
            target_start_idx=None, target_end_idx=None, number_of_step_perflow=1, return_all_layers=False, noise=None):
        assert target_start_idx is not None, "target_start_idx must be provided"
        target_end_idx = target_end_idx if target_end_idx is not None else (len(self.blocks)+1)
        sigmas_ = self.sigmas[target_start_idx:target_end_idx+1]
        sigmas_overlap_ = self.sigmas_overlap[target_start_idx:target_end_idx+1]
        
        intermediate_layers = []
        y = self.y_embedder(y, self.training)           # (B, D)
        
        # get RoPE frequences in advance, then calculate attention.
        if self.online_rope:    
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        
        for i in range(len(sigmas_)-1):
            sigma_next = sigmas_[i+1]
            if self.overlap:
                sigma_current = sigmas_overlap_[i]
            else:
                sigma_current = sigmas_[i]
            sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)

            for step in range(number_of_step_perflow):
                #if i_drop == 0:
                if i != 0:
                    t_input = sigma_list[step].expand(x.shape[0]).to(x.device)
                t = torch.clamp(self.time_shifting * t_input / (1  + (self.time_shifting - 1) * t_input), max=1.0)        
                t = t.float().to(x.dtype)
                t = self.t_embedder(t)
                c = t + y
                
                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                residual = x.clone()
                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
                if self.perlayer_embedder:
                    x = self.x_embedder[target_start_idx+i](x)                          # (B, N, C) -> (B, N, D)  
                else:
                    x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  

                if self.use_checkpoint:
                    for block in self.blocks[(target_start_idx + i)*self.number_of_layers_for_perflow: (target_start_idx + i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
                else:
                    for block in self.blocks[(target_start_idx + i)*self.number_of_layers_for_perflow: (target_start_idx + i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln)

                if self.perlayer_embedder:
                    x = self.final_layer[target_start_idx+i](x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
                else:
                    x = self.final_layer(x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                
                if return_all_layers:
                    intermediate_layers.append(x)

                x = (sigma_list[step+1].expand(x.shape[0]).to(x.device)[:, None, None] - t_input[:, None, None]) * x + residual

            if self.overlap and i != len(sigmas_) - 2:
                x = (self.sigmas_overlap[i+1] / self.sigmas[i+1]) * x + (((1-self.sigmas_overlap[i+1])**2 - ((1-self.sigmas[i+1])**2 * (self.sigmas_overlap[i+1])**2 / (self.sigmas[i+1])**2))).sqrt() * torch.randn_like(x)
        if return_all_layers:
            return x, intermediate_layers
        else:   
            return x
    
    def forward_cfg(self, x, t, cfg, y, grid, mask, size=None, number_of_step_perflow=1, noise=None, representation_noise=None):
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
        y = torch.cat([y, y_null], dim=0)
        # if self.perlayer_embedder:
        #     y_embed = self.y_embedder[0](y, self.training)           # (B, D)
        # else:
        y_embed = self.y_embedder(y, self.training)           # (B, D)
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
            if self.overlap:
                if i == 0:
                    sigma_current = self.sigmas[i]
                else:
                    sigma_current = self.sigmas_overlap[i]
            else:
                sigma_current = self.sigmas[i]
            sigma_list = torch.linspace(sigma_current, sigma_next, number_of_step_perflow+1)

            for step in range(number_of_step_perflow):
                t = sigma_list[step].expand(x.shape[0]*2).to(x.device)
                t = torch.clamp(self.time_shifting * t / (1  + (self.time_shifting - 1) * t), max=1.0)        
                t = t.float().to(x.dtype)
                # if self.perlayer_embedder:
                #     t = self.t_embedder[i](t)
                # else:
                t = self.t_embedder(t)
                c = t + y_embed

                if self.fourier_basis is not None:
                    t_next = self.sigmas[i+1].expand(x.shape[0]).to(x.device)
                    t_next = torch.clamp(self.time_shifting * t_next / (1  + (self.time_shifting - 1) * t_next), max=1.0)        
                    t_next = t_next.float().to(x.dtype)
                    basis = self.fourier_basis(t_next)
                    cos_basis, sin_basis = basis.chunk(2, dim=-1)
                    cos_basis = cos_basis.unsqueeze(1).expand(-1, x.shape[1], -1)
                    sin_basis = sin_basis.unsqueeze(1).expand(-1, x.shape[1], -1)

                residual = x.clone()
                x = torch.cat([x, x], dim=0)

                if self.number_of_representation_blocks > 1:
                    #assert representation_noise is not None, "representation_noise must be provided when representation_blocks > 1"
                    representation_noise = self.representation_x_embedder(x)
                    for rep_block in self.representation_blocks:
                        if not self.use_checkpoint:
                            representation_noise = rep_block(representation_noise, c, mask, freqs_cos, freqs_sin)
                        else:
                            representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, mask, freqs_cos, freqs_sin)
                    
                    representation_noise_mean = torch.mean(representation_noise, dim=1)
                    representation_noise_mean, _ = representation_noise_mean.chunk(2, dim=0)
                    representation_noise_mean = torch.cat([representation_noise_mean, torch.zeros_like(representation_noise_mean)], dim=0)
                    c = torch.cat([c, representation_noise_mean], dim=1)
                    #c = c + representation_noise_mean

                if self.global_adaLN_modulation != None:
                    global_adaln = self.global_adaLN_modulation(c)
                else: 
                    global_adaln = 0.0

                if not self.use_sit:
                    x = rearrange(x, 'B C N -> B N C')          # (B, C, N) -> (B, N, C), where C = p**2 * C_in
                if self.perlayer_embedder:
                    x = self.x_embedder[i](x)                          # (B, N, C) -> (B, N, D)  
                else:
                    x = self.x_embedder(x)                          # (B, N, C) -> (B, N, D)  
                
                # if self.number_of_representation_blocks > 1:
                #     x += representation_noise

                if self.use_checkpoint:
                    if self.number_of_shared_blocks > 0:
                        for block in self.start_shared_blocks:
                            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln)
                else:
                    if self.number_of_shared_blocks > 0:
                        for block in self.start_shared_blocks:
                            x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln)
                    for block in self.blocks[i*self.number_of_layers_for_perflow: (i+1)*self.number_of_layers_for_perflow]:
                        x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln)
                
                if self.perlayer_embedder:
                    x = self.final_layer[i](x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
                else:
                    x = self.final_layer(x, c)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.

                if self.fourier_basis is not None:
                    coeff_cos, coeff_sin = x.chunk(2, dim=-1)
                    x = coeff_cos * cos_basis + coeff_sin * sin_basis

                x = x * mask[..., None]                         # mask the padding tokens
                if not self.use_sit:
                    x = rearrange(x, 'B N C -> B C N')          # (B, N, C) -> (B, C, N), where C = p**2 * C_out
                
                x_cond, x_uncond = x.chunk(2, dim=0)
                x = x_uncond + cfg * (x_cond - x_uncond)
                x = (sigma_list[step+1] - sigma_list[step]) * x + residual

            if self.overlap and i != len(self.blocks) // self.number_of_layers_for_perflow - 1:
                if noise is not None:
                    add_noise = noise[i]
                else:
                    add_noise = torch.randn_like(x)
                x = (self.sigmas_overlap[i+1] / self.sigmas[i+1]) * x + (((1-self.sigmas_overlap[i+1])**2 - ((1-self.sigmas[i+1])**2 * (self.sigmas_overlap[i+1])**2 / (self.sigmas[i+1])**2))).sqrt() * add_noise
        return x
        
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
                    param.requires_grad = True
        