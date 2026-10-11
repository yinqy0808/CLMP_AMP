# -*- coding: utf-8 -*-
"""
阶段2模型（强化版，适配新版阶段1）
- 复用并冻结/微调阶段1 SharedEncoder，输入为 (u50, physchem)，返回 f_shared 及阶段1注意力分布 attn_s1
- RawBranch 使用与阶段1相同的 U50+五路理化卷积结构，再做 MHQPooling，返回 raw_feat 与 attn_raw
- 融合头支持 bottleneck/gating
- 支持四种微调模式：full / top / shallow / head_only
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1_model import SharedEncoder  # 复用阶段1实现
from stage1_config import CONFIG as S1C
from stage1_model import MHQPooling as S1MHQPooling


# ========= 多头查询池化（与 stage1 同实现/同分布族） =========
class MHQPooling(nn.Module):
    """
    ===== [MOD-C1] 直接对齐 stage1_model.MHQPooling 的实现 =====
    - q 在 query_dim
    - 对输入做 k_proj/v_proj
    - scores = (k*q).sum * scale
    - mask 方式一致
    - attn_mean = attn.mean(dim=1)
    返回:
      pooled_flat: (B, H*Q)
      attn_mean:   (B, L)
    """

    def __init__(self,
                 in_dim: int,
                 num_heads: int = 4,
                 query_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 use_bias: bool = True):
        super().__init__()
        self.num_heads = int(num_heads)

        if query_dim is None:
            # 与 stage1 相同的默认策略（不依赖 hidden）
            query_dim = max(32, int(in_dim) // max(1, self.num_heads))
        self.query_dim = int(query_dim)

        self.scale = self.query_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.q = nn.Parameter(torch.randn(self.num_heads, self.query_dim))
        self.k_proj = nn.Linear(in_dim, self.num_heads * self.query_dim, bias=use_bias)
        self.v_proj = nn.Linear(in_dim, self.num_heads * self.query_dim, bias=use_bias)

        self.out_dim = self.num_heads * self.query_dim

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if use_bias:
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
        nn.init.normal_(self.q, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B,L,D), mask: (B,L) True=valid
        B, L, _ = x.shape
        H, Q = self.num_heads, self.query_dim

        k = self.k_proj(x).view(B, L, H, Q).permute(0, 2, 1, 3).contiguous()  # (B,H,L,Q)
        v = self.v_proj(x).view(B, L, H, Q).permute(0, 2, 1, 3).contiguous()  # (B,H,L,Q)
        q = self.q.view(1, H, 1, Q)  # (1,H,1,Q)

        scores = (k * q).sum(dim=-1) * self.scale  # (B,H,L)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B,H,L)
        attn = self.dropout(attn)

        pooled = torch.einsum("bhl,bhlq->bhq", attn, v)  # (B,H,Q)
        pooled_flat = pooled.reshape(B, H * Q)  # (B,H*Q)
        attn_mean = attn.mean(dim=1)  # (B,L)
        return pooled_flat, attn_mean


# ========== 阶段1共享编码器包装 ==========

class Stage1FeatureExtractor(nn.Module):
    """
    用阶段1的共享编码器做特征提取：
    - 严格按 ckpt 的配置重建 SharedEncoder（u50 + per-residue physchem）
    - forward 输入: (u50, mask, physchem) -> 输出 (f_shared, attn_s1)

    额外增加:
    - ft_mode: {"full", "top", "shallow", "head_only"}，由阶段2训练脚本设置
      * "head_only" 下 forward 使用 no_grad，且由 Stage2AttrModel.train() 保证 encoder 处于 eval 模式
    """

    def __init__(self, ckpt_path: str, strict_cfg_check: bool = True):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        s1cfg = ckpt.get("config", {})
        state = ckpt["state_dict"]

        if not isinstance(s1cfg, dict):
            s1cfg = {}
        if strict_cfg_check:
            keys = ["esm_dim", "shared_hidden", "bilstm_hidden",
                    "mhq_heads", "mhq_query_dim",
                    "dim_aaindex", "dim_paac", "dim_pc6", "dim_blosum62", "dim_onehot", "feat_dim_phys",
                    "conv_channels_aaindex", "conv_channels_paac", "conv_channels_pc6",
                    "conv_channels_blosum62", "conv_channels_onehot"]
            for k in keys:
                if k in s1cfg and k in S1C:
                    if str(s1cfg[k]) != str(S1C[k]):
                        raise ValueError(f"[Stage1FeatureExtractor] S1C 与 ckpt config 不一致: {k} "
                                         f"S1C={S1C[k]} vs ckpt={s1cfg[k]}")

        cfg = {**S1C, **s1cfg}  # ckpt 覆盖同名键

        self.encoder = SharedEncoder(
            esm_dim=cfg.get("esm_dim", 1024),
            hand_dim=None,
            shared_hidden=cfg.get("shared_hidden", 512),
            cnn_out_channels=cfg.get("cnn_out_channels", 512),
            cnn_kernel_sizes=cfg.get("cnn_kernel_sizes", [3, 5, 7]),
            bilstm_hidden=cfg.get("bilstm_hidden", 256),
            attn_pooling_dim=cfg.get("attn_pooling_dim", 256),
            dropout=cfg.get("dropout", 0.2),
        )

        self.encoder.load_state_dict(state, strict=True)
        self.encoder.eval()

        self.ft_mode: Optional[str] = None
        self.output_dim = cfg.get("shared_hidden", 512)

        print(
            f"[S1-Extractor] shared_hidden={self.output_dim} | "
            f"dim_aaindex={cfg.get('dim_aaindex')} | dim_paac={cfg.get('dim_paac')} | "
            f"dim_pc6={cfg.get('dim_pc6')} | dim_blosum62={cfg.get('dim_blosum62')} | "
            f"dim_onehot={cfg.get('dim_onehot')}"
        )

    def set_ft_mode(self, mode: Optional[str]):
        self.ft_mode = mode

    def forward(self, u50: torch.Tensor, mask: torch.Tensor, physchem: torch.Tensor):
        if self.ft_mode == "head_only":
            with torch.no_grad():
                f_shared, attn_s1 = self.encoder(u50, mask, physchem)
        else:
            f_shared, attn_s1 = self.encoder(u50, mask, physchem)
        return f_shared, attn_s1  # (B, H), (B, L)


# ========== Raw 分支：U50+五路理化卷积 + MHQPooling ==========

class RawBranch(nn.Module):
    """
    与阶段1相同的 U50 + 五路理化卷积结构：
      1) [U50, physchem] -> LayerNorm
      2) 拆分为 U50 + AAIndex/PAAC/PC6/BLOSUM62/OneHot 五路
      3) 每一路: Conv1d + ReLU + LayerNorm
      4) 拼回 feat = [U50, aa_f, pa_f, pc_f, bl_f, oh_f]，再做 LayerNorm
      5) feat 上做 MHQPooling -> raw_feat 与 attn_raw

    """

    def __init__(self,
                 raw_hidden: int = 256,
                 dropout: float = 0.3,
                 mhq_heads: int = 4,
                 s1_encoder: Optional[nn.Module] = None,
                 freeze_pool: bool = False):
        super().__init__()
        self.esm_dim = S1C.get("esm_dim", 1024)
        self.dim_aaindex = S1C["dim_aaindex"]
        self.dim_paac = S1C["dim_paac"]
        self.dim_pc6 = S1C["dim_pc6"]
        self.dim_blosum62 = S1C["dim_blosum62"]
        self.dim_onehot = S1C["dim_onehot"]

        feat_dim_phys = (
                self.dim_aaindex + self.dim_paac + self.dim_pc6 +
                self.dim_blosum62 + self.dim_onehot
        )
        self.total_in_dim = self.esm_dim + feat_dim_phys

        ch_aa = S1C.get("conv_channels_aaindex", 64)
        ch_pa = S1C.get("conv_channels_paac", 64)
        ch_pc = S1C.get("conv_channels_pc6", 64)
        ch_bl = S1C.get("conv_channels_blosum62", 64)
        ch_oh = S1C.get("conv_channels_onehot", 64)

        self.norm_input = nn.LayerNorm(self.total_in_dim)

        def make_conv_branch(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.ReLU(inplace=True),
            )

        self.conv_aaindex = make_conv_branch(self.dim_aaindex, ch_aa, 3)
        self.conv_paac = make_conv_branch(self.dim_paac, ch_pa, 3)
        self.conv_pc6 = make_conv_branch(self.dim_pc6, ch_pc, 3)
        self.conv_blosum62 = make_conv_branch(self.dim_blosum62, ch_bl, 3)
        self.conv_onehot = make_conv_branch(self.dim_onehot, ch_oh, 3)

        self.norm_aaindex = nn.LayerNorm(ch_aa)
        self.norm_paac = nn.LayerNorm(ch_pa)
        self.norm_pc6 = nn.LayerNorm(ch_pc)
        self.norm_blosum62 = nn.LayerNorm(ch_bl)
        self.norm_onehot = nn.LayerNorm(ch_oh)

        self.concat_dim = self.esm_dim + ch_aa + ch_pa + ch_pc + ch_bl + ch_oh
        self.norm_concat = nn.LayerNorm(self.concat_dim)

        q_dim = int(S1C.get("mhq_query_dim", 64))
        self.pool = S1MHQPooling(
            in_dim=self.concat_dim,
            num_heads=int(mhq_heads),
            query_dim=q_dim,
            dropout=0.0,  # 对齐项建议 0.0
            use_bias=True
        )

        if s1_encoder is not None and hasattr(s1_encoder, "copy_attn_pool_to"):
            n_copied = int(s1_encoder.copy_attn_pool_to(self.pool, strict=False))
            # 可选：你也可以换成 logger.info
            print(f"[RawBranch] init pool from stage1 attn_pool: copied={n_copied}")

        # ===== [MOD-C3] 把 (H*Q) 投影回 concat_dim，保持你原来 RawBranch 后续结构不变 =====
        pool_out_dim = getattr(self.pool, "out_dim", int(mhq_heads) * q_dim)

        self.pool_proj = nn.Linear(pool_out_dim, self.concat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, raw_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = raw_hidden
        if freeze_pool:
            self.freeze_pool()

    # ===== padding 归零（与你上一轮一致）=====
    @staticmethod
    def _zero_pad_positions(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)

    def _conv_branch(self, x: torch.Tensor, conv: nn.Module, norm: nn.Module, mask: Optional[torch.Tensor]):
        x = self._zero_pad_positions(x, mask)
        x = x.transpose(1, 2)  # (B,Cin,L)
        x = conv(x)  # Conv+ReLU -> (B,Cout,L)
        x = x.transpose(1, 2)  # (B,L,Cout)
        x = norm(x)
        x = self._zero_pad_positions(x, mask)
        return x

    # ===== [MOD-R4] 新增：冻结/解冻 pool 的显式接口，供 Stage2AttrModel.set_ft_mode 调用 =====
    def freeze_pool(self):
        for p in self.pool.parameters():
            p.requires_grad = False

    def unfreeze_pool(self):
        for p in self.pool.parameters():
            p.requires_grad = True
    # ===== [MOD-R4] END =====

    def forward(self, u50: torch.Tensor, mask: torch.Tensor, physchem: torch.Tensor):
        x = torch.cat([u50, physchem], dim=-1)  # (B,L,total_in_dim)
        x = self.norm_input(x)
        x = self._zero_pad_positions(x, mask)

        splits = [
            self.esm_dim,
            self.dim_aaindex, self.dim_paac, self.dim_pc6,
            self.dim_blosum62, self.dim_onehot
        ]
        u50_part, aaidx, paac, pc6, bl62, oh = torch.split(x, splits, dim=-1)

        aa_f = self._conv_branch(aaidx, self.conv_aaindex, self.norm_aaindex, mask)
        pa_f = self._conv_branch(paac, self.conv_paac, self.norm_paac, mask)
        pc_f = self._conv_branch(pc6, self.conv_pc6, self.norm_pc6, mask)
        bl_f = self._conv_branch(bl62, self.conv_blosum62, self.norm_blosum62, mask)
        oh_f = self._conv_branch(oh, self.conv_onehot, self.norm_onehot, mask)

        feat = torch.cat([u50_part, aa_f, pa_f, pc_f, bl_f, oh_f], dim=-1)  # (B,L,concat_dim)
        feat = self.norm_concat(feat)
        feat = self._zero_pad_positions(feat, mask)

        # ===== [MOD-C4] pool 输出 (B,H*Q)，再投影回 concat_dim =====
        pooled_flat, attn_raw = self.pool(feat, mask)  # (B,H*Q), (B,L)
        pooled = self.pool_proj(pooled_flat)  # (B,concat_dim)

        raw_feat = self.mlp(pooled)  # (B,raw_hidden)
        return raw_feat, attn_raw


# ========== 融合头：瓶颈 / 门控 ==========

class Stage2Head(nn.Module):
    def __init__(self, fshared_dim: int, raw_dim: int,
                 hidden: int = 256, dropout: float = 0.3, mode: str = "bottleneck"):
        super().__init__()
        self.mode = mode
        self.f_mlp = nn.Sequential(
            nn.Linear(fshared_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        if mode == "bottleneck":
            fuse_in = hidden + raw_dim
            self.fuse = nn.Sequential(
                nn.Linear(fuse_in, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        elif mode == "gating":
            self.gate = nn.Sequential(
                nn.Linear(fshared_dim, raw_dim),
                nn.Sigmoid()
            )
            fuse_in = hidden + raw_dim
            self.fuse = nn.Sequential(
                nn.Linear(fuse_in, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, fshared: torch.Tensor, raw_feat: torch.Tensor):
        fz = self.f_mlp(fshared)
        if self.mode == "bottleneck":
            x = torch.cat([fz, raw_feat], dim=1)
        else:
            g = self.gate(fshared)
            x = torch.cat([fz, raw_feat * (1.0 - g)], dim=1)
        logit = self.fuse(x).squeeze(-1)
        return logit


# ========== 完整阶段2模型（单属性） ==========

class Stage2AttrModel(nn.Module):
    def __init__(self, stage1_extractor: Stage1FeatureExtractor,
                 mode: str = "bottleneck",
                 raw_hidden: int = 256,
                 fshared_hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.stage1 = stage1_extractor
        self.raw = RawBranch(
            raw_hidden=raw_hidden,
            dropout=dropout,
            mhq_heads=S1C.get("mhq_heads", 4),
            s1_encoder=self.stage1.encoder,  # <<< 关键
            freeze_pool=False
        )
        self.head = Stage2Head(
            fshared_dim=self.stage1.output_dim,
            raw_dim=self.raw.out_dim,
            hidden=fshared_hidden,
            dropout=dropout,
            mode=mode,
        )
        self.ft_mode: Optional[str] = None

    def set_ft_mode(self, mode: Optional[str]):
        self.ft_mode = mode
        if hasattr(self.stage1, "set_ft_mode"):
            self.stage1.set_ft_mode(mode)

        if mode in {"head_only", "top", "shallow"}:
            self.raw.freeze_pool()
        else:
            self.raw.unfreeze_pool()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.ft_mode == "head_only":
            self.stage1.eval()
            # self.raw.eval()
        return self

    def forward(self, u50: torch.Tensor, mask: torch.Tensor, physchem: torch.Tensor):
        fshared, attn_s1 = self.stage1(u50, mask, physchem)  # (B,H), (B,L)
        raw_feat, attn_raw = self.raw(u50, mask, physchem)  # (B,h), (B,L)
        logit = self.head(fshared, raw_feat)  # (B,)
        return logit, fshared, raw_feat, attn_s1, attn_raw
