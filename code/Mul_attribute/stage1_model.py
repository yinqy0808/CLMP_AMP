# -*- coding: utf-8 -*-
import os
import math
import random
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from typing import Dict, Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from stage1_config import CONFIG

logger = logging.getLogger("Stage1")


# ---------- 随机种子（仅提供工具函数，由外部显式调用） ----------
def set_seed(seed: int):
    """
    统一固定所有可控随机源：
      - Python 内置 random
      - NumPy
      - PyTorch CPU/GPU
      - PYTHONHASHSEED
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 实用函数
# =========================================================
def average_precision_score_scratch(y_true: np.ndarray, y_score: np.ndarray) -> float:
    idx = np.argsort(-y_score)
    y_true_sorted = y_true[idx]
    cum_tp = np.cumsum(y_true_sorted)
    precision_at_k = cum_tp / (np.arange(len(y_true_sorted)) + 1)
    total_pos = y_true.sum()
    if total_pos == 0:
        return np.nan
    ap = (precision_at_k * y_true_sorted).sum() / (total_pos + 1e-12)
    return float(ap)


# =========================================================
# per-residue 五路理化特征字典
# =========================================================
def load_physchem_dicts(AAINDEX_PATH: str,
                        PAAC_PATH: str,
                        PC6_PATH: str,
                        BLOSUM62_PATH: str):
    """
    构建 per-residue 理化特征字典：
      - AAIndex_dict[aa] -> (dim_AAIndex,)
      - PAAC_dict[aa]    -> (dim_PAAC,)
      - PC6_dict[aa]     -> (dim_PC6,)
      - BLOSUM62_dict[aa]-> (dim_BLOSUM62,)
      - OneHot_dict[aa]  -> (dim_OneHot,)
    """
    # --- AAIndex ---
    AAIndex_dict: Dict[str, np.ndarray] = {}
    with open(AAINDEX_PATH, "r", encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    header = lines[0].strip().split("\t")
    aa_list = [x for x in header[1:] if x]
    data_rows = []
    for row in lines[1:]:
        parts = [p for p in row.strip().split("\t") if p]
        if len(parts) < len(aa_list) + 1:
            continue
        values = [float(x) for x in parts[1:len(aa_list) + 1]]
        data_rows.append(values)
    index_matrix = np.array(data_rows, dtype=np.float32)  # (N, 20)
    for j, aa in enumerate(aa_list):
        AAIndex_dict[aa] = index_matrix[:, j]
    AAIndex_dict["X"] = np.zeros((index_matrix.shape[0],), dtype=np.float32)
    dim_AAIndex = index_matrix.shape[0]

    # --- PAAC ---
    PAAC_dict: Dict[str, np.ndarray] = {}
    with open(PAAC_PATH, "r", encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    header = lines[0].strip().split("\t")
    aa_list_paac = [x for x in header[1:] if x]
    paac_rows = []
    for row in lines[1:]:
        parts = [p for p in row.strip().split("\t") if p]
        if len(parts) < len(aa_list_paac) + 1:
            continue
        paac_rows.append([float(x) for x in parts[1:len(aa_list_paac) + 1]])
    mat = np.array(paac_rows, dtype=np.float32).T  # (20, dim_PAAC)
    for j, aa in enumerate(aa_list_paac):
        PAAC_dict[aa] = mat[j, :]
    PAAC_dict["X"] = np.zeros((mat.shape[1],), dtype=np.float32)
    dim_PAAC = mat.shape[1]

    # --- PC6 ---
    PC6_dict: Dict[str, np.ndarray] = {}
    with open(PC6_PATH, "r", encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    header_pc6 = lines[0].split()
    last_values = None
    for row in lines[1:]:
        parts = [p for p in row.strip().split() if p]
        if len(parts) < 2:
            continue
        aa = parts[0]
        values = [float(x) for x in parts[1:]]
        last_values = values
        PC6_dict[aa] = np.array(values, dtype=np.float32)
    if last_values is None:
        raise ValueError("PC6 文件解析失败，未读到有效行")
    PC6_dict["X"] = np.zeros((len(last_values),), dtype=np.float32)
    dim_PC6 = len(last_values)

    # --- BLOSUM62 ---
    BLOSUM62_dict: Dict[str, np.ndarray] = {}
    with open(BLOSUM62_PATH, "r", encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    aa_list_blosum = lines[0].split()
    blosum_rows = []
    for row in lines[1:]:
        parts = [p for p in row.strip().split() if p]
        if len(parts) < len(aa_list_blosum):
            continue
        blosum_rows.append([float(x) for x in parts[:len(aa_list_blosum)]])
    blosum_mat = np.array(blosum_rows, dtype=np.float32)
    if blosum_mat.shape[0] != len(aa_list_blosum):
        raise ValueError("BLOSUM62 矩阵行数与 aa_list 长度不匹配")
    for j, aa in enumerate(aa_list_blosum):
        BLOSUM62_dict[aa] = blosum_mat[j, :]
    BLOSUM62_dict["X"] = np.zeros((len(aa_list_blosum),), dtype=np.float32)
    dim_BLOSUM62 = len(aa_list_blosum)

    # --- One-Hot ---
    amino_chars = "ARNDCQEGHILKMFPSTWYVX"
    OneHot_dict: Dict[str, np.ndarray] = {}
    for i, aa in enumerate(amino_chars):
        vec = np.zeros((len(amino_chars),), dtype=np.float32)
        vec[i] = 1.0
        OneHot_dict[aa] = vec
    OneHot_dict["X"] = np.zeros((len(amino_chars),), dtype=np.float32)
    dim_OneHot = len(amino_chars)
    logger.info("OneHot_dict: 映射 %d 个氨基酸 → 向量长度 %d", len(OneHot_dict), dim_OneHot)

    feat_dim_phys = dim_AAIndex + dim_PAAC + dim_PC6 + dim_BLOSUM62 + dim_OneHot

    return (
        AAIndex_dict, PAAC_dict, PC6_dict, BLOSUM62_dict, OneHot_dict,
        dim_AAIndex, dim_PAAC, dim_PC6, dim_BLOSUM62, dim_OneHot, feat_dim_phys
    )


# =========================================================
# 多任务数据集：U50 + per-residue 五路理化特征
# =========================================================
class MultiTaskDataset(Dataset):
    """
    X_u50: list(np.ndarray)  每个元素 (Li, 1024)
    physchem: list(np.ndarray) 每个元素 (Li, feat_dim_phys)
    labels: np.ndarray (N, T)
    """

    def __init__(self, csv_path: str, u50_npy_path: str,
                 handcraft_builder: Optional["HandcraftFeatureBuilder"] = None):
        # handcraft_builder 参数仅保留兼容旧接口，阶段1不再使用
        self.df = pd.read_csv(csv_path)

        # 序列列
        seq_col = None
        for c in CONFIG["sequence_col_candidates"]:
            if c in self.df.columns:
                seq_col = c
                break
        if seq_col is None:
            raise ValueError("MultiTaskDataset: 未从 CSV 中找到序列列，请检查 sequence_col_candidates")
        self.sequences = self.df[seq_col].astype(str).tolist()

        # 标签列：非序列列、值域子集 {0,1}
        ignore_cols = set([seq_col])
        candidate_cols = [c for c in self.df.columns if c not in ignore_cols]
        final_label_cols = []
        for c in candidate_cols:
            uniq = set(pd.Series(self.df[c]).dropna().unique().tolist())
            if uniq.issubset({0, 1}):
                final_label_cols.append(c)
        self.label_cols = final_label_cols
        self.labels = self.df[self.label_cols].values.astype(np.float32)  # (N, T)

        # U50 嵌入
        arr = np.load(u50_npy_path, allow_pickle=True)
        self.u50 = [np.asarray(x, np.float32) for x in arr]
        assert len(self.u50) == len(self.labels), "U50 数量与标签数量不一致"

        # per-residue 五路理化特征字典（使用统一理化文件）
        (self.AAIndex_dict, self.PAAC_dict, self.PC6_dict,
         self.BLOSUM62_dict, self.OneHot_dict,
         self.dim_aaindex, self.dim_paac, self.dim_pc6,
         self.dim_blosum62, self.dim_onehot,
         self.feat_dim_phys) = load_physchem_dicts(
            CONFIG["aaindex_path"], CONFIG["paac_path"],
            CONFIG["pc6_path"], CONFIG["blosum62_path"]
        )

        # 构建每条序列的 per-residue 理化特征
        self.physchem = []
        for i, seq in enumerate(self.sequences):
            phys = self._seq_to_phys(seq)
            u = self.u50[i]
            if phys.shape[0] != u.shape[0]:
                raise ValueError(
                    f"序列与 U50 长度不一致：idx={i}, seq_len={phys.shape[0]}, u50_len={u.shape[0]}"
                )
            self.physchem.append(phys.astype(np.float32))

        # 为兼容旧代码，明确置空 handcraft
        self.handcraft_builder = None
        self.handcraft = None

    def _seq_to_phys(self, seq: str) -> np.ndarray:
        mats = []
        for c in seq:
            aa = c.upper()
            if aa not in self.AAIndex_dict:
                aa = "X"
            v_aa = self.AAIndex_dict[aa]
            v_pa = self.PAAC_dict[aa]
            v_pc = self.PC6_dict[aa]
            v_bl = self.BLOSUM62_dict[aa]
            v_oh = self.OneHot_dict[aa]
            mats.append(np.concatenate([v_aa, v_pa, v_pc, v_bl, v_oh], axis=-1))
        if not mats:
            return np.zeros((0, self.feat_dim_phys), dtype=np.float32)
        return np.stack(mats, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_u50 = self.u50[idx]          # (L, 1024)
        x_phys = self.physchem[idx]    # (L, feat_dim_phys)
        y = self.labels[idx]
        return x_u50, x_phys, y


# -------------------- 对齐打包（变长 -> padding） --------------------
def pad_collate_fn(batch):
    """
    batch: list of (u50[L,1024], phys[L,D_phys], y[T])
    返回:
      u50:  (B, Lmax, 1024)
      mask: (B, Lmax)  True=有效
      phys:(B, Lmax, D_phys)
      y:    (B, T)
    """
    u50_list, phys_list, y_list, lens = [], [], [], []
    for x_u50, x_phys, y in batch:
        u50_list.append(torch.from_numpy(x_u50))
        phys_list.append(torch.from_numpy(x_phys))
        lens.append(x_u50.shape[0])
        y_list.append(torch.from_numpy(y))

    max_len = max(lens)
    padded_u50, padded_phys = [], []
    for u, p, L in zip(u50_list, phys_list, lens):
        if u.shape[0] != L or p.shape[0] != L:
            raise RuntimeError("pad_collate_fn: 序列长度不一致")
        pad_len = max_len - L
        if pad_len > 0:
            u_pad = torch.zeros((pad_len, u.shape[1]), dtype=u.dtype)
            p_pad = torch.zeros((pad_len, p.shape[1]), dtype=p.dtype)
            u = torch.cat([u, u_pad], dim=0)
            p = torch.cat([p, p_pad], dim=0)
        padded_u50.append(u)
        padded_phys.append(p)

    u50 = torch.stack(padded_u50, dim=0)        # (B, Lmax, 1024)
    phys = torch.stack(padded_phys, dim=0)      # (B, Lmax, D_phys)

    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, L in enumerate(lens):
        mask[i, :L] = True

    y = torch.stack(y_list, dim=0)              # (B, T)
    return u50, mask, phys, y


# =========================================================
# 采样器：1) 正样本感知；2) 任务轮转采样
# =========================================================
class PositiveAwareBatchSampler(Sampler):
    def __init__(self, labels: np.ndarray, batch_size: int, drop_last=False,
                 pos_fraction: float = 0.5, seed: int = 2025,
                 prefer_unl_for_rest: bool = False):
        assert labels.ndim == 2
        self.labels = labels
        self.N = labels.shape[0]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.pos_fraction = float(np.clip(pos_fraction, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        any_pos = (labels == 1).any(axis=1)
        self.pos_idx = np.flatnonzero(any_pos)
        self.unl_idx = np.flatnonzero(~any_pos)
        self.prefer_unl_for_rest = prefer_unl_for_rest
        self._epoch_batches = max(1, math.ceil(self.N / self.batch_size))

    def __len__(self):
        return self._epoch_batches

    def __iter__(self):
        b_pos = int(round(self.batch_size * self.pos_fraction))
        b_rest = self.batch_size - b_pos
        for _ in range(self._epoch_batches):
            if len(self.pos_idx) > 0:
                pos = self.rng.choice(self.pos_idx, size=b_pos, replace=True)
            else:
                pos = self.rng.choice(self.N, size=b_pos, replace=True)
            rest_pool = self.unl_idx if self.prefer_unl_for_rest and len(self.unl_idx) > 0 else np.arange(self.N)
            rest = self.rng.choice(rest_pool, size=b_rest, replace=True) if b_rest > 0 else np.array([], int)
            batch = np.concatenate([pos, rest])
            self.rng.shuffle(batch)
            yield batch.tolist()


class TaskRotatingBatchSampler(Sampler):
    """
    逐 batch 轮转一个“焦点任务 t”
    - 从任务 t 的正样本抽取一定比例（tr_pos_fraction）
    - 其余样本从全体均匀抽取
    - 任务选择概率 ∝ (1/pos_count)^power（稀有任务权重更高）
    """

    def __init__(self, labels: np.ndarray, batch_size: int,
                 tr_pos_fraction: float = 0.5, power: float = 1.0,
                 seed: int = 2025):
        assert labels.ndim == 2
        self.labels = labels.astype(np.int64)
        self.N, self.T = labels.shape
        self.batch = int(batch_size)
        self.pf = float(np.clip(tr_pos_fraction, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

        # 每个任务的正样本索引
        self.pos_idx_per_task = []
        pos_counts = self.labels.sum(axis=0)
        self.active_tasks = [t for t in range(self.T) if pos_counts[t] > 0]
        for t in range(self.T):
            self.pos_idx_per_task.append(np.flatnonzero(self.labels[:, t] == 1))

        if len(self.active_tasks) == 0:
            self.task_probs = np.ones(self.T) / max(1, self.T)
        else:
            inv = np.zeros(self.T, dtype=np.float64)
            inv[self.active_tasks] = 1.0 / np.maximum(1.0, pos_counts[self.active_tasks])
            inv = np.power(inv, power)
            self.task_probs = inv / (inv.sum() + 1e-12)

        self.steps = max(1, math.ceil(self.N / self.batch))

    def __len__(self):
        return self.steps

    def __iter__(self):
        n_pos = int(round(self.batch * self.pf))
        n_rest = self.batch - n_pos
        all_idx = np.arange(self.N)
        for _ in range(self.steps):
            t = self.rng.choice(np.arange(self.T), p=self.task_probs)
            pool = self.pos_idx_per_task[t]
            if len(pool) == 0:
                pos = self.rng.choice(all_idx, size=n_pos, replace=True)
            else:
                pos = self.rng.choice(pool, size=n_pos, replace=True)
            rest = self.rng.choice(all_idx, size=n_rest, replace=True) if n_rest > 0 else np.array([], int)
            batch = np.concatenate([pos, rest])
            self.rng.shuffle(batch)
            yield batch.tolist()


# =========================================================
# 模型：SharedEncoder + TaskHeads（无不确定性加权与对抗）
# =========================================================
class MHQPooling(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_heads: int = 4,
                 query_dim: int = None,
                 hidden: int = None,
                 dropout: float = 0.0,
                 use_bias: bool = True,
                 **unused_kwargs):
        super().__init__()
        self.num_heads = int(num_heads)

        if query_dim is None:
            query_dim = hidden if hidden is not None else max(32, in_dim // max(1, self.num_heads))
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
        B, L, _ = x.shape
        H, Q = self.num_heads, self.query_dim

        k = self.k_proj(x).view(B, L, H, Q).permute(0, 2, 1, 3).contiguous()
        v = self.v_proj(x).view(B, L, H, Q).permute(0, 2, 1, 3).contiguous()

        q = self.q.view(1, H, 1, Q)

        scores = (k * q).sum(dim=-1) * self.scale  # (B, H, L)

        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, H, L)
        attn = self.dropout(attn)

        pooled = torch.einsum("bhl,bhlq->bhq", attn, v)
        pooled_flat = pooled.reshape(B, H * Q)
        attn_mean = attn.mean(dim=1)  # (B, L)

        return pooled_flat, attn_mean


class SharedEncoder(nn.Module):
    """
    新版 SharedEncoder：
      输入：
        u50:  (B, L, 1024)
        phys: (B, L, feat_dim_phys) = AAIndex+PAAC+PC6+BLOSUM62+OneHot
    """

    def __init__(self, esm_dim, hand_dim, shared_hidden,
                 cnn_out_channels, cnn_kernel_sizes, bilstm_hidden,
                 attn_pooling_dim, dropout):
        super().__init__()

        self.esm_dim = esm_dim
        self.dim_aaindex = CONFIG["dim_aaindex"]
        self.dim_paac = CONFIG["dim_paac"]
        self.dim_pc6 = CONFIG["dim_pc6"]
        self.dim_blosum62 = CONFIG["dim_blosum62"]
        self.dim_onehot = CONFIG["dim_onehot"]

        if any(d is None for d in [
            self.dim_aaindex, self.dim_paac, self.dim_pc6,
            self.dim_blosum62, self.dim_onehot
        ]):
            raise ValueError("SharedEncoder: per-residue 五路特征维度未在 CONFIG 中设置")

        self.total_in_dim = (
            self.esm_dim
            + self.dim_aaindex
            + self.dim_paac
            + self.dim_pc6
            + self.dim_blosum62
            + self.dim_onehot
        )

        self.norm_input = nn.LayerNorm(self.total_in_dim)

        ch_aa = CONFIG.get("conv_channels_aaindex", 64)
        ch_pa = CONFIG.get("conv_channels_paac", 64)
        ch_pc = CONFIG.get("conv_channels_pc6", 64)
        ch_bl = CONFIG.get("conv_channels_blosum62", 64)
        ch_oh = CONFIG.get("conv_channels_onehot", 64)

        self.conv_aaindex = nn.Conv1d(self.dim_aaindex, ch_aa, kernel_size=3, padding=1)
        self.norm_aaindex = nn.LayerNorm(ch_aa)

        self.conv_paac = nn.Conv1d(self.dim_paac, ch_pa, kernel_size=3, padding=1)
        self.norm_paac = nn.LayerNorm(ch_pa)

        self.conv_pc6 = nn.Conv1d(self.dim_pc6, ch_pc, kernel_size=3, padding=1)
        self.norm_pc6 = nn.LayerNorm(ch_pc)

        self.conv_blosum62 = nn.Conv1d(self.dim_blosum62, ch_bl, kernel_size=3, padding=1)
        self.norm_blosum62 = nn.LayerNorm(ch_bl)

        self.conv_onehot = nn.Conv1d(self.dim_onehot, ch_oh, kernel_size=3, padding=1)
        self.norm_onehot = nn.LayerNorm(ch_oh)

        self.concat_dim = self.esm_dim + ch_aa + ch_pa + ch_pc + ch_bl + ch_oh
        self.norm_concat = nn.LayerNorm(self.concat_dim)

        self.mhq_heads = int(CONFIG.get("mhq_heads", 4))
        self.mhq_query_dim = int(CONFIG.get("mhq_query_dim", 64))
        self.bilstm_hidden = int(bilstm_hidden)

        self.bilstm = nn.LSTM(
            input_size=self.concat_dim,
            hidden_size=self.bilstm_hidden,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
            num_layers=1,
        )

        n_heads = CONFIG.get("mhq_heads", 4)
        q_dim = CONFIG.get("mhq_query_dim", 64)
        self.attn_pool = MHQPooling(
            in_dim=2 * self.bilstm_hidden,
            num_heads=n_heads,
            query_dim=q_dim
        )

        fused_in = n_heads * q_dim
        self.fused = nn.Sequential(
            nn.Linear(fused_in, shared_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(shared_hidden),
        )

    # ===== [MOD-1] 新增：统一把 padding 位置强制置零，防止 LN/Conv 产生非零 padding =====
    @staticmethod
    def _zero_pad_positions(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        mask: (B, L) True=valid
        """
        if mask is None:
            return x
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)

    def _conv_branch(self, x: torch.Tensor, conv: nn.Module, norm: nn.Module, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, Din)
        """
        # 强制保证输入 padding 为 0（即使上游漏了，也兜底）
        x = self._zero_pad_positions(x, mask)

        x = x.transpose(1, 2)          # (B, Din, L)
        x = conv(x)                    # (B, Dout, L)
        x = F.relu(x, inplace=True)
        x = x.transpose(1, 2)          # (B, L, Dout)
        x = norm(x)                    # LN 会让全零 padding -> beta，所以下面要再置零
        x = self._zero_pad_positions(x, mask)
        return x

    # ===== [MOD-S2] 新增：导出/拷贝 stage1 的 MHQPooling 参数（给 stage2 RawBranch.pool 初始化用）=====
    def export_attn_pool_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.attn_pool.state_dict().items()}

    def copy_attn_pool_to(self, target_pool: nn.Module, strict: bool = False) -> int:
        """
        将本 encoder 的 attn_pool 参数拷贝到 target_pool：
        - 仅拷贝 “同名且形状一致” 的参数
        - strict=True 且无可拷贝参数 -> 报错
        返回：成功拷贝条目数
        """
        src = self.attn_pool.state_dict()
        dst = target_pool.state_dict()

        common = {}
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                common[k] = v

        if strict and len(common) == 0:
            raise ValueError("[SharedEncoder.copy_attn_pool_to] 目标 pool 与 stage1 attn_pool 不匹配，无法拷贝")

        if len(common) > 0:
            dst.update(common)
            target_pool.load_state_dict(dst, strict=False)

        return int(len(common))

    def forward(self, u50: torch.Tensor, mask: torch.Tensor, physchem: Optional[torch.Tensor] = None):
        if physchem is None:
            raise ValueError("SharedEncoder 需要 per-residue physchem 特征（形状 (B,L,D_phys)）")

        # (B, L, total_in_dim)
        x = torch.cat([u50, physchem], dim=-1)

        x = self.norm_input(x)
        x = self._zero_pad_positions(x, mask)  # <<< [MOD-3.1] 关键：LN 后把 padding 归零

        splits = [
            self.esm_dim,
            self.dim_aaindex,
            self.dim_paac,
            self.dim_pc6,
            self.dim_blosum62,
            self.dim_onehot,
        ]
        u50_part, aaidx, paac, pc6, bl62, oh = torch.split(x, splits, dim=-1)

        # Conv 分支：Conv/LN 后都把 padding 置零，避免 bias/LN 造成非零 padding
        aa_f = self._conv_branch(aaidx, self.conv_aaindex, self.norm_aaindex, mask)     # <<< [MOD-3.2]
        pa_f = self._conv_branch(paac, self.conv_paac, self.norm_paac, mask)            # <<< [MOD-3.2]
        pc_f = self._conv_branch(pc6, self.conv_pc6, self.norm_pc6, mask)               # <<< [MOD-3.2]
        bl_f = self._conv_branch(bl62, self.conv_blosum62, self.norm_blosum62, mask)    # <<< [MOD-3.2]
        oh_f = self._conv_branch(oh, self.conv_onehot, self.norm_onehot, mask)          # <<< [MOD-3.2]

        feat = torch.cat([u50_part, aa_f, pa_f, pc_f, bl_f, oh_f], dim=-1)
        feat = self.norm_concat(feat)
        feat = self._zero_pad_positions(feat, mask)  # <<< [MOD-3.3] concat-LN 后再归零

        # ===== [MOD-3.4] pack BiLSTM：彻底避免 LSTM 处理 padding steps =====
        if mask is None:
            feat, _ = self.bilstm(feat)
        else:
            lengths = mask.sum(dim=1).to(torch.long)           # (B,)
            lengths_cpu = lengths.detach().cpu()
            packed = pack_padded_sequence(feat, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.bilstm(packed)
            feat, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=mask.size(1))
        # ===== [MOD-3.4] END =====

        pooled, attn_w = self.attn_pool(feat, mask)

        f_shared = self.fused(pooled)
        return f_shared, attn_w


class TaskHeads(nn.Module):
    def __init__(self, shared_hidden, num_tasks, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(shared_hidden, num_tasks)
        self.final_linear = self.fc  # 供 laplacian_reg_from_head 使用

    def forward(self, f_shared):
        return self.fc(self.drop(f_shared))


def bce_per_task_losses(logits: torch.Tensor, labels: torch.Tensor):
    loss_mat = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    per_task = loss_mat.mean(dim=0)
    pos_counts = (labels == 1.0).sum(dim=0)
    return per_task, pos_counts


class Stage1Model(nn.Module):
    def __init__(self, num_tasks, hand_dim):
        super().__init__()
        # hand_dim 已不再使用，仅保留签名兼容
        self.encoder = SharedEncoder(
            esm_dim=CONFIG["esm_dim"], hand_dim=None,
            shared_hidden=CONFIG["shared_hidden"],
            cnn_out_channels=CONFIG["cnn_out_channels"],
            cnn_kernel_sizes=CONFIG["cnn_kernel_sizes"],
            bilstm_hidden=CONFIG["bilstm_hidden"],
            attn_pooling_dim=CONFIG["attn_pooling_dim"],
            dropout=CONFIG["dropout"],
        )
        self.heads = TaskHeads(CONFIG["shared_hidden"], num_tasks, CONFIG["dropout"])

    def forward(self, u50, mask, physchem=None):
        f_shared, attn_w = self.encoder(u50, mask, physchem)
        logits = self.heads(f_shared)
        return logits, f_shared, attn_w
