# stage1_train.py
# -*- coding: utf-8 -*-
import json
import math
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage1_config import CONFIG, get_paths
from stage1_model import (
    MultiTaskDataset, pad_collate_fn,
    PositiveAwareBatchSampler, TaskRotatingBatchSampler,
    Stage1Model,
    average_precision_score_scratch,
    set_seed,
)

# ============ 确定性设置（只控制 cuDNN 行为） ============
def setup_deterministic():
    """
    在 CONFIG['torch_deterministic'] 为 True 时：
      - cuDNN 使用确定性算法
      - 关闭 benchmark
    """
    if CONFIG.get("torch_deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 先设置 cuDNN 模式，再统一固定随机种子
setup_deterministic()
set_seed(CONFIG["seed"])

# ==============================
# 日志配置
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(Path(CONFIG["log_dir"]) / CONFIG["log_file"], mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("Stage1")


# ==============================
# 调度器：Lap 权重
# ==============================
def schedule_weight(epoch, total_epochs, max_w, warmup_frac):
    warm = max(1, int(total_epochs * warmup_frac))
    if epoch < warm:
        return max_w * (epoch + 1) / warm
    return max_w


# ==============================
# 训练/验证帮助
# ==============================
def build_loaders(train_dataset: MultiTaskDataset, val_dataset: MultiTaskDataset):
    """
    使用 train_dataset 作为训练集，val_dataset 作为验证集。
    """
    hand_dim = 0  # 阶段1不再使用 handcraft
    T = len(train_dataset.label_cols)
    labels = train_dataset.labels

    sampler_mode = CONFIG.get("sampler", "task_rotate").lower()

    if sampler_mode == "task_rotate":
        batch_sampler = TaskRotatingBatchSampler(
            labels=labels,
            batch_size=CONFIG["batch_size"],
            tr_pos_fraction=CONFIG.get("tr_pos_fraction", 0.65),
            power=CONFIG.get("tr_power", 1.5),
            seed=CONFIG["seed"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=CONFIG["num_workers"],
            pin_memory=CONFIG["pin_memory"],
            collate_fn=pad_collate_fn,
        )
    elif sampler_mode == "pos_aware":
        batch_sampler = PositiveAwareBatchSampler(
            labels=labels,
            batch_size=CONFIG["batch_size"],
            drop_last=False,
            pos_fraction=CONFIG.get("pos_sampler_pos_fraction", 0.5),
            seed=CONFIG["seed"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=CONFIG["num_workers"],
            pin_memory=CONFIG["pin_memory"],
            collate_fn=pad_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=CONFIG["pin_memory"],
            collate_fn=pad_collate_fn,
            drop_last=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        collate_fn=pad_collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, hand_dim, T


def auprc_on_val(model: Stage1Model, loader, device, T: int, label_cols):
    model.eval()
    all_scores = [[] for _ in range(T)]
    all_labels = [[] for _ in range(T)]
    with torch.no_grad():
        for u50, mask, phys, y in tqdm(loader, desc="Validate", unit="batch"):
            u50 = u50.to(device)
            mask = mask.to(device)
            y = y.to(device)
            phys = phys.to(device)
            logits, _, _ = model(u50, mask, phys)
            probs = torch.sigmoid(logits)
            for t in range(T):
                all_scores[t].append(probs[:, t].cpu().numpy())
                all_labels[t].append(y[:, t].cpu().numpy())
    ap_per_task = []
    for t in range(T):
        scores = np.concatenate(all_scores[t], axis=0) if all_scores[t] else np.array([], dtype=np.float32)
        lbls = np.concatenate(all_labels[t], axis=0) if all_labels[t] else np.array([], dtype=np.float32)
        if scores.size == 0:
            ap = float("nan")
        else:
            ap = average_precision_score_scratch(lbls, scores)
        ap_per_task.append(ap)
    valid_aps = [a for a in ap_per_task if not (np.isnan(a))]
    macro_ap = float(np.mean(valid_aps)) if len(valid_aps) > 0 else float("nan")
    topk = CONFIG["report_topk_tasks"]
    order = np.argsort([- (a if not np.isnan(a) else -1) for a in ap_per_task])[:topk]
    ap_report = [(label_cols[i], 0.0 if np.isnan(ap_per_task[i]) else float(ap_per_task[i])) for i in order]
    return macro_ap, ap_per_task, ap_report


# ==============================
# 标签图 & 拉普拉斯
# ==============================
def build_task_graph_and_laplacian(labels_train: np.ndarray, topk: int):
    X = labels_train.astype(np.float32)
    T = X.shape[1]
    if T <= 1:
        A = np.zeros((T, T), dtype=np.float32)
        L = np.zeros_like(A)
        return torch.from_numpy(A), torch.from_numpy(L)

    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)

    A = np.clip(C, 0.0, None)

    if topk is not None and topk > 0 and topk < T:
        idx_sorted = np.argsort(-A, axis=1)
        mask = np.zeros_like(A, dtype=bool)
        for i in range(T):
            top_idx = idx_sorted[i, :topk]
            mask[i, top_idx] = True
        A = A * mask.astype(np.float32)

    A = np.maximum(A, A.T)

    d = A.sum(axis=1)
    L = np.diag(d) - A

    A = A.astype(np.float32)
    L = L.astype(np.float32)
    return torch.from_numpy(A), torch.from_numpy(L)


def _find_task_head_linear(model, num_tasks: int) -> torch.nn.Linear:
    lin = getattr(model.heads, "final_linear", None)
    if isinstance(lin, torch.nn.Linear) and lin.out_features == num_tasks:
        return lin

    candidates = []
    for name, m in model.heads.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_tasks:
            candidates.append((name, m))

    if len(candidates) == 1:
        return candidates[0][1]
    elif len(candidates) > 1:
        from stage1_config import CONFIG as _C
        preferred = [m for n, m in candidates if getattr(m, "in_features", None) == _C.get("shared_hidden", None)]
        if len(preferred) > 0:
            return preferred[-1]
        return candidates[-1][1]

    try:
        lin = model.heads.head[1]
        if isinstance(lin, torch.nn.Linear) and lin.out_features == num_tasks:
            return lin
    except Exception:
        pass

    if isinstance(model.heads, torch.nn.Linear) and model.heads.out_features == num_tasks:
        return model.heads

    raise RuntimeError("未能在 model.heads 中找到输出维度为任务数的 nn.Linear 作为任务头。")


def laplacian_reg_from_head(model, L: torch.Tensor) -> torch.Tensor:
    T = L.size(0)
    lin = _find_task_head_linear(model, T)
    W = lin.weight
    L = L.to(W.device, dtype=W.dtype)
    reg = torch.trace(W.t().matmul(L).matmul(W)) / (W.size(1) + 1e-9)
    return reg


# ==============================
# 训练主循环（pos_weighted BCE + Laplacian）
# ==============================
def run_train():
    warnings.filterwarnings("ignore")

    # 训练集路径
    train_csv_path, train_u50_path, pc6_path, aaindex_path, blosum_path, paac_path = get_paths()
    # 验证集路径
    val_csv_path = Path(CONFIG["val_csv_path"])
    val_u50_path = Path(CONFIG["val_u50_npy_path"])

    logger.info(
        f"Random seed fixed at {CONFIG['seed']} | torch_deterministic={CONFIG.get('torch_deterministic', False)}"
    )

    print(">>> Stage1 run_train 启动", flush=True)
    print(f"    Train CSV : {train_csv_path}", flush=True)
    print(f"    Train U50 : {train_u50_path}", flush=True)
    print(f"    Val   CSV : {val_csv_path}", flush=True)
    print(f"    Val   U50 : {val_u50_path}", flush=True)

    train_dataset = MultiTaskDataset(str(train_csv_path), str(train_u50_path))
    val_dataset = MultiTaskDataset(str(val_csv_path), str(val_u50_path))

    print(f">>> 数据集加载完成: Train={len(train_dataset)}, Val={len(val_dataset)}", flush=True)

    T = len(train_dataset.label_cols)
    hand_dim = 0

    if list(val_dataset.label_cols) != list(train_dataset.label_cols):
        raise ValueError("验证集 label 列与训练集不一致，请检查 test1.csv 与 train.csv 的列名/顺序。")

    # 用训练集实际维度覆盖 CONFIG 中的占位值
    CONFIG["dim_aaindex"] = train_dataset.dim_aaindex
    CONFIG["dim_paac"] = train_dataset.dim_paac
    CONFIG["dim_pc6"] = train_dataset.dim_pc6
    CONFIG["dim_blosum62"] = train_dataset.dim_blosum62
    CONFIG["dim_onehot"] = train_dataset.dim_onehot
    CONFIG["feat_dim_phys"] = train_dataset.feat_dim_phys

    # 全局 per-task 正例权重（pos_weight），缓解类别不平衡（基于训练集）
    labels_all = train_dataset.labels  # (N_train, T)
    pos_counts_global = labels_all.sum(axis=0).astype(np.float32)
    neg_counts_global = labels_all.shape[0] - pos_counts_global
    pos_counts_global = np.clip(pos_counts_global, 1.0, None)
    neg_counts_global = np.clip(neg_counts_global, 1.0, None)
    pos_weight_np = neg_counts_global / pos_counts_global
    pos_weight_np = np.clip(pos_weight_np, 1.0, 50.0).astype(np.float32)
    pos_weight = torch.from_numpy(pos_weight_np).to(CONFIG["device"])
    logger.info(f"Stage1 per-task pos_weight (clipped to 50): {pos_weight_np.tolist()}")

    logger.info(f"数据规模：Train={len(train_dataset)} | Val={len(val_dataset)} | T={T}")
    logger.info(
        f"per-residue 物化特征维度：AAIndex={train_dataset.dim_aaindex}, PAAC={train_dataset.dim_paac}, "
        f"PC6={train_dataset.dim_pc6}, BLOSUM62={train_dataset.dim_blosum62}, OneHot={train_dataset.dim_onehot}, "
        f"总={train_dataset.feat_dim_phys}"
    )

    train_loader, val_loader, _, _ = build_loaders(train_dataset, val_dataset)

    sampler_mode = CONFIG.get("sampler", "task_rotate").lower()
    if sampler_mode == "task_rotate":
        sampler_info = f"TaskRotate(pos={CONFIG.get('tr_pos_fraction', 0.65)}, power={CONFIG.get('tr_power', 1.5)})"
    elif sampler_mode == "pos_aware":
        sampler_info = f"PosAware(pos_fraction={CONFIG.get('pos_sampler_pos_fraction', 0.5)})"
    else:
        sampler_info = "Vanilla(shuffle)"

    # 标签图 & Laplacian（基于训练集标签）
    A_np, L_np = build_task_graph_and_laplacian(train_dataset.labels, CONFIG["lap_topk"])
    L = L_np.to(CONFIG["device"])
    assert (L.diag() >= 0).all(), "Laplacian diag 应非负"

    model = Stage1Model(num_tasks=T, hand_dim=hand_dim).to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    logger.info(
        f"任务加权=none | 对抗=N | "
        f"Laplacian(topk={CONFIG['lap_topk']}, λ_max={CONFIG['lap_weight_max']}) | 采样器={sampler_info}"
    )

    print(f">>> 开始训练，共 {CONFIG['epochs']} 个 epoch", flush=True)

    best_metric = -float("inf")
    no_improve = 0

    def init_cov():
        return np.zeros(T, dtype=np.int64), 0, 0.0, 0.0

    cov_epoch_counts, cov_steps, cov_act_accum, cov_poslab_accum = init_cov()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_taskloss, epoch_rawbce, epoch_lap = 0.0, 0.0, 0.0

        # 仅 Laplacian 权重调度
        lap_w = schedule_weight(epoch, CONFIG["epochs"], CONFIG["lap_weight_max"], CONFIG["lap_warmup_frac"])

        cov_epoch_counts[:] = 0
        cov_steps = 0
        cov_act_accum = 0.0
        cov_poslab_accum = 0.0

        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}",
            unit="batch",
        )
        for u50, mask, phys, y in pbar:
            u50 = u50.to(CONFIG["device"])
            mask = mask.to(CONFIG["device"])
            y = y.to(CONFIG["device"])
            phys = phys.to(CONFIG["device"])

            optimizer.zero_grad(set_to_none=True)
            logits, f_shared, _ = model(u50, mask, phys)

            # 未加权 BCE（日志用）
            bce_mat_raw = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            raw_bce = bce_mat_raw.mean(dim=0).sum().detach()

            # 带 per-task pos_weight 的 BCE（训练用）
            pw = pos_weight.view(1, -1)  # (1, T)，广播到 (B, T)
            bce_mat = F.binary_cross_entropy_with_logits(
                logits, y,
                reduction='none',
                pos_weight=pw
            )  # (B, T)
            per_task_losses = bce_mat.mean(dim=0)  # (T,)

            # 任务覆盖统计
            pos_counts_batch = (y == 1.0).sum(dim=0)
            valid_mask = (pos_counts_batch > 0)
            cov_epoch_counts += (pos_counts_batch.detach().cpu().numpy() > 0).astype(np.int64)
            cov_steps += 1
            cov_act_accum += float(valid_mask.sum().item())
            cov_poslab_accum += float((y == 1.0).sum().item())

            # 无不确定性加权：直接求和
            loss_tasks = per_task_losses.sum()

            # Laplacian 正则
            lap = laplacian_reg_from_head(model, L)
            lap = torch.clamp(lap, min=0.0)
            loss_struct = lap_w * lap

            total_loss = loss_tasks + loss_struct
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()

            epoch_taskloss += float(loss_tasks.item())
            epoch_rawbce += float(raw_bce.item())
            epoch_lap += float(loss_struct.item())

            pbar.set_postfix({
                "TaskLoss": f"{loss_tasks.item():.4f}",
                "RawBCE": f"{raw_bce.item():.4f}",
                "Lap": f"{loss_struct.item():.4f}",
                "λ_lap": f"{lap_w:.3f}",
            })

        nb = max(1, len(train_loader))
        epoch_taskloss /= nb
        epoch_rawbce /= nb
        epoch_lap /= nb

        cov_ratio = (cov_epoch_counts / max(1, cov_steps)).tolist()
        bad_k = min(CONFIG["lap_report_topk_bad"], T)
        worst_ids = np.argsort(cov_ratio)[:bad_k]
        worst_report = [(train_dataset.label_cols[i], round(float(cov_ratio[i]), 4)) for i in worst_ids]
        avg_act_tasks = cov_act_accum / max(1, cov_steps)
        avg_pos_labels = cov_poslab_accum / max(1, cov_steps)

        logger.info(
            f"[Epoch {epoch + 1}] Train TaskLoss={epoch_taskloss:.6f} | RawBCE={epoch_rawbce:.6f} | "
            f"Lap={epoch_lap:.6f} | λ_lap={lap_w:.3f} "
            f"| avg_act_tasks/batch={avg_act_tasks:.2f} | avg_pos_labels/batch={avg_pos_labels:.2f}"
        )
        logger.info(f"[Epoch {epoch + 1}] 覆盖率最低 {bad_k} 个任务：{worst_report}")

        if CONFIG["compute_val"]:
            macro_ap, ap_per_task, ap_report = auprc_on_val(
                model, val_loader, CONFIG["device"], T, train_dataset.label_cols
            )
            logger.info(
                f"[Epoch {epoch + 1}] Val Macro-AUPRC={macro_ap:.6f} | "
                f"Top{CONFIG['report_topk_tasks']}={ap_report}"
            )

            improved = macro_ap > (best_metric + 1e-8)
            if improved:
                best_metric = macro_ap
                no_improve = 0
                save_path = Path(CONFIG["save_dir"]) / CONFIG["save_name"]
                torch.save({
                    "state_dict": model.encoder.state_dict(),
                    "config": CONFIG,
                    "label_cols": train_dataset.label_cols,
                    "class_priors": ap_per_task,
                    "handcraft_scaler_mean": None,
                    "handcraft_scaler_std": None,
                    "has_handcraft": False,
                }, save_path)
                logger.info(f"✅ 保存共享编码器到: {str(save_path)}（Val Macro-AUPRC={macro_ap:.6f}）")
            else:
                no_improve += 1
                logger.info(
                    f"早停计数: {no_improve}/{CONFIG['early_stop_patience']} "
                    f"(best={best_metric:.6f})"
                )
                if no_improve >= CONFIG["early_stop_patience"]:
                    logger.info("⛔ 触发早停。")
                    break
        else:
            save_path = Path(CONFIG["save_dir"]) / CONFIG["save_name"]
            torch.save({
                "state_dict": model.encoder.state_dict(),
                "config": CONFIG,
                "label_cols": train_dataset.label_cols
            }, save_path)
            logger.info(f"✅ 保存共享编码器到: {str(save_path)}（无验证模式）")

    logger.info("训练完成（阶段一）。最佳共享编码器已产出，可用于阶段二。")
    print(">>> Stage1 训练结束", flush=True)


if __name__ == "__main__":
    print(">>> 调用 stage1_train.py 主入口", flush=True)
    run_train()
