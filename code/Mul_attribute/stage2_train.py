
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单属性 Stage2 训练脚本（均衡二分类版本）
- 每次只训练一个属性
- train 作为训练集
- test 作为验证+测试：用于选取最优权重与阈值（按 ACC/AUC/pre/f1 偏好）
- 保持 stage1 适配与 stage2_model 架构不变
- 保留日志记录与数据保存功能

[MOD-TO] 支持 selection_policy.threshold_overrides：为指定属性单独设置 threshold_objective
"""

import json
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stage2_config import STAGE2
from stage2_model import Stage1FeatureExtractor, Stage2AttrModel
from stage1_config import CONFIG as S1C
from stage1_model import load_physchem_dicts

from stage2_untils import (
    AttrDataset,
    pad_collate,
    attention_kld,
    average_precision_score_scratch,
    roc_auc_scratch,
    fit_temperature_on_logits,
    scan_threshold_best,
    compute_epoch_score,
    get_selection_weights_for_attr,
    PhyschemFeatureBuilder,
    logger,
    set_seed,
)
TARGET_ATTR = [
    'anti_mammalian_cells', 'antibacterial', 'antibiofilm', 'anticancer', 'antifungal',
    'antigram-negative', 'antigram-positive', 'antihiv', 'antimrsa', 'antiparasitic', 'antiviral',
    'chemotactic', 'endotoxin', 'insecticidal'
]
# ========= 指定要训练的属性名与 JSON 路径 =========
# TARGET_ATTR = [
#     "insecticidal",
# ]
HP_JSON_PATH = Path(STAGE2["report_dir"]) / "stage2_attr_hparams_full.json"

set_seed(STAGE2["seed"])


def setup_deterministic():
    if S1C.get("torch_deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


setup_deterministic()


def build_scheduler_for_attr(optimizer, total_epochs: int, warmup_epochs: int, min_lr_ratio: float):
    if warmup_epochs <= 0 and (min_lr_ratio is None or min_lr_ratio <= 0.0):
        return None

    warm = max(0, int(warmup_epochs))
    min_ratio = float(min_lr_ratio)

    def lr_lambda(epoch: int):
        if warm > 0 and epoch < warm:
            return (epoch + 1) / max(1, warm)
        if total_epochs <= warm:
            return min_ratio
        t = (epoch - warm) / max(1, (total_epochs - warm))
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _tie_break_better(a: Dict[str, float], b: Dict[str, float], order: list) -> bool:
    for k in order:
        va = float(a.get(k, -1e18))
        vb = float(b.get(k, -1e18))
        if va > vb + 1e-12:
            return True
        if vb > va + 1e-12:
            return False
    return False


def calc_spe_mcc(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[float, float]:
    y_true = y_true.astype(np.int32)
    pred = (y_prob >= thr).astype(np.int32)

    TP = int(((pred == 1) & (y_true == 1)).sum())
    TN = int(((pred == 0) & (y_true == 0)).sum())
    FP = int(((pred == 1) & (y_true == 0)).sum())
    FN = int(((pred == 0) & (y_true == 1)).sum())

    eps = 1e-12
    spe = TN / (TN + FP + eps)
    mcc = (TP * TN - FP * FN) / math.sqrt(
        (TP + FP + eps) * (TP + FN + eps) * (TN + FP + eps) * (TN + FN + eps)
    )
    return float(spe), float(mcc)


# ===== [MOD-TO1] 新增：按属性获取 threshold_objective（支持 threshold_overrides）=====
def get_threshold_objective_for_attr(attr: str) -> Dict[str, float]:
    """
    返回该属性用于阈值扫描的 objective_weights（acc/pre/f1）。
    优先使用 selection_policy.threshold_overrides[attr]，
    否则使用 selection_policy.threshold_objective（全局默认）。
    """
    pol = STAGE2.get("selection_policy", {}) or {}
    base = dict(pol.get("threshold_objective", {"acc": 0.35, "pre": 0.25, "f1": 0.40}))
    overrides = pol.get("threshold_overrides", {}) or {}
    if attr in overrides and isinstance(overrides[attr], dict):
        return dict(overrides[attr])
    return base
# ===== [MOD-TO1] END =====


def evaluate_on_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attr: str,
    pos_tr: int,
) -> Dict[str, object]:
    """
    在 test 集合上：
    - 计算 probs（sigmoid logits）
    - 计算 AUC/AUPRC
    - 扫描阈值（按 threshold_objective 最大化）
    - 计算 epoch 选模分数（按 AUC+ACC+pre+F1 组合）
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for u50, mask, phys, y in test_loader:
            u50 = u50.to(device)
            mask = mask.to(device)
            phys = phys.to(device)
            logit, *_ = model(u50, mask, phys)
            p = torch.sigmoid(logit).detach().cpu().numpy()
            probs.append(p.astype(np.float32))
            labels.append(y.numpy().astype(np.int32))

    probs = np.concatenate(probs, axis=0).astype(np.float64)
    labels = np.concatenate(labels, axis=0).astype(np.int32)

    auprc = average_precision_score_scratch(labels, probs.astype(np.float32))
    auroc = roc_auc_scratch(labels, probs.astype(np.float32))

    # per-attr epoch selection weights
    sel_w = get_selection_weights_for_attr(attr, pos_tr=pos_tr)

    # ===== [MOD-TO2] 修改：thr_obj 支持 per-attr override =====
    thr_obj = get_threshold_objective_for_attr(attr)
    # ===== [MOD-TO2] END =====

    thr, m_thr, thr_score = scan_threshold_best(
        labels,
        probs,
        grid_points=STAGE2["threshold_scan"]["grid_points"],
        objective_weights=thr_obj,
        eps=float(STAGE2["threshold_scan"].get("eps", 1e-6)),
    )

    epoch_score = compute_epoch_score(auroc, m_thr, sel_w)

    return {
        "labels": labels,
        "probs": probs,
        "auprc": float(auprc) if np.isfinite(auprc) else float("nan"),
        "auroc": float(auroc) if np.isfinite(auroc) else float("nan"),
        "thr": float(thr),
        "thr_metrics": m_thr,
        "thr_score": float(thr_score),
        "epoch_score": float(epoch_score),
        "sel_weights": sel_w,
        "thr_objective": dict(thr_obj),
    }


def train_one_attr(attr: str, s1_ckpt_path: str, phys_builder: PhyschemFeatureBuilder):
    device = torch.device(STAGE2["device"])

    # ===== 数据 =====
    dtrain = AttrDataset(attr, STAGE2["train_root"], "train", phys_builder)
    dtest = AttrDataset(attr, STAGE2["test_root"], "test", phys_builder)

    ytr = np.array(dtrain.labels, dtype=np.int32)
    yte = np.array(dtest.labels, dtype=np.int32)

    Ntr = int(len(dtrain))
    Nte = int(len(dtest))
    pos_tr = int(ytr.sum())
    neg_tr = int(Ntr - pos_tr)
    pos_te = int(yte.sum())
    neg_te = int(Nte - pos_te)

    logger.info(
        f"[{attr}] N_train={Ntr} (pos={pos_tr}, neg={neg_tr}) | "
        f"N_test={Nte} (pos={pos_te}, neg={neg_te}) | "
        f"(NOTE: balanced data assumed; removed pos_weight & pos-aware sampler)"
    )

    train_loader = DataLoader(
        dtrain,
        batch_size=STAGE2["batch_size"],
        shuffle=True,
        num_workers=STAGE2["num_workers"],
        pin_memory=STAGE2["pin_memory"],
        collate_fn=pad_collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        dtest,
        batch_size=STAGE2["batch_size"],
        shuffle=False,
        num_workers=STAGE2["num_workers"],
        pin_memory=STAGE2["pin_memory"],
        collate_fn=pad_collate,
        drop_last=False,
    )

    # ===== 模型 =====
    s1_extractor = Stage1FeatureExtractor(s1_ckpt_path).to(device)
    model = Stage2AttrModel(
        stage1_extractor=s1_extractor,
        mode=STAGE2["mode_default"],
        raw_hidden=STAGE2["raw_branch_hidden"],
        fshared_hidden=STAGE2["fshared_hidden"],
        dropout=STAGE2["dropout"],
    ).to(device)

    def freeze_module(m: torch.nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    # 默认全可训练
    for p in model.parameters():
        p.requires_grad = True

    # ft_mode：小样本任务 head_only，大样本可 full/top
    thr_full = 2500
    thr_top = 500
    if pos_tr >= thr_full:
        ft_mode = "full"
    elif pos_tr >= thr_top:
        ft_mode = "top"
    else:
        ft_mode = "head_only"

    if ft_mode == "top":
        enc = model.stage1.encoder
        freeze_module(enc.norm_input)
        freeze_module(enc.conv_aaindex)
        freeze_module(enc.conv_paac)
        freeze_module(enc.conv_pc6)
        freeze_module(enc.conv_blosum62)
        freeze_module(enc.conv_onehot)
        freeze_module(enc.norm_aaindex)
        freeze_module(enc.norm_paac)
        freeze_module(enc.norm_pc6)
        freeze_module(enc.norm_blosum62)
        freeze_module(enc.norm_onehot)
        freeze_module(enc.norm_concat)
    elif ft_mode == "head_only":
        freeze_module(model.stage1.encoder)

    if hasattr(model, "set_ft_mode"):
        model.set_ft_mode(ft_mode)

    logger.info(f"[{attr}] ft_mode={ft_mode} | mode={STAGE2['mode_default']}")

    # ===== 优化器参数组 =====
    params_stage1, params_raw, params_head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("stage1."):
            params_stage1.append(p)
        elif name.startswith("raw."):
            params_raw.append(p)
        elif name.startswith("head."):
            params_head.append(p)
        else:
            params_head.append(p)

    base_lr = float(STAGE2["lr"])
    param_groups = []
    if params_head or params_raw:
        param_groups.append({"params": params_head + params_raw, "lr": base_lr})

    encoder_lr_scale_used = None
    if params_stage1:
        if ft_mode in {"top", "full"}:
            encoder_lr_scale_used = 0.5
        elif ft_mode == "shallow":
            encoder_lr_scale_used = 0.2

        if encoder_lr_scale_used is not None:
            param_groups.append({"params": params_stage1, "lr": base_lr * float(encoder_lr_scale_used)})

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=float(STAGE2["weight_decay"]))

    criterion = nn.BCEWithLogitsLoss()

    scheduler = build_scheduler_for_attr(
        optimizer,
        total_epochs=int(STAGE2["epochs"]),
        warmup_epochs=int(STAGE2["scheduler"].get("warmup_epochs", 0)),
        min_lr_ratio=float(STAGE2["scheduler"].get("min_lr_ratio", 1.0)),
    )

    raw_l2 = float(STAGE2["raw_feat_l2"])
    use_attn_kld = bool(STAGE2.get("use_attn_kld", False))
    attn_kld_weight = float(STAGE2.get("attn_kld_weight", 0.0))

    best = {
        "epoch": -1,
        "epoch_score": -float("inf"),
        "auroc": -float("inf"),
        "thr": 0.5,
        "thr_metrics": {"acc": 0.0, "pre": 0.0, "f1": 0.0, "sen": 0.0},
        "state": None,
        "tie": None,
    }
    no_improve = 0
    tiebreak_order = STAGE2["selection_policy"].get("epoch_tiebreak_order", ["f1", "pre", "acc", "auc"])

    # ===== [MOD-TO3] 新增：本属性固定使用的 thr_obj（训练评估与最终阈值一致）=====
    thr_obj_attr = get_threshold_objective_for_attr(attr)
    # ===== [MOD-TO3] END =====

    for epoch in range(int(STAGE2["epochs"])):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"[{attr}] Epoch {epoch + 1}/{STAGE2['epochs']}",
            unit="batch",
            leave=False,
        )
        for u50, mask, phys, y in pbar:
            u50 = u50.to(device)
            mask = mask.to(device)
            phys = phys.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logit, fshared, raw_feat, attn_s1, attn_raw = model(u50, mask, phys)

            loss = criterion(logit, y)
            loss = loss + raw_l2 * (raw_feat.pow(2).mean())
            if use_attn_kld:
                loss = loss + attn_kld_weight * attention_kld(attn_s1, attn_raw, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if scheduler is not None:
            scheduler.step()

        # ===== test 评估 + 阈值选择 + epoch_score =====
        eval_info = evaluate_on_test(model, test_loader, device, attr=attr, pos_tr=pos_tr)

        nb = max(1, len(train_loader))
        logger.info(
            f"[{attr}] Epoch {epoch + 1} | TrainLoss={epoch_loss / nb:.6f} | "
            f"AUROC={eval_info['auroc']:.6f} | "
            f"thr={eval_info['thr']:.4f} | "
            f"ACC={eval_info['thr_metrics']['acc']:.4f} | "
            f"PRE={eval_info['thr_metrics']['pre']:.4f} | "
            f"F1={eval_info['thr_metrics']['f1']:.4f} | "
            f"epoch_score={eval_info['epoch_score']:.6f} | "
            f"sel_w={eval_info['sel_weights']} | thr_obj={eval_info['thr_objective']}"
        )

        tie_dict = {
            "auc": eval_info["auroc"],
            "acc": eval_info["thr_metrics"]["acc"],
            "pre": eval_info["thr_metrics"]["pre"],
            "f1": eval_info["thr_metrics"]["f1"],
        }

        improved = (eval_info["epoch_score"] > best["epoch_score"] + 1e-12) or (
            abs(eval_info["epoch_score"] - best["epoch_score"]) <= 1e-12
            and _tie_break_better(tie_dict, best["tie"] or {}, tiebreak_order)
        )

        if improved:
            best["epoch"] = epoch + 1
            best["epoch_score"] = float(eval_info["epoch_score"])
            best["auroc"] = float(eval_info["auroc"])
            best["thr"] = float(eval_info["thr"])
            best["thr_metrics"] = dict(eval_info["thr_metrics"])
            best["tie"] = tie_dict
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            logger.info(
                f"[{attr}] 早停计数: {no_improve}/{STAGE2['early_stop_patience']} "
                f"(best_epoch={best['epoch']}, best_score={best['epoch_score']:.6f})"
            )
            if (epoch + 1) >= int(STAGE2["min_epochs"]) and no_improve >= int(STAGE2["early_stop_patience"]):
                logger.info(f"[{attr}] 提前停止（best_by=epoch_score on test）。")
                break

    # ===== 保存 best 权重 =====
    ckpt_dir = Path(STAGE2["ckpt_root"]) / attr
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    if best["state"] is not None:
        torch.save(best["state"], best_path)
    else:
        torch.save(model.state_dict(), best_path)

    # ===== 用 best 权重在 test 上：温度缩放 + 阈值再选择（最终推理用）=====
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.to(device).eval()

    logits_list, labels_list = [], []
    with torch.no_grad():
        for u50, mask, phys, y in test_loader:
            u50 = u50.to(device)
            mask = mask.to(device)
            phys = phys.to(device)
            logit, *_ = model(u50, mask, phys)
            logits_list.append(logit.detach().cpu().numpy().astype(np.float32))
            labels_list.append(y.numpy().astype(np.int32))

    logits_all = np.concatenate(logits_list, axis=0).astype(np.float32)
    labels_all = np.concatenate(labels_list, axis=0).astype(np.int32)

    T = fit_temperature_on_logits(logits_all, labels_all, device=device)

    probs_cal = 1.0 / (1.0 + np.exp(-(logits_all.astype(np.float64) / max(float(T), 1e-6))))
    auprc_cal = average_precision_score_scratch(labels_all, probs_cal.astype(np.float32))
    auroc_cal = roc_auc_scratch(labels_all, probs_cal.astype(np.float32))

    # ===== [MOD-TO4] 修改：最终阈值扫描也使用 per-attr thr_obj_attr =====
    thr_final, m_final, thr_score_final = scan_threshold_best(
        labels_all,
        probs_cal,
        grid_points=STAGE2["threshold_scan"]["grid_points"],
        objective_weights=thr_obj_attr,
        eps=float(STAGE2["threshold_scan"].get("eps", 1e-6)),
    )
    # ===== [MOD-TO4] END =====

    logger.info(
        f"[{attr}] FINAL(best_epoch={best['epoch']}) | "
        f"T={T:.4f} | AUROC_cal={auroc_cal:.6f} | "
        f"thr_final={thr_final:.4f} | ACC={m_final['acc']:.4f} | PRE={m_final['pre']:.4f} | F1={m_final['f1']:.4f} | "
        f"thr_obj={thr_obj_attr}"
    )

    # ===== 写回报告/配置 =====
    report_dir = Path(STAGE2["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / "stage2_metrics_summary.csv"
    spe_final, mcc_final = calc_spe_mcc(labels_all, probs_cal, thr_final)
    new_row = {
        "attr": attr,
        "Pre": round(float(m_final["pre"]), 4),
        "Sen": round(float(m_final["sen"]), 4),
        "Spe": round(float(spe_final), 4),
        "Acc": round(float(m_final["acc"]), 4),
        "F1": round(float(m_final["f1"]), 4),
        "MCC": round(float(mcc_final), 4),
        "AUC": round(float(auroc_cal), 4) if np.isfinite(auroc_cal) else float("nan"),
        "thr": float(thr_final),
        "T": float(T),
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["epoch_score"]),
        "ft_mode": ft_mode,
    }

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df = df[df["attr"] != attr]
        df = pd.concat([df, pd.DataFrame([new_row])], axis=0, ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(metrics_path, index=False)

    # stage2_hparams.json（推理用）
    hparams_simple_path = report_dir / "stage2_hparams.json"
    if hparams_simple_path.exists():
        with open(hparams_simple_path, "r", encoding="utf-8") as f:
            try:
                hparams_map: Dict[str, Dict[str, object]] = json.load(f)
            except json.JSONDecodeError:
                hparams_map = {}
    else:
        hparams_map = {}

    hparams_map[attr] = {
        "threshold": float(thr_final),
        "temperature": float(T),
        "mode": STAGE2["mode_default"],
        "ft_mode": ft_mode,
        "ckpt_best": str(best_path),
        # ===== [MOD-TO5] 可选：把 per-attr thr_obj 写进推理侧，便于复现 =====
        "threshold_objective": dict(thr_obj_attr),
        # ===== [MOD-TO5] END =====
    }
    with open(hparams_simple_path, "w", encoding="utf-8") as f:
        json.dump(hparams_map, f, ensure_ascii=False, indent=2)

    # stage2_attr_hparams_full.json（完整记录）
    if HP_JSON_PATH.exists():
        with open(HP_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                full_map: Dict[str, Dict[str, object]] = json.load(f)
            except json.JSONDecodeError:
                full_map = {}
    else:
        full_map = {}

    full_map[attr] = {
        "select_group": {
            "best_by": "epoch_score_on_test",
            "epoch_score_weights": get_selection_weights_for_attr(attr, pos_tr=pos_tr),
            "epoch_tiebreak_order": tiebreak_order,
            # ===== [MOD-TO6] 修改：记录实际使用的 per-attr thr_obj_attr =====
            "threshold_objective": dict(thr_obj_attr),
            # ===== [MOD-TO6] END =====
            "threshold_grid_points": int(STAGE2["threshold_scan"]["grid_points"]),
        },
        "opt_group": {
            "base_lr": float(base_lr),
            "encoder_lr_scale_used": float(encoder_lr_scale_used) if encoder_lr_scale_used is not None else 0.0,
            "scheduler_warmup_epochs": int(STAGE2["scheduler"].get("warmup_epochs", 0)),
            "scheduler_min_lr_ratio": float(STAGE2["scheduler"].get("min_lr_ratio", 1.0)),
            "min_epochs": int(STAGE2["min_epochs"]),
            "early_stop_patience": int(STAGE2["early_stop_patience"]),
        },
        "ft_group": {
            "ft_mode": ft_mode,
            "ft_mode_pos_thresholds": {"full": thr_full, "top": thr_top},
        },
        "reg_group": {
            "raw_feat_l2": float(raw_l2),
            "use_attn_kld": bool(use_attn_kld),
            "attn_kld_weight": float(attn_kld_weight),
        },
        "eval_group": {
            "temperature": float(T),
            "threshold": float(thr_final),
            "auroc_cal": float(auroc_cal) if np.isfinite(auroc_cal) else float("nan"),
            "auprc_cal": float(auprc_cal) if np.isfinite(auprc_cal) else float("nan"),
            "metrics_at_thr": dict(m_final),
        },
        "data_group": {
            "train_pos": int(pos_tr),
            "train_neg": int(neg_tr),
            "test_pos": int(pos_te),
            "test_neg": int(neg_te),
        },
        "arch_group": {
            "mode": STAGE2["mode_default"],
            "raw_branch_hidden": int(STAGE2["raw_branch_hidden"]),
            "fshared_hidden": int(STAGE2["fshared_hidden"]),
            "dropout": float(STAGE2["dropout"]),
        },
    }

    HP_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_map, f, ensure_ascii=False, indent=2)

    logger.info("===== 单属性训练完成 =====")
    logger.info(f"- metrics_summary: {metrics_path}")
    logger.info(f"- simple_hparams:  {hparams_simple_path}")
    logger.info(f"- full_hparams:    {HP_JSON_PATH}")

    return {
        "attr": attr,
        "best_path": str(best_path),
        "thr": float(thr_final),
        "T": float(T),
        "auroc_cal": float(auroc_cal) if np.isfinite(auroc_cal) else float("nan"),
        "metrics": dict(m_final),
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["epoch_score"]),
        "ft_mode": ft_mode,
    }


def run_single_attr(attr: str, hp_json_path: Path = HP_JSON_PATH):
    warnings.filterwarnings("ignore")

    # ===== 读取 stage1 ckpt config 并更新 S1C =====
    ckpt_path = STAGE2["stage1_ckpt_path"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    s1cfg = ckpt.get("config", {})
    if not isinstance(s1cfg, dict):
        raise ValueError("stage1 ckpt 中未找到合法的 config 字典")

    S1C.update(s1cfg)

    required_keys = ["aaindex_path", "paac_path", "pc6_path", "blosum62_path"]
    missing = [k for k in required_keys if k not in S1C or S1C[k] is None]
    if missing:
        raise KeyError(f"stage1 ckpt 的 config 缺少关键 physchem 路径键: {missing}")

    logger.info(f"[S1CFG from ckpt] aaindex_path={S1C['aaindex_path']}")
    logger.info(f"[S1CFG from ckpt] paac_path={S1C['paac_path']}")
    logger.info(f"[S1CFG from ckpt] pc6_path={S1C['pc6_path']}")
    logger.info(f"[S1CFG from ckpt] blosum62_path={S1C['blosum62_path']}")

    # ===== per-residue 理化特征 dict =====
    AAIndex_dict, PAAC_dict, PC6_dict, BLOSUM62_dict, OneHot_dict, \
        dim_aaindex, dim_paac, dim_pc6, dim_blosum62, dim_onehot, feat_dim_phys = load_physchem_dicts(
            S1C["aaindex_path"], S1C["paac_path"], S1C["pc6_path"], S1C["blosum62_path"]
        )

    # 校验维度一致
    for k, got in [
        ("dim_aaindex", dim_aaindex),
        ("dim_paac", dim_paac),
        ("dim_pc6", dim_pc6),
        ("dim_blosum62", dim_blosum62),
        ("dim_onehot", dim_onehot),
        ("feat_dim_phys", feat_dim_phys),
    ]:
        if k in S1C and S1C[k] is not None:
            if int(S1C[k]) != int(got):
                raise ValueError(f"physchem 维度不一致: {k} ckpt={S1C[k]} vs loaded={got}")

    phys_builder = PhyschemFeatureBuilder(
        AAIndex_dict, PAAC_dict, PC6_dict, BLOSUM62_dict, OneHot_dict, feat_dim_phys
    )

    train_one_attr(attr, STAGE2["stage1_ckpt_path"], phys_builder)


if __name__ == "__main__":
    for attr in TARGET_ATTR:
        run_single_attr(attr)
