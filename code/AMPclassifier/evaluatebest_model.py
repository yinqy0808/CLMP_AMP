#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
独立测试脚本（变量配置版）
- 复用训练时的数据管线（U50 标准化 + 理化特征 + 构图 + Ragged Dataset）
- 从 training.log 解析后处理所需的温度 T 与阈值 thr（解析失败回退 T=1.0, thr=0.5）
- 日志记录到 evaluate.log，并在控制台同步输出
- 可选导出每条样本的预测明细 CSV
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)

# =============== 配置区（按需修改） ===============

# 模型与日志
MODEL_DIR   = "best_model1"                      # 训练时保存的 SavedModel 目录（ModelCheckpoint 的目录）
TRAIN_LOG   = "best_model1/training.log"         # 训练日志路径（用于解析 T 与 thr）
EVAL_LOG    = "best_model1/evaluate.log"         # 本脚本的评估日志输出路径

# 数据路径
TRAIN_CSV_PATH = "../example_data/diff_train.csv"     # 训练集 CSV（用于计算标准化统计）
TRAIN_U50_NPY  = "../example_data/diff_U50_train.npy" # 训练集 U50（用于计算标准化统计）

TEST_CSV_PATH  = "../example_data/diff_val.csv"       # 测试集 CSV（包含 seq 与 label）
TEST_U50_NPY   = "../example_data/diff_U50_val.npy"   # 测试集 U50

# 理化字典文件
AAINDEX_PATH  = "/media/aita4090/yinqy/PGAT-ABPp-main/data_diff/AAindex.txt"
PAAC_PATH     = "/media/aita4090/yinqy/PGAT-ABPp-main/data_diff/PAAC.txt"
PC6_PATH      = "/media/aita4090/yinqy/PGAT-ABPp-main/data_diff/6-pc.txt"
BLOSUM62_PATH = "/media/aita4090/yinqy/PGAT-ABPp-main/data_diff/blosum62.txt"

# 推理与导出
BATCH_SIZE = 32
OUT_CSV    = None  # 例如 "predictions_test.csv"；为 None 则不导出

# =============== 依赖你工程的模块（需可导入） ===============
from GraphFromPDB_bestModel import build_edges_from_adj_matrix, Dataset
from physchem_features import load_physchem_dicts
# 如需自定义对象反序列化，可打开下面两行，并在 load_model 里传 custom_objects
# from Model_bestModel import GATModel
# CUSTOM_OBJECTS = {"GATModel": GATModel}
CUSTOM_OBJECTS = None

# =============== 日志配置 ===============
os.makedirs(os.path.dirname(EVAL_LOG) or ".", exist_ok=True)
logger = logging.getLogger("evaluate_test")
logger.setLevel(logging.INFO)
for h in list(logger.handlers):
    logger.removeHandler(h)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(EVAL_LOG, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)

np.random.seed(42)
tf.random.set_seed(42)


def split_embeddings(all_emb: np.ndarray, seqs):
    lens = [len(s) for s in seqs]
    assert all_emb.shape[0] == sum(lens), "U50 行数与序列长度不符"
    out, idx = [], 0
    for L in lens:
        out.append(all_emb[idx:idx + L])
        idx += L
    return out


def parse_calibration_from_log(log_path: str):
    """
    从 training.log 解析温度 T 与阈值 thr。
    匹配示例：
      最佳温度 T=1.234，对应 NLL=...
      最佳阈值 = 0.567，对应验证集 F1 = ...
    兼容英文关键字；失败则回退默认值。
    """
    T, thr = None, None
    if not os.path.isfile(log_path):
        logger.warning("未找到训练日志：%s，将使用默认 T=1.0, thr=0.5", log_path)
        return 1.0, 0.5

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # 温度
    m_T = re.search(r"(最佳温度|Best\s*temperature)[^T]*T\s*=\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if m_T:
        try:
            T = float(m_T.group(2))
        except Exception:
            T = None

    # 阈值
    m_thr = re.search(r"(最佳阈值|Best\s*threshold)[^=]*=\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if m_thr:
        try:
            thr = float(m_thr.group(2))
        except Exception:
            thr = None

    if T is None:
        logger.warning("日志中未解析到温度 T，使用默认 T=1.0")
        T = 1.0
    if thr is None:
        logger.warning("日志中未解析到阈值 thr，使用默认 thr=0.5")
        thr = 0.5

    logger.info("解析得到温度 T=%.6f，阈值 thr=%.6f", T, thr)
    return T, thr


def build_phys_maker(AAIndex_dict, PAAC_dict, PC6_dict, BLOSUM62_dict, OneHot_dict, feat_dim_phys: int):
    """返回 make_phys(seq) 的闭包，避免重复捕获大字典。"""
    def make_phys(seq: str):
        L = len(seq)
        arr = np.zeros((L, feat_dim_phys), dtype=np.float32)
        for i, c in enumerate(seq):
            AA = c.upper() if c.upper() in OneHot_dict else 'X'
            v1 = AAIndex_dict.get(AA, AAIndex_dict['X'])
            v2 = PAAC_dict.get(AA, PAAC_dict['X'])
            v3 = PC6_dict.get(AA, PC6_dict['X'])
            v4 = BLOSUM62_dict.get(AA, BLOSUM62_dict['X'])
            v5 = OneHot_dict.get(AA, OneHot_dict['X'])
            arr[i] = np.concatenate([v1, v2, v3, v4, v5], axis=0)
        return arr
    return make_phys


def compute_standardization_stats(df_train: pd.DataFrame,
                                  u50_train_all: np.ndarray,
                                  make_phys_fn):
    """用训练集计算 U50 与理化特征的均值/方差。"""
    logger.info("计算训练集 U50 标准化统计")
    u50_list_train = split_embeddings(u50_train_all, df_train["seq"].values)
    u50_concat = np.vstack(u50_list_train)
    u50_mean = u50_concat.mean(axis=0)
    u50_std = u50_concat.std(axis=0) + 1e-6

    logger.info("计算训练集理化特征标准化统计")
    raw_phys_train = [make_phys_fn(s) for s in df_train["seq"].values]
    phys_concat = np.vstack(raw_phys_train)
    phys_mean = phys_concat.mean(axis=0)
    phys_std = phys_concat.std(axis=0) + 1e-6
    return (u50_mean, u50_std), (phys_mean, phys_std)


def build_nodes_for_split(df_split: pd.DataFrame,
                          u50_all: np.ndarray,
                          make_phys_fn,
                          u50_mean, u50_std,
                          phys_mean, phys_std):
    """标准化并拼接 U50 + 理化特征，返回节点特征列表（每条序列一个 np.ndarray）。"""
    u50_list = split_embeddings(u50_all, df_split["seq"].values)
    u50_list = [(u - u50_mean) / u50_std for u in u50_list]

    raw_phys_list = [make_phys_fn(s) for s in df_split["seq"].values]
    phys_list = [(p - phys_mean) / phys_std for p in raw_phys_list]

    node_list = [np.concatenate([u50_list[i], phys_list[i]], axis=1) for i in range(len(df_split))]
    return node_list


def safe_metrics(y_true, y_pred, y_score):
    """计算一组稳健的二分类指标。"""
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spe = tn / max(1, (tn + fp))
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    f1 = f1_score(y_true, y_pred)

    return {
        "ACC": acc, "PRE": pre, "SEN": rec, "SPE": spe,
        "MCC": mcc, "F1": f1, "AUC": auc
    }


def main():
    script_name = os.path.basename(globals().get("__file__", "evaluate_test_vars.py"))
    logger.info("启动独立测试脚本：%s", script_name)

    # 1) 读取 CSV
    logger.info("读取训练/测试 CSV")
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_test  = pd.read_csv(TEST_CSV_PATH)
    logger.info("数据规模：训练 %d，测试 %d", len(df_train), len(df_test))

    # 2) 加载理化特征字典
    logger.info("加载理化特征字典")
    (AAIndex_dict, PAAC_dict, PC6_dict,
     BLOSUM62_dict, OneHot_dict,
     dim_AAIndex, dim_PAAC, dim_PC6,
     dim_BLOSUM62, dim_OneHot, feat_dim_phys) = load_physchem_dicts(
        AAINDEX_PATH, PAAC_PATH, PC6_PATH, BLOSUM62_PATH
    )
    make_phys_fn = build_phys_maker(AAIndex_dict, PAAC_dict, PC6_dict, BLOSUM62_dict, OneHot_dict, feat_dim_phys)

    # 3) 标准化统计（基于训练集）
    logger.info("加载训练集 U50（mmap 方式）")
    u50_train_all = np.load(TRAIN_U50_NPY, mmap_mode="r")
    (u50_mean, u50_std), (phys_mean, phys_std) = compute_standardization_stats(df_train, u50_train_all, make_phys_fn)

    # 4) 准备测试集节点特征（标准化 + 拼接）
    logger.info("加载测试集 U50（mmap 方式）并构建节点特征")
    u50_test_all = np.load(TEST_U50_NPY, mmap_mode="r")
    node_list_test = build_nodes_for_split(df_test, u50_test_all, make_phys_fn,
                                           u50_mean, u50_std, phys_mean, phys_std)
    logger.info("拼接后节点特征维度 = %d", node_list_test[0].shape[1])

    # 5) 构图（测试集）
    logger.info("构造测试集边（含权重/类型）")
    edges_test_rt, edgew_test_rt, edget_test_rt = build_edges_from_adj_matrix(df_test)

    # 6) Ragged Dataset
    logger.info("构造 Ragged Dataset（测试集）")
    rt_node_test = tf.ragged.constant(node_list_test, dtype=tf.float32)
    test_dataset = Dataset((rt_node_test, edges_test_rt, edgew_test_rt, edget_test_rt),
                           df_test["label"].values, BATCH_SIZE, shuffle=False)

    # 7) 加载模型
    logger.info("加载 SavedModel：%s", MODEL_DIR)
    try:
        model = keras.models.load_model(MODEL_DIR, custom_objects=CUSTOM_OBJECTS)
    except Exception as e1:
        logger.warning("含编译信息加载失败（%s），尝试 compile=False 方式", e1)
        try:
            model = keras.models.load_model(MODEL_DIR, custom_objects=CUSTOM_OBJECTS, compile=False)
        except Exception as e2:
            logger.error("模型加载失败：%s", e2)
            raise

    # 8) 从训练日志解析温度与阈值
    best_T, best_thr = parse_calibration_from_log(TRAIN_LOG)

    # 9) 推理 + 校准 + 阈值化
    logger.info("开始测试集推理")
    raw_prob = model.predict(test_dataset).ravel()
    eps = 1e-7
    raw_prob = np.clip(raw_prob, eps, 1 - eps)
    logits = np.log(raw_prob / (1 - raw_prob))
    p_cal  = 1.0 / (1.0 + np.exp(-logits / best_T))

    y_true = df_test["label"].values.astype(int)
    y_pred = (p_cal > best_thr).astype(int)

    # 10) 指标
    metrics = safe_metrics(y_true, y_pred, p_cal)
    logger.info("测试集指标（校准+最优阈值）："
                "ACC=%.4f, PRE=%.4f, SEN=%.4f, SPE=%.4f, MCC=%.4f, F1=%.4f, AUC=%.4f",
                metrics["ACC"], metrics["PRE"], metrics["SEN"], metrics["SPE"],
                metrics["MCC"], metrics["F1"], metrics["AUC"])

    # 记录一份报告 JSON（可选）
    report_path = os.path.join(MODEL_DIR, "test_report.json")
    payload = {
        "temperature": float(best_T),
        "threshold": float(best_thr),
        "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else float(v)) for k, v in metrics.items()},
        "n_test": int(len(df_test))
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("已保存测试报告：%s", report_path)

    # 11) 导出明细
    if OUT_CSV:
        out_df = df_test.copy()
        out_df["prob_raw"] = raw_prob
        out_df["logit"] = logits
        out_df["prob_calibrated"] = p_cal
        out_df["pred"] = y_pred
        out_df.to_csv(OUT_CSV, index=False)
        logger.info("已导出预测明细：%s", OUT_CSV)

    logger.info("评估完成，日志已写入：%s", EVAL_LOG)


if __name__ == "__main__":
    main()
