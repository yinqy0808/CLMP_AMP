"""
模块：加载理化特征字典并返回相应的映射和维度
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_physchem_dicts(AAINDEX_PATH, PAAC_PATH, PC6_PATH, BLOSUM62_PATH):
    """
    构建理化特征字典（AAIndex, PAAC, PC6, BLOSUM62, One-Hot），
    并计算各维度及总维度。
    返回：
        AAIndex_dict, PAAC_dict, PC6_dict, BLOSUM62_dict, OneHot_dict,
        dim_AAIndex, dim_PAAC, dim_PC6, dim_BLOSUM62, dim_OneHot,
        feat_dim_phys
    """
    # --- AAIndex 字典 ---
    AAIndex_dict = {}
    with open(AAINDEX_PATH, 'r') as f:
        lines = [l for l in f.read().split('\n') if l.strip()]
    header = lines[0].strip().split('\t')
    aa_list = [x for x in header[1:] if x]
    data_rows = []
    for row in lines[1:]:
        parts = [p for p in row.strip().split('\t') if p]
        if len(parts) < len(aa_list) + 1:
            continue  # 跳过格式不完整的行
        values = [float(x) for x in parts[1:len(aa_list)+1]]
        data_rows.append(values)
    index_matrix = np.array(data_rows, dtype=np.float32)  # (N,20)
    for j, aa in enumerate(aa_list):
        AAIndex_dict[aa] = index_matrix[:, j]
    AAIndex_dict['X'] = np.zeros((index_matrix.shape[0],), dtype=np.float32)
    dim_AAIndex = index_matrix.shape[0]

    # --- PAAC 字典 ---
    PAAC_dict = {}
    with open(PAAC_PATH, 'r') as f:
        lines = [l for l in f.read().split('\n') if l.strip()]
    header = lines[0].strip().split('\t')
    aa_list_paac = [x for x in header[1:] if x]
    paac_rows = []
    for row in lines[1:]:
        parts = [p for p in row.strip().split('\t') if p]
        if len(parts) < len(aa_list_paac) + 1:
            continue
        paac_rows.append([float(x) for x in parts[1:len(aa_list_paac)+1]])
    mat = np.array(paac_rows, dtype=np.float32).T  # (20,3)
    for j, aa in enumerate(aa_list_paac):
        PAAC_dict[aa] = mat[j, :]
    PAAC_dict['X'] = np.zeros((mat.shape[1],), dtype=np.float32)
    dim_PAAC = mat.shape[1]

    # --- PC6 字典 ---
    PC6_dict = {}
    with open(PC6_PATH, 'r') as f:
        lines = [l for l in f.read().split('\n') if l.strip()]
    header_pc6 = lines[0].split()
    for row in lines[1:]:
        parts = [p for p in row.strip().split() if p]
        aa = parts[0]
        values = [float(x) for x in parts[1:]]
        PC6_dict[aa] = np.array(values, dtype=np.float32)
    PC6_dict['X'] = np.zeros((len(values),), dtype=np.float32)
    dim_PC6 = len(values)

    # --- BLOSUM62 字典 ---
    BLOSUM62_dict = {}
    with open(BLOSUM62_PATH, 'r') as f:
        lines = [l for l in f.read().split('\n') if l.strip()]
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
    BLOSUM62_dict['X'] = np.zeros((len(aa_list_blosum),), dtype=np.float32)
    dim_BLOSUM62 = len(aa_list_blosum)

    # --- One-Hot 字典 ---
    amino_chars = "ARNDCQEGHILKMFPSTWYVX"
    OneHot_dict = {}
    for i, aa in enumerate(amino_chars):
        vec = np.zeros((len(amino_chars),), dtype=np.float32)
        vec[i] = 1.0
        OneHot_dict[aa] = vec
    OneHot_dict['X'] = np.zeros((len(amino_chars),), dtype=np.float32)
    dim_OneHot = len(amino_chars)
    logger.info("  OneHot_dict: 映射 %d 个氨基酸 → 向量长度 %d", len(OneHot_dict), dim_OneHot)

    # 计算总维度
    feat_dim_phys = dim_AAIndex + dim_PAAC + dim_PC6 + dim_BLOSUM62 + dim_OneHot

    return (
        AAIndex_dict, PAAC_dict, PC6_dict,
        BLOSUM62_dict, OneHot_dict,
        dim_AAIndex, dim_PAAC, dim_PC6,
        dim_BLOSUM62, dim_OneHot,
        feat_dim_phys
    )