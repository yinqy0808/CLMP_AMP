###Process PDB files to obtain CSV format file
import csv
import warnings
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import os

from pandas.tests.io.formats.style.test_bar import df_pos

warnings.filterwarnings("ignore")



def load_pdb(pdbfile):
    try:
        parser = PDBParser()
        structure = parser.get_structure(pdbfile.split('.')[0], pdbfile)
        residues = [r for r in structure.get_residues()]

        records = SeqIO.parse(pdbfile, 'pdb-atom')
        seqs = [str(r.seq) for r in records]

        distances = np.empty((len(residues), len(residues)))
        for x in range(len(residues)):
            for y in range(len(residues)):
                try:
                    one = residues[x]["CA"].get_coord()  # 确保CA原子存在
                    two = residues[y]["CA"].get_coord()
                    distances[x, y] = np.linalg.norm(one - two)
                except KeyError:
                    distances[x, y] = np.nan  # 如果没有CA原子，距离设置为NaN
        return distances, seqs[0]
    except Exception as e:
        print(f"Error loading {pdbfile}: {e}")
        return None, None

def load_cmap(pdbfile, cmap_thresh):
    D, seq = load_pdb(pdbfile)
    if D is None:
        return None, None
    A = np.zeros(D.shape, dtype=np.float32)
    A[D < cmap_thresh] = 1
    print(f"Generated adjacency matrix for {pdbfile}, shape: {A.shape}, unique values: {np.unique(A)}")
    return A, seq



def read_files(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            A, seq = load_cmap(file_path, CMAP_THRESH)
            if A is not None and seq is not None:
                data.append({
                    'filename': filename,
                    'A': str(A.tolist()),
                    'seq': seq,
                    'label': label
                })
            else:
                print(f"Skipping {filename} due to invalid contact map.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    print(f"Processed {len(data)} files from {folder_path}")
    return data




# 参数配置
CMAP_THRESH = 10
# positive_folder = '../dataset/test/abp'
# negative_folder = '../dataset/test/nonabp'
# positive_folder = '../dataset/main/abp'
# negative_folder = '../dataset/main/nonabp'
# positive_folder = './data_diff/processed_data/val/amp'
# negative_folder = './data_diff/processed_data/val/nonamp'
# positive_folder = './data_diff/processed_data/train/amp'
# negative_folder = './data_diff/processed_data/train/nonamp'
positive_folder = '../data_diff/processed_data/independent_data/amp'
negative_folder = '../data_diff/processed_data/independent_data/nonamp'

output_folder = 'example_data/diff_independent_data.csv'
# 数据收集
positive_data = read_files(positive_folder, 1)
negative_data = read_files(negative_folder, 0)


combined_data = positive_data + negative_data

df = pd.DataFrame(combined_data)
print(f"DataFrame has {len(df)} rows and columns: {df.columns.tolist()}")
df.to_csv(output_folder, index=False)
print(f"CSV saved to {output_folder}")

# 打印验证列名
print("生成的CSV列名：", df.columns.tolist())