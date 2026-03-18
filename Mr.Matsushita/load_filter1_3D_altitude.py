import os
import numpy as np
import higra as hg
from _experiment_utils import create_experiment_output
import matplotlib.pyplot as plt

# ==================================
# 前処理
# ==================================
def minmax_normalize(volume):
    vmin, vmax = np.min(volume), np.max(volume)
    if vmax - vmin < 1e-6:
        return volume
    return (volume - vmin) / (vmax - vmin)

def save_histogram(volume, output_path, bins=100):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = volume[volume > 1e-5].flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins)
    plt.title("Intensity Histogram (Normalized)")
    plt.xlabel("Intensity (0-1)")
    plt.ylabel("Voxel Count")
    plt.savefig(output_path)
    plt.close()

# ==================================
# データ読み込み (.npy)
# ==================================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    
    data = np.load(path).astype(np.float32)
    
    if data.ndim == 4:
        print(f"   - 4Dデータ検出: {data.shape} → 最初のチャンネル使用")
        data = data[0, :, :, :]
        
    return data

# ==================================
# 3Dインスタンス抽出
# ==================================
def extract_instances_3d(volume, voxel_threshold, intensity_threshold):
    print("   - 3D Max-tree構築中...")
    graph = hg.get_6_adjacency_graph(volume.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, volume)

    n_nodes = tree.num_vertices()
    area = hg.attribute_area(tree)
    parent = tree.parents()
    root = tree.root()

    candidates_mask = (area > voxel_threshold) & (altitudes > intensity_threshold)
    candidates_mask[root] = False
    candidate_indices = np.where(candidates_mask)[0]

    if len(candidate_indices) == 0:
        return [], None

    is_candidate = np.zeros(n_nodes, dtype=bool)
    is_candidate[candidate_indices] = True
    selected_nodes = [n for n in candidate_indices if not is_candidate[parent[n]]]

    print(f"   - 抽出インスタンス数: {len(selected_nodes)}")

    labelmap = np.zeros(volume.shape, dtype=np.int32)
    instances = []

    for i, node in enumerate(selected_nodes):
        node_indicator = np.zeros(n_nodes, dtype=np.uint8)
        node_indicator[node] = 1
        deleted_nodes = np.ones(n_nodes, dtype=bool)
        deleted_nodes[node] = False
        deleted_nodes[root] = False
        mask_1d = hg.reconstruct_leaf_data(tree, node_indicator, deleted_nodes)
        mask_3d = mask_1d.reshape(volume.shape) > 0
        labelmap[mask_3d] = i + 1
        instances.append(mask_3d)

    return instances, labelmap

# ==================================
# メイン処理
# ==================================
if __name__ == "__main__":

    root_output = create_experiment_output()
    path = r"Mr.Matsushita\HU_A0001_pt.npy"

    voxel_threshold = 10
    intensity_threshold = 0.60  # ★ 手動調整
    print("▶ 1. データ読み込み")
    volume = load_data(path)

    print("▶ 2. 正規化")
    volume = minmax_normalize(volume)

    print("▶ 3. ヒストグラム保存")
    histogram_path = os.path.join(root_output, "histogram.png")
    save_histogram(volume, histogram_path)

    print("▶ 4. インスタンス抽出")
    instances, labelmap = extract_instances_3d(
        volume,
        voxel_threshold,
        intensity_threshold
    )

    if instances:
        np.save(os.path.join(root_output, "labelmap.npy"), labelmap)
        print(f"▶ 完了: {root_output}")
    else:
        print("⚠ 条件に合うインスタンスがありません")