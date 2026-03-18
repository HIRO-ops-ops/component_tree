import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import higra as hg
from _experiment_utils import create_experiment_output
import matplotlib.pyplot as plt

try:
    from utils import * # imshow, locate_resource
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

def load_nifti(filepath):
    """NIfTIデータを読み込む"""
    return nib.load(filepath).get_fdata().astype(np.float32)

def apply_gaussian_smoothing(volume, sigma=0.5):
    return gaussian_filter(volume, sigma=sigma)

def minmax_normalize(volume):
    vmin, vmax = np.min(volume), np.max(volume)
    if vmax - vmin < 1e-6: return volume
    return (volume - vmin) / (vmax - vmin)

def find_first_valley_threshold(volume, window_size=7, bins=100):
    data = volume[volume > 1e-5].flatten()
    counts, bin_edges = np.histogram(data, bins=bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(counts, window, mode='same')
    peak_idx = np.argmax(smoothed)
    valley_idx = peak_idx
    for i in range(peak_idx + 1, len(smoothed) - 1):
        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
            valley_idx = i
            break
    if valley_idx == peak_idx:
        valley_idx = min(peak_idx + 10, len(smoothed) - 1)
    return bin_centers[valley_idx], bin_centers, counts, smoothed

def save_histogram_with_threshold(bin_centers, counts, smoothed, threshold, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=0.01, color='blue', alpha=0.3, label="Original")
    plt.plot(bin_centers, smoothed, color='red', label="Moving Average")
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f"Auto Threshold: {threshold:.3f}")
    plt.title("Threshold Detection Result")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def save_overlay_images_3d(instances, volume, output_dir):
    """3Dインスタンスの最大断断面をオーバーレイ保存"""
    os.makedirs(output_dir, exist_ok=True)
    for i, inst in enumerate(instances):
        mask = inst["mask"]
        # インスタンスが存在する各スライスの面積を計算
        slice_areas = np.sum(mask, axis=(0, 1))
        max_z = np.argmax(slice_areas) # 最も面積が大きいスライスを選択
        
        plt.figure()
        plt.imshow(volume[:, :, max_z].T, cmap="gray", origin="lower")
        m = np.ma.masked_where(mask[:, :, max_z].T == 0, mask[:, :, max_z].T)
        plt.imshow(m, alpha=0.6, cmap="autumn", origin="lower")
        plt.title(f"Inst {i+1} - Max Slice {max_z} - Node {inst['node_id']}")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"instance_{i:03d}_node_{inst['node_id']}_slice{max_z}.png"))
        plt.close()

def save_numpy_instances_3d(instances, output_dir):
    """個別の3Dバイナリマスクを保存"""
    os.makedirs(output_dir, exist_ok=True)
    for i, inst in enumerate(instances):
        save_path = os.path.join(output_dir, f"instance_{i:03d}_node_{inst['node_id']}.npy")
        np.save(save_path, inst["mask"])


def extract_instances_3d(volume, voxel_threshold, intensity_threshold):
    print("   - 3D Max-tree構築中...")
    graph = hg.get_6_adjacency_graph(volume.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, volume)

    n_nodes = tree.num_vertices()
    area = hg.attribute_area(tree)
    parent = tree.parents()
    root = tree.root()

    # 属性計算 (2D版と同じ項目) 
    print("   - 属性(Depth, Mean, Persistence)計算中...")
    # Depth
    depth = np.zeros(n_nodes, dtype=np.int32)
    for n in reversed(range(n_nodes)): # ルートから順に
        if n != root:
            depth[n] = depth[parent[n]] + 1
    
    # Persistence
    persistence = np.zeros(n_nodes, dtype=np.float32)
    persistence[np.arange(n_nodes) != root] = altitudes[np.arange(n_nodes) != root] - altitudes[parent[np.arange(n_nodes) != root]]
    
    # Mean Intensity
    if hasattr(hg, "attribute_mean"):
        mean_intensity = hg.attribute_mean(tree, volume)
    else:
        mean_intensity = hg.attribute_mean_vertex_weights(tree, volume)

    #候補選択 
    candidates_mask = (area > voxel_threshold) & (altitudes > intensity_threshold)
    candidates_mask[root] = False
    candidate_set = set(np.where(candidates_mask)[0])

    # 入れ子構造の整理（親が候補なら自分は除外 = 最上位ノードのみ抽出）
    selected_nodes = [n for n in candidate_set if parent[n] not in candidate_set]
    print(f"   - 抽出された独立インスタンス数: {len(selected_nodes)}")

    #ラベルマップ & インスタンス詳細作成
    labelmap = np.zeros(volume.shape, dtype=np.int32)
    instances = []
    
    # 子リスト構築（再構成の高速化のため）
    children = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        if i != root:
            children[parent[i]].append(i)

    for i, node in enumerate(selected_nodes):
        # サブツリーのノードをすべて取得
        stack = [node]
        subtree = []
        while stack:
            curr = stack.pop()
            subtree.append(curr)
            stack.extend(children[curr])
        

        node_indicator = np.zeros(n_nodes, dtype=np.uint8)
        node_indicator[subtree] = 1
        deleted_nodes = np.ones(n_nodes, dtype=bool)
        deleted_nodes[subtree] = False
        deleted_nodes[root] = False
        
        mask_1d = hg.reconstruct_leaf_data(tree, node_indicator, deleted_nodes)
        mask_3d = mask_1d.reshape(volume.shape) > 0
        
        labelmap[mask_3d] = i + 1
        instances.append({
            "mask": mask_3d.astype(np.uint8),
            "node_id": int(node),
            "volume": int(area[node]),
            "mean_intensity": float(mean_intensity[node])
        })



    node_table = []
    tree_edges = []
    for n in range(n_nodes):
        node_table.append([
            0, n, parent[n], depth[n], altitudes[n], 
            persistence[n], area[n], mean_intensity[n]
        ])
        if n != root:
            tree_edges.append([0, parent[n], n])

    return instances, labelmap, np.array(node_table), np.array(tree_edges)



if __name__ == "__main__":
    root_output = create_experiment_output()
    path = r"BraTS20_Training_001_flair.nii"
    voxel_threshold = 500
    sigma = 0.5

    print("▶ 1. NIfTI読み込み")
    volume_raw = nib.load(path).get_fdata().astype(np.float32)
    
    print("▶ 2. 正規化")
    volume = minmax_normalize(volume_raw)
    
    print(f"▶ 3. ガウシアン平滑化 (sigma={sigma})")
    volume = apply_gaussian_smoothing(volume, sigma=sigma)
    
    print("▶ 4. 自動閾値検出")
    thresh, centers, counts, smoothed = find_first_valley_threshold(volume, window_size=7)
    print(f"   自動算出閾値: {thresh:.4f}")
    
    print("▶ 5. ヒストグラム保存")
    histogram_path = os.path.join(root_output, "histogram.png")
    save_histogram_with_threshold(centers, counts, smoothed, thresh, histogram_path)
    
    print("▶ 6. インスタンス抽出 (3D)")
    instances, labelmap, node_table, tree_edges = extract_instances_3d(volume, voxel_threshold, thresh)
    
    print(f"    抽出されたインスタンス数: {len(instances)}")
    if len(instances) > 0:
        print("▶ 7. インスタンス保存 (Overlay & Numpy)")
        overlays_dir = os.path.join(root_output, "overlays")
        numpy_dir = os.path.join(root_output, "numpy")
        save_overlay_images_3d(instances, volume_raw, overlays_dir)
        save_numpy_instances_3d(instances, numpy_dir)
        

        np.save(os.path.join(root_output, "labelmap.npy"), labelmap)
        
        print("▶ 8. ノード階層情報保存")
        print(f"    総ノード数: {len(node_table)}")
        np.save(os.path.join(root_output, "node_table.npy"), node_table)
        np.save(os.path.join(root_output, "tree_edges.npy"), tree_edges)
        
        print(f"▶ 保存完了: {root_output}")
    else:
        print("⚠️ 条件に合うインスタンスが見つかりませんでした。")