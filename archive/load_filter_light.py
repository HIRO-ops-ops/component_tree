import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import higra as hg
from scipy.ndimage import gaussian_filter
from _experiment_utils import create_experiment_output

try:
    from utils import * # imshow, locate_resource
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

def apply_gaussian_smoothing(volume, sigma=0.5):
    """3Dガウシアンフィルタでノイズを平滑化"""
    return gaussian_filter(volume, sigma=sigma)

def find_first_valley_threshold(volume, window_size=5, bins=100):
    """ヒストグラムから自動閾値を決定"""
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

def load_nifti(filepath):
    """NIfTI読み込み"""
    return nib.load(filepath).get_fdata().astype(np.float32)

def minmax_normalize(volume):
    """0.0-1.0正規化"""
    v_min, v_max = np.min(volume), np.max(volume)
    return (volume - v_min) / (v_max - v_min) if v_max - v_min > 1e-6 else volume

def extract_each_instance_axial(volume, area_threshold, intensity_threshold):
    H, W, D = volume.shape
    all_instances = []
    
    mid = D // 2
    target_slices = range(mid - 5, mid + 6) 
    print(f"   [検証モード] スライス {mid-5} 〜 {mid+5} を個別に抽出中...")

    for z in target_slices:
        slice2d = volume[:, :, z]
        if np.std(slice2d) < 1e-5: continue
        
        graph = hg.get_4_adjacency_graph(slice2d.shape)
        tree, altitudes = hg.component_tree_max_tree(graph, slice2d)
        area = hg.attribute_area(tree)
        
        if hasattr(hg, "attribute_mean"):
            mean_intensity = hg.attribute_mean(tree, slice2d)
        else:
            mean_intensity = hg.attribute_mean_vertex_weights(tree, slice2d)
        
        parent = tree.parents()
        root = tree.root()
        n_nodes = tree.num_vertices()

        # 1. まず条件に合うノードをすべて見つける
        candidate_indices = np.where((area > area_threshold) & (mean_intensity > intensity_threshold))[0]
        
        if len(candidate_indices) == 0: continue

        # 2. 【重要】マトリョーシカ現象を防ぐフィルタリング
        # 「条件を満たし、かつその親は条件を満たさない」ノードだけを選ぶ
        # これにより、入れ子構造の中の一番大きい（外側の）マスクだけが残ります
        candidate_set = set(candidate_indices)
        top_level_nodes = [n for n in candidate_indices if parent[n] not in candidate_set]

        print(f"      スライス {z}: 候補 {len(candidate_indices)}個 -> 個 {len(top_level_nodes)}選出")

        for node in top_level_nodes:
            if node == root: continue

            # マスクの作成
            deleted_nodes = np.ones(n_nodes, dtype=bool)
            deleted_nodes[node] = False
            deleted_nodes[root] = False
            node_altitudes = np.zeros(n_nodes, dtype=np.int32)
            node_altitudes[node] = 1
            
            mask_1d = hg.reconstruct_leaf_data(tree, node_altitudes, deleted_nodes)
            mask = mask_1d.reshape(slice2d.shape) > 0

            if np.sum(mask) == 0: continue

            all_instances.append({
                "slice": z,
                "mask": mask.astype(np.uint8),
                "node_id": int(node),
                "area": int(area[node]),
                "intensity": float(mean_intensity[node])
            })
            
    return all_instances

# ==================================
# 3. 画像保存（インスタンスごと）
# ==================================
def save_each_instance_images(instances, volume, output_dir):
    """
    インスタンスごとに独立した画像ファイルを生成する。
    ファイル名に Node ID を含めることで区別する。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, inst in enumerate(instances):
        z = inst["slice"]
        mask = inst["mask"]
        node_id = inst["node_id"]
        
        plt.figure(figsize=(6, 6))
        # 下地に白黒のMRIスライスを表示
        plt.imshow(volume[:, :, z], cmap="gray")
        
        # 抽出された特定のインスタンスだけを赤色で重ねる
        m = np.ma.masked_where(mask == 0, mask)
        plt.imshow(m, alpha=0.6, cmap="autumn", interpolation="none")
        
        plt.title(f"Slice: {z} | Instance Index: {i} | Node ID: {node_id}\nArea: {inst['area']} | Mean: {inst['intensity']:.3f}")
        plt.axis("off")
        
        # ファイル名にスライス番号とインデックスを含める
        save_path = os.path.join(output_dir, f"slice_{z}_instance_{i}_node_{node_id}.png")
        plt.savefig(save_path)
        plt.close()

# ==================================
# 4. メイン処理
# ==================================
if __name__ == "__main__":
    
    root_output = create_experiment_output()

    path = r"C:\Users\nishi\OneDrive - 近畿大学\MedICoT_Lab\BraTS\BraTS_project\BraTS20_Training_001_flair.nii"
    
    # 設定
    area_thresh = 200
    sigma_val = 0.5
    
    # 1. 前処理
    vol = load_nifti(path)
    vol = minmax_normalize(vol)
    vol = apply_gaussian_smoothing(vol, sigma=sigma_val)

    # 2. 自動閾値検出
    thresh, centers, counts, smoothed = find_first_valley_threshold(vol, window_size=7)
    print(f"▶ 自動算出された閾値: {thresh:.4f}")

    # 3. インスタンス別抽出
    # スライスごとにまとめず、全ての候補ノードを個別に保持します
    instances = extract_each_instance_axial(vol, area_threshold=area_thresh, intensity_threshold=thresh)
    print(f"▶ 抽出された総インスタンス数: {len(instances)}")

    # 4. 画像保存
    if len(instances) > 0:
        output_path = os.path.join(root_output, "each_instances")
        print(f"▶ 画像を保存中 (root): {root_output}")
        save_each_instance_images(instances, vol, output_path)
    else:
        print("⚠️ インスタンスが見つかりませんでした。")

    print("✅ 完了")