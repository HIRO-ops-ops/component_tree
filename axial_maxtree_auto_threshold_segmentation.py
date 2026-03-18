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

def load_nifti(filepath):
    """NIfTIデータを読み込む"""
    return nib.load(filepath).get_fdata().astype(np.float32)

def minmax_normalize(volume):
    """0.0-1.0に正規化"""
    v_min, v_max = np.min(volume), np.max(volume)
    return (volume - v_min) / (v_max - v_min) if v_max - v_min > 1e-6 else volume

def apply_gaussian_smoothing(volume, sigma=0.5):
    """3Dガウシアンフィルタでノイズを平滑化"""
    return gaussian_filter(volume, sigma=sigma)

def find_first_valley_threshold(volume, window_size=5, bins=100):
    """ヒストグラムから自動的に「谷」を見つけ、閾値を決定する"""
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
    """検出された閾値を含むヒストグラムを画像保存"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=0.01, color='blue', alpha=0.3, label="Original")
    plt.plot(bin_centers, smoothed, color='red', label="Moving Average")
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f"Auto Threshold: {threshold:.3f}")
    plt.title("Threshold Detection Result")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def extract_instances_axial(volume, area_threshold, intensity_threshold):

    H, W, D = volume.shape
    all_instances = []
    node_table = []
    tree_edges = []

    for z in range(D):

        slice2d = volume[:, :, z]
        if np.std(slice2d) < 1e-5:
            continue

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

        # ---深さ(ルートを0として子に向かって増加)
        depth = np.zeros(n_nodes, dtype=np.int32)
        # Higraのノード順序（トポロジカル順）を利用して効率的に計算
        for node in reversed(range(n_nodes)):
            if node != root:
                depth[node] = depth[parent[node]] + 1

        #（ベクトル化）
        persistence = np.zeros(n_nodes, dtype=np.float32)
        mask = np.arange(n_nodes) != root
        persistence[mask] = altitudes[parent[mask]] - altitudes[mask]

        #node_table / tree_edges
        for node in range(n_nodes):
            node_table.append([
                z, node, parent[node],
                depth[node],
                altitudes[node],
                persistence[node],
                area[node],
                mean_intensity[node]
            ])
            tree_edges.append([z, parent[node], node])
        # 閾値判定ノード抽出
        
        candidates = np.where(
            (area > area_threshold) &
            (altitudes > intensity_threshold) &
            (np.arange(n_nodes) != root)
        )[0]

        candidate_set = set(candidates)
        
        #子リスト構築 
        children = [[] for _ in range(n_nodes)]
        for i in range(n_nodes):
            if i != root:
                children[parent[i]].append(i)

        #親が候補に含まれないノードのみ採用（＝最上位ノード） 
        selected_nodes = [n for n in candidates if parent[n] not in candidate_set]

        #各ノードを根とする部分木全体を再構成
        for node in selected_nodes:
            stack = [node]
            subtree = []
            while stack:
                n = stack.pop()
                subtree.append(n)
                stack.extend(children[n])
            node_mask = np.zeros(n_nodes, dtype=np.int32)
            node_mask[subtree] = 1
            deleted_nodes = np.ones(n_nodes, dtype=bool)
            deleted_nodes[subtree] = False
            deleted_nodes[root] = False
            mask_1d = hg.reconstruct_leaf_data(tree, node_mask, deleted_nodes)
            mask2d = mask_1d.reshape(slice2d.shape) > 0
            if np.sum(mask2d) == 0:
                continue
            all_instances.append({
                "slice": z,
                "mask": mask2d.astype(np.uint8),
                "node_id": int(node),
                "area": int(area[node]),
                "mean_intensity": float(mean_intensity[node])
            })

    return all_instances, np.array(node_table), np.array(tree_edges)

def get_subtree_nodes(node, children):
    stack = [node]
    subtree = []
    while stack:
        n = stack.pop()
        subtree.append(n)
        stack.extend(children[n])
    return subtree

def save_overlay_images(instances, volume, output_dir):
    """元画像にマスクを重ねて画像保存"""
    os.makedirs(output_dir, exist_ok=True)
    for i, inst in enumerate(instances):
        z, mask = inst["slice"], inst["mask"]
        plt.figure()
        plt.imshow(volume[:, :, z], cmap="gray")
        m = np.ma.masked_where(mask == 0, mask)
        plt.imshow(m, alpha=0.6, cmap="autumn")
        plt.title(f"not_filtered_Slice {z} - Instance {i+1} - Node {inst['node_id']}")
        plt.axis("off")
        save_path = os.path.join(output_dir, f"not_filtered_slice_{z}_instance_{i}.png")
        plt.savefig(save_path)
        plt.close()

def save_numpy_instances(instances, output_dir):
    """個別のバイナリマスクを保存"""
    os.makedirs(output_dir, exist_ok=True)
    for i, inst in enumerate(instances):
        np.save(os.path.join(output_dir, f"not_filtered_slice_{inst['slice']}_instance_{i}.npy"), inst["mask"])


if __name__ == "__main__":

    root_output = create_experiment_output()
    
    #パラメータ設定
    path = r""
    area_threshold = 50
    sigma = 0.5
    
    print("▶ 1. NIfTI読み込み")
    volume = load_nifti(path)
    print("▶ 2. 正規化")
    volume = minmax_normalize(volume)
    print(f"▶ 3. ガウシアン平滑化 (sigma={sigma})")
    volume = apply_gaussian_smoothing(volume, sigma=sigma)
    print("▶ 4. 自動閾値検出")
    thresh, centers, counts, smoothed = find_first_valley_threshold(volume, window_size=7)
    print(f"   自動算出された閾値: {thresh:.4f}")
    print("▶ 5. ヒストグラム保存")
    histogram_path = os.path.join(root_output, "histogram.png")
    save_histogram_with_threshold(centers, counts, smoothed, thresh, histogram_path)
    print("▶ 6. インスタンス抽出 (altitude)")
    instances, node_table, tree_edges = extract_instances_axial(
        volume,
        area_threshold,
        thresh)
    print(f"   抽出インスタンス数: {len(instances)}")
    if len(instances) > 0:
        overlays_dir = os.path.join(root_output, "overlays")
        numpy_dir = os.path.join(root_output, "numpy")
        save_overlay_images(instances, volume, overlays_dir)
        save_numpy_instances(instances, numpy_dir)
        labelmap = np.zeros(volume.shape, dtype=np.int32)
        for i, inst in enumerate(instances):
            labelmap[:, :, inst["slice"]][inst["mask"].astype(bool)] = i + 1
        np.save(os.path.join(root_output, "labelmap.npy"), labelmap)
    print("▶ 7. ノード階層情報保存")
    np.save(os.path.join(root_output, "node_table.npy"), node_table)
    np.save(os.path.join(root_output, "tree_edges.npy"), tree_edges)

    print("▶ すべての処理が完了しました。")
    print(f"   総ノード数: {len(node_table)}")
