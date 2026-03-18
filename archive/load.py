import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import higra as hg
from _experiment_utils import create_experiment_output

# ==================================
# 1. 自動閾値検出 (移動平均 + 谷検出)
# ==================================
def find_first_valley_threshold(volume, window_size=5, bins=100):
    """
    ヒストグラムの移動平均を計算し、最大の山（正常組織）の右側にある
    最初の「谷」を抽出対象の閾値として自動決定する。
    """
    # 0に近い値（背景）を除外して、脳領域の画素値のみをフラットな配列にする
    data = volume[volume > 1e-5].flatten()
    
    # 0.0〜1.0の範囲でヒストグラムを作成
    counts, bin_edges = np.histogram(data, bins=bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 移動平均（Moving Average）による平滑化：細かいギザギザ（ノイズ）を除去する
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(counts, window, mode='same')

    # 1. ヒストグラム内で最も頻度が高い点（ピーク）を特定
    # 通常、Flair画像では健康な脳組織がこのピークに該当する
    peak_idx = np.argmax(smoothed)
    
    # 2. ピーク位置から右側（明るい方向）へ向かって、最初の「谷（極小値）」を探す
    # 谷 = 前後の値よりも小さい点
    valley_idx = peak_idx
    for i in range(peak_idx + 1, len(smoothed) - 1):
        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
            valley_idx = i
            break
    
    # 谷が見つからなかった場合の安全策（ピークの少し右側を閾値とする）
    if valley_idx == peak_idx:
        valley_idx = min(peak_idx + 10, len(smoothed) - 1)
        print("  ⚠️ 谷が検出できなかったため、フォールバック値を使用します。")

    # インデックスを実際の画素値(0.0-1.0)に変換
    threshold_val = bin_centers[valley_idx]
    return threshold_val, bin_centers, counts, smoothed

# ==================================
# 2. ヒストグラム出力 (可視化)
# ==================================
def save_histogram_with_threshold(bin_centers, counts, smoothed, threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, counts, label="Histogram")
    plt.plot(bin_centers, smoothed, label="Smoothed")
    plt.axvline(threshold, linestyle="--", label=f"Threshold: {threshold:.3f}")
    plt.legend()
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    save_path = os.path.join(output_dir, "histogram.png")
    plt.savefig(save_path)
    plt.close()

# ==================================
# 3. NIfTI読み込みと正規化
# ==================================
def load_nifti(filepath):
    """NIfTIファイルを読み込み、numpy配列(float32)として返す"""
    nii = nib.load(filepath)
    return nii.get_fdata().astype(np.float32)

def minmax_normalize(volume):
    """全体の画素値を 0.0 〜 1.0 の範囲にスケーリングする"""
    v_min, v_max = np.min(volume), np.max(volume)
    if v_max - v_min < 1e-6:
        return volume
    return (volume - v_min) / (v_max - v_min)

# ==================================
# 4. Axial 2Dインスタンス抽出 (Higra使用)
# ==================================
def extract_instances_axial(volume, area_threshold, intensity_threshold):
    """
    各スライスからMax-treeを用いて、
    条件に合う各ノードを個別インスタンスとして抽出
    """
    H, W, D = volume.shape
    all_instances = []

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

        # 条件を満たすノード取得
        condition = (area > area_threshold) & (mean_intensity > intensity_threshold)
        selected_nodes = np.where(condition)[0]

        for node in selected_nodes:

            # ルート除外（必要なら）
            if node == tree.root():
                continue

            # --- ノード単体マスク再構築 ---
            deleted_nodes = np.ones(tree.num_vertices(), dtype=bool)
            deleted_nodes[node] = False
            deleted_nodes[tree.root()] = False

            node_altitudes = np.zeros(tree.num_vertices(), dtype=np.int32)
            node_altitudes[node] = 1

            mask_1d = hg.reconstruct_leaf_data(tree, node_altitudes, deleted_nodes)
            mask = mask_1d.reshape(slice2d.shape) > 0

            # 面積0は除外
            if np.sum(mask) == 0:
                continue

            all_instances.append({
                "slice": z,
                "mask": mask.astype(np.uint8),
                "node_id": int(node),
                "area": int(area[node]),
                "mean_intensity": float(mean_intensity[node])
            })

    return all_instances

# ==================================
# 5. 結果の保存（画像・NumPy・ラベル）
# ==================================
def save_overlay_images(instances, volume, output_dir="output/overlays"):
    """元画像の上に抽出マスクを重ねて保存する（背景は透明）"""
    os.makedirs(output_dir, exist_ok=True)
    for i, inst in enumerate(instances):
        z, mask = inst["slice"], inst["mask"]
        plt.figure()
        plt.imshow(volume[:,:,z], cmap="gray")
        # マスクの0（背景）を透明にし、1の部分を暖色で表示
        m = np.ma.masked_where(mask == 0, mask)
        plt.imshow(m, alpha=0.6, cmap="autumn", interpolation="none")
        plt.title(f"Slice {z} - Instance {i}")
        plt.axis("off")
        plt.savefig(f"{output_dir}/slice_{z}_instance_{i}.png")
        plt.close()

def save_numpy_instances(instances, output_dir="output/numpy"):
    """各スライスのマスクをバイナリのNumPy形式で個別に保存する"""
    os.makedirs(output_dir, exist_ok=True)
    for i, inst in enumerate(instances):
        np.save(f"{output_dir}/slice_{inst['slice']}_instance_{i}.npy", inst["mask"])

def instances_to_labelmap(instances, shape):
    """抽出された全インスタンスを1つの3Dボリューム（ラベルマップ）に統合する"""
    labelmap = np.zeros(shape, dtype=np.int32)
    for i, inst in enumerate(instances):
        # 抽出された順に 1, 2, 3... とIDを振る
        labelmap[:, :, inst["slice"]][inst["mask"].astype(bool)] = i + 1
    return labelmap

# ==================================
# 6. メインパイプライン
# ==================================
def run_pipeline(nifti_path, area_threshold=200):
    """全工程を統合して実行する関数"""
    print(f"▶ 1. NIfTI読み込み: {os.path.basename(nifti_path)}")
    volume = load_nifti(nifti_path)
    
    print("▶ 2. 正規化 (Min-Max)")
    volume = minmax_normalize(volume)

    print("▶ 3. 自動閾値検出 (移動平均法)")
    # window_sizeを大きくするとより滑らかになり、小さくするとより詳細に追従する
    thresh, centers, counts, smoothed = find_first_valley_threshold(volume, window_size=7)
    print(f"   検出された自動閾値: {thresh:.4f}")

    print("▶ 4. ヒストグラム保存")
    save_histogram_with_threshold(centers, counts, smoothed, thresh, output_dir=root_output)

    print("▶ 5. インスタンス抽出 (Higra Max-tree)")
    instances = extract_instances_axial(volume, area_threshold, thresh)
    print(f"   抽出されたスライス数: {len(instances)}")

    if len(instances) > 0:
        print("▶ 6. 結果保存中 (overlays / numpy / labelmap)")
        save_overlay_images(instances, volume, output_dir=os.path.join(root_output, "overlays"))
        save_numpy_instances(instances, output_dir=os.path.join(root_output, "numpy"))
        
        # 3Dラベルマップの生成と保存
        labelmap = instances_to_labelmap(instances, volume.shape)
        np.save(os.path.join(root_output, "labelmap.npy"), labelmap)
    else:
        print("▶ 条件を満たす領域が見つかりませんでした。")
    
    print("▶ 全工程が完了しました。")
    return instances


# ==================================
# スクリプト実行
# ==================================
if __name__ == "__main__":
    
    root_output = create_experiment_output()
    # NIfTIファイルのフルパスを指定
    path = r"C:\Users\nishi\OneDrive - 近畿大学\MedICoT_Lab\BraTS\BraTS_project\BraTS20_Training_001_flair.nii"
    
    # パイプライン実行（面積閾値は状況に応じて調整）
    run_pipeline(path, area_threshold=200)