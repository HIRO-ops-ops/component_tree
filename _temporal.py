import numpy as np
import higra as hg
import nibabel as nib
import matplotlib.pyplot as plt

def segment_brats_axial_multimodal(path_dict, slice_idx=80, n_regions=60):
    """
    BraTSデータのAxial断面をマルチモーダルでセグメンテーションする
    """
    # 1. データの読み込みと正規化（内部関数）
    def load_and_norm(path):
        data = nib.load(path).get_fdata()[:, :, slice_idx]
        # 背景（0）を除いた部分で標準化を行うのがコツ
        mask = data > 0
        if np.any(mask):
            mean = data[mask].mean()
            std = data[mask].std()
            data[mask] = (data[mask] - mean) / (std + 1e-8)
        return data

    # 4つの画像を読み込み（ここから関数の内部なのでインデントを揃える）
    imgs = {k: load_and_norm(v) for k, v in path_dict.items()}

    # 全チャネルをスタック (H, W, 4)
    # これにより各ピクセルが4次元の特徴ベクトルを持つ
    multimodal_axial = np.stack([imgs['flair'], imgs['t1ce'], imgs['t2'], imgs['t1']], axis=-1)

    # 2. グラフ構築
    size = multimodal_axial.shape[:2]
    graph = hg.get_4_adjacency_graph(size)

    # 3. エッジの重み付け (4次元空間でのL2距離)
    edge_weights = hg.weight_graph(graph, multimodal_axial, hg.WeightFunction.L2)

    # 4. 階層の構築
    # Higraで最も標準的で確実に動作する single_linkage を使用します
    # (より複雑な hierarchical_clustering 系が環境によりエラーになるため)
    tree, altitudes = hg.binary_partition_tree_single_linkage(graph, edge_weights)

    # 5. ノイズ除去（小さなノードをフィルタリング）
    # 30ピクセル以下の小さな領域は脳の構造として無視する
    tree, altitudes = hg.filter_small_nodes_from_tree(tree, altitudes, 30)
    
    # 6. 領域抽出（水平カット）
    labels = hg.labelisation_horizontal_cut_from_num_regions(tree, altitudes, n_regions)

    # 7. 表示
    plt.figure(figsize=(24, 6))
    
    # (1) FLAIR
    plt.subplot(1, 5, 1); plt.imshow(imgs['flair'], cmap='gray')
    plt.title("FLAIR (Edema)"); plt.axis('off')

    # (2) T1ce
    plt.subplot(1, 5, 2); plt.imshow(imgs['t1ce'], cmap='gray')
    plt.title("T1ce (Core/Enhancing)"); plt.axis('off')

    # (3) Multi-modal Saliency Map
    sm = hg.saliency(tree, altitudes)
    sm_k = hg.graph_4_adjacency_2_khalimsky(graph, sm)
    plt.subplot(1, 5, 3); plt.imshow(sm_k**0.15, cmap='hot')
    plt.title("Integrated Boundaries"); plt.axis('off')

    # (4) Segmentation Labels
    plt.subplot(1, 5, 4); plt.imshow(labels, cmap='nipy_spectral')
    plt.title(f"Segmentation (k={n_regions})"); plt.axis('off')

    # (5) Overlay
    plt.subplot(1, 5, 5); plt.imshow(imgs['flair'], cmap='gray')
    # labelsを整数型に変換して等高線を描画
    plt.contour(labels.astype(np.int32), colors='red', linewidths=0.5)
    plt.title("Overlay on FLAIR"); plt.axis('off')

    plt.tight_layout()
    plt.show()

# パス設定（raw文字列 r'' を使用）
paths = {
    'flair': r'C:\Users\nishi\OneDrive - 近畿大学\MedICoT_Lab\BraTS\BraTS_project\dataset\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii',
    't1ce':  r'C:\Users\nishi\OneDrive - 近畿大学\MedICoT_Lab\BraTS\BraTS_project\dataset\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii',
    't2':    r'C:\Users\nishi\OneDrive - 近畿大学\MedICoT_Lab\BraTS\BraTS_project\dataset\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t2.nii',
    't1':    r'C:\Users\nishi\OneDrive - 近畿大学\MedICoT_Lab\BraTS\BraTS_project\dataset\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1.nii'
}

# 関数の実行
segment_brats_axial_multimodal(paths)