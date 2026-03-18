import nibabel as nib
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from _experiment_utils import create_experiment_output

root_output = create_experiment_output()

# 1. フォルダパスの設定
folder_path = r""
file_list = sorted(glob.glob(os.path.join(folder_path, '*.nii*')))

# 2. グラフのレイアウト設定
num_files = len(file_list)
cols = 3
rows = (num_files + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

# 3. ループで処理
for i, file_path in enumerate(file_list):
    filename = os.path.basename(file_path)
    img = nib.load(file_path)
    data = img.get_fdata()
    
    mask = data > 0
    foreground_data = data[mask]
    
    if 'seg' in filename:
        # ラベルデータはそのまま

        plot_data = foreground_data
        title_label = "Original (Label)"
    else:

        # 1. 外れ値の除去 (上位0.5%の値をカットして分布を安定させる)
        upper_limit = np.percentile(foreground_data, 99.5)
        clipped_data = np.clip(foreground_data, a_min=None, a_max=upper_limit)
        
        # 2. Min-Max正規化 (0.0 ~ 1.0 に変換)
        min_val = np.min(clipped_data)
        max_val = np.max(clipped_data)
        plot_data = (clipped_data - min_val) / (max_val - min_val)
        
        title_label = "Min-Max Normalized (0-1)"
    
    # ヒストグラム作成
    axes[i].hist(plot_data, bins=50, color='orange' if 'seg' in filename else 'lightblue', edgecolor='black')
    axes[i].set_title(f"{title_label}\n{filename}", fontsize=10)
    axes[i].grid(True, alpha=0.3)

# 空白のグラフを非表示
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.savefig(os.path.join(root_output, "histgram.png"))
plt.tight_layout()
