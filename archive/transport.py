import SimpleITK as sitk
import numpy as np
import os

mhd_path = r"Mr.Matsushita\HU_A0001_ans.mhd"
npy_path = mhd_path.replace(".mhd", ".npy")

# =====================
# 1. mhd経由で読み込み
# =====================
image = sitk.ReadImage(mhd_path)

# numpy配列へ変換
volume = sitk.GetArrayFromImage(image)
# 形状: (z, y, x)

# =====================
# 2. npyとして保存
# =====================
np.save(npy_path, volume)

print("保存完了:", npy_path)
print("shape:", volume.shape)
print("dtype:", volume.dtype)
print("spacing:", image.GetSpacing())