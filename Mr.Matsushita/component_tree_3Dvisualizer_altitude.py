import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import higra as hg

# ============================================
# 1. データ読み込み
# ============================================
npy_path = r"Mr.Matsushita\HU_A0001_pt.npy"

data = np.load(npy_path)
print(f"Data Loaded: {data.shape}")

volume = data.astype(np.float32)

# ============================================
# 2. 3D Max-Tree 構築
# ============================================
print("Building 3D Max-Tree...")
graph = hg.get_6_adjacency_graph(volume.shape)
tree, altitudes = hg.component_tree_max_tree(graph, volume)

num_nodes = tree.num_vertices()
num_leaves = tree.num_leaves()
parents = tree.parents()
areas = hg.attribute_area(tree)

print(f"Total nodes: {num_nodes:,}")

# ============================================
# 3. 面積剪定（ノイズ除去）
# ============================================
print("Pruning small components...")
pruned_altitudes = np.copy(altitudes)

for i in range(num_nodes - 1, -1, -1):
    p = parents[i]
    if p != i and areas[i] < 50:
        pruned_altitudes[i] = pruned_altitudes[p]

# ============================================
# 4. レイアウト計算
# ============================================
print("Calculating layout...")

node_weights = np.zeros(num_nodes, dtype=np.float64)
node_weights[:num_leaves] = np.arange(num_leaves)

sum_x = hg.accumulate_parallel(tree, node_weights, hg.Accumulators.sum)

x_coords = sum_x / areas
y_coords = pruned_altitudes.astype(np.float64)

# ---- x軸正規化（0-1）----
x_min, x_max = x_coords.min(), x_coords.max()
x_coords_norm = (x_coords - x_min) / (x_max - x_min + 1e-8)

# ---- y軸：外れ値除去 → 正規化 ----
y_low, y_high = np.percentile(y_coords, [1, 99])
y_clip = np.clip(y_coords, y_low, y_high)

y_min, y_max = y_clip.min(), y_clip.max()
y_coords_norm = (y_clip - y_min) / (y_max - y_min + 1e-8)

# ============================================
# 5. エッジ抽出
# ============================================
print("Extracting edges...")

edges_trunk = []
edges_branch = []

for i in range(num_nodes):
    p = parents[i]
    if p != i and y_coords_norm[i] > y_coords_norm[p]:
        line = [(x_coords_norm[i], y_coords_norm[i]),
                (x_coords_norm[p], y_coords_norm[p])]

        if areas[i] > 2000:
            edges_trunk.append(line)
        elif areas[i] > 100:
            edges_branch.append(line)

# 分岐点抽出
out_degree = np.zeros(num_nodes, dtype=np.int32)
for p in parents:
    out_degree[p] += 1

branch_nodes = (out_degree > 1) & (areas > 100)

# ============================================
# 6. 可視化
# ============================================
fig, ax = plt.subplots(figsize=(12, 7), dpi=110)

# 細枝
ax.add_collection(LineCollection(edges_branch, linewidths=0.6, alpha=0.3))

# 幹
ax.add_collection(LineCollection(edges_trunk, linewidths=1.8, alpha=0.7))

# 分岐点
ax.scatter(
    x_coords_norm[branch_nodes],
    y_coords_norm[branch_nodes],
    s=3,
    alpha=0.6
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_title(
    f"3D Max-Tree (Area Pruned, Normalized)\nNodes: {num_nodes:,} | Volume: {volume.shape}",
    fontsize=16
)

ax.set_xlabel("Spatial Distribution (Normalized)", fontsize=12)
ax.set_ylabel("Normalized Intensity (0-1)", fontsize=12)

ax.tick_params(labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()