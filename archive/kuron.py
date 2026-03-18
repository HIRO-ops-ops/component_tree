import numpy as np
import matplotlib.pyplot as plt
import higra as hg
import networkx as nx

# ============================================
# 1. Numpy 配列(.npy)の読み込みと前処理
# ============================================
npy_path = r"C:\Users\nishi\OneDrive - 近畿大学\MedICoT_Lab\BraTS\BraTS_project\HU_A0001_pt.npy"  # ★ ここに .npy のパスを指定

try:
    data = np.load(npy_path)
except FileNotFoundError:
    print(f"エラー: {npy_path} が見つかりません。パスを確認してください。")
    exit()

# 3D画像の場合は z 軸方向のスライスを選択できるようにする
if data.ndim >= 3:
    z_max = data.shape[2] - 1
    z_center = data.shape[2] // 2
else:
    z_max = 0
    z_center = 0

# スライス処理関数（NIfTI版と同じ）
def process_slice(z_index):
    if data.ndim >= 3:
        image_2d = data[:, :, int(z_index)]
    else:
        image_2d = data

    # 量子化（0〜255）でHigra用に整数化
    img_min = np.min(image_2d)
    img_max = np.max(image_2d)
    if img_max > img_min:
        image_proc = (image_2d - img_min) / (img_max - img_min)
    else:
        image_proc = image_2d * 0
    image_proc = (image_proc * 255).astype(np.uint8)

    # グラフ／ツリー構築
    graph = hg.get_4_adjacency_graph(image_proc.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, image_proc)

    num_vertices_loc = tree.num_vertices()
    num_leaves_loc = tree.num_leaves()
    children_loc = [[] for _ in range(num_vertices_loc)]
    G_loc = nx.DiGraph()
    for node in range(num_vertices_loc):
        parent = tree.parent(node)
        if parent != node:
            G_loc.add_edge(parent, node)
            children_loc[parent].append(node)

    stack_loc = [(tree.root(), 0)]
    post_order_loc = []
    while stack_loc:
        node, child_idx = stack_loc.pop()
        if child_idx < len(children_loc[node]):
            stack_loc.append((node, child_idx + 1))
            stack_loc.append((children_loc[node][child_idx], 0))
        else:
            post_order_loc.append(node)

    x_coords_loc = np.zeros(num_vertices_loc, dtype=np.float32)
    leaf_idx_loc = 0
    for node in post_order_loc:
        if not children_loc[node]:
            x_coords_loc[node] = leaf_idx_loc
            leaf_idx_loc += 1
        else:
            x_coords_loc[node] = sum(x_coords_loc[c] for c in children_loc[node]) / len(children_loc[node])

    y_coords_loc = np.array(altitudes, dtype=np.float32)
    pos_loc = {node: (x_coords_loc[node], y_coords_loc[node]) for node in range(num_vertices_loc)}

    x_range_loc = leaf_idx_loc if leaf_idx_loc > 0 else 1
    y_range_loc = max(altitudes) - min(altitudes) or 1

    return {
        "image": image_proc,
        "tree": tree,
        "altitudes": altitudes,
        "G": G_loc,
        "children": children_loc,
        "pos": pos_loc,
        "x_coords": x_coords_loc,
        "y_coords": y_coords_loc,
        "x_range": x_range_loc,
        "y_range": y_range_loc,
        "num_leaves": num_leaves_loc,
        "num_vertices": num_vertices_loc,
    }

# 初期スライス計算
state = process_slice(z_center)

# 以下は既存のインタラクティブ表示部分と同じ
from matplotlib.widgets import Slider

fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios':[1,2,1]})

img_ax = axes[0]
img_obj = img_ax.imshow(state["image"].T, cmap='gray', origin='lower')
img_ax.set_title(f"Numpy Slice {z_center}")

tree_ax = axes[1]
tree_ax.set_title("Component Tree (Click Node)")
tree_ax.set_ylabel("Intensity (0-255)")
nx.draw(state["G"], state["pos"], ax=tree_ax,
        node_size=2, edge_color='gray', node_color='blue',
        arrows=False, alpha=0.3)

reg_ax = axes[2]
reg_ax.imshow(state["image"].T, cmap='gray', origin='lower', alpha=0.5)
region_display = reg_ax.imshow(np.full_like(state["image"].T, np.nan, dtype=float),
                               cmap='Reds', alpha=0.6, vmin=0, vmax=1, origin='lower')
reg_ax.set_title("Selected Region")

selected_node_marker = None

def on_click(event):
    global selected_node_marker
    if event.inaxes != tree_ax or event.xdata is None or event.ydata is None:
        return

    x_coords = state["x_coords"]
    y_coords = state["y_coords"]
    x_range = state["x_range"]
    y_range = state["y_range"]
    children = state["children"]
    num_leaves = state["num_leaves"]

    dx = (x_coords - event.xdata) / x_range
    dy = (y_coords - event.ydata) / y_range
    closest_node = np.argmin(dx**2 + dy**2)

    region = np.zeros(num_leaves, dtype=float)
    search_stack = [closest_node]
    while search_stack:
        n = search_stack.pop()
        if not children[n]:
            region[n] = 1.0
        else:
            search_stack.extend(children[n])

    region = region.reshape(state["image"].shape)
    region_display.set_data(np.where(region.T == 1.0, 1.0, np.nan))

    if selected_node_marker:
        selected_node_marker.remove()
    selected_node_marker = tree_ax.scatter(
        state["x_coords"][closest_node],
        state["y_coords"][closest_node],
        s=100, edgecolors='red', facecolors='none', linewidths=2, zorder=5
    )
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)

axslice = fig.add_axes([0.25, 0.02, 0.50, 0.03])
slice_slider = Slider(axslice, 'Slice', 0, z_max,
                      valinit=z_center, valstep=1, valfmt='%0.0f')

def update(val):
    global state, selected_node_marker, region_display
    z = int(slice_slider.val)
    state = process_slice(z)

    img_obj.set_data(state["image"].T)
    img_ax.set_title(f"Numpy Slice {z}")

    tree_ax.clear()
    tree_ax.set_title("Component Tree (Click Node)")
    tree_ax.set_ylabel("Intensity (0-255)")
    nx.draw(state["G"], state["pos"], ax=tree_ax,
            node_size=2, edge_color='gray', node_color='blue',
            arrows=False, alpha=0.3)

    reg_ax.clear()
    reg_ax.imshow(state["image"].T, cmap='gray', origin='lower', alpha=0.5)
    region_display = reg_ax.imshow(np.full_like(state["image"].T, np.nan, dtype=float),
                                   cmap='Reds', alpha=0.6, vmin=0, vmax=1, origin='lower')
    reg_ax.set_title("Selected Region")

    selected_node_marker = None
    fig.canvas.draw_idle()

slice_slider.on_changed(update)

plt.show()