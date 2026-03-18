# Brain Tumor Segmentation via Component Tree (Max-tree)

このリポジトリは、BraTS（Brain Tumor Segmentation）データセットに対して、 **Max-tree（Component Tree）** を用いて脳腫瘍領域のセグメンテーションを行う。`higra` ライブラリを使用し、3Dおよび2D（Axial断面）での解析を実装している。

## ファイル構成と役割

### メイン・セグメンテーション・スクリプト
- **`3D_maxtree_auto_shreshold_segmentation.py`**  
  MRIボリューム全体（3D）に対してMax-treeを構築し、自動閾値判定（Valley Detection）を用いて腫瘍候補となる独立したインスタンスを抽出する。
- **`axial_maxtree_auto_threshold_segmentation.py`**  
  スライス単位（Axial断面）で2D Max-treeを構築し、セグメンテーションを行います。各断面の結果を統合してラベルマップを生成する。

### マルチモーダル・解析
- **`_histgram.py`**  
  NIfTIファイルの輝度分布を可視化する。正規化処理（Min-Max正規化や外れ値カット）の前後の分布を確認するために使用する。

### ユーティリティ・可視化
- **`_viewer.py`**  
  `napari` を使用した3Dビューア。元のMRI画像に生成したセグメンテーションラベル（labelmap.npy）を重ねて3次元的に確認する。
- **`_experiment_utils.py`**  
  実験結果を整理するためのユーティリティ。実行のたびに `output/` 内にタイムスタンプ付きのフォルダを自動生成し、ログや画像を保存する。

### その他
- **`.gitignore`**  
  GitHubにアップロードしない巨大なデータセット（`.nii`）や、実行時に生成されるキャッシュ（`__pycache__`）、出力結果（`output/`）を除外するための設定ファイル。

## 技術スタック
- **Language:** Python 3.x
- **Core Library:** [higra](https://github.com/higra/higra) (Hierarchical Graph Analysis)
- **Medical Imaging:** nibabel, napari
- **Data Science:** numpy, scipy, matplotlib