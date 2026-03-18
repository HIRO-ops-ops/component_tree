import os
from datetime import datetime

def create_experiment_output():
    """
    output/
        YYYYMMDD_HHMMSS_scriptname/
    を自動生成してパスを返す
    """
    # 実行スクリプトの場所
    project_root = os.path.dirname(os.path.abspath(__file__))

    # output フォルダ
    output_root = os.path.join(project_root, "output")
    os.makedirs(output_root, exist_ok=True)

    # スクリプト名取得
    import sys
    script_path = sys.argv[0]
    script_name = os.path.splitext(os.path.basename(script_path))[0]

    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 実験フォルダ
    experiment_dir = os.path.join(
        output_root,
        f"{timestamp}_{script_name}"
    )

    os.makedirs(experiment_dir, exist_ok=True)

    print(f"[Experiment Output] {experiment_dir}")

    return experiment_dir