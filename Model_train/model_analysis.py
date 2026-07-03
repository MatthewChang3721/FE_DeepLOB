import os
import sys
from pathlib import Path

import torch
import config
from dataset import create_dataloader
from FE_DeepLOB import FE_DeepLOB
from visualization import (
    visual_df_prepare,
    signal_hist,
    visualization_engine,
    model_signal_split_plot,
    model_signal_scatter,
    net_directional_signal_plot,
    PRAUC,
)


def run_analysis(model, device, loader, loader_name, startfile, numfiles, processed_data_path):
    """对给定的 DataLoader 运行完整的模型分析流程"""
    avg_accuracy, preds, labels, probs = visualization_engine(
        model, loader, device=device, confidence_threshold=0.5
    )

    print(f"\n{'='*40}")
    print(f"{loader_name} set Average accuracy: {avg_accuracy:.2f}%")

    from sklearn.metrics import classification_report
    print(classification_report(labels, preds, target_names=['Down', 'Stationary', 'Up']))

    ap_results = PRAUC(labels, probs, save_name=f"{loader_name}_PRAUC")
    print(f"{loader_name} PR-AUC results: {ap_results}")

    # 合并可视化 DataFrame
    df_visual = visual_df_prepare(
        processed_data_path,
        config.normalize_window,
        config.window_size,
        startfile,
        numfiles,
        preds,
        probs,
    )

    # 保存 DataFrame 到 sample/ 目录
    sample_dir = Path("sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    df_visual.to_csv(sample_dir / f"{loader_name}_visual_df.csv", index=False)

    # 画图（按名称存入 picture/，可覆盖）
    confidence_level = 0.5
    signal_hist(df_visual, save_name=f"{loader_name}_signal_hist")
    model_signal_split_plot(confidence_level, df_visual, save_name=f"{loader_name}_signal_split")
    model_signal_scatter(confidence_level, df_visual, save_name=f"{loader_name}_signal_scatter")
    net_directional_signal_plot(df_visual, save_name=f"{loader_name}_net_directional")

    return df_visual


if __name__ == "__main__":
    # 校验命令行参数
    if len(sys.argv) < 2:
        print("Usage: python model_analysis.py <model_filename.pt>")
        print("Example: python model_analysis.py 20260702_10_1bps_150.pt")
        sys.exit(1)

    pt_filename = sys.argv[1]
    pt_path = os.path.join("Model/torch", pt_filename)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FE_DeepLOB(num_features=config.num_features, num_classes=config.num_classes)
    model.to(device)

    state_dict = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from {pt_path}, device={device}")

    # 创建 DataLoaders
    val_loader, _ = create_dataloader(
        config.normalized_data_path,
        config.val_start_file,
        config.val_num_files,
        config.window_size,
        config.target_size,
        config.batch_size * 2,
        False,
        False,
    )
    test_loader, _ = create_dataloader(
        config.normalized_data_path,
        config.train_num_files + config.val_num_files,
        3,
        config.window_size,
        config.target_size,
        config.batch_size * 2,
        False,
        False,
    )

    # 分析验证集
    run_analysis(
        model, device, val_loader,
        "Validation",
        config.val_start_file,
        config.val_num_files,
        config.processed_data_path,
    )

    # 分析测试集
    run_analysis(
        model, device, test_loader,
        "Test",
        config.train_num_files + config.val_num_files,
        3,
        config.processed_data_path,
    )
