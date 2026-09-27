import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class TimeSeriesDataset(Dataset):
    # Proprietary core algorithm redacted for confidentiality.
        return X, Y


def create_dataloader(
    inputpath: str,
    start_files: int,
    num_files: int,
    window_size: int = 200,
    target_size: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """从标准化后的 CSV 文件创建 DataLoader。

    Args:
        inputpath:   包含 normalized_*.csv 的目录
        start_files: 起始文件索引 (0-based)
        num_files:   使用的文件数量
        window_size: 时间窗口
        target_size: 预测目标步数
        batch_size:  batch 大小
        shuffle:     是否打乱
        drop_last:   是否丢弃最后不完整 batch
    """
    all_daily_ds = []
    csv_files = sorted(Path(inputpath).glob('normalized_*.csv'))
    total_files = len(csv_files)

    if total_files == 0:
        raise FileNotFoundError(f'No normalized_*.csv files found in {inputpath}')

    end_files = start_files + num_files
    if end_files > total_files:
        raise ValueError(
            f'Requested file range [{start_files}:{end_files}] '
            f'exceeds total available files ({total_files}).'
        )

    selected_files = csv_files[start_files:end_files]
    print(f'Found {len(selected_files)} CSV files for this dataloader '
          f'(indices {start_files}-{end_files - 1}).')

    label_counts_overall = np.zeros(3, dtype=int)

    for csv_file in selected_files:
        df = pd.read_csv(csv_file)

        # 防御性检查：长度不足以支撑窗口时跳过（并提示）
        if len(df) < window_size + target_size:
            print(f'Warning: {csv_file.name} has only {len(df)} rows '
                  f'(need >= {window_size + target_size}). Skipping.')
            continue

        try:
            ts_dataset = TimeSeriesDataset(
                df, window_size=window_size, target_size=target_size
            )
        except ValueError as e:
            print(f'Error building dataset from {csv_file.name}: {e}')
            raise

        all_daily_ds.append(ts_dataset)
        label_counts_overall += (
            (df['price_move_label'] + 1)
            .value_counts()
            .sort_index()
            .reindex([0, 1, 2], fill_value=0)
            .values
        )

    if not all_daily_ds:
        raise RuntimeError(
            f'No valid daily datasets created from files '
            f'{selected_files[0].name}..{selected_files[-1].name}. '
            f'Check file lengths.'
        )

    final_dataset = ConcatDataset(all_daily_ds)
    data_loader = DataLoader(
        final_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    label_names = ['Down', 'Neutral', 'Up']
    label_df = pd.DataFrame(
        label_counts_overall, index=label_names, columns=['Count']
    )

    return data_loader, label_df
