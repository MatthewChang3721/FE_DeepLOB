import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class TimeSeriesDataset(Dataset):
    """时间序列滑动窗口 Dataset，通过列名（而非列索引）访问特征。

    特征分组：
      - momentum:  MidPrice_diff       (1 列)
      - pic:       Bid_G1..Ask_G5      (10 列)
      - lt_sensor: MidPrice            (1 列)
      - st_sensor: OBI_3, OBI_delta    (2 列)

    列顺序若变化，只需修改下方列名常量，无需改动切片逻辑。
    """

    # ── 列名常量 ──────────────────────────────────────────────
    _MOMENTUM_COL   = 'MidPrice_diff'
    _LT_SENSOR_COL  = 'MidPrice'
    _PIC_COLS       = [f'{side}_G{i}' for i in range(1, 6)
                       for side in ('Bid', 'Ask')]          # Bid_G1 .. Ask_G5
    _ST_SENSOR_COLS = ['OBI_3', 'OBI_delta']
    _LABEL_COL      = 'price_move_label'

    _REQUIRED_COLS = (
        [_MOMENTUM_COL, _LT_SENSOR_COL]
        + _PIC_COLS
        + _ST_SENSOR_COLS
        + [_LABEL_COL]
    )

    # 非特征列：允许存在但不会被使用
    _NON_FEATURE_COLS = {'timestamp', 'price_move_bps', 'hit_times'}

    def __init__(self, data: pd.DataFrame, window_size: int = 200,
                 target_size: int = 1, label_offset: int = -1):
        """
        Args:
            data:         单日 DataFrame，需包含所有 REQUIRED_COLS
            window_size:  输入窗口长度 (ticks)
            target_size:  预测目标长度   (ticks)
            label_offset: 标签相对于窗口末端的偏移。
                          默认 -1 = 标签取自窗口 end-1，即预测"窗口内最后
                          一个已知状态"之后的目标，是交易场景的常见做法。
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f'Expected pd.DataFrame, got {type(data)}')

        # 1. 列存在性校验
        missing = set(self._REQUIRED_COLS) - set(data.columns)
        if missing:
            raise ValueError(
                f'Missing required columns: {sorted(missing)}. '
                f'Available: {list(data.columns)}'
            )

        # 2. 额外列校验：schema 漂移时立即报错
        extra = [c for c in data.columns
                 if c not in self._REQUIRED_COLS
                 and c not in self._NON_FEATURE_COLS]
        if extra:
            raise ValueError(
                f'Unexpected extra columns (not used as features): {extra}. '
                f'Schema may have changed.'
            )

        # 3. 构建特征 & 标签张量（label: -1,0,1 → 0,1,2）
        features = data[self._REQUIRED_COLS].copy()
        features[self._LABEL_COL] = features[self._LABEL_COL] + 1

        tensor_all = torch.tensor(features.values, dtype=torch.float32)
        self.X_tensor = tensor_all[:, :-1].float().contiguous()
        self.Y_tensor = tensor_all[:, -1].long().contiguous()

        self.window_size = window_size
        self.target_size = target_size
        self.label_offset = label_offset

        # 4. 按列名解析特征索引
        self._col_index = {name: i for i, name in enumerate(features.columns[:-1])}
        self._momentum_idx  = self._col_index[self._MOMENTUM_COL]
        self._lt_sensor_idx = self._col_index[self._LT_SENSOR_COL]
        self._pic_idx       = [self._col_index[name] for name in self._PIC_COLS]
        self._st_sensor_idx = [self._col_index[name] for name in self._ST_SENSOR_COLS]

        # 5. 维度防御性校验
        if len(self._pic_idx) != 10:
            raise ValueError(f'Expected 10 pic columns, got {len(self._pic_idx)}')
        if len(self._st_sensor_idx) != 2:
            raise ValueError(f'Expected 2 st_sensor columns, got {len(self._st_sensor_idx)}')

        self.num_features = self.X_tensor.shape[1]

        # 6. 有效样本数 & 数据长度校验
        self.max_idx = (
            len(self.X_tensor)
            - self.window_size
            - self.target_size
            - self.label_offset
        )
        if self.max_idx < 0:
            raise ValueError(
                f'Data has {len(self.X_tensor)} rows, '
                f'but need at least '
                f'{self.window_size + self.target_size + self.label_offset} '
                f'(window={self.window_size}, target={self.target_size}, '
                f'offset={self.label_offset}).'
            )

    # ── 属性 ──────────────────────────────────────────────────
    @property
    def pic_dim(self):
        return len(self._pic_idx)

    # ── 切片 ──────────────────────────────────────────────────
    def __len__(self):
        return self.max_idx + 1

    def __getitem__(self, idx):
        x_end = idx + self.window_size

        X_momentum  = self.X_tensor[idx:x_end, self._momentum_idx:self._momentum_idx + 1]
        X_pic       = self.X_tensor[idx:x_end, self._pic_idx]
        x_lt_sensor = self.X_tensor[idx:x_end, self._lt_sensor_idx:self._lt_sensor_idx + 1]
        x_st_sensor = self.X_tensor[idx:x_end, self._st_sensor_idx]

        y_start = x_end + self.label_offset
        y_end   = y_start + self.target_size
        Y = self.Y_tensor[y_start:y_end].view(-1)

        X = {
            'momentum':   X_momentum.contiguous(),
            'pic':        X_pic.contiguous(),
            'lt_sensor':  x_lt_sensor.contiguous(),
            'st_sensor':  x_st_sensor.contiguous(),
        }
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