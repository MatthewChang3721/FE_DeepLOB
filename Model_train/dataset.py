import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, window_size: int = 200, target_size: int = 1, label_offest: int = -1):
        if not isinstance(data, torch.Tensor):
            raise TypeError(f'Expected data too be torch.Tensor, got {type(data)}')

        self.X_data = data[:, :-1].float().contiguous()
        self.Y_data = data[:, -1].long().contiguous()
        self.window_size = window_size
        self.target_size = target_size
        self.label_offest = label_offest

        self.max_idx = len(data) - self.window_size - self.target_size - self.label_offest

    def __len__(self):
        return self.max_idx + 1
    
    def __getitem__(self, idx):
        x_end = idx + self.window_size

        X_momentum = self.X_data[idx: x_end, 0:1].contiguous()
        X_pic = self.X_data[idx: x_end, 1:11].contiguous()  # Features
        x_lt_sensor = self.X_data[idx: x_end, 11:12].contiguous()
        x_st_sensor = self.X_data[idx: x_end, 12:14].contiguous()
        y_start = x_end + self.label_offest
        y_end = y_start + self.target_size
        Y = self.Y_data[y_start:y_end].view(-1)  # Target label
        
        X = {'momentum': X_momentum,'pic':X_pic, 'lt_sensor': x_lt_sensor, 'st_sensor': x_st_sensor}

        return X, Y

def create_dataloader(inputpath, start_files: int, num_files: int, window_size: int = 200, target_size: int = 1, batch_size: int = 32, shuffle: bool = True, drop_last: bool = True):
    all_daily_ds = []

    csv_files = sorted(Path(inputpath).glob('normalized_*.csv'))[start_files:start_files + num_files]
    print(f"Found {len(csv_files)} CSV files for this dataloader.")

    label_counts_overall = np.zeros(3, dtype=int)  # To count occurrences of each label across all files
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        data['price_move_label'] = data['price_move_label'] + 1  # Shift labels to be 0, 1, 2 instead of -1, 0, 1
        label_counts_overall += data['price_move_label'].value_counts().sort_index().values
        tc_data = torch.tensor(data.values, dtype=torch.float32)

        # Denfensively check length of df is sufficient for windowing
        if len(data) < window_size + target_size:
            print(f"Warning: {csv_file} has insufficient data ({len(data)} rows). Skipping this file.")
            continue

        ts_dataset = TimeSeriesDataset(tc_data, window_size=window_size, target_size=target_size)
        all_daily_ds.append(ts_dataset)
    
    label_names = ['Down', 'Neutral', 'Up']

    final_dataset = ConcatDataset(all_daily_ds)

    # DataLoader
    data_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle, drop_last= drop_last)
    
    label_df = pd.DataFrame(label_counts_overall, index=label_names, columns=['Count'])

    return data_loader, label_df