import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from numba import njit

@njit
def apply_triple_barrier(bids, asks, window, alpha):
    """Numba 加速的三重屏障标签 (path 方法)"""
    n = len(bids)
    labels = np.zeros(n, dtype=np.int8)
    price_move_bps = np.zeros(n ,dtype=np.float64)
    hit_times = np.zeros(n, dtype=np.int8)

    for t in range(n - window):
        ask_t = asks[t]
        bid_t = bids[t]

        ub = ask_t * (1 + alpha)   # 向上屏障
        lb = bid_t * (1 - alpha)   # 向下屏障

        hit_bps = 0.0
        time = 0

        for i in range(1, window + 1):
            future_bid = bids[t + i]
            future_ask = asks[t + i]

            if future_bid > ub:
                hit_bps = (future_bid - ask_t) / ask_t * 10000.0
                time = i
                labels[t] = 1
                break
            elif future_ask < lb:
                hit_bps = (future_ask - bid_t) / bid_t * 10000.0
                time = i
                labels[t] = -1
                break
        price_move_bps[t] = hit_bps
        hit_times[t] = time
    return price_move_bps, hit_times, labels

def process_data(inputfile, outputfile, label_method: str = 'trend',
                 label_window: int = 10, alpha: float = 0.0001,
                 levels: int = 5, epsilon: float = 1e-8):
    """
    处理 LOB 原始 CSV 文件（含特征工程版）：
    1. 保留 timestamp + bid/ask/bidSize/askSize 列
    2. 计算 MidPrice 和 price_gravity (Bid_G/Ask_G)
    3. 计算 MidPrice_diff（百分比差分）
    4. 用指定方法生成 price_move_label
    5. 截掉头部/尾部 NaN
    6. 输出 timestamp + Bid_G* + Ask_G* + MidPrice_diff + label

    Args:
        inputfile:     输入 CSV 路径
        outputfile:    输出 CSV 路径
        label_method:  标签方法 ('midprice' | 'execution')
        label_window:  标签计算窗口（ticks）
        alpha:         阈值 / 交易成本比率 (bps)
        levels:        gravity 计算档位数
        epsilon:       防除零常数
    """
    valid_methods = ['trend', 'path']
    if label_method not in valid_methods:
        raise ValueError(f"label_method must be one of {valid_methods}, got '{label_method}'")

    # 1. 读取 timestamp + bid/ask + bidSize/askSize 列
    df = pd.read_csv(inputfile)
    df = df.filter(regex='timestamp|^bid|^ask')

    # 2. 计算 MidPrice（gravity 依赖）
    df['MidPrice'] = (df['ask1'] + df['bid1']) / 2

    # 3. price_gravity 特征
    for i in range(1, levels + 1):
        bid_dev = df['MidPrice'] - df[f'bid{i}']
        ask_dev = df['MidPrice'] - df[f'ask{i}']
        df[f'Bid_G{i}'] = df[f'bidSize{i}'] / (bid_dev ** 2 + epsilon)
        df[f'Ask_G{i}'] = -df[f'askSize{i}'] / (ask_dev ** 2 + epsilon)

    # 4. OBI (order book imbalance)
    bid_cum_3 = df[['bidSize1', 'bidSize2', 'bidSize3']].sum(axis=1)
    ask_cum_3 = df[['askSize1', 'askSize2', 'askSize3']].sum(axis=1)
    denom_3 = bid_cum_3 + ask_cum_3 + epsilon
    obi3 = (bid_cum_3 - ask_cum_3) / denom_3
    df['OBI_3'] = obi3

    bid_cum_5 = bid_cum_3 + df['bidSize4'] + df['bidSize5']
    ask_cum_5 = ask_cum_3 + df['askSize4'] + df['askSize5']
    denom_5 = bid_cum_5 + ask_cum_5 + epsilon
    obi5 = (bid_cum_5 - ask_cum_5) / denom_5
    df['OBI_delta'] = obi5 - obi3

    # 5. MidPrice 百分比差分（当前 tick vs 前一个 tick），乘 10000 以 bps 为单位
    df['MidPrice_diff'] = df['MidPrice'].pct_change().fillna(0) * 10000.0

    # 5. 标签
    if label_method == 'path':
        bid = df['bid1'].values.astype(np.float64)
        ask = df['ask1'].values.astype(np.float64)
        df['price_move_bps'], df['hit_times'], df['price_move_label'] = apply_triple_barrier(bid, ask, label_window, alpha)

    elif label_method == 'trend':
        m_plus_t = df['MidPrice'].rolling(window=abs(label_window)).mean().shift(-label_window)
        price_move_pctg = (m_plus_t - df['MidPrice']) / df['MidPrice']
        df['price_move_bps'] = price_move_pctg * 10000
        df['price_move_label'] = 0
        df.loc[price_move_pctg > alpha, 'price_move_label'] = 1
        df.loc[price_move_pctg < -alpha, 'price_move_label'] = -1

    # 6. 截掉尾部无效数据
    df = df.iloc[: -label_window] 

    # 7. 只保留 timestamp + Bid_G* + Ask_G* + OBI* + MidPrice_diff + label
    df = df.filter(regex='timestamp|MidPrice|^Bid_G|^Ask_G|^OBI_|MidPrice_diff|^price_move_|hit_times')

    # 8. 保存
    df.to_csv(outputfile, index=False)

def window_normalize(inputpath, outputpath, window_size=5):
    """
    使用前 window_size 天的统计量对当日 Bid_G/Ask_G / MidPrice特征做标准化。
    输出所有列（timestamp + MidPrice + Bid_G* + Ask_G* + OBI_* + MidPrice_diff + price_move_label）。

    Args:
        inputpath:  输入目录（process_data_fe 的输出）
        outputpath: 输出目录
        window_size: 用于计算统计量的历史天数
    """
    folder = Path(inputpath)
    output_dir = Path(outputpath)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(folder.glob('*.csv'))
    buffer = deque(maxlen=window_size + 1)

    for file in csv_files:
        df = pd.read_csv(file)

        # 自动检测需要标准化的列：Bid_G* / Ask_G*（不标准化 OBI，保持原始量纲）
        normalize_cols = [c for c in df.columns if c.startswith(('Bid_G', 'Ask_G'))]
        normalize_cols.append('MidPrice')
        processing_date = file.stem[-4:]
        buffer.append(df)

        if len(buffer) == window_size + 1:
            stats_list = list(buffer)[:-1]
            stats_df = pd.concat(stats_list, ignore_index=True)

            # price_gravity: 仅除以历史窗口 std（保留正负号，不做中心化）
            gravity_cols = normalize_cols[:-1]   # Bid_G* + Ask_G*（末尾是 MidPrice）
            hist_std_gravity = stats_df[gravity_cols].std()
            hist_std_gravity[hist_std_gravity < 1e-8] = 1   # 防除零

            # MidPrice: 保持 z-score（减均值 + 除 std）
            mid_col = 'MidPrice'
            hist_mean_mid = stats_df[mid_col].mean()
            hist_std_mid = stats_df[mid_col].std()
            if hist_std_mid < 1e-8:
                hist_std_mid = 1

            target_df = buffer[-1].copy()
            target_df[mid_col] = (target_df[mid_col] - hist_mean_mid) / hist_std_mid
            target_df[gravity_cols] = target_df[gravity_cols] / hist_std_gravity

            file_name = f'normalized_{processing_date}.csv'
            save_path = output_dir / file_name
            target_df.to_csv(save_path, index=False)

def run_pipeline(
    label_method: str = 'trend',
    label_window: int = 10,
    alpha: float = 0.0001,
    levels: int = 5,
    normalize_window: int = 5,
    inputpath: str = 'Data/Raw_data/',
    processedpath: str = 'Data/FE_DeepLOB_data/Processed_data/',
    outputpath: str = 'Data/FE_DeepLOB_data/Normalized_data/'
):
    """
    完整数据处理流水线（含 price_gravity 特征工程）：
    Raw CSV → process_data (gravity + label) → processed_*.csv → window_normalize → normalized_*.csv
    """
    Path(processedpath).mkdir(parents=True, exist_ok=True)
    Path(outputpath).mkdir(parents=True, exist_ok=True)

    raw_files = sorted(Path(inputpath).glob('*.csv'))

    for raw_file in raw_files:
        outputfile = f'{processedpath}processed_{raw_file.stem[-4:]}.csv'
        process_data(
            raw_file, outputfile,
            label_method=label_method, label_window=label_window,
            alpha=alpha, levels=levels
        )

    window_normalize(processedpath, outputpath, window_size=normalize_window)

if __name__ == "__main__":
    label_method = 'path'  
    label_window = 10       # ticks 0.5s
    alpha = 2 * 1e-4        # 2 bps default
    levels = 5              # LOB data LEVELS
    normalize_window = 5    # Length of Normalized Window (days)
    inputpath = 'Data/Raw_data/'
    processpath = 'Data/FE_DeepLOB_data/Processed_data/'
    outputpath = 'Data/FE_DeepLOB_data/Normalized_data/'
    run_pipeline(label_method, label_window, alpha, levels, normalize_window, inputpath, processpath, outputpath)