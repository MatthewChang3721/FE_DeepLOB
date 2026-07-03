# 基于 FE_DeepLOB 与 Optuna 的期货高频方向预测

本项目利用高频限价订单簿（LOB）数据，使用带特征工程的 DeepLOB 神经网络模型（FE_DeepLOB）预测期货（IM）短期价格方向（上涨/下跌/持平），并通过 **Optuna** 贝叶斯优化自动搜索最优超参数组合，提升模型在指定评价指标（Down & Up PR-AUC 之和）上的表现。

---

## 项目文件说明

### `config.py` — 全局静态配置

存放训练过程中不会随 Optuna 搜索变化的基础参数。当你需要快速调整训练骨架时在此修改：

| 参数 | 默认值 | 含义 |
|------|:---:|------|
| **数据路径** | | |
| `copy_data_path` | `/yourdatabase
` | 原始数据远程路径 |
| `fetch_file_path` | `"Data/SC_Primary.csv"` | 拉取的中间文件路径 |
| `raw_data_path` | `"Data/Raw_data/"` | 原始 LOB CSV 目录 |
| `processed_data_path` | `"Data/FE_DeepLOB_data/Processed_data/"` | 特征工程后的临时数据目录 |
| `normalized_data_path` | `"Data/FE_DeepLOB_data/Normalized_data/"` | 归一化后的最终训练数据目录 |
| `model_save_dir` | `"Model/cache"` | 模型保存路径 |
| `model_save_name` | `"best_model.pt"` | 最佳模型文件名 |
| `log_dir` | `"Logs/"` | 日志输出目录 |
| **数据处理** | | |
| `label_method` | `"l2"` | 标签生成方法（l1/l2/l3） |
| `alpha` | `1e-4` | 标签生成中价格变动阈值（1bps） |
| `label_window` | `10` | 预测未来多少个 tick 的价格方向 |
| `normalize_window` | `5` | 滚动归一化使用的历史天数 |
| **DataLoader 划分** | | |
| `train_start_file` | `0` | 训练集起始文件索引 |
| `train_num_files` | `25` | 训练集使用文件数 |
| `val_start_file` | `25` | 验证集起始文件索引 |
| `val_num_files` | `5` | 验证集使用文件数 |
| `batch_size` | `1024` | 训练 batch size |
| `target_size` | `1` | 预测步长（固定1） |
| `shuffle_train` | `True` | 训练集是否 shuffle |
| `drop_last` | `True` | 是否丢弃最后一个不完整 batch |
| **模型** | | |
| `window_size` | `150` | 输入模型的历史窗口长度（tick），会被 Optuna 动态覆盖 |
| `num_features` | `5` | 输入特征数（MidPrice_diff + 2档重力值 × 2方向，经 spatial conv 后另算） |
| `num_classes` | `3` | 分类数（Down/Neutral/Up） |
| **可复现性** | | |
| `seed` | `42` | 随机种子 |
| **训练** | | |
| `num_epochs` | `200` | 每轮 trial 最大训练轮数 |
| `learning_rate` | `0.001` | 初始学习率 |
| `weight_decay` | `5e-3` | 优化器权重衰减 |
| `warmup_epochs` | `5` | 前5个epoch使用 CrossEntropyLoss 预热 |
| `warmup_ratio` | `0.05` | warmup 步数占总步数的比例 |
| `eta_min` | `5e-5` | CosineAnnealingLR 最小学习率 |
| `log_interval` | `10` | 日志打印间隔（epoch） |
| **损失函数** | | |
| `gamma` | `2.0` | Focal Loss 聚焦参数，会被 Optuna 动态覆盖 |
| `soft_matrix` | 3×3 矩阵 | 软标签平滑矩阵，会被 Optuna 动态覆盖 |
| `pnl_matrix` | 3×3 矩阵 | 盈亏惩罚矩阵，会被 Optuna 动态覆盖 |
| **早停** | | |
| `early_stop_patience` | `15` | 早停等待轮数 |
| `early_stop_verbose` | `False` | 是否打印早停细节 |
| `monitor_loss` | `True` | 是否监控验证 loss（否则监控指标） |

---

### `process_data.py` — 数据预处理流水线

#### `process_data(inputfile, outputfile, label_method, label_window, alpha)`

不展示

#### `window_normalize_FE(inputpath, outputpath, window_size)`

对处理后的CSV文件进行跨日滚动归一化，**严格保证没有未来信息泄露**。

- 按文件日期排序（`sorted(glob)`）
- 维护一个容量为 `window_size + 1` 的 deque 缓冲区
- 每次取缓冲区中前 `window_size` 天计算各特征的均值和标准差
- 用历史统计量归一化第 `window_size+1` 天的 `Bid_G*`、`Ask_G*`、`MidPrice` 列
- 输出最终特征列（共 14 列）：

  `MidPrice_diff`
  `Bid_G1, Ask_G1, Bid_G2, Ask_G2, Bid_G3, Ask_G3, Bid_G4, Ask_G4, Bid_G5, Ask_G5`
  `MidPrice, OBI_level3, OBI_delta`
  `price_move_label`

#### 独立执行

`process_data.py` 也可作为独立脚本直接运行：

```bash
python process_data.py
```

此时使用默认参数（`label_window=10, alpha=1e-4`）从 `Data/SC_Raw_data/` 读取原始 CSV 并执行完整的数据预处理流程。

---

### `dataset.py` — 时间序列数据集与 DataLoader

#### `TimeSeriesDataset(Dataset)`

继承 PyTorch `Dataset` 的时序切片数据集：

- 从 tensor 中分离特征列（`data[:, :-1]`）和标签列（`data[:, -1]`）
- 以滑动窗口方式生成 `(X, Y)` 样本对，偏移量 `label_offset = -1` 确保标签取自特征窗口的**最后一个时间点**
- 自动限制 `max_idx` 避免越界

#### `create_dataloader(inputpath, start_files, num_files, window_size, target_size, batch_size, shuffle, drop_last)`

- 从 `inputpath` 读取已归一化的 CSV 文件（匹配 `normalized_*.csv`）
- 按文件索引（排序后）切片，支持灵活划分训练/验证集（例如训练集取前25个文件，验证集取后5个）
- 标签值从 `{-1, 0, 1}` 转换为 `{0, 1, 2}` 以适应 PyTorch 分类要求
- 统计标签分布并作为 DataFrame 返回（用于计算类别权重）
- 返回 `(DataLoader, label_distribution_df)`

---

### `FE_DeepLOB.py` — 神经网络模型与训练/验证引擎

#### `FE_DeepLOB(nn.Module)`

带特征工程的高频订单簿深度学习模型架构。**输入由4个独立张量组成**，不再拼接为单一输入：
涉及公司知识产权，未展示。

单 epoch 训练函数：

- 从 DataLoader 的 dict 中提取 `data['momentum']`、`data['pic']`、`data['lt_sensor']`、`data['st_sensor']` 四个张量
- 梯度裁剪（max_norm=1.0）
- 返回 `(avg_loss, avg_acc)`

#### `validate_engine(model, val_loader, criterion, device)`

单 epoch 验证函数：

- 与 train_engine 相同的多输入接口
- 计算 Down（类别0）和 Up（类别2）的 **PR-AUC**（Precision-Recall AUC）
- 返回 `(avg_loss, pr_auc_down, pr_auc_up)`

> PR-AUC 相比准确率更适合不平衡分类场景，能更好地衡量模型对少数类（涨/跌）的区分能力。

---

### `train_artifact.py` — 训练辅助组件

#### `SoftFocalLoss(nn.Module)`

融合软标签和不对称盈亏惩罚的 Focal Loss：

- **软标签平滑**（`soft_targets`）：对真实标签周围的邻域赋予小概率，减少过拟合，提高泛化性
- **盈亏惩罚矩阵**（`penalty_matrix`）：对不同类型的分类错误施加不对称惩罚，例如将"上涨"误判为"下跌"的代价远大于误判为"持平"
- **Focal Loss**：`gamma` 参数控制对难分类样本的关注程度
- 损失函数完整形式：`Loss = FocalLoss × (1 + ExpectedPenalty)`

#### `EarlyStopping`

基于验证 loss 的早停机制：

- `patience` 由 `config.early_stop_patience` 控制（默认10）
- `verbose` 由 `config.early_stop_verbose` 控制（默认关闭）
- 可选 `monitor_loss`：监控验证 loss（越小越好）或其它指标（越大越好）

---

### `train.py` — Optuna 主搜索入口

核心控制脚本，执行流程：

1. **数据目录清理**：运行开始时自动删除并重建 `config.processed_data_path` 和 `config.normalized_data_path`，确保每次运行从干净状态开始
2. **外层环境遍历**：遍历 `target_windows = [10, 20, 50, 100]`，对每个窗口动态缩放 alpha 阈值
   - 缩放因子：`τ = sqrt(target_window / 10)`
   - 以基础阈值 `[1.0, 1.5, 2.0] bps` 乘以 τ，四舍五入保留一位小数：

     | label_window | τ | 缩放后的 alpha 集 |
     |:---:|:---:|:---:|
     | 10 | 1.000 | [1.0, 1.5, 2.0] |
     | 20 | 1.414 | [1.4, 2.1, 2.8] |
     | 50 | 2.236 | [2.2, 3.4, 4.5] |
     | 100 | 3.162 | [3.2, 4.7, 6.3] |

   - 更长的预测窗口对应更大的阈值，使标签分布在不同时间尺度下保持合理
3. **数据制备**：对每个环境组合，运行 `process_data()` 和 `window_normalize_FE()` 生成对应标签的文件
4. **Optuna 搜索**：对每个环境创建独立的 study，优化目标函数 `objective()`，每环境运行 50 个 trial
5. **日志记录**：所有输出写入 `Logs/Optuna_Campaign_{timestamp}.log`

所有训练超参数（数据划分、batch size、早停等）均从 `config.py` 读取，方便统一调整。

---

## 超参数搜索空间

Optuna 在每个 trial 中搜索的超参数及范围如下：

| 参数 | 类型 | 搜索范围 | 步长 | 含义 | 对应矩阵位置 |
|------|:---:|:--------:|:----:|------|:----------:|
| `window_size` | `int` | [100, 500] | 5 | 模型输入的历史窗口长度（tick） | — |
| `gamma` | `float` | [1.2, 3.0] | — | Focal Loss 聚焦参数 | — |
| `target_conf` | `float` | [0.85, 0.99] | — | 软标签对角置信度（主导类别） | `soft_matrix[0][0]`, `soft_matrix[2][2]` |
| `target_conf_mid` | `float` | [0.85, 0.99] | — | 软标签对角置信度（中间"平"类） | `soft_matrix[1][1]` |
| `max_penalty` | `float` | [1.0, 3.0] | — | 最大错判惩罚（涨→跌或跌→涨） | `pnl_matrix[0][2]`, `pnl_matrix[2][0]` |
| `min_penalty` | `float` | [0.1, 1.0] | — | 最小错判惩罚（涨→平或平→涨的边缘误判） | `pnl_matrix[0][1]`, `pnl_matrix[2][1]` |
| `mid_penalty` | `float` | [0.1, 2.0] | — | 中间"平"类的错判惩罚 | `pnl_matrix[1][0]`, `pnl_matrix[1][2]` |

### 搜索空间构造逻辑

#### soft_matrix（软标签平滑矩阵）

由 `target_conf` 和 `target_conf_mid` 动态构造：

```
rem      = 1.0 - target_conf
rem_mid  = (1.0 - target_conf_mid) / 2

soft_matrix = [
    [target_conf,     rem,         0.0],
    [rem_mid,   target_conf_mid, rem_mid],
    [0.0,             rem,    target_conf]
]
```

- 对角线上：各类别自身的置信度（`target_conf` 或 `target_conf_mid`）
- 非对角线上：分配给邻接类别的概率（错误分布在相邻类别，不会跨级分配）

#### pnl_matrix（盈亏惩罚矩阵）

由 `max_penalty`、`min_penalty`、`mid_penalty` 动态构造：

```
pnl_matrix = [
    [0.0,      min_p,    max_p],
    [mid_p,     0.0,     mid_p],
    [max_p,    min_p,     0.0 ]
]
```

- `pnl_matrix[i][j]` 表示真实类别为 `i`、预测为 `j` 的额外惩罚系数
- 三类标签顺序：`[Down, Neutral, Up]`
- 对角线上为 `0.0`（正确预测无惩罚）
- 对称性：涨→跌 与 跌→涨 对称（`max_p`），涨→平 与 平→涨 对称（`min_p`）

---

## 如何设置超参数搜索空间

### 修改搜索范围

在 `train.py` 的 `objective()` 函数中调整 `trial.suggest_*` 参数：

```python
# 例：将 window_size 搜索范围从 [100, 500] 改为 [50, 800]
window_size = trial.suggest_int('window_size', 50, 800, step=10)

# 例：将 gamma 范围从 [1.2, 3.0] 改为 [0.5, 5.0]
gamma = trial.suggest_float("gamma", 0.5, 5.0)

# 例：调整 max_penalty 范围
max_p = trial.suggest_float("max_penalty", 2.0, 5.0)
```

### 新增/移除搜索参数

如需固定某个参数而不参与搜索，直接在 `objective()` 中赋固定值即可，无需使用 `trial.suggest_*` 方法：

```python
# 不搜索 gamma，固定为 2.5
gamma = 2.5
```

如需新增搜索参数，按以下格式添加：

```python
# 例：搜索 dropout 率
dropout = trial.suggest_float('dropout', 0.1, 0.5)

# 例：搜索优化器权重衰减
weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)

# 在创建模型时传入
# FE_DeepLOB_model = FE_DeepLOB(num_features=num_features, num_classes=3, dropout=dropout)
```

> `log=True` 适用于在数量级跨度较大的范围内进行对数均匀采样（如学习率、权重衰减等）。

### 修改外层环境遍历

在 `train.py` 的 `main()` 函数中调整 `target_windows` 和 `target_alphas`：

```python
# 例：只测试 label_window = 20 和 50
target_windows = [20, 50]

# 例：增加 alpha = 0.5 阈值
target_alphas = [0.5, 1.0, 1.5, 2.0]
```

### 调整每个环境的 trial 数量

修改 `study.optimize()` 的 `n_trials` 参数：

```python
# 每环境运行 100 个 trial
study.optimize(frozen_objective, n_trials=100)
```

---

## 运行流程

参考 `Run.txt`：

1. 创建并激活 Python 虚拟环境
2. 安装依赖：`pip install -r requirements.txt`
3. 安装 GPU 版 PyTorch（推荐 CUDA 12.1）
4. 确认 GPU 可用：`python -c "import torch; print(torch.cuda.is_available())"`
5. **独立运行数据预处理**（可选，如需单独查看中间结果）：`python process_data.py`
6. **启动全流程**（预处理 + Optuna 搜索）：`python train.py`
7. 查看结果日志：`Logs/Optuna_Campaign_{timestamp}.log`

> `train.py` 启动时会自动清空并重建 `Processed_data/` 和 `Normalized_data/` 目录，确保每次运行从干净状态开始。

---

## 目录结构

```
Optuna_Hyperparameter_searching/
├── config.py               # 全局静态配置
├── process_data.py         # 数据预处理（引力特征 + 标签生成 + 滚动归一化）
├── dataset.py              # 时间序列数据集与 DataLoader
├── FE_DeepLOB.py           # 神经网络模型 + 训练/验证引擎
├── train_artifact.py       # SoftFocalLoss + EarlyStopping
├── train.py                # Optuna 主入口
├── requirements.txt        # Python 依赖
├── Run.txt                 # 运行指引
├── README.md               # 本文件
│
├── Data/
│   ├── Raw_data/           # 原始 LOB CSV（由 data_fetch.py 生成）
│   └── FE_DeepLOB_data/
│       ├── Processed_data/ # 特征工程后的数据（临时）
│       └── Normalized_data/ # 归一化后的最终训练数据
│
└── Logs/                   # Optuna 搜索日志