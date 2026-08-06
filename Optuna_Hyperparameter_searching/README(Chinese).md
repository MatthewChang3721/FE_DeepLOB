# 基于 FE_DeepLOB 与 Optuna 的期货高频方向预测

本项目利用高频限价订单簿（LOB）数据，使用带特征工程的神经网络模型预测期货短期价格方向（上涨/下跌/持平），并通过 **Optuna** 贝叶斯优化自动搜索最优超参数组合，提升模型在指定评价指标上的表现。

---

## 项目文件说明

### `config.py` — 全局静态配置

存放训练过程中不会随 Optuna 搜索变化的基础参数。当你需要快速调整训练骨架时在此修改：

| 参数 | 默认值 | 含义 |
|------|:---:|------|
| **数据路径** | | |
| `copy_data_path` | `/yourdatabase` | 原始数据远程路径 |
| `fetch_file_path` | `"Data/SC_Primary.csv"` | 拉取的中间文件路径 |
| `raw_data_path` | `"Data/Raw_data/"` | 原始 LOB CSV 目录 |
| `processed_data_path` | `"Data/FE_DeepLOB_data/Processed_data/"` | 特征工程后的临时数据目录 |
| `normalized_data_path` | `"Data/FE_DeepLOB_data/Normalized_data/"` | 归一化后的最终训练数据目录 |
| `model_save_dir` | `"Model/cache"` | 模型保存路径 |
| `model_save_name` | `"best_model.pt"` | 最佳模型文件名 |
| `log_dir` | `"Logs/"` | 日志输出目录 |
| **数据处理** | | |
| `label_method` | `"trend"` | 标签生成方法（`'trend'` = midprice 滚动均值，`'path'` = 三重屏障） |
| `alpha` | `1e-4` | 标签生成中价格变动阈值 |
| `label_window` | `10` | 预测未来多少个 tick 的价格方向 |
| `normalize_window` | `5` | 滚动归一化使用的历史天数 |
| **DataLoader 划分** | | |
| `train_start_file` | `0` | 训练集起始文件索引 |
| `train_num_files` | `25` | 训练集使用文件数 |
| `val_start_file` | `25` | 验证集起始文件索引 |
| `val_num_files` | `5` | 验证集使用文件数 |
| `batch_size` | `1024` | 训练 batch size |
| `target_size` | `1` | 预测步长 |
| `shuffle_train` | `True` | 训练集是否 shuffle |
| `drop_last` | `True` | 是否丢弃最后一个不完整 batch |
| **模型** | | |
| `window_size` | `150` | 输入模型的历史窗口长度（tick），会被 Optuna 动态覆盖 |
| `num_features` | `5` | 模型内部特征维度参数 |
| `num_classes` | `3` | 分类数 |
| **可复现性** | | |
| `seed` | `42` | 随机种子 |
| **训练** | | |
| `num_epochs` | `200` | 每轮 trial 最大训练轮数 |
| `learning_rate` | `0.001` | 初始学习率 |
| `weight_decay` | `5e-3` | 优化器权重衰减 |
| `warmup_epochs` | `5` | warmup 阶段轮数 |
| `warmup_ratio` | `0.05` | warmup 步数占总步数的比例 |
| `eta_min` | `5e-5` | 学习率调度器的最小学习率 |
| `log_interval` | `10` | 日志打印间隔（epoch） |
| **损失函数** | | |
| `gamma` | `2.0` | 损失函数参数，会被 Optuna 动态覆盖 |
| `soft_matrix` | 3×3 矩阵 | 标签平滑矩阵，会被 Optuna 动态覆盖 |
| `pnl_matrix` | 3×3 矩阵 | 惩罚矩阵，会被 Optuna 动态覆盖 |
| **早停** | | |
| `early_stop_patience` | `10` | 早停等待轮数 |
| `early_stop_verbose` | `False` | 是否打印早停细节 |
| `monitor_loss` | `True` | 是否监控验证 loss |

---

### `process_data.py` — 数据预处理流水线

#### `process_data(inputfile, outputfile, label_method, label_window, alpha)`

对单日原始 LOB CSV 文件进行特征工程和标签生成，处理后保存到指定路径。

**特征工程**：涉及公司知识产权，未展示具体公式。

**标签生成**：
- 使用 `label_method` 参数选择标签生成策略（共两种预定义方法），基于未来窗口的价格走势计算 `price_move_label`（取值 -1 / 0 / 1，分别代表下跌/持平/上涨），通过 `alpha` 阈值控制判定的灵敏度。
- `'trend'`（默认）：基于未来窗口内 MidPrice 滚动均值相对当前价格的变动超过 `alpha`（bps）来判定标签。
- `'path'`：基于三重屏障方法（Numba 加速），标签由窗口内先触及的上/下屏障方向决定。

标签值：`1`=上涨、`0`=持平、`-1`=下跌

#### `window_normalize(inputpath, outputpath, window_size)`

对处理后的CSV文件进行跨日滚动归一化，**严格保证没有未来信息泄露**。

- 按文件日期排序（`sorted(glob)`）
- 维护一个容量为 `window_size + 1` 的 deque 缓冲区
- 每次取缓冲区中前 `window_size` 天计算各特征的均值和标准差
- 用历史统计量归一化第 `window_size+1` 天的数据
- 输出预处理后的训练特征文件和标签

#### 独立执行

`process_data.py` 也可作为独立脚本直接运行：

```bash
python process_data.py
```

此时使用默认参数从 `Data/Raw_data/` 读取原始 CSV 并执行完整的数据预处理流程。

---

### `dataset.py` — 时间序列数据集与 DataLoader

#### `TimeSeriesDataset(Dataset)`

继承 PyTorch `Dataset` 的时序切片数据集：

- **按列名（而非位置索引）访问特征**，并带 schema 校验：必需列缺失或出现未知列时立即抛出明确错误
- 特征分为四组命名张量：`momentum`（MidPrice_diff）、`pic`（Bid_G1..Ask_G5）、`lt_sensor`（MidPrice）、`st_sensor`（OBI_3, OBI_delta）
- 以滑动窗口方式生成 `(X, Y)` 样本对，偏移量确保标签取自特征窗口的最后一个时间点
- 自动限制索引范围避免越界，并校验数据长度是否满足窗口要求

#### `create_dataloader(inputpath, start_files, num_files, window_size, target_size, batch_size, shuffle, drop_last)`

- 从 `inputpath` 读取已归一化的 CSV 文件
- 按文件索引（排序后）切片，支持灵活划分训练/验证集
- 标签值转换为适应 PyTorch 分类要求的格式
- 统计标签分布并返回（用于计算类别权重）
- 返回 `(DataLoader, label_distribution_df)`

---

### `FE_DeepLOB.py` — 神经网络模型与训练/验证引擎

#### `FE_DeepLOB(nn.Module)`

涉及公司知识产权，未展示。

#### `train_engine(model, train_loader, optimizer, criterion, device, lr_scheduler)`

单 epoch 训练函数：

- 从 DataLoader 获取分批数据
- 梯度裁剪
- 返回 `(avg_loss, avg_acc)`

#### `validate_engine(model, val_loader, criterion, device)`

单 epoch 验证函数：

- 计算 Down（类别0）和 Up（类别2）的 PR-AUC
- 返回 `(avg_loss, pr_auc_down, pr_auc_up)`

> PR-AUC 相比准确率更适合不平衡分类场景，能更好地衡量模型对少数类（涨/跌）的区分能力。

---

### `train_artifact.py` — 训练辅助组件

#### `SoftFocalLoss(nn.Module)`

融合标签平滑与盈亏感知的加权损失函数，涉及公司知识产权，未展示公式。在 Optuna 搜索中通过 `gamma` 以及两个 3×3 矩阵参数参与调优。

#### `EarlyStopping`

基于验证 loss 的早停机制：

- `patience` 由 `config.early_stop_patience` 控制
- `verbose` 由 `config.early_stop_verbose` 控制
- 可选 `monitor_loss`：监控验证 loss 或其它指标

---

### `opt_searching.py` — Optuna 主搜索入口

核心控制脚本（旧版 `train.py` 的继任者），执行流程：

1. **数据目录清理**：运行开始时自动删除并重建数据缓存目录，确保每次运行从干净状态开始
2. **外层环境遍历**：遍历 `target_windows`，对每个窗口动态缩放 alpha 阈值（缩放因子基于窗口大小计算），使标签分布在不同时间尺度下保持合理
3. **数据制备**：对每个环境组合，运行 `process_data()` 和 `window_normalize()` 生成对应标签的文件
4. **Optuna 搜索**：对每个环境创建独立的 study，优化目标函数，每环境运行多个 trial
5. **日志记录**：所有输出写入日志文件

所有训练超参数均从 `config.py` 读取，方便统一调整。

---

## 超参数搜索空间

Optuna 在每个 trial 中搜索的超参数及范围如下：

| 参数 | 类型 | 搜索范围 | 含义 |
|------|:---:|:--------:|------|
| `window_size` | `int` | 区间搜索 | 模型输入的历史窗口长度（tick） |
| `gamma` | `float` | 区间搜索 | 损失函数聚焦参数 |
| `target_conf` | `float` | 区间搜索 | 标签平滑矩阵对角置信度参数 |
| `target_conf_mid` | `float` | 区间搜索 | 标签平滑矩阵中间类置信度参数 |
| `max_penalty` | `float` | 区间搜索 | 惩罚矩阵参数 |
| `min_penalty` | `float` | 区间搜索 | 惩罚矩阵参数 |
| `mid_penalty` | `float` | 区间搜索 | 惩罚矩阵参数 |

各矩阵参数的具体构造方式涉及公司知识产权，未展示。

---

## 如何设置超参数搜索空间

### 修改搜索范围

在 `opt_searching.py` 的 `objective()` 函数中调整 `trial.suggest_*` 参数。

### 新增/移除搜索参数

如需固定某个参数而不参与搜索，直接在 `objective()` 中赋固定值即可，无需使用 `trial.suggest_*` 方法。

如需新增搜索参数，按以下格式添加：

```python
# 例：搜索 dropout 率
dropout = trial.suggest_float('dropout', 0.1, 0.5)
```

### 调整每个环境的 trial 数量

修改 `study.optimize()` 的 `n_trials` 参数。

---

## 运行流程

参考 `Run.txt`：

1. 创建并激活 Python 虚拟环境
2. 安装依赖：`pip install -r requirements.txt`
3. 安装 GPU 版 PyTorch（推荐 CUDA 12.1）
4. 确认 GPU 可用：`python -c "import torch; print(torch.cuda.is_available())"`
5. **独立运行数据预处理**（可选）：`python process_data.py`
6. **启动全流程**：`python opt_searching.py`
7. 查看结果日志

> `opt_searching.py` 启动时会自动清空并重建数据缓存目录，确保每次运行从干净状态开始。

---

## 目录结构

```
Optuna_Hyperparameter_searching/
├── config.py               # 全局静态配置
├── process_data.py         # 数据预处理
├── dataset.py              # 时间序列数据集与 DataLoader
├── FE_DeepLOB.py           # 神经网络模型 + 训练/验证引擎
├── train_artifact.py       # 训练辅助组件
├── train.py                # （旧版）Optuna 入口
├── opt_searching.py        # Optuna 主入口
├── requirements.txt        # Python 依赖
├── Run.txt                 # 运行指引
├── README.md               # 本文件
│
├── Data/
│   ├── Raw_data/           # 原始 LOB CSV
│   └── FE_DeepLOB_data/
│       ├── Processed_data/ # 特征工程后的数据（临时）
│       └── Normalized_data/ # 归一化后的最终训练数据
│
└── Logs/                   # Optuna 搜索日志