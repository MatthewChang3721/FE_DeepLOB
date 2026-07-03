# FE-DeepLOB 原油期货 LOB 价格方向预测

基于深度学习（FE-DeepLOB 架构）的上海原油期货（SC）L2 订单簿价格走势预测项目。模型使用 CNN + Inception + LSTM 混合架构，通过 4 个模态（momentum、price snapshot、long-term sensor、short-term sensor）对未来价格方向进行三分类预测（下跌/平稳/上涨）。

---

## 目录结构

```
.
├── config.py                  # 全局配置（数据处理、模型、训练超参）
├── train.py                   # 主训练脚本
├── export_model.py            # PyTorch → ONNX 导出
├── data_fetch.py              # 从远端服务器拉取原始数据
├── process_data.py            # 特征工程 + 标签生成 + 窗口归一化
├── dataset.py                 # 滑动窗口数据集 + DataLoader
├── FE_DeepLOB.py              # 模型定义 + train/validate engine
├── train_artifact.py          # SoftFocalLoss + EarlyStopping
├── README.md
│
├── Data/
│   ├── SC_Primary.csv          # 交易日-合约映射表
│   ├── Raw_data/               # 原始 L2 行情 CSV（由 data_fetch.py 生成）
│   └── FE_DeepLOB_data/
│       ├── Processed_data/     # 特征工程处理后的 CSV
│       └── Normalized_data/    # 窗口 Z-score 归一化后的 CSV
│
├── Model/
│   ├── cache/                  # 训练过程中保存的 best_model.pt
│   ├── torch/                  # 最终按时间戳命名的 .pt 模型文件
│   └── onnx/                   # ONNX 导出模型（C++ 推理用）
│
└── Logs/                       # 训练日志
```

---

## 核心文件说明

### `config.py` — 全局配置中心

所有关键超参和配置集中于此文件。以下按功能分组说明：

#### 数据路径

| 变量 | 默认值 | 说明 |
|---|---|---|
| `copy_data_path` | `T:/microvast-zx_920/Future/INE/sc` | 远端原始数据源目录路径 |
| `fetch_file_path` | `Data/SC_Primary.csv` | 交易日-合约映射表 |
| `raw_data_path` | `Data/Raw_data/` | 原始行情 CSV 存放目录 |
| `processed_data_path` | `Data/FE_DeepLOB_data/Processed_data/` | 特征工程后数据输出目录 |
| `normalized_data_path` | `Data/FE_DeepLOB_data/Normalized_data/` | 归一化后数据输出目录 |
| `model_save_dir` | `Model/cache` | 训练中 best model 缓存路径 |
| `model_save_name` | `best_model.pt` | 缓存模型文件名 |
| `log_dir` | `Logs/` | 训练日志输出目录 |

#### 数据处理参数

> **⚠️ 对模型训练效果影响极大，调参重点关注**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `label_method` | `"l2"` | 标签生成方法：`l1`=未来价格直接差分、`l2`=未来滚动均价差分（推荐）、`l3`=考虑交易成本 |
| `alpha` | `1e-4` | 标签阈值（bps）。价格变化超过此值标记为涨/跌，否则为平稳。例如 `1e-4` = 1bps |
| `label_window` | `10` | 标签窗口（tick 数）。预测未来多少个 tick 的价格方向 |
| `normalize_window` | `5` | 归一化滑动窗口。使用前 N 日的数据做 Z-score 统计量 |

#### DataLoader 切分参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `train_start_file` | `0` | 从第几个文件开始作为训练集（0-indexed） |
| `train_num_files` | `25` | 使用几个文件作为训练集 |
| `val_start_file` | `25` | 从第几个文件开始作为验证集 |
| `val_num_files` | `5` | 使用几个文件作为验证集 |
| `batch_size` | `1024` | 训练 batch size |
| `window_size` | `150` | 模型输入序列长度（历史 tick 数） |
| `shuffle_train` | `True` | 训练集是否 shuffle |
| `drop_last` | `True` | 是否丢弃训练集最后不足 batch 的样本 |

> **文件分割释义**：以上默认值表示使用第 0 ~ 24 号（共 25 个）归一化文件作为训练集，第 25 ~ 29 号（共 5 个）作为验证集。结合 `normalize_window=5`，前 5 个文件仅用于计算归一化统计量而不参与训练/验证。分割时需确保日期连续性，避免未来信息泄露。

#### 模型结构参数（参数设定为Optuna Hyperparameter Searching的结果）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `num_features` | `5` | snapshot 模态的原始特征数（10 个 price gravity 列经 stride=2 conv2D 压缩后为 5） |
| `num_classes` | `3` | 分类数：0=下跌(Down)、1=平稳(Stationary)、2=上涨(Up) |

#### 训练超参

| 变量 | 默认值 | 说明 |
|---|---|---|
| `num_epochs` | `200` | 总训练 epoch 数 |
| `learning_rate` | `0.001` | 初始学习率 |
| `weight_decay` | `5e-3` | AdamW 的 weight decay（L2 正则化） |
| `warmup_epochs` | `5` | 使用 CrossEntropyLoss 进行 warmup 的 epoch 数，之后切换到 SoftFocalLoss |
| `warmup_ratio` | `0.05` | warmup 阶段占总训练步数的比例（用于 CosineAnnealing 之前的 LinearLR） |
| `eta_min` | `5e-5` | CosineAnnealing 学习率最小值 |
| `log_interval` | `10` | 每多少 epoch 打印一次日志 |

#### 损失函数参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `gamma` | `2.0` | Focal Loss 的 gamma 参数，控制难易样本权重 |
| `soft_matrix` | 3×3 矩阵（见下文） | 软标签矩阵：行=真实标签，列=预测标签概率分布 |
| `pnl_matrix` | 3×3 矩阵（见下文） | PnL 惩罚矩阵：错误分类的惩罚权重（考虑了交易成本不对称性） |

**soft_matrix**（软标签平滑）：
```
真实\预测：   下跌     平稳     上涨
下跌        0.90    0.10    0.00
平稳        0.05    0.90    0.05
上涨        0.00    0.10    0.90
```

**pnl_matrix**（PnL 惩罚，不对称反映了涨/跌的盈亏不对称）：
```
真实\预测：   下跌     平稳     上涨
下跌        0.0     0.5     3.0
平稳        1.0     0.0     1.0
上涨        3.0     0.5     0.0
```
> 例如：真实是"上涨"却预测为"下跌"，惩罚 = 3.0（做反了亏钱最多）

#### Early Stopping

| 变量 | 默认值 | 说明 |
|---|---|---|
| `early_stop_patience` | `15` | 验证 loss 连续多少 epoch 未改善则停止训练 |
| `early_stop_verbose` | `False` | 是否打印早停信息 |
| `monitor_loss` | `True` | 监控验证 loss（`False` 则监控其他指标） |

---

### `train.py` — 训练主流程

执行以下完整 pipeline：

1. **日志初始化** — 同时输出到 `Logs/` 文件和终端
2. **数据准备** — 读取 `Raw_data/` 中的 CSVs，调用 `process_data()` 做特征工程，再调用 `window_normalize_FE()` 做滚动 Z-score 归一化
3. **创建 DataLoader** — 按 `train_start_file` / `val_start_file` 划分训练/验证集，输出标签分布统计
4. **模型初始化** — 实例化 `FE_DeepLOB`，若 CUDA 可用则 `torch.compile` 加速
5. **优化器与调度器** — AdamW + LinearLR(Warmup) → CosineAnnealingLR
6. **两阶段训练**：
   - **Warmup 阶段**（前 `warmup_epochs` 轮）：使用标准 CrossEntropyLoss
   - **正式阶段**：切换到 SoftFocalLoss（含软标签 + PnL 惩罚）
7. **早停** — 若验证 loss `early_stop_patience` 轮未改善则停止
8. **模型导出** — 将 `cache/best_model.pt` 复制到 `Model/torch/{日期}_{参数}.pt`

**启动训练**：
```bash
python train.py
```

---

### `FE_DeepLOB.py` — 模型定义

**模型架构**（按 forward 顺序）：

```
                                      Input (4 modalities)
                                           │
          ┌────────────────┬────────────────┬────────────────┐
          │                │                │                │
      Momentum        Snapshot(PIC)    LT Sensor      ST Sensor
     (batch,T,1)      (batch,T,10)   (batch,T,1)    (batch,T,2)
          │                │                │                │
    causal Conv2D×3    Conv2D(feat)     Dilated Conv×5   (直接拼接)
    (RF=10)          causal Conv2D×6         │                │
          │           (RF=19)                │                │
          │                │                 │                │
          │           Inception Block        │                │
          │          (3 path concat)         │                │
          └───────────────┬──────────────────┴────────────────┘
                          │
                      Concatenate → LSTM(hidden=32)
                          │
                     Linear(32→3)
                          │
                       Logits
```

- **Momentum** — `MidPrice_diff`（对数收益×10000），3 层因果卷积捕捉短时动量
- **PIC (Price Snapshot)** — 10 维 price gravity 特征 → 2 倍步长压缩 → 6 层因果时间卷积 → Inception 模块（3 条路径）→ bottleneck
- **LT Sensor** — `MidPrice`（归一化后），5 层空洞因果卷积（dilation=1,2,4,8,8），感受野大
- **ST Sensor** — `OBI_level3` + `OBI_delta`，直接拼接到 LSTM 输入

此文件还包含两个辅助函数：
- `train_engine()` — 单个 epoch 训练循环，含梯度裁剪
- `validate_engine()` — 验证循环，返回 average loss + Down/Up 的 PR-AUC

---

### `export_model.py` — ONNX 导出

将训练好的 `.pt` 权重加载到纯净模型，导出为 ONNX 格式供 C++ 推理引擎使用。

```bash
python export_model.py
```

**ONNX 文件命名规则**（与 `train.py` 的 `.pt` 文件保持一致）：
```
{YYYYMMDD}_{label_window}_{alpha_bps}bps_{window_size}.onnx
```
例如：`20260702_10_1bps_150.onnx`

> 使用前需在 `export_model.py` 中修改 `torch.load()` 的权重路径，指向实际训练输出的 `.pt` 文件。

**ONNX 输入/输出节点**：

| 方向 | 名称 | 形状 |
|---|---|---|
| Input | `momentum` | `(batch, T, 1)` |
| Input | `pic` | `(batch, T, 10)` |
| Input | `lt_sensor` | `(batch, T, 1)` |
| Input | `st_sensor` | `(batch, T, 2)` |
| Output | `logits` | `(batch, 3)` |

> **注意**：导出前需确认 `export_model.py` 中的模型参数（`num_features=5`）与训练时一致，且加载的权重路径正确。

---

### `process_data.py` — 特征工程

将原始 L2 行情 CSVs 处理为模型可用的特征矩阵：

**原始输入**（CSV 列）：
- `timestamp`, `bid1-5`, `ask1-5`, `bidSize1-5`, `askSize1-5`

**生成特征**（14 列）：
| 特征 | 列名 | 数量 | 描述 |
|---|---|---|---|
| MidPrice_diff | `MidPrice_diff` | 1 | ln(P_t / P_{t-1}) × 10000 |
| Bid Gravity | `Bid_G1` ~ `Bid_G5` | 5 | 买方价格引力 |
| Ask Gravity | `Ask_G1` ~ `Ask_G5` | 5 | 卖方价格引力 |
| MidPrice | `MidPrice` | 1 | 归一化后的中间价 |
| OBI_level3 | `OBI_level3` | 1 | 3 档订单簿不平衡 |
| OBI_delta | `OBI_delta` | 1 | OBI_5 - OBI_3 |
| Label | `price_move_label` | 1 | `-1`=下跌, `0`=平稳, `1`=上涨 |

---

### `dataset.py` — 滑动窗口数据集

- `TimeSeriesDataset`：以滑动窗口方式从 14 列数据中切分 4 个输入模态
- `create_dataloader()`：读入多个归一化文件，统计标签分布，返回 `DataLoader` + 标签统计 DataFrame

### `data_fetch.py` — 数据拉取

根据 `SC_Primary.csv` 中的交易日和合约代码，从远端目录复制所需原始行情文件到本地 `Raw_data/`。自动跳过缺失文件。

**所需文件数计算**：
```python
required_num_files = train_num_files + val_num_files + 5 + 3
#                    |____________训练+验证________||_归一化_||_OOS_|
```
- `train_num_files + val_num_files`：训练集和验证集的交易日文件数（当前默认 `25 + 5 = 30`）
- `+5`：`normalize_window=5`，规范化滑动窗口需要前 5 个文件作为统计量计算，不参与实际训练
- `+3`：预留 3 个文件用作训练后的样本外测试（OOS）评估
- 默认 `required_num_files = 38`，即始终从 `SC_Primary.csv` 最新日期向前取 38 个交易日的数据

```bash
python data_fetch.py
```

### `train_artifact.py` — 训练辅助模块

- **`SoftFocalLoss`**：自定义损失函数，结合 soft label（标签平滑）和 PnL 惩罚矩阵，使模型在分类时考虑交易成本的不对称性
- **`EarlyStopping`**：监控验证 loss，若连续 `patience` 轮未改善则触发早停，并自动保存最佳模型

---

## 使用流程

### 1. 拉取数据
```bash
python data_fetch.py
```

### 2. 训练模型
```bash
python train.py
```

### 3. 导出 ONNX
```bash
python export_model.py
```

> 导出前需确认 `export_model.py` 中的模型参数与训练时一致，并修改权重加载路径为实际训练输出的 `.pt` 文件