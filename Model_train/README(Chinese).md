# FE-DeepLOB 期货价格方向预测

基于深度学习（FE-DeepLOB 架构）的期货 L2 订单簿价格走势预测项目。模型使用 CNN + Inception + LSTM 混合架构，通过 4 个模态对未来价格方向进行三分类预测。

---

## 目录结构

```
.
├── config.py                  # 全局配置
├── train.py                   # 主训练脚本
├── export_model.py            # PyTorch → ONNX 导出
├── data_fetch.py              # 从远端服务器拉取原始数据
├── process_data.py            # 特征工程 + 标签生成 + 窗口归一化
├── dataset.py                 # 滑动窗口数据集 + DataLoader
├── FE_DeepLOB.py              # 模型定义 + train/validate engine
├── train_artifact.py          # 自定义损失函数 + EarlyStopping
├── README.md
│
├── Data/
│   ├── Primary.csv             # 交易日-合约映射表
│   ├── Raw_data/               # 原始 L2 行情 CSV
│   └── FE_DeepLOB_data/
│       ├── Processed_data/     # 特征工程处理后的 CSV
│       └── Normalized_data/    # 窗口归一化后的 CSV
│
├── Model/
│   ├── cache/                  # 训练过程中保存的 best_model.pt
│   ├── torch/                  # 最终按参数命名的 .pt 模型文件
│   └── onnx/                   # ONNX 导出模型
│
└── Logs/                       # 训练日志
```

---

## 核心文件说明

### `config.py` — 全局配置中心

#### 数据路径

| 变量 | 说明 |
|---|---|
| `copy_data_path` | 远端原始数据源目录路径 |
| `fetch_file_path` | 交易日-合约映射表 |
| `raw_data_path` | 原始行情 CSV 存放目录 |
| `processed_data_path` | 特征工程后数据输出目录 |
| `normalized_data_path` | 归一化后数据输出目录 |
| `model_save_dir` | 训练中 best model 缓存路径 |
| `model_save_name` | 缓存模型文件名 |
| `log_dir` | 训练日志输出目录 |

#### 数据处理参数

| 变量 | 说明 |
|---|---|
| `label_method` | 标签生成方法：支持多种标签策略 |
| `alpha` | 标签阈值，价格变化超过此值标记为涨/跌，否则为平稳 |
| `label_window` | 标签窗口（tick 数），可调整 |
| `normalize_window` | 归一化滑动窗口大小，可调整 |

#### DataLoader 切分参数

| 变量 | 说明 |
|---|---|
| `train_start_file` | 训练集起始文件索引 |
| `train_num_files` | 训练集文件数，可调整 |
| `val_start_file` | 验证集起始文件索引 |
| `val_num_files` | 验证集文件数，可调整 |
| `batch_size` | 训练 batch size |
| `window_size` | 模型输入序列长度（历史 tick 数），可调整 |
| `shuffle_train` | 训练集是否 shuffle |
| `drop_last` | 是否丢弃训练集最后不足 batch 的样本 |

#### 模型结构参数（通过超参搜索获得）

| 变量 | 说明 |
|---|---|
| `num_features` | snapshot 模态的原始特征数 |
| `num_classes` | 分类数：0=下跌、1=平稳、2=上涨 |

#### 训练超参

| 变量 | 说明 |
|---|---|
| `num_epochs` | 总训练 epoch 数 |
| `learning_rate` | 初始学习率 |
| `weight_decay` | AdamW 的 weight decay |
| `warmup_epochs` | 预热阶段 epoch 数 |
| `warmup_ratio` | warmup 阶段占总训练步数的比例 |
| `eta_min` | 学习率调度器最小值 |
| `log_interval` | 日志打印间隔 |

#### 损失函数参数

| 变量 | 说明 |
|---|---|
| `gamma` | Focal Loss 的 gamma 参数，控制难易样本权重 |
| `soft_matrix` | 自定义软标签平滑矩阵 |
| `pnl_matrix` | 自定义 PnL 惩罚矩阵，错误分类的惩罚权重（考虑了交易成本不对称性） |

> 损失函数结合 soft label（标签平滑）和 PnL 惩罚矩阵，使模型在分类时考虑交易成本的不对称性。具体矩阵系数属于核心配置参数。

#### Early Stopping

| 变量 | 说明 |
|---|---|
| `early_stop_patience` | 验证 loss 连续多 epoch 未改善则停止训练 |
| `early_stop_verbose` | 是否打印早停信息 |
| `monitor_loss` | 监控验证 loss |

---

### `train.py` — 训练主流程

执行以下完整 pipeline：

1. **日志初始化**
2. **数据准备** — 特征工程 + 窗口归一化
3. **创建 DataLoader** — 按配置划分训练/验证集
4. **模型初始化** — 可选 `torch.compile` 加速
5. **优化器与调度器** — AdamW + 学习率调度
6. **两阶段训练**：
   - **Warmup 阶段**：使用标准 CrossEntropyLoss
   - **正式阶段**：切换到自定义损失函数
7. **早停** — 监控验证 loss
8. **模型导出**

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
    ├── Momentum ──→ Causal Conv Blocks ──→
    ├── Snapshot ──→ Conv + Causal Conv + Inception ──→
    ├── LT Sensor ──→ Dilated Causal Conv ──→
    └── ST Sensor ──→ (direct concatenation)
                           │
                      Concatenate → LSTM
                           │
                      Linear → Logits
```

此文件还包含：
- `train_engine()` — 训练循环
- `validate_engine()` — 验证循环，返回 loss + Down/Up 的 PR-AUC

---

### `export_model.py` — ONNX 导出

将训练好的 `.pt` 权重加载为纯净模型，导出为 ONNX 格式。

```bash
python export_model.py
```

> 导出前需确认模型参数与训练时一致，并修改权重加载路径。

---

### `dataset.py` — 滑动窗口数据集

- `TimeSeriesDataset`：以滑动窗口方式从数据中切分多个输入模态
- `create_dataloader()`：读入多个归一化文件，返回 `DataLoader`

### `data_fetch.py` — 数据拉取

根据 `Primary.csv` 中的交易日和合约代码，从远端目录复制所需原始行情文件到本地。

```bash
python data_fetch.py
```

### `train_artifact.py` — 训练辅助模块

- **自定义损失函数**：结合 soft label 和 PnL 惩罚矩阵
- **`EarlyStopping`**：早停机制

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