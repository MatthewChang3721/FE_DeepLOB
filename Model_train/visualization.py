import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from pathlib import Path


def visual_df_prepare(inputpath, window_normalize, window_size, startfile, numfiles, preds, probs):
    startfile += window_normalize
    csv_files = sorted(Path(inputpath).glob('processed_*.csv'))[startfile: startfile+numfiles]
    print(f"Found {len(csv_files)} CSV files for this dataloader.")

    def load_and_transform(file):
        df = pd.read_csv(file)[window_size - 1:]
        # 你的逻辑：标签 + 1
        df['price_move_label'] = df['price_move_label'] + 1
        return df
    data_frames = [load_and_transform(f) for f in csv_files]

    big_df = pd.concat(data_frames, ignore_index=True)
    big_df['predicted_label'] = preds
    big_df['Net_directional_score'] = probs[:,2] - probs[:,0]
    big_df['Pred_prob_Down'] = probs[:,0]
    big_df['Pred_prob_Stationary'] = probs[:,1]
    big_df['Pred_prob_UP'] = probs[:,2]
    return big_df


def visualization_engine(model, val_loader, device, confidence_threshold: float = 0.6):
    model.eval() # Switch to evluation
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad(): # stop gradient
        for features, label in val_loader:
            x_momentum = features['momentum'].to(device)
            x_pic = features['pic'].to(device) 
            x_lt_sensor = features['lt_sensor'].to(device)
            x_st_sensor = features['st_sensor'].to(device)
            label = label.to(device).long().squeeze()
            
            output = model(x_momentum, x_pic, x_lt_sensor, x_st_sensor)

            probs = F.softmax(output.data, dim=1)
            max_probs, raw_predicted = torch.max(probs, 1)

            # Confidence threshold
            trade_predicted = raw_predicted.clone()
            weak_signals_mask = (raw_predicted != 1) & (max_probs < confidence_threshold)
            trade_predicted[weak_signals_mask] = 1

            # collect all the data model generate and push to cpu
            all_labels.append(label.cpu().numpy())
            all_preds.append(trade_predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Calculate monitor accuarcy for training with raw_predicted label
            total += label.size(0)
            correct += (raw_predicted == label).sum().item()

    # concat all data
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    accuracy = 100 * correct / total
    
    # accuracy -> Model converging; pr_auc_down, pr_auc_up -> Model effectiveness. 
    return accuracy, all_preds, all_labels, all_probs


def signal_hist(df, friction_cost: float = 0.0002, x_lim: float = 0.001, save_name: str = None):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    class_labels = [0, 1, 2]

    for i, label in enumerate(class_labels):
        ax = axes[i]
        
        # 获取该预测标签下的真实价格移动序列
        subset = df[df['predicted_label'] == label]['price_move_pctg'].dropna()
        
        # 【可视化关键】：金融高频数据呈显著的尖峰厚尾分布
        # 如果不截断极端离群值，少数几个极值会把 X 轴撑得极大，导致中间 99% 的分布缩成一根针
        # 这里截取 0.5% 到 99.5% 的分位数区间进行绘图，不影响整体分布的形态判断
        p_low = subset.quantile(0.005)
        p_high = subset.quantile(0.995)
        subset_plot = subset[(subset >= p_low) & (subset <= p_high)]
        
        # 绘制直方图与 KDE（密度）线
        sns.histplot(
            subset_plot, 
            bins=100, 
            kde=True, 
            ax=ax, 
            stat='density', # 使用密度而不是绝对频数，消除类别间样本量不平衡带来的视觉误差
            color='steelblue', 
            alpha=0.6,
            line_kws={'linewidth': 2}
        )
        
        # 画出绝对 0 轴（平盘线）
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
        
        # 画出致命的 ±friction_cost 摩擦成本边界
        ax.axvline(x=friction_cost, color='red', linestyle='--', linewidth=1.5, label=f'+{friction_cost*10000:.0f}bp Friction')
        ax.axvline(x=-friction_cost, color='red', linestyle='--', linewidth=1.5, label=f'-{friction_cost*10000:.0f}bp Friction')
        
        # 标题标注该类别的具体样本量，用于确认类别不平衡的程度
        ax.set_title(f'Predicted Label: {label}\n(n={len(subset)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('True Price Move (price_move_pctg)', fontsize=11)
        
        if i == 0:
            ax.set_ylabel('Density', fontsize=11)
        
        ax.legend(loc='upper right')

    # 动态调整 X 轴显示范围，聚焦于核心波动区域
    plt.xlim(-x_lim, x_lim)

    plt.tight_layout()
    if save_name:
        Path("picture").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path("picture") / f"{save_name}.png", dpi=150, bbox_inches='tight')
    #plt.show()


def model_signal_split_plot(confidence_level, df, save_name: str = None):
    CONFIDENCE_LEVEL = confidence_level

    # 直接从 df 中读取预测标签和概率，避免重复计算
    raw_preds = df['predicted_label'].values
    max_probs = np.max(df[['Pred_prob_Down', 'Pred_prob_Stationary', 'Pred_prob_UP']].values, axis=1)

    # 生成多空掩码 (Mask)，并加入置信度过滤条件
    mask_down_valid = (raw_preds == 0) & (max_probs >= CONFIDENCE_LEVEL)
    mask_up_valid = (raw_preds == 2) & (max_probs >= CONFIDENCE_LEVEL)

    # 提取有效预测的横坐标 (置信度) 和 纵坐标 (真实价格移动bp)
    conf_down = df['Pred_prob_Down'].values[mask_down_valid]
    pctg_down = df['price_move_pctg'].values[mask_down_valid] * 10000

    conf_up = df['Pred_prob_UP'].values[mask_up_valid]
    pctg_up = df['price_move_pctg'].values[mask_up_valid] * 10000

    # --- 打印一下过滤结果 ---
    print(f"当前置信度阈值: {CONFIDENCE_LEVEL}")
    print(f"有效做空点 (Down) 数量: {len(conf_down)}")
    print(f"有效做多点 (Up) 数量: {len(conf_up)}")
    print("-" * 30)

    # 4. 开始绘图 (1行2列，共享Y轴方便对比)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # --- 左图：做空 (Down) 预测散点 ---
    if len(conf_down) > 0:
        ax1.scatter(conf_down, pctg_down, alpha=0.5, s=5, color='forestgreen')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8) 
    ax1.set_title(fr"Down Predictions (Conf $\geq$ {CONFIDENCE_LEVEL})", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Softmax Probability (Down)", fontsize=11)
    ax1.set_ylabel("Actual Price Move (bps)", fontsize=11)
    ax1.set_xlim(left=CONFIDENCE_LEVEL - 0.02, right=1.02)
    ax1.set_ylim(-20, 20)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 右图：做多 (Up) 预测散点 ---
    if len(conf_up) > 0:
        ax2.scatter(conf_up, pctg_up, alpha=0.5, s=5, color='firebrick')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8) 
    ax2.set_title(fr"Up Predictions (Conf $\geq$ {CONFIDENCE_LEVEL})", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Softmax Probability (Up)", fontsize=11)
    ax2.set_xlim(left=CONFIDENCE_LEVEL - 0.02, right=1.02)
    ax2.set_ylim(-20, 20)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"Model Edge Validation (Sniper Threshold: {CONFIDENCE_LEVEL})", fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    if save_name:
        Path("picture").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path("picture") / f"{save_name}.png", dpi=150, bbox_inches='tight')
    #plt.show()


def model_signal_scatter(confidence_level, df, save_name: str = None):
    CONFIDENCE_LEVEL = confidence_level

    # 直接从 df 读取预测标签和各类概率
    raw_preds = df['predicted_label'].values
    prob_down = df['Pred_prob_Down'].values
    prob_stationary = df['Pred_prob_Stationary'].values
    prob_up = df['Pred_prob_UP'].values
    max_probs = np.max([prob_down, prob_stationary, prob_up], axis=0)

    # 构建连续的横坐标 (Directional Score)
    x_coords = np.zeros(len(df))

    # 预测为跌 (Down)：置信度乘 -1
    mask_pred_down = (raw_preds == 0)
    x_coords[mask_pred_down] = max_probs[mask_pred_down] * -1.0

    # 预测为涨 (Up)：置信度乘 1
    mask_pred_up = (raw_preds == 2)
    x_coords[mask_pred_up] = max_probs[mask_pred_up] * 1.0

    # 预测为平盘 (Stationary)：使用 (涨概率 - 跌概率)
    mask_pred_stat = (raw_preds == 1)
    x_coords[mask_pred_stat] = prob_up[mask_pred_stat] - prob_down[mask_pred_stat]

    # 提取纵坐标 (真实收益率 bps)
    y_coords = df['price_move_pctg'].values * 10000

    # 根据置信度阈值进行终极切分
    mask_strong_down = mask_pred_down & (max_probs >= CONFIDENCE_LEVEL)
    mask_strong_up = mask_pred_up & (max_probs >= CONFIDENCE_LEVEL)
    x_down_count = np.sum(mask_strong_down)
    x_up_count = np.sum(mask_strong_up)
    mask_grey_zone = ~(mask_strong_down | mask_strong_up)

    # ==========================================
    # 工业级单图全景绘制
    # ==========================================
    plt.figure(figsize=(16, 9))

    # 第一层：灰色噪音云
    plt.scatter(x_coords[mask_grey_zone], y_coords[mask_grey_zone], 
                alpha=0.15, s=3, color='dimgray', zorder=1, label='Weak Signals & Stationary (Filtered)')

    # 第二层：强做空点
    if np.any(mask_strong_down):
        plt.scatter(x_coords[mask_strong_down], y_coords[mask_strong_down], 
                    alpha=0.4, s=6, color='forestgreen', zorder=2, label=fr'Strong Short (Conf $\geq$ {CONFIDENCE_LEVEL})')

    # 第三层：强做多点
    if np.any(mask_strong_up):
        plt.scatter(x_coords[mask_strong_up], y_coords[mask_strong_up], 
                    alpha=0.4, s=6, color='firebrick', zorder=2, label=fr'Strong Long (Conf $\geq$ {CONFIDENCE_LEVEL})')

    # 绘制十字基准线
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=3)  
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=3)  

    # 绘制置信度隔离墙
    plt.axvline(CONFIDENCE_LEVEL, color='firebrick', linestyle='--', linewidth=2, alpha=0.8, zorder=3)
    plt.axvline(-CONFIDENCE_LEVEL, color='forestgreen', linestyle='--', linewidth=2, alpha=0.8, zorder=3)

    # 涂灰无交易区
    plt.axvspan(-CONFIDENCE_LEVEL, CONFIDENCE_LEVEL, color='gray', alpha=0.05, zorder=0) 

    # 图表美化与标签
    plt.title(fr"Full-Spectrum Alpha Landscape (Threshold: $\pm${CONFIDENCE_LEVEL})", fontweight='bold', fontsize=16, pad=15)
    plt.xlabel("Directional Confidence (Negative = Short, Positive = Long)", fontsize=12)
    plt.ylabel("Realized Price Move (bps)", fontsize=12)
    plt.xlim(-1.05, 1.05)
    plt.ylim(-30, 30)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    if save_name:
        Path("picture").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path("picture") / f"{save_name}.png", dpi=150, bbox_inches='tight')
    #plt.show()

    # 打印战报
    print(f"📊 信号总览 (阈值 {CONFIDENCE_LEVEL})")
    print(f"做空开仓: {x_down_count} 次")
    print(f"做多开仓: {x_up_count} 次")
    print(f"平盘观望 (灰色区): {len(raw_preds) - x_down_count - x_up_count} 次")


def net_directional_signal_plot(df, y_lim: int = 30, save_name: str = None):
    net_directional_score = df['Net_directional_score'].values
    actual_returns = df['price_move_pctg'].values * 10000 # in bps

    # ==========================================
    # 架构师级定制：A股/国内高频专属视觉引擎
    # ==========================================
    vibrant_colors = ['#00C800', '#D3D3D3', '#FF0000'] 
    cn_cmap = LinearSegmentedColormap.from_list('CN_HighFreq', vibrant_colors)

    plt.figure(figsize=(12, 8))

    # 绘制散点
    scatter = plt.scatter(
        net_directional_score, 
        actual_returns, 
        c=net_directional_score, 
        cmap=cn_cmap,
        alpha=0.25,
        s=4
    )

    # 绘制十字基准线
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)  
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)  

    # 先设置坐标范围，再标注文字（避免 text 定位到默认坐标）
    plt.xlim(-1.05, 1.05)
    plt.ylim(-y_lim, y_lim)

    plt.text(0.8, y_lim * 0.8, 'Strong Buy', color='#FF0000', fontsize=12, fontweight='bold', ha='center')
    plt.text(-0.8, -y_lim * 0.8, 'Strong Sell', color='#00C800', fontsize=12, fontweight='bold', ha='center')

    plt.title("Net Directional Score vs. Realized Alpha (CN Market Visuals)", fontweight='bold', fontsize=16, pad=15)
    plt.xlabel("Net Directional Score: P(Up) - P(Down)", fontsize=13)
    plt.ylabel("Realized Price Move bps", fontsize=13)

    plt.grid(True, linestyle=':', alpha=0.6)

    # 添加颜色条
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Signal Polarity (Green=Short, Gray=Neutral, Red=Long)', rotation=270, labelpad=20)

    plt.tight_layout()
    if save_name:
        Path("picture").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path("picture") / f"{save_name}.png", dpi=150, bbox_inches='tight')
    #plt.show()

    # 打印皮尔逊相关系数
    pearson_r = np.corrcoef(net_directional_score, actual_returns)[0, 1]
    print(f"📊 净方向得分与真实收益的皮尔逊相关系数: {pearson_r:.4f}")

    # 打印斯皮尔曼秩相关系数（对异常值更稳健）
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(net_directional_score, actual_returns)
    print(f"📊 净方向得分与真实收益的斯皮尔曼秩相关系数: {spearman_r:.4f} (p-value: {spearman_p:.2e})")


def PRAUC(labels, probs, save_name: str = None):
    # ==========================================
    # 1. 标签二值化 (One-vs-Rest)
    # ==========================================
    Y_bin = label_binarize(labels, classes=[0, 1, 2])
    n_classes = Y_bin.shape[1]

    # ==========================================
    # 2. 视觉配置
    # ==========================================
    colors = ['#00C800', '#A9A9A9', '#FF0000']
    class_names = ['Short / Down (Class 0)', 'Neutral / Stationary (Class 1)', 'Long / Up (Class 2)']

    plt.figure(figsize=(10, 8))

    # ==========================================
    # 3. 逐个类别计算并绘制 PR 曲线
    # ==========================================
    ap_scores = {}
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(Y_bin[:, i], probs[:, i])
        ap = average_precision_score(Y_bin[:, i], probs[:, i])
        ap_scores[class_names[i]] = ap
        
        plt.plot(recall, precision, color=colors[i], lw=2.5, alpha=0.8,
                label=f'{class_names[i]} (PR-AUC = {ap:.4f})')

    # ==========================================
    # 4. 图表美化
    # ==========================================
    plt.title('Precision-Recall Landscape (Micro-structural Momentum)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    if save_name:
        Path("picture").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path("picture") / f"{save_name}.png", dpi=150, bbox_inches='tight')
    #plt.show()
    
    return ap_scores
