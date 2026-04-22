# 基于可靠性知识图谱的故障预测系统

西安交通大学硕士学位论文相关研究项目，实现知识图谱与深度学习结合的变速箱故障预测。

## 项目结构

```
fault-prediction/
├── config.yaml              # 配置文件（包含贝叶斯优化后的最佳参数）
├── requirements.txt         # Python依赖
├── train.py                 # 模型训练脚本
├── evaluate.py              # 模型评估脚本
├── README.md               # 本文档

├── data/                    # 数据目录
│   └── processed/          # 处理后的数据
│       ├── processed_features.csv    # 特征数据
│       ├── knowledge_graph.json      # 知识图谱数据
│       └── kg_embeddings.json        # KG嵌入向量

├── models/                  # 保存的模型文件
│   ├── mlp_model.pt
│   ├── kg_enhanced_mlp_v1_model.pt
│   └── kg_enhanced_mlp_v2_model.pt

├── results/                 # 结果输出目录
│   ├── training_results.json    # 训练结果
│   ├── evaluation_results.json # 评估结果
│   ├── figures/               # 可视化图表
│   └── visualization_report.html

├── src/
│   ├── data/
│   │   └── preprocess.py       # 数据预处理
│   ├── kg/
│   │   ├── neo4j_manager.py    # Neo4j管理
│   │   └── kg_embedder.py      # 嵌入生成
│   ├── models/
│   │   └── mlp_model.py        # MLP模型定义
│   ├── train/
│   │   └── train.py           # 训练子模块
│   └── visualization/
│       └── visualize.py       # 可视化
└── logs/                     # 日志目录
```

## 环境配置

```bash
# 创建虚拟环境
conda create -n fault-prediction python=3.10
conda activate fault-prediction

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 训练模型

```bash
# 训练所有模型
python train.py --all

# 仅训练特定模型
python train.py --mlp          # 仅训练普通MLP
python train.py --kg-v1        # 仅训练KG-MLP V1
python train.py --kg-v2        # 仅训练KG-MLP V2
```

### 评估模型

```bash
# 自动检测并评估models文件夹中已存在的模型
python evaluate.py
```

## 实现的模型

### 1. PlainMLP
普通多层感知机，作为基准对比模型。

### 2. KG-Enhanced MLP V1（加性融合）
```
输入特征 → 特征编码器 → + → 融合层 → 分类头 → 输出
KG嵌入   → KG编码器  → ↗
```
- 特点：特征与KG嵌入分别编码后加性融合

### 3. KG-Deep Fusion MLP V2（深度融合）
```
输入特征 → 特征编码器 ─┐
                         ├→ 双线性交互 → 门控融合 → 深度融合 → 分类头 → 输出
KG嵌入   → KG编码器  ─┘
```
- 特点：
  - 双线性交互层学习特征与KG的高阶交互
  - 门控机制自适应调整特征与KG信息权重
  - 更深的融合层（2层MLP）

## 知识图谱嵌入

从Neo4j知识图谱中提取全局结构特征：

| 特征 | 维度 | 来源 |
|------|------|------|
| 故障节点数 | 1 | KG node_counts |
| 部件节点数 | 1 | KG node_counts |
| 特征节点数 | 1 | KG node_counts |
| CAUSED_BY边数 | 1 | KG edge_counts |
| LOCATED_AT边数 | 1 | KG edge_counts |
| HAS_FEATURE边数 | 1 | KG edge_counts |
| 随机投影 | 58 | 用于填充至64维 |

## 超参数优化

使用贝叶斯优化（Optuna + TPESampler）得到的最佳配置：

```yaml
hidden_dim: 128
dropout: 0.215
learning_rate: 0.0064
weight_decay: 0.00021
```

## 数据集

使用西安交通大学变速箱故障数据集，包含9种故障类型：

| 故障类型 | 描述 |
|---------|------|
| 1ndBearing_ball | 轴承球故障 |
| 1ndBearing_inner | 轴承内圈故障 |
| 1ndBearing_outer | 轴承外圈故障 |
| 1ndBearing_mix | 轴承混合故障 |
| 2ndPlanetary_brokentooth | 行星齿轮断齿 |
| 2ndPlanetary_missingtooth | 行星齿轮缺齿 |
| 2ndPlanetary_rootcracks | 行星齿轮根裂纹 |
| 2ndPlanetary_toothwear | 行星齿轮齿磨损 |
| 2ndPlanetary_normalstate | 正常状态 |

## 设备支持

自动检测并使用可用设备：
- Apple MPS (Metal Performance Shaders) - Mac GPU
- NVIDIA CUDA - GPU
- CPU - 回退方案

## 输出文件说明

### 模型文件 (models/)
- `mlp_model.pt` - 普通MLP模型
- `kg_enhanced_mlp_v1_model.pt` - KG增强V1模型
- `kg_enhanced_mlp_v2_model.pt` - KG深度融合V2模型

### 结果文件 (results/)
- `training_results.json` - 训练过程结果
- `evaluation_results.json` - 评估结果（含验证集和测试集指标）
- `evaluation_results.json` 格式：
```json
{
  "timestamp": "2026-04-22 20:28:00",
  "best_config": {...},
  "validation": {
    "MLP": {"accuracy": 0.xx, "f1": 0.xx},
    "KG_Enhanced_MLP_V1": {...},
    "KG_Enhanced_MLP_V2": {...}
  },
  "test": {...}
}
```

## 依赖

```
torch
numpy
pandas
scikit-learn
pyyaml
optuna
joblib
tqdm
matplotlib
seaborn
```

## 作者

韩旭彬 - 西安交通大学