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
│       ├── kg_embeddings.json        # 全局KG嵌入（旧版）
│       └── fault_embeddings.json     # 故障级别嵌入（V2/V3使用）

├── models/                  # 保存的模型文件
│   ├── mlp_model.pt
│   ├── kg_enhanced_mlp_v1_model.pt   # V1: 全局嵌入
│   ├── kg_enhanced_mlp_v2_model.pt   # V2: 深度融合
│   └── kg_enhanced_mlp_v3_model.pt   # V3: 故障级别嵌入

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
python train.py --kg-v1        # 仅训练KG-MLP V1 (全局嵌入)
python train.py --kg-v2        # 仅训练KG-MLP V2 (深度融合)
python train.py --kg-v3        # 仅训练KG-MLP V3 (故障级别嵌入)
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
- **嵌入类型**：全局KG统计（6维），所有样本共享相同嵌入
- **嵌入维度**：64维（6维投影）
- 特点：特征与KG嵌入分别编码后加性融合

### 3. KG-Deep Fusion MLP V2（深度融合）
```
输入特征 → 特征编码器 ─┐
                         ├→ 双线性交互 → 门控融合 → 深度融合 → 分类头 → 输出
KG嵌入   → KG编码器  ─┘
```
- **嵌入类型**：故障级别嵌入（33维），与V3相同
- 特点：
  - 双线性交互层学习特征与KG的高阶交互
  - 门控机制自适应调整特征与KG信息权重
  - 更深的融合层（2层MLP）

### 4. KG-Enhanced MLP V3（故障级别嵌入）
- **架构**：与V1相同（加性融合）
- **嵌入类型**：故障级别嵌入（33维），每个故障类型不同
- 特点：
  - 故障相似度向量（该故障与其他9种故障的相似度）
  - 故障-部件关联（4维）
  - 故障-特征权重（14维）
  - 全局KG统计（6维）

## 知识图谱嵌入

### V1嵌入（全局统计）
从Neo4j知识图谱中提取6个粗粒度统计量：

| 特征 | 维度 |
|------|------|
| 故障节点数 | 1 |
| 部件节点数 | 1 |
| 特征节点数 | 1 |
| CAUSED_BY边数 | 1 |
| LOCATED_AT边数 | 1 |
| HAS_FEATURE边数 | 1 |

投影至64维，所有样本共享。

### V2/V3嵌入（故障级别，33维）

从Neo4j知识图谱中提取故障类型级别的结构特征：

| 特征 | 维度 | 来源 |
|------|------|------|
| 故障相似度向量 | 9 | 该故障与其他9种故障的Jaccard相似度 |
| 故障-部件关联 | 4 | 故障→部件的CAUSED_BY关系 |
| 故障-特征权重 | 14 | 故障→特征的HAS_FEATURE关系 |
| 全局KG统计 | 6 | KG节点/边总数 |
| **总计** | **33** | |

### 故障相似度矩阵

基于部件和特征共享计算的9种故障类型间的相似度：

| 故障 | Ball | Broken | Inner | Missing | Mixed | Normal | Outer | Root | Wear |
|------|------|--------|-------|---------|-------|--------|-------|------|------|
| Ball_Fault | 1.0 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| Broken_Tooth | 0.4 | 1.0 | 0.4 | 1.0 | 0.4 | 0.4 | 0.4 | 1.0 | 1.0 |

齿轮类故障（Broken, Missing, Root, Wear）彼此相似度为1.0，因为都关联Gear_Tooth部件。

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

韩旭彬 - 天津大学