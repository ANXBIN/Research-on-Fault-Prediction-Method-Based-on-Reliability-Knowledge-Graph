# 基于可靠性知识图谱的故障预测系统

## 项目概述

利用知识图谱增强深度学习模型的故障诊断能力，实现变速箱故障类型的精准分类。系统集成了MLP、CNN、GNN三大类深度学习模型，并支持知识图谱融合方案。

**数据集**：西安交通大学变速箱故障数据集（82,606样本，9类故障）

---

## 项目结构

```
fault-prediction/
├── config.yaml              # 配置文件
├── train.py                 # 模型训练脚本
├── evaluate.py              # 模型评估脚本
├── predict.py               # 交互式预测脚本（含LLM解释）
├── requirements.txt         # Python依赖
├── README.md               # 本文档

├── data/
│   └── processed/
│       ├── processed_features.csv    # 特征数据（20维）
│       ├── knowledge_graph.json      # 知识图谱数据
│       ├── kg_embeddings.json         # KG嵌入
│       └── fault_embeddings.json      # 故障级别嵌入

├── models/
│   ├── mlp_model.pt     # 普通MLP
│   ├── mlp_kg_model.pt  # MLP-KG融合
│   ├── cnn_model.pt     # 普通CNN
│   ├── cnn_kg_model.pt  # CNN-KG融合
│   ├── gnn_model.pt     # 图神经网络
│   └── gnn_kg_model.pt  # GNN-KG融合

├── results/
│   ├── training_results.json
│   ├── evaluation_results.json
│   └── figures/

└── src/
    ├── models/
    │   ├── mlp_model.py     # MLP模型
    │   ├── cnn_model.py      # CNN模型
    │   └── gnn_model.py       # GNN模型
    └── visualization/
        └── plot_results.py   # 可视化
```

---

## 快速开始

### 1. 环境配置

```bash
conda create -n fault-prediction python=3.10
conda activate fault-prediction
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 训练所有模型（默认100轮）
python train.py --all

# 指定训练轮数
python train.py --all --epochs 200

# 仅训练特定模型
python train.py --mlp              # 普通MLP
python train.py --mlp-kg           # MLP-KG融合
python train.py --cnn              # 普通CNN
python train.py --cnn-kg           # CNN-KG融合
python train.py --gnn              # 图神经网络
python train.py --gnn-kg          # GNN-KG融合
```

### 3. 评估模型

```bash
python evaluate.py
```

### 4. 交互式预测

```bash
# 使用随机示例进行预测
python predict.py --sample

# 使用指定模型
python predict.py --model GNN-KG --sample

# 使用指定样本索引
python predict.py --idx 100

# 禁用LLM解释
python predict.py --sample --no-llm
```

---

## 评估结果

### 测试集准确率

| 模型 | 准确率 | F1分数 |
|------|--------|--------|
| MLP | 65.1% | 0.643 |
| MLP-KG | 74.0% | 0.739 |
| CNN | 69.1% | 0.682 |
| CNN-KG | 79.2% | 0.785 |
| GNN | 76.6% | 0.765 |
| **GNN-KG** | **83.3%** | **0.831** |

### 最佳模型：GNN-KG

GNN-KG达到最高准确率 **83.3%**，相比基准GNN提升6.7个百分点。知识图谱嵌入为GNN模型提供了关键的故障语义信息，显著提升了分类能力。

---

## 实现的模型

### MLP系列

| 模型 | 嵌入类型 | 描述 |
|------|----------|------|
| MLP | 无 | 普通多层感知机（基准） |
| MLP-KG | KNN样本嵌入(64维) | 基于K近邻的样本级嵌入融合 |

### CNN系列

| 模型 | 融合方案 | 描述 |
|------|----------|------|
| CNN | 无 | 1D卷积神经网络（基准） |
| CNN-KG | 残差连接 | 双塔交互 + 多层残差融合 |

### GNN系列

| 模型 | 描述 |
|------|------|
| GNN | 图神经网络（KNN图结构） |
| GNN-KG | GNN + 故障级别KG嵌入融合 |

**GNN核心思想**：
- 基于样本特征构建KNN邻接矩阵
- 通过图卷积传播相似样本的信息
- KG嵌入参与图卷积过程

---

## 知识图谱嵌入

### 故障级别嵌入（33维）- CNN/GNN使用

| 特征 | 维度 | 说明 |
|------|------|------|
| 故障相似度向量 | 9 | 与其他9种故障的Jaccard相似度 |
| 故障-部件关联 | 4 | 故障→部件的CAUSED_BY关系 |
| 故障-特征权重 | 14 | 故障→特征的HAS_FEATURE关系 |
| 全局KG统计 | 6 | KG节点/边总数 |

### KNN样本嵌入（64维）- MLP使用

| 特征 | 维度 | 说明 |
|------|------|------|
| 归一化原始特征 | 20 | L2归一化后的原始特征 |
| 邻居特征加权和 | 20 | K近邻的距离倒数加权平均 |
| 特征差异 | 20 | 原始特征与邻居加权特征的差异 |
| 填充 | 4 | 补齐到64维 |

**优势**：每个样本有独特嵌入，基于其k近邻样本计算，避免了故障级别嵌入的映射冲突问题。

---

## 数据集

### 故障类型（9类）

| 故障类型 | 说明 |
|----------|------|
| Ball_Fault | 滚动体故障 |
| Broken_Tooth | 断齿故障 |
| Inner_Race_Fault | 内圈故障 |
| Missing_Tooth | 缺齿故障 |
| Mixed_Fault | 混合故障 |
| Normal | 正常 |
| Outer_Race_Fault | 外圈故障 |
| Root_Crack | 齿根裂纹 |
| Tooth_Wear | 齿面磨损 |

### 特征（20维）

| 类别 | 特征 |
|------|------|
| 时域统计 | mean, std, rms, max, min, peak, skewness, kurtosis |
| 指标 | crest_factor, shape_factor, impulse_factor |
| 频域特征 | spectral_energy, spectral_centroid, spectral_entropy, dominant_frequency |
| 频带能量 | band_energy_1~5 |

### 数据规模

| 数据集 | 样本数 |
|--------|--------|
| 训练集 | 52,867 |
| 验证集 | 13,217 |
| 测试集 | 16,522 |
| **总计** | **82,606** |

---

## 设备支持

自动检测并使用可用设备：
- Apple MPS (Metal Performance Shaders) - Mac GPU
- NVIDIA CUDA - GPU
- CPU - 回退方案

---

## 可视化

运行可视化生成评估图表：

```bash
python -m src.visualization.plot_results
```

生成图表：
- `model_comparison.png` - 模型准确率对比
- `metrics_heatmap.png` - 性能热力图
- `kg_enhancement_comparison.png` - KG增强效果对比
- `summary_report.png` - 综合评估报告

---

## 交互式预测

predict.py 提供交互式故障预测，支持LLM（Ollama）解释：

```bash
python predict.py --sample
```

输出示例：
```
============================================================
变速箱故障预测系统
============================================================

使用随机样本 12345 (实际故障: Broken_Tooth)

============================================================
预测结果
============================================================
预测故障: Broken_Tooth
Top-3预测:
  1. Broken_Tooth: 52.8% <-- 实际故障
  2. Missing_Tooth: 39.5%
  3. Tooth_Wear: 5.2%

============================================================
LLM 解释
============================================================
断齿故障特征：振动信号中明显的高频冲击成分。可能原因：过载、疲劳断裂。建议：检查齿轮啮合情况，及时更换受损齿轮。

实际故障: Broken_Tooth
预测结果: 正确
```

---

## 依赖

```
torch>=1.9
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
pyyaml>=5.4
tqdm>=4.62
matplotlib>=3.5
seaborn>=0.11
requests>=2.28
joblib>=1.1
```

---

## 作者

韩旭彬 - 天津大学
