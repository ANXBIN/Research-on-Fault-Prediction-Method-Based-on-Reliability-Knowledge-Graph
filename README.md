# 基于可靠性知识图谱的故障预测系统

## 项目概述

利用知识图谱增强深度学习模型的故障诊断能力，实现变速箱故障类型的精准分类。系统集成了MLP、CNN、GNN三大类深度学习模型，并支持知识图谱融合方案。同时提供基于规则的推理方法和交互式Web可视化界面。

**数据集**：西安交通大学变速箱故障数据集（82,606样本，9类故障）

---

## 核心特性

- **深度学习模型**：MLP、CNN、GNN及其知识图谱增强版本
- **规则推理引擎**：基于知识图谱的故障规则推理系统
- **注意力机制**：GNN-KG模型采用多头自注意力和图注意力
- **Focal Loss优化**：解决类别混淆问题，提升Mixed_Fault识别能力
- **Web可视化界面**：基于Streamlit的交互式故障预测平台
- **AI智能解释**：集成LLM（Ollama）生成故障诊断说明
- **设备支持**：Apple MPS (Metal)、NVIDIA CUDA、CPU

---

## 项目结构

```
fault-prediction/
├── config.yaml              # 配置文件
├── train.py                 # 模型训练脚本
├── evaluate.py              # 模型评估脚本
├── predict.py               # 交互式预测脚本
├── app.py                   # Streamlit Web可视化界面
├── requirements.txt         # Python依赖
├── README.md               # 本文档

├── data/
│   └── processed/
│       ├── processed_features.csv    # 特征数据（20维）
│       ├── knowledge_graph.json     # 知识图谱数据
│       ├── kg_embeddings.json       # KG嵌入
│       └── fault_embeddings.json     # 故障级别嵌入

├── models/                  # 训练好的模型文件
│   ├── mlp_model.pt
│   ├── mlp_kg_model.pt
│   ├── cnn_model.pt
│   ├── cnn_kg_model.pt
│   ├── gnn_model.pt
│   └── gnn_kg_model.pt

├── src/
│   ├── models/             # 模型定义
│   │   ├── mlp_model.py
│   │   ├── cnn_model.py
│   │   └── gnn_model.py
│   ├── reasoning/          # 规则推理模块
│   │   └── rule_reasoner.py
│   ├── data/               # 数据处理
│   │   └── loader.py
│   └── visualization/       # 可视化
│       └── plot_results.py

└── results/                # 评估结果
    ├── training_results.json
    ├── evaluation_results.json
    └── figures/
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
python train.py --all --epochs 150

# 仅训练特定模型
python train.py --mlp              # 普通MLP
python train.py --mlp-kg          # MLP-KG融合
python train.py --cnn             # 普通CNN
python train.py --cnn-kg          # CNN-KG融合
python train.py --gnn             # 图神经网络
python train.py --gnn-kg          # GNN-KG融合
```

### 3. 评估模型

```bash
python evaluate.py
```

### 4. Web可视化界面

```bash
# 启动Streamlit界面
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`

### 5. 命令行预测

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
| GNN-KG | GNN + 故障级别KG嵌入 + 注意力机制 |

**GNN核心思想**：
- 基于样本特征构建KNN邻接矩阵
- 通过图卷积传播相似样本的信息
- KG嵌入参与图卷积过程
- **多头自注意力**捕获全局样本关系
- **图注意力层**学习邻居节点的重要性权重

---

## 知识图谱嵌入

### 故障级别嵌入（33维）

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

### 知识图谱结构

| 节点类型 | 说明 |
|----------|------|
| Fault | 82,606个故障样本节点 |
| Feature | 1,652,120个特征节点 |
| Component | 5个部件节点（Rolling_Element, Inner_Race, Outer_Race, Gear_Tooth, Unknown） |

| 边类型 | 说明 |
|--------|------|
| HAS_FEATURE | 故障→特征关系 |
| LOCATED_AT | 故障→部件关系 |
| CAUSED_BY | 故障原因部件关系 |

---

## 故障类型（9类）

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

---

## 特征（20维）

| 类别 | 特征 |
|------|------|
| 时域统计 | mean, std, rms, max, min, peak, skewness, kurtosis |
| 指标 | crest_factor, shape_factor, impulse_factor |
| 频域特征 | spectral_energy, spectral_centroid, spectral_entropy, dominant_frequency |
| 频带能量 | band_energy_1~5 |

---

## Web可视化界面功能

### 系统首页
- 项目概述和架构图
- 故障类型说明

### 模型对比
- 测试集性能对比柱状图
- 最佳模型雷达图
- 详细评估结果表格

### 知识图谱
- 节点/边数量统计
- 节点类型分布饼图
- 边类型分布柱状图
- 故障相似度矩阵热力图
- 知识图谱样本展示

### 故障预测
- **深度学习模型预测**：支持多模型同时预测
- **规则推理预测**：基于知识图谱规则的推理
- **AI智能解释**：集成Ollama LLM生成诊断说明
- 预测结果可视化对比

### 训练历史
- 各模型验证集准确率
- 训练配置展示

---

## 规则推理系统

### 推理方法

系统实现了基于规则的故障推理，结合知识图谱中的故障-部件-特征关系：

1. **特征阈值匹配**：为每种故障类型定义特征统计范围
2. **部件关联推理**：基于故障→部件的因果关系
3. **KG相似度增强**：利用故障类型间的Jaccard相似度

### 置信度计算

```
总置信度 = 特征匹配分数 × 0.7
         + 部件关联分数 × 0.2
         + KG相似度增强 × 0.1
```

### 使用示例

```python
from src.reasoning.rule_reasoner import RuleBasedReasoner

reasoner = RuleBasedReasoner()
features = [100000, 50000, 80000, 200000, -200000, 300000, 0.5, 3.0,
            2.5, 1.1, 3.0, 1e8, 500, 3.0, 500, 1e7, 1e7, 1e7, 1e7, 1e7]

fault, probs, details = reasoner.diagnose(features)
explanation = reasoner.explain(fault, features)

print(f"推理结果: {fault}")
print(f"相关部件: {explanation['related_components']}")
```

---

## 训练优化技术

### Focal Loss

采用Focal Loss替代标准交叉熵损失，解决类别混淆问题：

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        # alpha: 类别权重
        # gamma: 聚焦参数，越大越关注难分类样本
```

### 类别权重调整

为Mixed_Fault类别增加50%额外权重：

```python
weights[i] *= 1.5  # Mixed_Fault类别权重增加
```

### 学习率调度

使用余弦退火调度器：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

---

## 依赖

```
torch>=2.0.0
torch_geometric>=2.3.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
streamlit>=1.30.0
pyyaml>=6.0
tqdm>=4.65.0
joblib>=1.3.0
```

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    基于知识图谱的变速箱故障预测系统                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   数据输入    │────▶│  特征提取    │────▶│  知识图谱    │     │
│  │  (振动信号)   │     │  (20维)      │     │   构建       │     │
│  └──────────────┘     └──────────────┘     └──────┬───────┘     │
│                                                   │              │
│  ┌──────────────────────────────────────────────┐│              │
│  │              知识图谱嵌入 (33维)              │┘              │
│  └──────────────────────────────────────────────┘               │
│                           │                                      │
│  ┌──────────────┬─────────┴─────────┬──────────────┐              │
│  │     MLP      │       CNN        │     GNN      │              │
│  │   (KNN嵌入)   │    (残差融合)    │   (注意力)   │              │
│  └──────┬───────┘─────────┬─────────└──────┬───────┘              │
│         │                 │                │                    │
│         └────────────┬────┴───────────────┘                     │
│                      ▼                                          │
│            ┌─────────────────┐                                  │
│            │    故障分类      │                                  │
│            │   (9类输出)      │                                  │
│            └────────┬────────┘                                  │
│                     │                                            │
│         ┌───────────┴───────────┐                               │
│         ▼                       ▼                               │
│  ┌─────────────┐        ┌─────────────┐                         │
│  │  规则推理   │        │  LLM解释    │                         │
│  │  (阈值匹配) │        │ (Ollama)   │                         │
│  └─────────────┘        └─────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 评估结果

模型性能对比（测试集）：

| 模型 | 准确率 | F1分数 |
|------|--------|--------|
| MLP | 65.1% | 0.643 |
| MLP-KG | 74.0% | 0.739 |
| CNN | 69.1% | 0.682 |
| CNN-KG | 79.2% | 0.785 |
| GNN | 76.6% | 0.765 |
| **GNN-KG** | **83.3%+** | **0.831+** |

---

## 作者

韩旭彬 - 天津大学
