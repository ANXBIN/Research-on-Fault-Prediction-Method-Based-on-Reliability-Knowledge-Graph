# 基于可靠性知识图谱的故障预测系统

## 项目结构

```
fault-prediction/
├── config.yaml              # 配置文件
├── train.py                 # 模型训练脚本
├── evaluate.py              # 模型评估脚本
├── predict.py               # 交互式预测脚本（含LLM解释）
├── requirements.txt         # Python依赖
├── README.md               # 本文档

├── data/                    # 数据目录
│   └── processed/          # 处理后的数据
│       ├── processed_features.csv    # 特征数据
│       ├── knowledge_graph.json      # 知识图谱数据
│       ├── kg_embeddings.json       # 全局KG嵌入
│       └── fault_embeddings.json    # 故障级别嵌入

├── models/                  # 保存的模型文件
│   ├── mlp_model.pt
│   ├── kg_enhanced_mlp_v1_model.pt   # V1: 全局嵌入
│   ├── kg_enhanced_mlp_v2_model.pt   # V2: 故障级别嵌入
│   ├── cnn_model.pt
│   ├── cnn_kg_model.pt             # CNN + KG拼接融合
│   ├── cnn_kg_v2_model.pt         # CNN + KG门控融合
│   └── cnn_kg_v3_model.pt         # CNN + KG残差连接

├── results/                 # 结果输出目录
│   ├── training_results.json
│   ├── evaluation_results.json
│   ├── batch_tuning_results.json   # 批量调优结果
│   └── figures/               # 可视化图表

└── src/
    ├── models/
    │   ├── mlp_model.py        # MLP模型
    │   └── cnn_model.py        # CNN模型
    └── visualization/
        └── plot_results.py     # 可视化
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

### 1. 训练模型

```bash
# 训练所有模型（默认100轮）
python train.py --all

# 指定训练轮数
python train.py --all --epochs 200

# 仅训练特定模型
python train.py --mlp              # 普通MLP
python train.py --kg-v1            # KG-MLP V1 (全局嵌入)
python train.py --kg-v2            # KG-MLP V2 (故障级别嵌入)
python train.py --cnn             # 普通CNN
python train.py --cnn-kg          # CNN + KG (拼接融合)
python train.py --cnn-kg-v2       # CNN + KG (门控融合)
python train.py --cnn-kg-v3       # CNN + KG (残差连接)
```

### 2. 批量调优

使用贝叶斯优化对所有模型进行超参数搜索：

```bash
# 批量调优所有模型（每模型20次试验）
python train.py --tune-all

# 指定试验次数
python train.py --tune-all --n-trials 30
```

### 3. 评估模型

```bash
# 自动检测并评估models文件夹中已存在的模型
python evaluate.py
```

### 4. 交互式预测

```bash
# 使用随机示例进行预测
python predict.py --sample

# 使用指定样本索引
python predict.py --idx 100

# 使用其他模型
python predict.py --model CNN-KG V2 --sample

# 禁用LLM解释
python predict.py --sample --no-llm
```

## 实现的模型

### MLP系列

| 模型 | 描述 | 准确率 |
|------|------|--------|
| MLP | 普通多层感知机（基准） | ~68% |
| KG-MLP V1 | KG全局嵌入融合 | ~76% |
| KG-MLP V2 | KG故障级别嵌入融合 | **~83%** |

### CNN系列

| 模型 | 描述 | 准确率 |
|------|------|--------|
| CNN | 1D卷积神经网络 | ~70% |
| CNN-KG | CNN + KG拼接融合 | ~79% |
| CNN-KG V2 | CNN + KG门控融合 | ~81% |
| CNN-KG V3 | CNN + KG残差连接 | ~81% |

## 知识图谱嵌入

### V1嵌入（全局统计，64维→6维）
从Neo4j知识图谱中提取6个粗粒度统计量，所有样本共享。

### V2嵌入（故障级别，33维）

| 特征 | 维度 | 说明 |
|------|------|------|
| 故障相似度向量 | 9 | 与其他9种故障的Jaccard相似度 |
| 故障-部件关联 | 4 | 故障→部件的CAUSED_BY关系 |
| 故障-特征权重 | 14 | 故障→特征的HAS_FEATURE关系 |
| 全局KG统计 | 6 | KG节点/边总数 |

### 故障相似度矩阵

基于部件和特征共享计算，齿轮类故障（Broken, Missing, Root, Wear）彼此相似度为1.0。

## 数据集

西安交通大学变速箱故障数据集，包含9种故障类型：
- Ball_Fault, Broken_Tooth, Inner_Race_Fault, Missing_Tooth
- Mixed_Fault, Normal, Outer_Race_Fault, Root_Crack, Tooth_Wear

## 设备支持

自动检测并使用可用设备：
- Apple MPS (Metal Performance Shaders) - Mac GPU
- NVIDIA CUDA - GPU
- CPU - 回退方案

## 可视化

生成4种评估图表：
- `model_comparison.png` - 模型准确率对比
- `metrics_heatmap.png` - 性能热力图
- `improvement_comparison.png` - KG增强效果对比
- `kg_enhancement_comparison.png` - MLP/CNN分别与KG增强对比
- `summary_report.png` - 综合评估报告

运行可视化：
```bash
python -m src.visualization.plot_results
```

## 交互式预测功能

predict.py 提供交互式故障预测：

```bash
python predict.py --sample
```

输出示例：
```
============================================================
变速箱故障预测系统
============================================================

使用随机样本 12345 (实际故障: Inner_Race_Fault)

============================================================
预测结果
============================================================
预测故障: Inner_Race_Fault
Top-3预测:
  1. Inner_Race_Fault: 97.9% <-- 实际故障
  2. Root_Crack: 0.5%
  3. Normal: 0.4%

============================================================
LLM 解释
============================================================
内圈故障特征：表面裂纹剥落，振动噪音异常。可能原因：过载、润滑不足、安装不当。建议：立即检查内圈，补充润滑，必要时更换轴承。（78字）

实际故障: Inner_Race_Fault
预测结果: 正确
```

## 依赖

```
torch
numpy
pandas
scikit-learn
pyyaml
optuna
tqdm
matplotlib
seaborn
requests
```

## 作者

韩旭彬 - 天津大学