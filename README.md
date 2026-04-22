# 基于知识图谱的变速箱故障预测系统

本项目是西安交通大学变速箱故障数据集的故障预测研究，实现了知识图谱与深度学习/机器学习方法结合的故障预测，并与传统方法进行对比。

## 项目结构

```
fault-prediction/
├── config.yaml              # 配置文件
├── requirements.txt         # Python依赖
├── README.md               # 本文档

├── data/
│   ├── raw/                # 原始数据
│   │   └── XJTU_Gearbox/   # 西安交通大学变速箱数据集
│   └── processed/          # 处理后的数据
│       ├── processed_features.csv    # 提取的特征
│       ├── knowledge_graph.json      # 知识图谱数据
│       └── kg_embeddings.json         # KG嵌入向量

├── src/
│   ├── data/
│   │   └── preprocess.py   # 数据预处理脚本
│   ├── kg/
│   │   ├── neo4j_manager.py    # Neo4j容器管理
│   │   └── kg_embedder.py      # 知识图谱嵌入生成
│   ├── models/
│   │   ├── gnn_model.py           # GNN模型
│   │   ├── rf_model.py            # 随机森林模型
│   │   └── kg_enhanced_models.py   # 知识图谱增强模型
│   ├── train/
│   │   └── train.py       # 训练主程序
│   └── visualization/
│       └── visualize.py   # 可视化脚本

├── results/                # 训练结果
│   ├── figures/           # 可视化图表
│   ├── models/            # 保存的模型
│   └── training_results.json  # 训练结果JSON
└── logs/                  # 日志文件
```

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n fault-prediction python=3.10
conda activate fault-prediction
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. Docker配置

确保Docker已安装并运行，本项目使用Neo4j Docker容器存储知识图谱。

## 数据集说明

使用的是西安交通大学变速箱故障数据集，包含以下故障类型：

### 轴承故障
- `1ndBearing_ball` - 球故障
- `1ndBearing_inner` - 内圈故障
- `1ndBearing_outer` - 外圈故障
- `1ndBearing_mix` - 混合故障

### 行星齿轮故障
- `2ndPlanetary_brokentooth` - 断齿
- `2ndPlanetary_missingtooth` - 缺齿
- `2ndPlanetary_rootcracks` - 根裂纹
- `2ndPlanetary_toothwear` - 齿磨损
- `2ndPlanetary_normalstate` - 正常状态

## 使用方法

### 1. 数据预处理

```bash
python src/data/preprocess.py \
    --data_path data/raw/XJTU_Gearbox \
    --output_path data/processed \
    --sampling_rate 25600
```

这将：
- 加载原始传感器数据
- 使用滑动窗口提取时域和频域特征
- 构建知识图谱结构
- 保存处理后的数据

### 2. 启动Neo4j并导入知识图谱

```bash
# 完整流程：启动容器 -> 清除旧数据 -> 导入新数据 -> 创建索引
python src/kg/neo4j_manager.py --action full \
    --kg_file data/processed/knowledge_graph.json \
    --password your_password
```

其他选项：
```bash
# 仅启动容器
python src/kg/neo4j_manager.py --action start --password your_password

# 仅清除数据
python src/kg/neo4j_manager.py --action clear

# 仅导入数据
python src/kg/neo4j_manager.py --action import
```

### 3. 生成知识图谱嵌入

```bash
python src/kg/kg_embedder.py \
    --embedding_path data/processed/kg_embeddings.json
```

### 4. 模型训练

```bash
# 训练所有模型
python src/train/train.py

# 跳过特定模型
python src/train/train.py --skip_gnn  # 跳过GNN
python src/train/train.py --skip_rf   # 跳过RF
python src/train/train.py --skip_kg  # 跳过KG增强
```

### 5. 生成可视化结果

```bash
# 生成所有可视化图表
python src/visualization/visualize.py --all

# 生成指定图表
python src/visualization/visualize.py \
    --results results/training_results.json
```

## 实现的方法

### 传统方法
1. **随机森林 (RF)** - 使用sklearn实现的随机森林分类器
2. **梯度提升 (GB)** - 作为对比的梯度提升方法

### 知识图谱增强方法
1. **KG-Enhanced GNN** - 结合知识图谱结构的图神经网络
2. **KG-Enhanced RF** - 将知识图谱嵌入与传统随机森林结合

### 知识图谱增强技术
- 从Neo4j提取故障关联关系
- 生成节点嵌入向量（谱嵌入方法）
- 构建故障相似度矩阵
- 融合结构特征到原始特征空间

## 提取的特征

### 时域特征
- 均值、标准差、RMS
- 最大值、最小值、峰值
- 偏度、峰度
- 峰值因子、形状因子、脉冲因子

### 频域特征
- 频谱能量
- 频谱质心
- 频谱熵
- 主频率
- 频带能量分布

## 知识图谱结构

### 节点类型
- `Fault` - 故障实例节点
- `Component` - 部件节点（轴承、齿轮等）
- `Feature` - 特征节点

### 关系类型
- `CAUSED_BY` - 故障由某部件引起
- `AFFECTS` - 故障影响某部件
- `HAS_FEATURE` - 故障具有某特征
- `LOCATED_AT` - 故障位于某部件
- `SIMILAR_TO` - 故障与其他故障相似

## 配置文件说明

`config.yaml`包含所有配置项：

```yaml
# Neo4j配置
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"
  container_name: "neo4j_knowledge_graph"

# 数据预处理配置
preprocessing:
  window_size: 1024
  overlap: 512
  sampling_rate: 25600

# 模型配置
models:
  gnn:
    hidden_dim: 128
    num_layers: 3
    dropout: 0.3
    learning_rate: 0.001
    epochs: 100

  rf:
    n_estimators: 200
    max_depth: 15
    learning_rate: 0.001
    epochs: 100
```

## 实验结果

训练完成后，结果保存在`results/training_results.json`中，可视化图表在`results/figures/`目录下。

### 输出文件
- `results/models/*.joblib` - 保存的模型文件
- `results/training_results.json` - 训练指标
- `results/figures/model_comparison.png` - 模型对比图
- `results/figures/kg_structure.png` - 知识图谱结构图
- `results/figures/feature_importance.png` - 特征重要性图
- `results/visualization_report.html` - HTML总结报告

## 故障类型分布

| 故障类型 | 描述 |
|---------|------|
| Ball_Fault | 滚动体球故障 |
| Inner_Race_Fault | 内圈故障 |
| Outer_Race_Fault | 外圈故障 |
| Mixed_Fault | 混合故障 |
| Broken_Tooth | 齿轮断齿 |
| Missing_Tooth | 齿轮缺齿 |
| Root_Crack | 齿根裂纹 |
| Tooth_Wear | 齿面磨损 |
| Normal | 正常状态 |

## 注意事项

1. **Neo4j密码**：首次运行需要设置Neo4j密码，建议在`config.yaml`中修改默认密码
2. **Docker权限**：确保当前用户有Docker运行权限
3. **内存**：处理大规模数据时确保有足够内存（建议16GB+）
4. **GPU**：如使用GPU加速，确保PyTorch支持CUDA

## 故障排除

### Neo4j容器启动失败
```bash
# 检查Docker状态
docker ps -a

# 查看容器日志
docker logs neo4j_knowledge_graph
```

### 内存不足
```bash
# 增加Docker内存限制，或减少batch_size
```

### 特征提取过慢
```bash
# 减少滑动窗口步长或采样率
python src/data/preprocess.py --help
```

## 作者

西安交通大学 - 毕业设计

## 参考

- 西安交通大学变速箱数据集
- PyTorch Geometric文档
- Neo4j图数据库文档
- Scikit-learn机器学习库
