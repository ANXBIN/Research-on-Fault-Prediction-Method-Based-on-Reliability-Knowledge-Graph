#!/usr/bin/env python3
"""
快速测试脚本 - 验证项目各模块是否正常工作
"""

import sys
import numpy as np
from pathlib import Path

print("=" * 60)
print("项目快速测试")
print("=" * 60)

# 测试1: 配置文件
print("\n[测试1] 检查配置文件...")
try:
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"  ✓ 配置文件加载成功")
    print(f"    - 项目名称: {config['project']['name']}")
    print(f"    - Neo4j URI: {config['neo4j']['uri']}")
    print(f"    - 模型数量: {len(config['models'])}")
except Exception as e:
    print(f"  ✗ 配置文件加载失败: {e}")
    sys.exit(1)

# 测试2: 数据加载
print("\n[测试2] 检查数据文件...")
try:
    data_path = Path('data/raw/XJTU_Gearbox')
    fault_types = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"  ✓ 找到 {len(fault_types)} 种故障类型:")
    for ft in fault_types:
        print(f"    - {ft}")
except Exception as e:
    print(f"  ✗ 数据检查失败: {e}")
    sys.exit(1)

# 测试3: 特征提取
print("\n[测试3] 测试特征提取...")
try:
    from scipy import stats
    from scipy.fftpack import fft

    # 创建测试信号
    test_signal = np.random.randn(1024)

    # 时域特征
    mean_val = np.mean(test_signal)
    std_val = np.std(test_signal)
    rms_val = np.sqrt(np.mean(test_signal**2))
    skewness = stats.skew(test_signal)
    kurtosis = stats.kurtosis(test_signal)

    # 频域特征
    fft_vals = fft(test_signal)
    fft_magnitude = np.abs(fft_vals[:512])
    spectral_energy = np.sum(fft_magnitude**2)

    print(f"  ✓ 特征提取计算正常")
    print(f"    - RMS: {rms_val:.4f}")
    print(f"    - 频谱能量: {spectral_energy:.4f}")
except Exception as e:
    print(f"  ✗ 特征提取失败: {e}")
    sys.exit(1)

# 测试4: 知识图谱节点创建
print("\n[测试4] 测试知识图谱构建...")
try:
    import json
    from collections import defaultdict

    nodes = []
    edges = []

    # 创建故障节点
    fault_node = {
        'id': 'Fault_Ball_Fault_1',
        'type': 'Fault',
        'label': 'Ball_Fault',
        'properties': {'rms': 0.5, 'kurtosis': 3.0}
    }
    nodes.append(fault_node)

    # 创建部件节点
    component_node = {
        'id': 'Component_Bearing',
        'type': 'Component',
        'label': 'Bearing',
        'properties': {}
    }
    nodes.append(component_node)

    # 创建边
    edge = {
        'source': 'Fault_Ball_Fault_1',
        'target': 'Component_Bearing',
        'type': 'LOCATED_AT',
        'properties': {}
    }
    edges.append(edge)

    kg_data = {'nodes': nodes, 'edges': edges}
    print(f"  ✓ 知识图谱结构正常")
    print(f"    - 节点数: {len(nodes)}")
    print(f"    - 边数: {len(edges)}")
except Exception as e:
    print(f"  ✗ 知识图谱构建失败: {e}")
    sys.exit(1)

# 测试5: 机器学习模型
print("\n[测试5] 测试随机森林模型...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # 生成随机数据
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42)

    # 训练小型RF
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X, y)
    score = rf.score(X, y)

    print(f"  ✓ 随机森林模型正常")
    print(f"    - 训练准确率: {score:.4f}")
except Exception as e:
    print(f"  ✗ RF模型测试失败: {e}")
    sys.exit(1)

# 测试6: PyTorch
print("\n[测试6] 测试PyTorch...")
try:
    import torch
    import torch.nn as nn

    x = torch.randn(10, 5)
    linear = nn.Linear(5, 3)
    y = linear(x)

    print(f"  ✓ PyTorch计算正常")
    print(f"    - 输入形状: {x.shape}")
    print(f"    - 输出形状: {y.shape}")
    print(f"    - 设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except Exception as e:
    print(f"  ✗ PyTorch测试失败: {e}")
    sys.exit(1)

# 测试7: PyG
print("\n[测试7] 测试PyTorch Geometric...")
try:
    import torch_geometric
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(2, 8)

    data = Data(x=x, edge_index=edge_index)

    print(f"  ✓ PyTorch Geometric正常")
    print(f"    - 数据节点: {data.num_nodes}")
    print(f"    - 数据边数: {data.num_edges}")
except Exception as e:
    print(f"  ✗ PyG测试失败: {e}")
    sys.exit(1)

# 测试8: 可视化
print("\n[测试8] 测试可视化库...")
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    plt.savefig('/tmp/test_plot.png', dpi=50)
    plt.close()

    print(f"  ✓ 可视化库正常")
except Exception as e:
    print(f"  ✗ 可视化测试失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("所有测试通过! 项目环境配置正确。")
print("=" * 60)
print("""
下一步:
1. 运行数据预处理: python src/data/preprocess.py
2. 启动Neo4j: python src/kg/neo4j_manager.py --action start
3. 训练模型: python src/train/train.py
""")
