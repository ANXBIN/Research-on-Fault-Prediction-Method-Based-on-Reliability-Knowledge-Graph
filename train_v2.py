#!/usr/bin/env python3
"""仅训练KG-Enhanced MLP V2模型"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.models.mlp_model import KGEnhancedMLPV2Model, load_kg_embeddings_v3

def main():
    print("=" * 60)
    print("训练 KG-Enhanced MLP V2 模型 (深度融合)")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv('data/processed/processed_features.csv')
    fault_types = df['fault_type'].unique()
    print(f"数据集大小: {len(df)} 样本")
    print(f"故障类型数: {len(fault_types)}")

    # 标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(fault_types)
    y = label_encoder.transform(df['fault_type'])
    fault_to_idx = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    # 特征
    feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]
    X = df[feature_cols].values

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"特征维度: {X.shape[1]}")

    # 划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")

    # 加载KG嵌入
    print("\n加载KG嵌入...")
    kg_train_emb = load_kg_embeddings_v3(
        'data/processed/kg_embeddings.json', len(X_train), None, X_train
    )
    kg_val_emb = load_kg_embeddings_v3(
        'data/processed/kg_embeddings.json', len(X_val), None, X_train, X_val
    )
    print(f"KG嵌入维度: {kg_train_emb.shape}")

    # 设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    print("\n创建模型...")
    kg_mlp_v2 = KGEnhancedMLPV2Model(config_path='config.yaml')
    kg_mlp_v2.fault_to_idx = fault_to_idx
    kg_mlp_v2.build_model(X_train.shape[1], len(fault_types))
    print(f"模型参数: {sum(p.numel() for p in kg_mlp_v2.model.parameters())}")

    # 训练
    print("\n开始训练...")
    for epoch in range(kg_mlp_v2.epochs):
        train_loss, train_acc = kg_mlp_v2.train_epoch(X_train, y_train, kg_train_emb)
        if epoch % 20 == 0 or epoch == kg_mlp_v2.epochs - 1:
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    # 评估
    train_metrics, _ = kg_mlp_v2.evaluate(X_train, y_train, kg_train_emb)
    val_metrics, _ = kg_mlp_v2.evaluate(X_val, y_val, kg_val_emb)

    print(f"\n训练集 - 准确率: {train_metrics['accuracy']:.4f}")
    print(f"验证集 - 准确率: {val_metrics['accuracy']:.4f}")

    # 保存结果
    results = {
        'train_accuracy': float(train_metrics['accuracy']),
        'val_accuracy': float(val_metrics['accuracy'])
    }
    with open('results/kg_mlp_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至: results/kg_mlp_v2_results.json")

if __name__ == '__main__':
    main()