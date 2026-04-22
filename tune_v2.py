#!/usr/bin/env python3
"""V2模型超参数调优"""

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

def tune_v2():
    print("=" * 60)
    print("V2 模型超参数调优")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv('data/processed/processed_features.csv')
    fault_types = df['fault_type'].unique()

    label_encoder = LabelEncoder()
    label_encoder.fit(fault_types)
    y = label_encoder.transform(df['fault_type'])
    fault_to_idx = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"数据集: {len(df)} 样本, 特征: {X.shape[1]}")

    # 划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 加载KG嵌入
    kg_train_emb = load_kg_embeddings_v3(
        'data/processed/kg_embeddings.json', len(X_train), None, X_train
    )
    kg_val_emb = load_kg_embeddings_v3(
        'data/processed/kg_embeddings.json', len(X_val), None, X_train, X_val
    )

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 超参数搜索
    configs = [
        # 原始配置
        {'hidden_dim': 128, 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 1e-4},
        # 增加dropout
        {'hidden_dim': 128, 'dropout': 0.4, 'lr': 0.001, 'weight_decay': 1e-4},
        {'hidden_dim': 128, 'dropout': 0.5, 'lr': 0.001, 'weight_decay': 1e-4},
        # 增加weight_decay
        {'hidden_dim': 128, 'dropout': 0.4, 'lr': 0.001, 'weight_decay': 5e-4},
        # 减小hidden_dim
        {'hidden_dim': 64, 'dropout': 0.4, 'lr': 0.001, 'weight_decay': 1e-4},
        {'hidden_dim': 64, 'dropout': 0.5, 'lr': 0.001, 'weight_decay': 1e-4},
        # 降低学习率
        {'hidden_dim': 128, 'dropout': 0.4, 'lr': 0.0005, 'weight_decay': 1e-4},
        # 更强正则化
        {'hidden_dim': 96, 'dropout': 0.5, 'lr': 0.001, 'weight_decay': 1e-3},
    ]

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n--- 配置 {i+1}/{len(configs)}: {cfg} ---")

        # 创建模型
        model = KGEnhancedMLPV2Model(config_path='config.yaml')
        model.hidden_dim = cfg['hidden_dim']
        model.dropout = cfg['dropout']
        model.learning_rate = cfg['lr']
        model.fault_to_idx = fault_to_idx
        model.build_model(X_train.shape[1], len(fault_types))

        # 使用weight decay
        model.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay']
        )

        # 训练
        best_val_acc = 0
        patience = 15
        patience_counter = 0

        for epoch in range(80):
            train_loss, train_acc = model.train_epoch(X_train, y_train, kg_train_emb)
            val_metrics, _ = model.evaluate(X_val, y_val, kg_val_emb)
            val_acc = val_metrics['accuracy']

            if epoch % 20 == 0 or epoch == 79:
                print(f"Epoch {epoch}: Train={train_acc:.4f}, Val={val_acc:.4f}")

            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        results.append({
            'config': cfg,
            'best_val_acc': best_val_acc,
            'epochs_trained': epoch + 1
        })
        print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 打印结果
    print("\n" + "=" * 60)
    print("调优结果汇总")
    print("=" * 60)
    print(f"{'配置':<50} | {'验证准确率':<10}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x['best_val_acc'], reverse=True):
        cfg_str = f"hidden={r['config']['hidden_dim']}, drop={r['config']['dropout']}, wd={r['config']['weight_decay']}"
        print(f"{cfg_str:<50} | {r['best_val_acc']:.4f}")

    # 保存结果
    best = max(results, key=lambda x: x['best_val_acc'])
    print(f"\n最佳配置: {best['config']}")
    print(f"最佳验证准确率: {best['best_val_acc']:.4f}")

    with open('results/v2_tuning_results.json', 'w') as f:
        json.dump({
            'results': results,
            'best': best
        }, f, indent=2)
    print(f"结果已保存至: results/v2_tuning_results.json")

if __name__ == '__main__':
    tune_v2()