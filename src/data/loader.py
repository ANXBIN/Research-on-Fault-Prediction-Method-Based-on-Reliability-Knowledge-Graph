#!/usr/bin/env python3
"""
共享数据加载与预处理模块
train.py / evaluate.py / predict.py 共用
"""

import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.models.mlp_model import load_kg_embeddings_v4, load_kg_embeddings_mlp


def load_and_split_data(config_path='config.yaml'):
    """加载数据、划分数据集、构建KG嵌入

    Returns:
        dict: 包含以下字段
            - X_train, X_val, X_test: 标准化后的特征
            - y_train, y_val, y_test: 标签
            - fault_types: 故障类型数组
            - fault_to_idx: 故障名->索引映射
            - label_encoder: LabelEncoder实例
            - scaler: StandardScaler实例（仅在训练集上fit）
            - kg_train_emb, kg_val_emb, kg_test_emb: V2故障级KG嵌入 (33维)
            - kg_train_emb_mlp, kg_val_emb_emb_mlp, kg_test_emb_mlp: MLP专用KNN嵌入 (64维)
            - feature_cols: 特征列名列表
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    df = pd.read_csv('data/processed/processed_features.csv')
    fault_types = df['fault_type'].unique()

    # 标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(fault_types)
    y = label_encoder.transform(df['fault_type'])
    fault_to_idx = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    # 特征列
    feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]
    X = df[feature_cols].values

    # 划分数据集
    test_size = config.get('training', {}).get('test_size', 0.2)
    val_size = config.get('training', {}).get('val_size', 0.1)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=42, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=42, stratify=y[train_idx]
    )

    # 标准化：仅在训练集上fit
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # 保存scaler供预测时使用
    Path('models').mkdir(exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')

    # 获取划分后的故障类型标签（字符串）
    fault_labels_all = df['fault_type'].values
    fault_labels_train = fault_labels_all[train_idx]
    fault_labels_val = fault_labels_all[val_idx]
    fault_labels_test = fault_labels_all[test_idx]

    # V2嵌入：故障级别KG嵌入（33维）
    kg_train_emb = load_kg_embeddings_v4(
        'data/processed/fault_embeddings.json',
        fault_labels_train,
        'data/processed/kg_embeddings.json'
    )
    kg_val_emb = load_kg_embeddings_v4(
        'data/processed/fault_embeddings.json',
        fault_labels_val,
        'data/processed/kg_embeddings.json'
    )
    kg_test_emb = load_kg_embeddings_v4(
        'data/processed/fault_embeddings.json',
        fault_labels_test,
        'data/processed/kg_embeddings.json'
    )

    # MLP专用嵌入：基于KNN的样本级嵌入（64维）
    kg_train_emb_mlp, kg_val_emb_mlp, kg_test_emb_mlp = load_kg_embeddings_mlp(
        X_train, X_val, X_test, k=20
    )

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'fault_types': fault_types,
        'fault_to_idx': fault_to_idx,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'kg_train_emb': kg_train_emb, 'kg_val_emb': kg_val_emb, 'kg_test_emb': kg_test_emb,
        'kg_train_emb_mlp': kg_train_emb_mlp, 'kg_val_emb_mlp': kg_val_emb_mlp, 'kg_test_emb_mlp': kg_test_emb_mlp,
        'feature_cols': feature_cols,
    }
