#!/usr/bin/env python3
"""
知识图谱增强的故障预测模型
结合知识图谱嵌入和传统ML/DL方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class KGEnhancedGNN(nn.Module):
    """知识图谱增强的GNN模型"""

    def __init__(self, in_channels, kg_embedding_dim, hidden_channels,
                 out_channels, num_layers=3, dropout=0.3):
        super(KGEnhancedGNN, self).__init__()

        self.kg_embedding_dim = kg_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 知识图谱嵌入投影层
        self.kg_projection = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 原始特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_channels * 2, hidden_channels, heads=4, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * 4))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch, kg_embedding):
        """
        前向传播
        x: 原始特征
        edge_index: 图边索引
        batch: 批次索引
        kg_embedding: 知识图谱嵌入
        """
        # 投影
        x_feat = self.feature_projection(x)
        x_kg = self.kg_projection(kg_embedding)

        # 融合原始特征和KG嵌入
        if x_feat.size(0) != x_kg.size(0):
            # 广播kg_embedding到匹配x_feat的维度
            x_kg = x_kg.expand(x_feat.size(0), -1)

        x_fused = torch.cat([x_feat, x_kg], dim=1)
        x_fused = self.fusion(x_fused)

        # GNN层
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_fused = conv(x_fused, edge_index)
            x_fused = bn(x_fused)
            x_fused = F.relu(x_fused)
            x_fused = F.dropout(x_fused, p=self.dropout, training=self.training)

        # 图池化
        x_mean = global_mean_pool(x_fused, batch)
        x_max = global_max_pool(x_fused, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # 分类
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class KGEnhancedRF:
    """知识图谱增强的随机森林模型"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['models']['kg_enhanced_rf']
        self.n_estimators = self.model_config['n_estimators']
        self.max_depth = self.model_config['max_depth']
        self.kg_embedding_dim = self.model_config['kg_embedding_dim']

        self.model = None
        self.fault_to_idx = {}

    def build_model(self, n_features, n_classes):
        """构建增强随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        return self.model

    def enhance_features(self, X, kg_embeddings):
        """
        将知识图谱嵌入与原始特征融合
        X: 原始特征 [n_samples, n_features]
        kg_embeddings: 知识图谱嵌入 [n_samples, kg_embedding_dim]
        """
        # 确保维度匹配
        if len(X) != len(kg_embeddings):
            kg_embeddings = kg_embeddings.repeat(len(X), 1)

        # 归一化KG嵌入
        kg_embeddings_norm = kg_embeddings / (np.linalg.norm(kg_embeddings, axis=1, keepdims=True) + 1e-8)

        # 拼接
        X_enhanced = np.concatenate([X, kg_embeddings_norm], axis=1)
        return X_enhanced

    def fit(self, X, y, kg_embeddings=None):
        """训练模型"""
        if kg_embeddings is not None:
            X = self.enhance_features(X, kg_embeddings)

        if self.model is None:
            self.build_model(X.shape[1], len(np.unique(y)))

        self.model.fit(X, y)
        return self

    def predict(self, X, kg_embeddings=None):
        """预测"""
        if kg_embeddings is not None:
            X = self.enhance_features(X, kg_embeddings)
        return self.model.predict(X)

    def predict_proba(self, X, kg_embeddings=None):
        """预测概率"""
        if kg_embeddings is not None:
            X = self.enhance_features(X, kg_embeddings)
        return self.model.predict_proba(X)

    def evaluate(self, X, y, kg_embeddings=None):
        """评估模型"""
        y_pred = self.predict(X, kg_embeddings)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }

        return metrics, y_pred

    def save_model(self, path):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'fault_to_idx': self.fault_to_idx,
            'config': self.model_config
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.fault_to_idx = data['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")


class HybridPredictor:
    """混合预测器：结合KG增强的GNN和RF"""

    def __init__(self, gnn_model, rf_model):
        self.gnn_model = gnn_model
        self.rf_model = rf_model

    def predict_with_ensemble(self, X, kg_embeddings, edge_index=None, batch=None):
        """
        使用集成方法预测
        """
        # RF预测
        rf_proba = self.rf_model.predict_proba(X, kg_embeddings)

        # GNN预测（如果提供图数据）
        if edge_index is not None and batch is not None:
            # 需要将数据转换为张量
            device = next(self.gnn_model.parameters()).device
            x_tensor = torch.tensor(X, dtype=torch.float).to(device)
            edge_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)
            batch_tensor = torch.tensor(batch, dtype=torch.long).to(device)
            kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(device)

            with torch.no_grad():
                gnn_out = self.gnn_model(x_tensor, edge_tensor, batch_tensor, kg_tensor)
                gnn_proba = torch.softmax(gnn_out, dim=1).cpu().numpy()

            # 加权平均
            ensemble_proba = 0.5 * rf_proba + 0.5 * gnn_proba
        else:
            ensemble_proba = rf_proba

        return np.argmax(ensemble_proba, axis=1), ensemble_proba


def create_kg_enhanced_features(X, kg_embedding_path):
    """
    从知识图谱嵌入文件创建增强特征
    """
    import json

    with open(kg_embedding_path, 'r') as f:
        kg_data = json.load(f)

    structural_features = kg_data.get('structural_features', {})
    fault_similarity = kg_data.get('fault_similarity', {})

    # 构建知识增强特征
    kg_features = []

    # 节点计数特征
    if 'node_counts' in structural_features:
        for node_type in ['Fault', 'Component', 'Feature']:
            count = structural_features['node_counts'].get(node_type, 0)
            kg_features.append(count / 1000)

    # 度数特征
    if 'avg_degree' in structural_features:
        for node_type in ['Fault', 'Component', 'Feature']:
            degree = structural_features['avg_degree'].get(node_type, 0)
            kg_features.append(degree / 10)

    # 相似度特征
    if fault_similarity:
        sim_values = list(fault_similarity.values())
        kg_features.append(np.mean(sim_values) / 100 if sim_values else 0)
        kg_features.append(np.max(sim_values) / 100 if sim_values else 0)
        kg_features.append(np.min(sim_values) / 100 if sim_values else 0)
    else:
        kg_features.extend([0, 0, 0])

    # 填充到固定长度
    target_len = 20
    while len(kg_features) < target_len:
        kg_features.append(0)
    kg_features = kg_features[:target_len]

    # 为每个样本创建KG增强特征
    n_samples = len(X)
    kg_feature_matrix = np.tile(kg_features, (n_samples, 1))

    # 拼接
    X_enhanced = np.concatenate([X, kg_feature_matrix], axis=1)

    return X_enhanced
