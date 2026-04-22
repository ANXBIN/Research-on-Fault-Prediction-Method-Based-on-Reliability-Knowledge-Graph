#!/usr/bin/env python3
"""
MLP模型：普通MLP和知识图谱增强MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
import yaml
import joblib
import json


class PlainMLP(nn.Module):
    """普通MLP模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(PlainMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class KGEnhancedMLP(nn.Module):
    """知识图谱增强MLP模型"""

    def __init__(self, in_channels, kg_embedding_dim, hidden_channels,
                 out_channels, dropout=0.3):
        super(KGEnhancedMLP, self).__init__()

        # 原始特征投影
        self.feature_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # KG嵌入投影
        self.kg_net = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 加性融合 + 层归一化
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, kg_embedding):
        x_feat = self.feature_net(x)
        x_kg = self.kg_net(kg_embedding)

        # 加性融合
        x_fused = x_feat + x_kg

        # 拼接用于进一步变换
        combined = torch.cat([x_fused, x_feat, x_kg], dim=1)
        x_fused = self.fusion(combined)

        return self.classifier(x_fused)


class MLPModel:
    """MLP模型包装类"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['models'].get('mlp', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 128)
        self.dropout = model_cfg.get('dropout', 0.3)
        self.learning_rate = model_cfg.get('learning_rate', 0.001)
        self.epochs = model_cfg.get('epochs', 100)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = None
        self.optimizer = None
        self.fault_to_idx = {}

    def build_model(self, n_features, n_classes):
        """构建模型"""
        self.model = PlainMLP(
            in_channels=n_features,
            hidden_channels=self.hidden_dim,
            out_channels=n_classes,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model

    def train_epoch(self, X, y):
        """训练一个epoch"""
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        self.optimizer.zero_grad()
        out = self.model(X_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        loss.backward()
        self.optimizer.step()

        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)

        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, X, y):
        """评估模型"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        out = self.model(X_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)

        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)

        return {'loss': loss.item(), 'accuracy': acc}, pred.cpu().numpy()

    def predict(self, X):
        """预测"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        out = self.model(X_tensor)
        return out.argmax(dim=1).cpu().numpy()

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")


class KGEnhancedMLPModel:
    """知识图谱增强MLP模型包装类"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['models'].get('kg_enhanced_mlp', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 128)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 64)
        self.dropout = model_cfg.get('dropout', 0.3)
        self.learning_rate = model_cfg.get('learning_rate', 0.001)
        self.epochs = model_cfg.get('epochs', 100)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = None
        self.optimizer = None
        self.fault_to_idx = {}

    def build_model(self, n_features, n_classes):
        """构建模型"""
        self.model = KGEnhancedMLP(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            hidden_channels=self.hidden_dim,
            out_channels=n_classes,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model

    def train_epoch(self, X, y, kg_embeddings):
        """训练一个epoch"""
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        self.optimizer.zero_grad()
        out = self.model(X_tensor, kg_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        loss.backward()
        self.optimizer.step()

        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)

        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings):
        """评估模型"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        out = self.model(X_tensor, kg_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)

        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)

        return {'loss': loss.item(), 'accuracy': acc}, pred.cpu().numpy()

    def predict(self, X, kg_embeddings):
        """预测"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        out = self.model(X_tensor, kg_tensor)
        return out.argmax(dim=1).cpu().numpy()

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")


def build_knn_graph_features(X, k=10):
    """
    基于特征的K近邻图构建样本嵌入

    每个样本的嵌入 = [归一化原始特征(n_features), 邻居特征加权和(n_features)]
    填充到64维
    """
    n_samples = len(X)
    n_features = X.shape[1]
    k = min(k, max(1, n_samples - 1))  # 确保k至少为1

    # 归一化特征
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    row_norms = np.linalg.norm(X_norm, axis=1)
    zero_rows = row_norms < 1e-8
    if np.any(zero_rows):
        X_norm[zero_rows] = 1e-8

    # KNN找最近邻
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn_model.fit(X_norm)
    distances, indices = nn_model.kneighbors(X_norm)

    # 构建64维嵌入: 前20维=归一化特征，中间20维=邻居加权，后24维=0填充
    kg_embeddings = np.zeros((n_samples, 64), dtype=np.float32)

    for i in range(n_samples):
        neighbor_indices = indices[i, 1:]  # 排除自己
        neighbor_features = X_norm[neighbor_indices]  # shape: (k, n_features)

        # 距离倒数加权
        dists = np.maximum(distances[i, 1:], 1e-8)
        weights = 1.0 / dists
        weights = weights / weights.sum()

        # 加权平均 -> shape (n_features,)
        neighbor_weighted = np.dot(weights, neighbor_features)  # (n_features,)

        # 前20维: 归一化原始特征
        kg_embeddings[i, :n_features] = X_norm[i]
        # 中间20维: 邻居特征加权和
        kg_embeddings[i, n_features:2*n_features] = neighbor_weighted

    return kg_embeddings


def load_kg_embeddings_v3(kg_embed_path, n_samples, fault_labels, X_train, X_test=None, fault_types=None, fault_to_idx=None):
    """
    纯知识图谱嵌入 - 只使用KG的全局结构信息，不基于样本特征
    """
    with open(kg_embed_path, 'r') as f:
        kg_data = json.load(f)

    structural_features = kg_data.get('structural_features', {})

    # 从KG获取全局结构特征
    node_counts = structural_features.get('node_counts', {})
    edge_counts = structural_features.get('edge_counts', {})

    # 全局KG特征 (6维)
    global_features = np.array([
        np.log1p(node_counts.get('Fault', 1)) / 20,
        np.log1p(node_counts.get('Component', 1)) / 5,
        np.log1p(node_counts.get('Feature', 1)) / 20,
        edge_counts.get('CAUSED_BY', 0) / 100,
        edge_counts.get('LOCATED_AT', 0) / 100,
        edge_counts.get('HAS_FEATURE', 0) / 500,
    ], dtype=np.float32)

    # 填充到64维: 前6维是KG全局特征，后58维用不同的随机投影填充
    np.random.seed(42)  # 固定种子保证可重复性
    random_projection = np.random.randn(6, 58).astype(np.float32) * 0.01

    # 最终嵌入 = 全局特征 @ 随机投影
    kg_final = np.zeros((n_samples, 64), dtype=np.float32)
    kg_final[:, :6] = global_features

    # 广播global_features到所有样本，然后通过随机投影
    for i in range(n_samples):
        kg_final[i, 6:] = np.dot(global_features, random_projection)

    return kg_final